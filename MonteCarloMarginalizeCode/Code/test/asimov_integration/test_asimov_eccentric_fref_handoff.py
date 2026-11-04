"""Current pe-configurator postprocessing -> Asimov config -> real RIFT DAG.

Only the ledger persistence boundary is a fixture. The installed external
plugin's after_completion(), Asimov's make_config(), and RIFT's workflow builder
run normally. No PE-configurator build_dag(), scheduler submission, or ILE runs.
"""
import configparser
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

CODE = Path(__file__).resolve().parents[2]
REPO = CODE.parents[1]
TEMPLATE = CODE / "RIFT/asimov/rift.ini"


def _base_meta():
    """Small event metadata shared by both supported RIFT branch fixtures."""
    return {
        "name": "rift-eccentric", "engine": "RIFT", "interferometers": ["H1", "L1"],
        "event time": 1240340820.676,
        "data": {
            "channels": {ifo: ifo + ":DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01" for ifo in ("H1", "L1")},
            "frame types": {ifo: ifo + "_HOFT_CLEAN_SUB60HZ_C01" for ifo in ("H1", "L1")},
            "segment length": 4,
        },
        "quality": {"minimum frequency": {"H1": 20, "L1": 20},
                    "maximum frequency": {"H1": 1700, "L1": 1700}},
        "likelihood": {"start frequency": 20, "sample rate": 1024, "assume": {},
                       "marginalization": {"distance": False}},
        "waveform": {"approximant": "SEOBNRv5EHM", "pn amplitude order": 5, "maximum mode": 4},
        "priors": {
            "spin 1": {"maximum": .99}, "spin 2": {"maximum": .99},
            "chirp mass": {"minimum": 33.4, "maximum": 233.4},
            "mass ratio": {"minimum": .05, "maximum": 1.},
            "mass 1": {"minimum": 1, "maximum": 1000},
            "luminosity distance": {"minimum": 10, "maximum": 20000,
                                    "type": "bilby.gw.prior.UniformSourceFrame"},
        },
        "sampler": {"cip": {"sampling method": "AV", "explode jobs": 5},
                    "ile": {"n eff": 12, "copies": 2, "sampling method": "AV",
                            "freezeadapt": False, "jobs per worker": 30}},
        "scheduler": {"accounting group": "ligo.dev.o4.cbc.pe.rift", "osg": False},
    }


def _render_context(meta):
    production = SimpleNamespace(name=meta["name"], meta=meta, category="C01_offline",
        event=SimpleNamespace(name="GW190426_190642",
            repository=SimpleNamespace(directory="/tmp/rift-asimov-repo")),
        xml_psds={ifo: f"/tmp/rift-asimov-repo/C01_offline/psds/psd_{ifo}.xml.gz"
                  for ifo in meta["interferometers"]})
    return {"production": production}


def _postprocess_recommendations(tmp_path, monkeypatch, explicit_reference):
    """Exercise the actual external plugin, including downstream inheritance."""
    pytest.importorskip("asimov")
    version = importlib.metadata.version("asimov")
    if tuple(int(x) for x in version.split(".")[:2]) < (0, 8):
        pytest.skip("the current Asimov postprocessing fixture requires Asimov >=0.8")
    pe = pytest.importorskip("peconfigurator.asimov")
    from asimov import analysis
    from RIFT.asimov.rift import Rift

    settings = configparser.ConfigParser()
    settings.read_dict({
        "general": {"webroot": str(tmp_path / "pages")},
        "pipelines": {"environment": str(Path(sys.executable).parent.parent)},
        "condor": {"user": "rift-ci"},
    })
    monkeypatch.setattr(pe, "config", settings)
    monkeypatch.setattr(analysis, "config", settings)

    meta = _base_meta()
    meta["name"] = "rift-eccentric"
    meta["waveform"].pop("gwsignal arguments", None)
    meta["waveform"]["approximant"] = "SEOBNRv5EHM"
    meta["likelihood"]["assume"] = {"eccentric": True, "nonprecessing": True}
    extra = {"eccentric-phase-sampling": True, "force-ecc-min": .01, "force-ecc-max": .3}
    if explicit_reference is not None:
        extra["eccentric-phase-reference-frequency"] = explicit_reference
    meta["sampler"]["extra eccentric arguments"] = extra
    meta["sampler"]["cip"]["fitting method"] = "rf"
    meta["sampler"]["cip"]["explode jobs auto"] = False
    meta["sampler"]["ile"]["sampling method"] = "AV"

    context = _render_context(deepcopy(meta))
    downstream = context["production"]
    event = downstream.event
    event.meta = deepcopy(meta)
    # Stored production metadata is the override relative to the event. In
    # particular it contains no stale start/reference-frequency override.
    event.meta["productions"] = [
        {"pe-settings": {}},
        {downstream.name: {"sampler": {"extra eccentric arguments": deepcopy(extra)}}},
    ]
    updates = []
    event.update_data = lambda: updates.append(True)
    upstream = SimpleNamespace(name="pe-settings", event=event, meta={},
                               dependencies=[], rundir=str(tmp_path / "pe-settings"))
    downstream.dependencies = [upstream.name]
    event.productions = [upstream, downstream]

    results = Path(upstream.rundir) / "results"
    results.mkdir(parents=True)
    recommendations = {"f_start": 15., "f_ref": 25., "srate": 1024, "seglen": 4,
                       "chirpmass_min": 21., "chirpmass_max": 42.}
    (results / "recommendations.json").write_text(json.dumps(recommendations))
    (results / "report.html").write_text("<html><body>Fixture recommendations</body></html>")
    pe.AsimovPipeline(upstream).after_completion()
    assert upstream.status == "uploaded" and updates
    for updated in (event.meta, downstream.meta):
        assert updated["likelihood"]["start frequency"] == 15.
        assert updated["waveform"]["reference frequency"] == 25.
    assert downstream.meta["sampler"]["extra eccentric arguments"] == extra

    # Use Asimov's actual configuration-generation method. Its persistence
    # object is minimal because this test does not create or update a ledger.
    downstream.pipeline = Rift.__new__(Rift)
    downstream.pipeline.production = downstream
    downstream.pipeline.config_template = str(TEMPLATE)
    ini = tmp_path / "asimov-rift.ini"
    analysis.Analysis.make_config(downstream, str(ini))
    parser = configparser.RawConfigParser()
    parser.optionxform = str
    parser.read(ini)
    assert parser.getfloat("engine", "fmin-template") == 15.
    assert parser.getfloat("engine", "fref") == 25.
    assert parser.getboolean("rift-pseudo-pipe", "eccentric-phase-sampling")
    if explicit_reference is None:
        assert not parser.has_option("rift-pseudo-pipe", "eccentric-phase-reference-frequency")
    else:
        assert parser.getfloat("rift-pseudo-pipe", "eccentric-phase-reference-frequency") == explicit_reference
    return ini


@pytest.mark.parametrize("explicit_reference", [None, 10.])
def test_current_peconfigurator_asimov_reference_reaches_workers(tmp_path, monkeypatch, explicit_reference):
    ini = _postprocess_recommendations(tmp_path, monkeypatch, explicit_reference)
    import lal
    from RIFT import lalsimutils
    from RIFT.misc import eccentric_phase_pipeline as pipeline

    point = lalsimutils.ChooseWaveformParams()
    point.m1 = 35 * lal.MSUN_SI
    point.m2 = 30 * lal.MSUN_SI
    point.eccentricity = .1
    point.meanPerAno = .2
    point.fref = 25.
    point.fmin = 15.
    lalsimutils.ChooseWaveformParams_array_to_xml(
        [point, point.manual_copy()], fname=str(tmp_path / "seed"), fref=25.)
    cache = tmp_path / "empty.cache"
    cache.write_text("")
    run = tmp_path / "run"
    env = dict(os.environ,
        PYTHONPATH=str(CODE) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        PATH=os.pathsep.join([str(CODE / "bin"), str(Path(sys.executable).parent), os.environ.get("PATH", "")]),
        OMP_NUM_THREADS="1", RIFT_LOWLATENCY="True", SINGULARITY_RIFT_IMAGE="foo",
        SINGULARITY_BASE_EXE_DIR="/usr/bin/")
    command = [sys.executable, str(CODE / "bin/util_RIFT_pseudo_pipe.py"),
        "--use-ini", str(ini), "--use-coinc", str(REPO / ".travis/ref_ini/coinc.xml"),
        "--use-rundir", str(run), "--fake-data-cache", str(cache),
        "--manual-initial-grid", str(tmp_path / "seed.xml.gz"), "--skip-reproducibility",
        "--ile-no-gpu"]
    result = subprocess.run(command, cwd=tmp_path, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=240)
    (tmp_path / "build.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-16000:]

    expected = 25. if explicit_reference is None else explicit_reference
    receipt = json.loads((run / "eccentric_phase_build.json").read_text())
    assert receipt["reference_frequency_hz"] == expected
    assert receipt["waveform_start_hz"] == 15.
    assert receipt["waveform_reference_hz"] == 25.
    assert receipt["native_epoch_source"] == ("waveform-reference" if explicit_reference is None else "explicit")

    def assert_reference(line):
        assert float(pipeline.values(pipeline.option_tokens(line), pipeline.REFERENCE)[-1]) == expected

    for filename in ("helper_cip_arg_list.txt", "args_cip_list.txt"):
        stages = (run / filename).read_text().splitlines()
        eligible = [line for line in stages if pipeline.eligible(line)]
        assert eligible
        for line in eligible:
            assert pipeline.SAMPLING in pipeline.option_tokens(line)
            assert_reference(line)
    for filename in ("helper_puff_args.txt", "args_puff.txt"):
        line = (run / filename).read_text()
        assert pipeline.SAMPLING not in pipeline.option_tokens(line)
        assert_reference(line)
    ile = (run / "args_ile.txt").read_text()
    tokens = pipeline.option_tokens(ile)
    assert float(pipeline.values(tokens, "--fmin-template")[-1]) == 15.
    assert float(pipeline.values(tokens, "--reference-freq")[-1]) == 25.
    assert all(flag in tokens for flag in ("--time-marginalization", "--vectorized", "--gpu", "--force-xpy"))
    assert list(run.glob("*.dag"))
