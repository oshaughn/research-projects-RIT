import copy
import configparser
from pathlib import Path
from types import SimpleNamespace

import pytest

from RIFT.rimsky import (
    RimskyIntegrationError,
    build_analysis,
    normalize_event_metadata,
    write_analysis,
)


def _rimsky_config(tmp_path):
    return {
        "detectors": ["H1", "L1", "V1"],
        "output_dir": "online-output",
        "event_sink": {
            "bilby_pipe_defaults": {
                "minimum_frequency": 18,
                "maximum_frequency": {"H1": 1024, "L1": 1024, "V1": 896},
            }
        },
        "rift": {
            "name": "rift-low-latency",
            "waveform": {"approximant": "SEOBNRv5PHM"},
        },
    }


def test_build_analysis_targets_rimsky_pesummary_output(tmp_path):
    config_path = tmp_path / "configs" / "rimsky.yaml"
    analysis = build_analysis(_rimsky_config(tmp_path), config_path=config_path)

    expected = (
        config_path.parent
        / "online-output"
        / "*"
        / "*"
        / "{event}"
        / "results_page"
        / "metafile.hdf5"
    ).resolve()
    assert analysis["kind"] == "analysis"
    assert analysis["pipeline"] == "RIFT"
    assert analysis["name"] == "rift-low-latency"
    assert analysis["dataset"] == "bilby-online"
    assert analysis["scheduler"]["bootstrap file"] == str(expected)
    assert analysis["quality"]["minimum frequency"] == {
        "H1": 18,
        "L1": 18,
        "V1": 18,
    }
    assert analysis["quality"]["maximum frequency"]["V1"] == 896
    assert analysis["waveform"]["approximant"] == "SEOBNRv5PHM"


def test_normalize_rimsky_event_priors_is_additive():
    event = {
        "name": "S260305df",
        "interferometers": ["H1", "L1"],
        "likelihood": {"sample rate": 2048},
        "psds": {"H1": "/tmp/H1.txt", "L1": "/tmp/L1.txt"},
        "priors": {
            "chirp_mass": {"minimum": 10, "maximum": 20},
            "mass_ratio": {"minimum": 0.1, "maximum": 1},
            "luminosity_distance": {
                "minimum": 10,
                "maximum": 5000,
                "type": "bilby.gw.prior.UniformSourceFrame",
            },
            "a_1": {"minimum": 0, "maximum": 0.8},
            "a_2": {"minimum": 0, "maximum": 0.7},
        },
    }
    original = copy.deepcopy(event)
    normalized = normalize_event_metadata(event)

    assert event == original
    assert normalized["priors"]["chirp mass"] == event["priors"]["chirp_mass"]
    assert normalized["priors"]["mass ratio"] == event["priors"]["mass_ratio"]
    assert normalized["priors"]["luminosity distance"]["maximum"] == 5000
    assert normalized["priors"]["spin 1"] == {"maximum": 0.8}
    assert normalized["priors"]["spin 2"] == {"maximum": 0.7}
    assert "a_1" in normalized["priors"]
    assert normalized["psds"] == {2048: {"H1": "/tmp/H1.txt", "L1": "/tmp/L1.txt"}}


def test_normalize_does_not_replace_explicit_rift_prior():
    event = {"priors": {"chirp_mass": {"maximum": 20}, "chirp mass": {"maximum": 30}}}
    assert normalize_event_metadata(event)["priors"]["chirp mass"]["maximum"] == 30


def test_rift_pipeline_normalizes_rimsky_metadata_before_templating():
    from RIFT.asimov.rift import Rift

    pipeline = object.__new__(Rift)
    pipeline.production = SimpleNamespace(
        meta={
            "priors": {
                "chirp_mass": {"minimum": 10, "maximum": 20},
                "a_1": {"maximum": 0.8},
            },
            "likelihood": {},
        }
    )
    pipeline._create_ledger_entries()

    assert pipeline.production.meta["priors"]["chirp mass"]["maximum"] == 20
    assert pipeline.production.meta["priors"]["spin 1"]["maximum"] == 0.8
    assert pipeline.production.meta["sampler"] == {"cip": {}, "ile": {}}
    assert pipeline.production.meta["likelihood"] == {
        "assume": {},
        "marginalization": {},
    }


def test_rift_pipeline_builds_lal_caches_for_rimsky_frames(tmp_path):
    from RIFT.asimov.rift import Rift

    frames = []
    for start in (1456739148, 1456739152):
        frame = tmp_path / "S260305df-H1-{}-4.gwf".format(start)
        frame.touch()
        frames.append(str(frame))

    work_dir = tmp_path / "work"
    pipeline = object.__new__(Rift)
    pipeline.production = SimpleNamespace(
        name="rift-online",
        event=SimpleNamespace(work_dir=str(work_dir)),
        meta={"data": {"data files": {"H1": frames}}},
    )
    caches = pipeline._prepare_frame_caches()

    cache = Path(caches["H1"])
    assert cache == work_dir / "H1-rimsky.cache"
    lines = cache.read_text().splitlines()
    assert lines == [
        "H RIMSKY 1456739148 4 {}".format(Path(frames[0]).as_uri()),
        "H RIMSKY 1456739152 4 {}".format(Path(frames[1]).as_uri()),
    ]
    assert pipeline.production.meta["data"]["frame cache"] == caches


def test_rift_template_passes_generated_frame_caches():
    template = (
        Path(__file__).resolve().parents[1] / "RIFT" / "asimov" / "rift.ini"
    ).read_text()
    assert "fake-cache" in template
    assert "data['frame cache'][ifo]" in template


def test_generated_rimsky_analysis_renders_rift_template(tmp_path):
    liquid = pytest.importorskip("liquid")
    analysis = build_analysis(
        _rimsky_config(tmp_path), config_path=tmp_path / "rimsky.yaml"
    )
    event = {
        "engine": "RIFT",
        "interferometers": ["H1", "L1", "V1"],
        "data": {
            "segment length": 8,
            "channels": {ifo: "{}:STRAIN".format(ifo) for ifo in ("H1", "L1", "V1")},
            "frame types": {ifo: "gwf" for ifo in ("H1", "L1", "V1")},
            "frame cache": {
                ifo: "/tmp/{}-rimsky.cache".format(ifo) for ifo in ("H1", "L1", "V1")
            },
        },
        "likelihood": {"sample rate": 2048},
        "priors": {
            "chirp_mass": {"minimum": 10, "maximum": 20},
            "mass_ratio": {"minimum": 0.1, "maximum": 1},
            "luminosity_distance": {
                "minimum": 10,
                "maximum": 5000,
                "type": "bilby.gw.prior.UniformSourceFrame",
            },
            "a_1": {"maximum": 0.8},
            "a_2": {"maximum": 0.7},
        },
    }
    for key, value in analysis.items():
        if isinstance(value, dict) and isinstance(event.get(key), dict):
            event[key].update(copy.deepcopy(value))
        else:
            event[key] = copy.deepcopy(value)
    meta = normalize_event_metadata(event)

    production = SimpleNamespace(
        name=meta["name"],
        meta=meta,
        category="C01_offline",
        event=SimpleNamespace(name="S260305df", repository=None),
        xml_psds={
            ifo: "/tmp/{}-psd.xml.gz".format(ifo) for ifo in meta["interferometers"]
        },
    )
    context = {
        "production": production,
        "config": {
            "general": {"webroot": "/tmp/rift-web"},
            "pipelines": {"environment": "/opt/igwn"},
            "condor": {"user": "riftci"},
        },
    }
    template_text = (
        Path(__file__).resolve().parents[1] / "RIFT" / "asimov" / "rift.ini"
    ).read_text()
    if hasattr(liquid, "Environment"):
        rendered = liquid.Environment().from_string(template_text).render(**context)
    elif hasattr(liquid, "Liquid"):
        rendered = liquid.Liquid(template_text, from_file=False).render(**context)
    else:
        rendered = liquid.Template(template_text).render(**context)

    parser = configparser.RawConfigParser()
    parser.read_string(rendered)
    assert parser.get("engine", "chirpmass-min") == "10"
    assert parser.get("engine", "comp-max") == "1000"
    assert parser.get("engine", "a_spin1-max") == "0.8"
    assert '"V1":"/tmp/V1-rimsky.cache"' in parser.get("lalinference", "fake-cache")


def test_write_analysis_round_trips(tmp_path):
    yaml = pytest.importorskip("yaml")
    analysis = build_analysis(
        _rimsky_config(tmp_path), config_path=tmp_path / "rimsky.yaml"
    )
    destination = write_analysis(analysis, tmp_path / "rift-followup.yaml")
    assert yaml.safe_load(destination.read_text()) == analysis


@pytest.mark.parametrize("detectors", [[], {"H1": "bad"}, ["H1", 2]])
def test_invalid_detectors_fail_early(tmp_path, detectors):
    config = _rimsky_config(tmp_path)
    config["detectors"] = detectors
    with pytest.raises(RimskyIntegrationError, match="detector"):
        build_analysis(config, config_path=tmp_path / "rimsky.yaml")


def test_frequency_mapping_must_cover_all_detectors(tmp_path):
    config = _rimsky_config(tmp_path)
    config["event_sink"]["bilby_pipe_defaults"]["maximum_frequency"] = {"H1": 1024}
    with pytest.raises(RimskyIntegrationError, match="L1, V1"):
        build_analysis(config, config_path=tmp_path / "rimsky.yaml")
