#!/usr/bin/env python3
"""Cross-version contract tests for the RIFT ASIMOV adapter."""

import os
import types

import pytest

pytest.importorskip("asimov")
rift_asimov = pytest.importorskip("RIFT.asimov.rift")

Rift = rift_asimov.Rift
PipelineException = rift_asimov.PipelineException


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _pipe(production):
    pipe = Rift.__new__(Rift)
    pipe.production = production
    pipe.category = production.category
    pipe.logger = _Logger()
    return pipe


def test_asimov_07_completion_defers_to_separate_postprocessing(monkeypatch):
    production = types.SimpleNamespace(
        status="processing", category="C01_offline", meta={"job id": 12}
    )
    pipe = _pipe(production)
    monkeypatch.setattr(rift_asimov, "PESummaryPipeline", None)

    pipe.after_completion()

    assert production.status == "finished"


def test_legacy_completion_submits_pesummary_once(monkeypatch):
    calls = []

    class _LegacyPESummary:
        def __init__(self, production, category=None):
            calls.append((production, category))

        def submit_dag(self):
            return 314

    production = types.SimpleNamespace(
        status="running", category="C01_offline", meta={}
    )
    pipe = _pipe(production)
    monkeypatch.setattr(rift_asimov, "PESummaryPipeline", _LegacyPESummary)

    pipe.after_completion()

    assert calls == [(production, "C01_offline")]
    assert production.meta["job id"] == 314
    assert production.status == "processing"


def test_collect_assets_publishes_pesummary_inputs(tmp_path):
    rundir = tmp_path / "run"
    rundir.mkdir()
    samples = rundir / "extrinsic_posterior_samples.dat"
    samples.write_text("# samples\n")
    config = tmp_path / "repository" / "C01_offline" / "rift.ini"
    config.parent.mkdir(parents=True)
    config.write_text("[analysis]\n")
    psd = tmp_path / "H1-psd.dat"
    psd.write_text("20 1e-46\n")
    calibration = tmp_path / "H1-calibration.dat"
    calibration.write_text("20 0 0\n")

    repository = types.SimpleNamespace(directory=str(tmp_path / "repository"))
    event = types.SimpleNamespace(name="S250202cu", repository=repository)
    production = types.SimpleNamespace(
        name="rift-SEOBNRv5PHM",
        category="C01_offline",
        rundir=str(rundir),
        event=event,
        psds={"H1": str(psd)},
        xml_psds={},
        meta={"data": {"calibration": {"H1": str(calibration)}}},
        get_configuration=lambda: types.SimpleNamespace(ini_loc="rift.ini"),
    )

    assets = _pipe(production).collect_assets(absolute=True)

    assert assets["asset_contract"] == "rift-assets/v1"
    assert assets["samples"] == [str(samples)]
    assert assets["config"] == str(config)
    assert assets["psds"] == {"H1": str(psd)}
    assert assets["calibration"] == {"H1": str(calibration)}
    assert assets["provenance"] == {
        "pipeline": "rift",
        "event": "S250202cu",
        "analysis": "rift-SEOBNRv5PHM",
    }


def test_reweighted_samples_keep_list_contract(tmp_path):
    rundir = tmp_path / "run"
    rundir.mkdir()
    reweighted = rundir / "reweighted_posterior_samples.dat"
    reweighted.write_text("# samples\n")
    event = types.SimpleNamespace(
        name="S250202cu", repository=types.SimpleNamespace(directory=str(tmp_path))
    )
    production = types.SimpleNamespace(
        name="rift-calmarg",
        category="C01_offline",
        rundir=str(rundir),
        event=event,
        psds={},
        meta={"data": {}},
        get_configuration=lambda: (_ for _ in ()).throw(ValueError()),
    )

    assets = _pipe(production).collect_assets(absolute=True)

    assert assets["samples"] == [str(reweighted)]
    assert assets["samples_calmarg"] == str(reweighted)


def test_asimov_07_psd_attributes_replace_legacy_getter():
    production = types.SimpleNamespace(
        category="C01_offline",
        psds={"H1": "/tmp/H1.dat"},
        xml_psds={"H1": "/tmp/H1.xml.gz"},
    )
    pipe = _pipe(production)

    assert pipe._get_psds("ascii") == production.psds
    assert pipe._get_psds("xml") == ["/tmp/H1.xml.gz"]


def test_single_sample_list_is_unwrapped_for_bootstrap(monkeypatch):
    dependency = types.SimpleNamespace(
        name="pesummary",
        pipeline=types.SimpleNamespace(
            collect_assets=lambda: {"samples": ["combined.h5"]}
        ),
    )
    event = types.SimpleNamespace(productions=[dependency])
    production = types.SimpleNamespace(
        name="rift-bootstrap",
        category="C01_offline",
        dependencies=["pesummary"],
        event=event,
        meta={"scheduler": {}},
    )
    pipe = _pipe(production)
    monkeypatch.setattr(pipe, "_dataset_label", lambda path: "rift-source")

    assert pipe._find_posterior() == "combined.h5"
    assert production.meta["dataset"] == "rift-source"


def test_multiple_sample_files_are_rejected_for_bootstrap():
    dependency = types.SimpleNamespace(
        name="pesummary",
        pipeline=types.SimpleNamespace(
            collect_assets=lambda: {"samples": ["a.h5", "b.h5"]}
        ),
    )
    event = types.SimpleNamespace(productions=[dependency])
    production = types.SimpleNamespace(
        name="rift-bootstrap",
        category="C01_offline",
        dependencies=["pesummary"],
        event=event,
        meta={"scheduler": {}},
    )

    with pytest.raises(PipelineException, match="exactly one PESummary metafile"):
        _pipe(production)._find_posterior()


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))
