#!/usr/bin/env python3
"""
Unit tests for explicit bootstrap-source selection in RIFT.asimov.rift.

``scheduler: bootstrap file:`` lets a blueprint name the PESummary metafile to
bootstrap from directly, instead of routing it through an asimov dependency.
The dependency route only works for pipelines whose ``collect_assets()``
returns ``samples`` as a single path to a PESummary metafile; the bilby
pipeline returns a *list* of raw bilby result files, which used to fail
silently and leave the run with no bootstrap at all.

These tests drive ``Rift._resolve_bootstrap_file`` and ``Rift._dataset_label``
against a stub production, so they need no asimov project on disk.
"""

import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
# asimov registers RIFT.asimov.rift:Rift as an entry point and loads it while
# building known_pipelines; importing asimov first keeps that from re-entering
# our own in-flight import of the same module.
pytest.importorskip("asimov")
rift_asimov = pytest.importorskip("RIFT.asimov.rift")

Rift = rift_asimov.Rift
PipelineException = rift_asimov.PipelineException


class _StubEvent:
    def __init__(self, name):
        self.name = name


class _StubProduction:
    def __init__(self, meta, event="S250202cu", name="rift-SEOBNRv5PHM"):
        self.meta = meta
        self.name = name
        self.event = _StubEvent(event)


class _StubRift:
    """Bind the methods under test to a stub, bypassing Rift.__init__."""

    _PESUMMARY_RESERVED = Rift._PESUMMARY_RESERVED
    _resolve_bootstrap_file = Rift._resolve_bootstrap_file
    _dataset_label = Rift._dataset_label

    def __init__(self, meta, event="S250202cu"):
        self.production = _StubProduction(meta, event=event)
        self.logger = _NullLogger()


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _write_metafile(path, labels, n_samples=32):
    """A minimal PESummary-shaped metafile."""
    columns = ("mass_1", "mass_2", "chirp_mass", "luminosity_distance",
               "geocent_time", "iota", "psi", "ra", "dec", "phase",
               "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z")
    dtype = np.dtype([(name, float) for name in columns])
    samples = np.zeros(n_samples, dtype=dtype)
    with h5py.File(path, "w") as handle:
        handle.create_group("version")
        handle.create_group("history")
        for label in labels:
            handle.create_group(label).create_dataset("posterior_samples", data=samples)
    return str(path)


def _write_raw_bilby(path, n_samples=32):
    """A raw bilby result file: samples at the root, no analysis label."""
    with h5py.File(path, "w") as handle:
        handle.create_dataset("label", data=b"bilby-SEOBNRv5PHM")
        posterior = handle.create_group("posterior")
        for name in ("mass_1", "mass_2", "chirp_mass"):
            posterior.create_dataset(name, data=np.zeros(n_samples))
    return str(path)


# --- resolution -----------------------------------------------------------

def test_absent_setting_returns_none():
    assert _StubRift({"scheduler": {}})._resolve_bootstrap_file() is None
    assert _StubRift({})._resolve_bootstrap_file() is None


def test_explicit_path_is_returned(tmp_path):
    target = _write_metafile(tmp_path / "posterior_samples.h5", ["bilby-SEOBNRv5PHM"])
    stub = _StubRift({"scheduler": {"bootstrap file": target}})
    assert stub._resolve_bootstrap_file() == target


@pytest.mark.parametrize("token", ["{event}", "<event>"])
def test_event_substitution(tmp_path, token):
    event_dir = tmp_path / "S250202cu"
    event_dir.mkdir()
    target = _write_metafile(event_dir / "posterior_samples.h5", ["bilby-SEOBNRv5PHM"])
    template = os.path.join(str(tmp_path), token, "posterior_samples.h5")
    stub = _StubRift({"scheduler": {"bootstrap file": template}}, event="S250202cu")
    assert stub._resolve_bootstrap_file() == target


def test_analysis_substitution(tmp_path):
    analysis_dir = tmp_path / "rift-SEOBNRv5PHM"
    analysis_dir.mkdir()
    target = _write_metafile(analysis_dir / "posterior_samples.h5", ["x"])
    template = os.path.join(str(tmp_path), "{analysis}", "posterior_samples.h5")
    assert _StubRift({"scheduler": {"bootstrap file": template}})._resolve_bootstrap_file() == target


def test_unique_glob_is_accepted(tmp_path):
    target = _write_metafile(tmp_path / "posterior_samples.h5", ["x"])
    pattern = os.path.join(str(tmp_path), "*.h5")
    assert _StubRift({"scheduler": {"bootstrap file": pattern}})._resolve_bootstrap_file() == target


def test_ambiguous_glob_raises(tmp_path):
    _write_metafile(tmp_path / "a.h5", ["x"])
    _write_metafile(tmp_path / "b.h5", ["x"])
    stub = _StubRift({"scheduler": {"bootstrap file": os.path.join(str(tmp_path), "*.h5")}})
    with pytest.raises(PipelineException):
        stub._resolve_bootstrap_file()


def test_glob_matching_nothing_raises(tmp_path):
    stub = _StubRift({"scheduler": {"bootstrap file": os.path.join(str(tmp_path), "*.h5")}})
    with pytest.raises(PipelineException):
        stub._resolve_bootstrap_file()


def test_missing_file_raises_rather_than_falling_back(tmp_path):
    """An explicit request that cannot be honoured must never be silent."""
    stub = _StubRift({"scheduler": {"bootstrap file": str(tmp_path / "nope.h5")}})
    with pytest.raises(PipelineException):
        stub._resolve_bootstrap_file()


# --- label selection ------------------------------------------------------

def test_single_label_is_auto_derived(tmp_path):
    target = _write_metafile(tmp_path / "m.h5", ["bilby-SEOBNRv5PHM"])
    stub = _StubRift({"scheduler": {}})
    assert stub._dataset_label(target) == "bilby-SEOBNRv5PHM"


def test_explicit_dataset_is_honoured(tmp_path):
    target = _write_metafile(tmp_path / "m.h5", ["a", "b"])
    stub = _StubRift({"scheduler": {}, "dataset": "b"})
    assert stub._dataset_label(target) == "b"


def test_explicit_dataset_does_not_open_the_file():
    """
    Backwards compatibility: the previous code only opened the metafile when
    `dataset` was absent.  A ledger that pins a dataset must keep building even
    if the source file has since moved, so long as the grid already exists.
    """
    stub = _StubRift({"scheduler": {}, "dataset": "pinned"})
    assert stub._dataset_label("/nonexistent/never/opened.h5") == "pinned"


def test_ambiguous_labels_raise_without_explicit_dataset(tmp_path):
    target = _write_metafile(tmp_path / "m.h5", ["a", "b"])
    with pytest.raises(PipelineException):
        _StubRift({"scheduler": {}})._dataset_label(target)


def test_metafile_without_version_and_history_still_works(tmp_path):
    """
    The previous implementation used list.remove('version'), which raises
    ValueError when those groups are absent - swallowed by a bare except.
    """
    path = str(tmp_path / "m.h5")
    columns = np.dtype([("mass_1", float)])
    with h5py.File(path, "w") as handle:
        handle.create_group("only-label").create_dataset(
            "posterior_samples", data=np.zeros(4, dtype=columns)
        )
    assert _StubRift({"scheduler": {}})._dataset_label(path) == "only-label"


def test_raw_bilby_result_is_rejected_with_a_useful_message(tmp_path):
    """RIFT consumes PESummary metafiles, not raw bilby result files."""
    target = _write_raw_bilby(tmp_path / "raw_result.hdf5")
    with pytest.raises(PipelineException) as excinfo:
        _StubRift({"scheduler": {}})._dataset_label(target)
    assert "raw bilby" in str(excinfo.value).lower()
    assert "pesummary" in str(excinfo.value).lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
