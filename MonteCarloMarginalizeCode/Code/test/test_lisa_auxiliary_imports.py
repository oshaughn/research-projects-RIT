"""Smoke and contract tests for imported LISA auxiliary modules."""

import os

import numpy as np
import pytest


def test_lisa_auxiliary_modules_import_without_cli_side_effects():
    import RIFT.LISA.initial_grid.fisher_errors as fisher_errors
    import RIFT.LISA.injections.LISA_injections as lisa_injections
    import RIFT.LISA.injections.create_injections as create_injections
    import RIFT.LISA.psd_generation.generate_LISA_psd as generate_LISA_psd
    import RIFT.LISA.sangria_test.pycbc_to_rift as pycbc_to_rift
    import RIFT.LISA.utils.utils as lisa_utils

    assert hasattr(fisher_errors, "get_error_bounds")
    assert hasattr(lisa_injections, "generate_lisa_TDI_dict")
    assert hasattr(create_injections, "parameter_dict_from_xml")
    assert hasattr(generate_LISA_psd, "write_lisa_psd")
    assert hasattr(pycbc_to_rift, "create_injection_from_time_series")
    assert hasattr(pycbc_to_rift, "read_gwpy_channels")
    assert hasattr(lisa_utils, "SSB_to_LISA")


def test_lisa_sky_frame_round_trip():
    from RIFT.LISA.utils import utils as lisa_utils

    t_ssb = np.array([1024.0])
    lam = np.array([1.2])
    beta = np.array([0.3])
    psi = np.array([0.4])

    lisa_frame = lisa_utils.SSB_to_LISA(t_ssb, lam, beta, psi)
    ssb_frame = lisa_utils.LISA_to_SSB(
        lisa_frame[:, 0],
        lisa_frame[:, 1],
        lisa_frame[:, 2],
        lisa_frame[:, 3],
    )

    assert np.allclose(ssb_frame[:, 0], t_ssb, rtol=0, atol=1e-4)
    assert np.allclose(ssb_frame[:, 1], lam, rtol=0, atol=1e-10)
    assert np.allclose(ssb_frame[:, 2], beta, rtol=0, atol=1e-10)


def test_lisa_psd_generator_writes_small_ascii_products(tmp_path):
    from RIFT.LISA.psd_generation import generate_LISA_psd

    txt_path, png_path = generate_LISA_psd.write_lisa_psd(
        os.fspath(tmp_path),
        fmin=1.0e-4,
        fmax=1.0e-3,
        npts=16,
        write_xml=False,
    )

    psd = np.loadtxt(txt_path)
    assert psd.shape == (16, 2)
    assert np.all(np.isfinite(psd))
    assert np.all(psd[:, 1] > 0)
    assert os.path.exists(png_path)


def test_sangria_converter_prefers_gwpy_reader(monkeypatch):
    from RIFT.LISA.sangria_test import pycbc_to_rift

    calls = []

    def fake_read(frame_path, channel):
        calls.append((frame_path, channel))
        return channel

    monkeypatch.setattr(pycbc_to_rift, "_read_gwpy_frame", fake_read)
    assert pycbc_to_rift.read_gwpy_channels("frame.gwf", channels=("A", "E")) == {"A": "A", "E": "E"}
    assert calls == [("frame.gwf", "A"), ("frame.gwf", "E")]


def test_sangria_converter_reports_missing_optional_gwpy(monkeypatch):
    from RIFT.LISA.sangria_test import pycbc_to_rift

    def missing_gwpy():
        raise ImportError("gwpy is required to read Sangria frame files")

    monkeypatch.setattr(pycbc_to_rift, "_import_gwpy_timeseries", missing_gwpy)
    with pytest.raises(ImportError, match="gwpy is required"):
        pycbc_to_rift.read_gwpy_channels("missing.gwf")
