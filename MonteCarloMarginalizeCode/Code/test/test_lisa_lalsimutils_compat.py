#!/usr/bin/env python
"""Smoke tests for LISA lalsimutils compatibility helpers."""

import os
import tempfile

import h5py
import numpy as np

import RIFT.LISA.lalsimutils_compat as compat
import RIFT.lalsimutils as lalsimutils


def _write_cache(directory, h5_path):
    cache_path = os.path.join(directory, "frames.cache")
    with open(cache_path, "w") as cache_file:
        cache_file.write(f"A A 0 1 file://localhost{h5_path}\n")
    return cache_path


def test_frame_h5_to_hoff_reads_frequency_series():
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "fd.h5")
        with h5py.File(h5_path, "w") as h5_file:
            h5_file.attrs["epoch"] = 0.0
            h5_file.attrs["f0"] = 0.0
            h5_file.attrs["deltaF"] = 0.25
            h5_file.attrs["length"] = 4
            h5_file.create_dataset("data", data=np.array([1, 2, 3, 4], dtype=np.complex128))

        hoff = compat.frame_h5_to_hoff(_write_cache(tmpdir, h5_path), "A", verbose=False)

    assert hoff.data.length == 4
    assert hoff.deltaF == 0.25
    np.testing.assert_array_equal(hoff.data.data, np.array([1, 2, 3, 4], dtype=np.complex128))


def test_h5_frame_to_non_herm_hoff_reads_time_series():
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "td.h5")
        with h5py.File(h5_path, "w") as h5_file:
            h5_file.attrs["epoch"] = 0.0
            h5_file.attrs["f0"] = 0.0
            h5_file.attrs["deltaT"] = 0.5
            h5_file.attrs["length"] = 4
            h5_file.create_dataset("data", data=np.array([1, 0, 0, 0], dtype=np.float64))

        hoff = compat.frame_data_to_non_herm_hoff(
            _write_cache(tmpdir, h5_path),
            "A:TD",
            TDlen=-1,
            h5_frame=True,
            verbose=False,
        )

    assert hoff.data.length == 4
    assert hoff.deltaF == 0.5
    np.testing.assert_allclose(hoff.data.data, 0.5 * np.ones(4, dtype=np.complex128))


def test_lisa_print_params_installer_is_local_to_compat_call():
    compat.install_choose_waveform_print_params_lisa()

    assert lalsimutils.ChooseWaveformParams.print_params_lisa is compat.print_params_lisa
