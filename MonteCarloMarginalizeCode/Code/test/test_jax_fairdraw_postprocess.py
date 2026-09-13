"""Strict conversion of JAX-ILE tabular fair draws."""

import subprocess
import sys
import json
from pathlib import Path

import numpy as np


CODE = Path(__file__).resolve().parents[1]
CONVERTER = CODE / "bin" / "util_ConvertJAXILEFairdraws.py"


def _write_pair(directory, event, draws=2, spins=None):
    stem = "EXTR_out-{}.xml_0".format(event)
    spins = [0, 0, 0, 0, 0, 0] if spins is None else spins
    intrinsic = np.array([[event, 30.0, 20.0] + list(spins) + [
                           12.0, 0.1, 1000, 25]])
    np.savetxt(directory / (stem + "_.dat"), intrinsic)
    values = np.column_stack([
        np.linspace(1.0, 1.1, draws), np.linspace(0.2, 0.3, draws),
        np.linspace(300, 320, draws), np.linspace(0.4, 0.5, draws),
        np.linspace(0.6, 0.7, draws), np.linspace(0.8, 0.9, draws),
        np.linspace(10, 11, draws),
    ])
    np.savetxt(directory / (stem + "_samples.dat"), values,
               header="right_ascension declination distance inclination psi phi_orb loglikelihood")


def test_converter_joins_intrinsic_and_extrinsic_rows(tmpdir):
    tmp_path = Path(str(tmpdir))
    _write_pair(tmp_path, 0)
    _write_pair(tmp_path, 1)
    output = tmp_path / "posterior.dat"
    result = subprocess.run([
        sys.executable, str(CONVERTER), "--directory", str(tmp_path),
        "--draws-per-intrinsic", "2", "--expected-intrinsic", "2",
        "--output", str(output),
    ], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    assert " time " not in (" " + output.read_text().splitlines()[0] + " ")
    table = np.loadtxt(output)
    assert table.shape == (4, 25)
    assert np.allclose(table[:, 0], 30.0)
    assert np.allclose(table[:, 1], 20.0)
    assert np.allclose(table[:, 20], 25.0)
    provenance = json.loads((tmp_path / "posterior.provenance.json").read_text())
    assert provenance["intrinsic_points"] == 2
    assert provenance["posterior_rows"] == 4
    assert provenance["equal_weight_columns"] == {"p": 1.0, "ps": 1.0}
    assert provenance["omitted_unavailable_coordinates"] == [
        "time", "redshift", "source_frame_masses"]


def test_converter_refuses_wrong_draw_count(tmpdir):
    tmp_path = Path(str(tmpdir))
    _write_pair(tmp_path, 0, draws=1)
    output = tmp_path / "bad.dat"
    provenance = tmp_path / "bad.provenance.json"
    output.write_text("stale\n")
    provenance.write_text("stale\n")
    result = subprocess.run([
        sys.executable, str(CONVERTER), "--directory", str(tmp_path),
        "--draws-per-intrinsic", "2", "--output", str(output),
    ], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode != 0
    assert "expected 2 rows" in result.stderr
    assert not output.exists()
    assert not provenance.exists()


def test_converter_refuses_intrinsic_record_without_samples(tmpdir):
    tmp_path = Path(str(tmpdir))
    _write_pair(tmp_path, 0)
    _write_pair(tmp_path, 1)
    (tmp_path / "EXTR_out-1.xml_0_samples.dat").unlink()
    result = subprocess.run([
        sys.executable, str(CONVERTER), "--directory", str(tmp_path),
        "--draws-per-intrinsic", "2", "--output", str(tmp_path / "bad.dat"),
    ], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode != 0
    assert "records_without_samples" in result.stderr


def test_converter_refuses_missing_both_and_stale_excess_pairs(tmpdir):
    tmp_path = Path(str(tmpdir))
    _write_pair(tmp_path, 0)
    result = subprocess.run([
        sys.executable, str(CONVERTER), "--directory", str(tmp_path),
        "--expected-intrinsic", "2", "--output", str(tmp_path / "missing.dat"),
    ], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode != 0
    assert "missing=[1]" in result.stderr

    _write_pair(tmp_path, 1)
    _write_pair(tmp_path, 2)
    result = subprocess.run([
        sys.executable, str(CONVERTER), "--directory", str(tmp_path),
        "--expected-intrinsic", "2", "--output", str(tmp_path / "excess.dat"),
    ], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode != 0
    assert "excess=[2]" in result.stderr


def test_converter_computes_spin_summaries(tmpdir):
    tmp_path = Path(str(tmpdir))
    spins = [0.3, 0.4, 0.2, 0.0, 0.6, -0.1]
    _write_pair(tmp_path, 0, spins=spins)
    output = tmp_path / "spinning.dat"
    result = subprocess.run([
        sys.executable, str(CONVERTER), "--directory", str(tmp_path),
        "--draws-per-intrinsic", "2", "--expected-intrinsic", "1",
        "--output", str(output),
    ], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    table = np.loadtxt(output)
    assert np.allclose(table[:, 23], (30 * 0.2 + 20 * -0.1) / 50)
    q = 20.0 / 30.0
    expected_chip = max((2 + 1.5 * q) * 30**2 * 0.5,
                        (2 + 1.5 / q) * 20**2 * 0.6) / ((2 + 1.5 * q) * 30**2)
    assert np.allclose(table[:, 24], expected_chip)
