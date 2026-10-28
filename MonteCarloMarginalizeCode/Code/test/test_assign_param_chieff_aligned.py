"""assign_param('chieff_aligned', x) must set xi to x and hold chiminus = (m2 s1z - m1 s2z)/M fixed.

Covers unequal and exactly equal masses, the no-change identity, repeated assignment, and a
near-zero-factor (1e-8; 0 is singular) util_ParameterPuffball run in (mc, delta_mc, chieff_aligned), which must return its input.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from RIFT import lalsimutils as lsu

N = 50


def _chiminus(P):
    return (P.m2*P.s1z - P.m1*P.s2z)/(P.m1 + P.m2)


def _truth(seed, equal_mass=False):
    rng = np.random.default_rng(seed)
    Ps = []
    for _ in range(N):
        P = lsu.ChooseWaveformParams()
        m1, m2 = np.sort(rng.uniform(2, 60, 2))[::-1]
        if equal_mass:
            m2 = m1
        P.m1, P.m2 = m1*lsu.lsu_MSUN, m2*lsu.lsu_MSUN
        P.s1z, P.s2z = rng.uniform(-0.9, 0.9, 2)
        Ps.append(P)
    return Ps


@pytest.mark.parametrize("equal_mass", [False, True])
def test_identity(equal_mass):
    for P in _truth(1, equal_mass):
        s1z, s2z = P.s1z, P.s2z
        P.assign_param('chieff_aligned', P.extract_param('xi'))
        assert np.allclose([P.s1z, P.s2z], [s1z, s2z], rtol=0, atol=1e-12)


@pytest.mark.parametrize("equal_mass", [False, True])
def test_requested_value_and_held_chiminus(equal_mass):
    rng = np.random.default_rng(2)
    for P in _truth(3, equal_mass):
        dm = _chiminus(P)
        x = rng.uniform(-0.8, 0.8)
        P.assign_param('chieff_aligned', x)
        assert P.extract_param('xi') == pytest.approx(x, abs=1e-12)
        assert _chiminus(P) == pytest.approx(dm, abs=1e-12)


def test_repeated_assignment_is_idempotent():
    for P in _truth(4):
        P.assign_param('chieff_aligned', 0.3)
        once = (P.s1z, P.s2z)
        P.assign_param('chieff_aligned', 0.3)
        assert np.allclose([P.s1z, P.s2z], once, rtol=0, atol=1e-12)


def test_precessing_spins_keep_inplane_components():
    rng = np.random.default_rng(6)
    for P in _truth(7):
        P.s1x, P.s1y, P.s2x, P.s2y = rng.uniform(-0.3, 0.3, 4)
        inplane = (P.s1x, P.s1y, P.s2x, P.s2y)
        dm = _chiminus(P)
        P.assign_param('chieff_aligned', 0.1)
        assert P.extract_param('xi') == pytest.approx(0.1, abs=1e-12)
        assert _chiminus(P) == pytest.approx(dm, abs=1e-12)
        assert (P.s1x, P.s1y, P.s2x, P.s2y) == inplane


def _puff(tmp_path, factor, extra=(), seed=5, m2_range=(20, 30), pair=('mc', 'delta_mc'), ecc=False, inplane=False):
    rng = np.random.default_rng(seed)
    Ps = []
    for _ in range(N*4):
        P = lsu.ChooseWaveformParams()
        P.m1, P.m2 = rng.uniform(34, 38)*lsu.lsu_MSUN, rng.uniform(*m2_range)*lsu.lsu_MSUN
        P.s1z, P.s2z = rng.uniform(-0.9, 0.9, 2)
        if inplane:
            P.s1x, P.s1y, P.s2x, P.s2y = rng.uniform(-0.3, 0.3, 4)
        P.fmin = P.fref = 20.
        if ecc:
            P.eccentricity, P.meanPerAno = rng.uniform(0.05, 0.2), rng.uniform(0, 2*np.pi)
        Ps.append(P)
    lsu.ChooseWaveformParams_array_to_xml(Ps, fname=str(tmp_path/"in"))
    exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "bin", "util_ParameterPuffball.py")
    cmd = [sys.executable, exe, "--inj-file", str(tmp_path/"in.xml.gz"), "--inj-file-out", str(tmp_path/"out"),
           "--puff-factor", str(factor), "--fmin", "20", "--fref", "20", "--random-seed", "7",
           ] + sum([["--parameter", p] for p in pair], []) + ([] if "chieff_aligned" in pair else ["--parameter", "chieff_aligned"]) + list(extra)
    if ecc:
        cmd += ["--parameter", "eccentricity", "--parameter", "meanPerAno", "--eccentric-phase-coordinates"]
    proc = subprocess.run(cmd, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-3000:]
    return Ps, lsu.xml_to_ChooseWaveformParams_array(str(tmp_path/"out.xml.gz")), proc.stdout


def test_puffball_tiny_factor_identity(tmp_path):
    Ps, Pout, log = _puff(tmp_path, 1e-8)   # 0 makes the covariance singular
    assert len(Pout) == len(Ps), log[-3000:]
    got = np.array([[p.s1z, p.s2z] for p in Pout])
    want = np.array([[p.s1z, p.s2z] for p in Ps])
    assert np.allclose(got, want, rtol=0, atol=1e-6)


def test_puffball_spin_downselect_uses_written_spins(tmp_path):
    cap = 0.5
    Ps, Pout, log = _puff(tmp_path, 1, ["--downselect-parameter", "chi1", "--downselect-parameter-range", "[0,%g]" % cap,
                                        "--downselect-parameter", "chi2", "--downselect-parameter-range", "[0,%g]" % cap,
                                        "--downselect-parameter", "s2z", "--downselect-parameter-range", "[-0.3,%g]" % cap,
                                        "--downselect-parameter", "mc_ecc", "--downselect-parameter-range", "[1,100]"])
    assert len(Pout) > 0, log[-3000:]
    chi = np.array([[p.extract_param('chi1'), p.extract_param('chi2')] for p in Pout])
    assert (chi <= cap + 1e-12).all()
    assert "Increment downselect : chi1" in log   # the vectorized spin test ran
    assert all(-0.3 - 1e-12 <= p.s2z for p in Pout)


def test_vectorized_inverse_and_spin_magnitudes_match_assign_param():
    rng = np.random.default_rng(8)
    Ps = _truth(9)
    for P in Ps:
        P.s1x, P.s1y, P.s2x, P.s2y = rng.uniform(-0.3, 0.3, 4)
    x = rng.uniform(-0.8, 0.8, N)
    m1, m2 = np.array([P.m1 for P in Ps])/lsu.lsu_MSUN, np.array([P.m2 for P in Ps])/lsu.lsu_MSUN
    sp = np.array([[P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z] for P in Ps])
    sp[:, 2], sp[:, 5] = lsu.chieff_aligned_spins(m1, m2, sp[:, 2], sp[:, 5], x)
    names = ['chi1', 'chi2', 's1z', 's2z', 'xi']
    got = lsu.convert_waveform_coordinates(np.c_[m1, m2, sp], coord_names=names,
                                           low_level_coord_names=['m1', 'm2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z'])
    for P, xv in zip(Ps, x):
        P.assign_param('chieff_aligned', xv)
    want = np.array([[P.extract_param(n) for n in names] for P in Ps])
    assert np.allclose(got, want, rtol=0, atol=1e-12)


@pytest.mark.parametrize("pair,ecc", [(["mc", "delta_mc"], False), (["mc", "eta"], False), (["m1", "m2"], False),
                                      (["mc", "delta_mc"], True), (["chieff_aligned", "mc", "delta_mc"], True)])
def test_puffball_vectorized_spin_downselect_matches_per_point(tmp_path, pair, ecc):
    # a wide puff, so some rows leave the delta_mc/eta range and stay per point;
    # in-plane spins and the chiz_minus cut make a wrong assignment order change the exported rows
    cuts = ["--downselect-parameter", "chi1", "--downselect-parameter-range", "[0,0.5]",
            "--downselect-parameter", "chi2", "--downselect-parameter-range", "[0,0.6]",
            "--downselect-parameter", "s1z", "--downselect-parameter-range", "[-0.4,0.45]",
            "--downselect-parameter", "xi", "--downselect-parameter-range", "[-0.3,0.3]",
            "--downselect-parameter", "chiz_minus", "--downselect-parameter-range", "[-0.2,0.2]"]
    def run(sub, extra):
        d = tmp_path/sub
        d.mkdir()
        Ps, Pout, log = _puff(d, 3, extra, m2_range=(10, 38), pair=pair, ecc=ecc, inplane=True)
        return np.array([[p.m1, p.m2, p.s1x, p.s1y, p.s1z, p.s2x, p.s2y, p.s2z] for p in Pout]), log
    vec, log_vec = run("vec", cuts)
    ref, _ = run("ref", cuts + ["--internal-scalar-spin-downselect"])
    assert ("Increment downselect : chi1" in log_vec) == (pair[0] != "chieff_aligned")
    assert len(vec) > 0 and vec.shape == ref.shape
    assert np.array_equal(vec, ref)
