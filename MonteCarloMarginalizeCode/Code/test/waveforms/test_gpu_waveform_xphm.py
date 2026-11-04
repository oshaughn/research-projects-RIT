"""Known-answer gate for RIFT.likelihood.gpu_waveform_xphm against LALSimulation.

Compares, point by point over 50 random precessing binaries (Mtot 10-150 Msun,
q 0.1-1, |chi| <= 0.95 isotropic, fmin = fref = 20 Hz):

* the five J-frame (2,m) modes with ``SimInspiralChooseFDModes`` for
  IMRPhenomXPHM, PrecVersion 102, co-precessing ModeArray (2,+-2), no
  multibanding;
* h+ and hx with ``SimInspiralChooseFDWaveform`` for IMRPhenomXP, PrecVersion 102;
* the aligned-spin (2,2) mode with ``SimIMRPhenomXASGenerateFD``.

Mismatches are maximized over time and phase with the aLIGO zero-detuned
high-power PSD.  Run with ``-s`` to print the measured distributions.  The
NumPy backend is tested everywhere; the CuPy check runs only where a GPU is
usable (set CUDA_PATH for CuPy's headers).
"""

import math

import numpy as np
import pytest

lal = pytest.importorskip("lal")
lalsim = pytest.importorskip("lalsimulation")

from RIFT.likelihood import gpu_waveform_xphm as X

DELTA_F = 1.0 / 16.0
F_MIN = 20.0
F_REF = 20.0
F_MAX = 2048.0
N_FREQ = int(F_MAX / DELTA_F) + 1
N_POINTS = 50
DIST = 500.0e6 * 3.085677581491367e16

MISMATCH_TOL = 1e-6      # target 1e-4; measured <= 1.2e-15 (the round-off floor of 1 - match)
AMP_REL_TOL = 1e-9       # measured <= 6.4e-12


def _random_points(n, seed, precessing=True):
    rng = np.random.default_rng(seed)
    mtot = rng.uniform(10.0, 150.0, n)
    q = rng.uniform(0.1, 1.0, n)
    m1 = mtot / (1.0 + q)
    m2 = mtot - m1

    def spins():
        mag = rng.uniform(0.0, 0.95, n)
        if not precessing:
            return np.stack([np.zeros(n), np.zeros(n), mag * rng.choice([-1.0, 1.0], n)], axis=-1)
        cos_t = rng.uniform(-1.0, 1.0, n)
        ph = rng.uniform(0.0, 2 * np.pi, n)
        sin_t = np.sqrt(1.0 - cos_t ** 2)
        return mag[:, None] * np.stack([sin_t * np.cos(ph), sin_t * np.sin(ph), cos_t], axis=-1)

    chi1, chi2 = spins(), spins()
    # Present a quarter of the points with m1 < m2 to exercise LAL's relabelling.
    flip = rng.uniform(size=n) < 0.25
    m1f = np.where(flip, m2, m1)
    m2f = np.where(flip, m1, m2)
    chi1f = np.where(flip[:, None], chi2, chi1)
    chi2f = np.where(flip[:, None], chi1, chi2)
    incl = np.arccos(rng.uniform(-1.0, 1.0, n))
    phi = rng.uniform(0.0, 2 * np.pi, n)
    return m1f, m2f, chi1f, chi2f, incl, phi


def _dict(prec_version=102):
    d = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d, prec_version)
    return d


def _lal_modes(m1, m2, c1, c2, incl, phi):
    d = _dict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(d, 0.0)
    ma = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateMode(ma, 2, 2)
    lalsim.SimInspiralModeArrayActivateMode(ma, 2, -2)
    lalsim.SimInspiralWaveformParamsInsertModeArray(d, ma)
    mj = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateAllModesAtL(mj, 2)
    lalsim.SimInspiralWaveformParamsInsertModeArrayJframe(d, mj)
    hlms = lalsim.SimInspiralChooseFDModes(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *c1, *c2, DELTA_F, F_MIN, F_MAX, F_REF,
        phi, DIST, incl, d, lalsim.IMRPhenomXPHM)
    out = []
    for m in range(-2, 3):
        h = lalsim.SphHarmFrequencySeriesGetMode(hlms, 2, m).data.data
        c = len(h) // 2
        out.append(h[c - (N_FREQ - 1): c + N_FREQ])
    return np.array(out)


def _lal_polarizations(m1, m2, c1, c2, incl, phi):
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *c1, *c2, DIST, incl, phi, 0.0, 0.0, 0.0,
        DELTA_F, F_MIN, F_MAX, F_REF, _dict(), lalsim.IMRPhenomXP)
    return hp.data.data[:N_FREQ], hc.data.data[:N_FREQ]


def _psd_one_sided():
    f = np.arange(N_FREQ) * DELTA_F
    S = np.full(N_FREQ, np.inf)
    band = f >= F_MIN
    S[band] = [lalsim.SimNoisePSDaLIGOZeroDetHighPower(x) for x in f[band]]
    return 1.0 / S


INV_PSD = None


def _inv_psd():
    global INV_PSD
    if INV_PSD is None:
        INV_PSD = _psd_one_sided()
    return INV_PSD


def _mismatch_one_sided(a, b):
    w = _inv_psd()
    z = np.fft.ifft(np.conj(a) * b * w, n=4 * len(a))
    norm = math.sqrt(np.sum(np.abs(a) ** 2 * w) * np.sum(np.abs(b) ** 2 * w))
    return 1.0 - np.max(np.abs(z)) * len(z) / norm


def _mismatch_two_sided(a, b):
    w1 = _inv_psd()
    w = np.concatenate([w1[:0:-1], w1])
    prod = np.conj(a) * b * w
    z = np.fft.ifft(np.fft.ifftshift(np.concatenate([prod, [0.0]])), n=4 * (len(prod) + 1))
    norm = math.sqrt(np.sum(np.abs(a) ** 2 * w) * np.sum(np.abs(b) ** 2 * w))
    return 1.0 - np.max(np.abs(z)) * len(z) / norm


def _max_rel(a, b):
    support = np.abs(a) > 1e-8 * np.max(np.abs(a))
    return float(np.max(np.abs(a[support] - b[support]) / np.abs(a[support])))


@pytest.fixture(scope="module")
def precessing_case():
    m1, m2, c1, c2, incl, phi = _random_points(N_POINTS, 20261018)
    modes = X.imrphenomxp_nnlo_modes(m1, m2, c1, c2, DIST, phi, F_MIN, F_REF,
                                     DELTA_F, N_FREQ, f_max=F_MAX, inclination=incl, xp=np)
    hp, hc = X.imrphenomxp_nnlo_polarizations(m1, m2, c1, c2, DIST, incl, phi, F_MIN, F_REF,
                                              DELTA_F, N_FREQ, f_max=F_MAX, xp=np)
    return dict(params=(m1, m2, c1, c2, incl, phi), modes=modes, hp=hp, hc=hc)


def _summary(label, values):
    v = np.asarray(values)
    print("%s: median %.2e  p90 %.2e  max %.2e" % (label, np.median(v), np.percentile(v, 90), np.max(v)))


def test_modes_match_lal_xphm_prec102(precessing_case):
    m1, m2, c1, c2, incl, phi = precessing_case["params"]
    mism, rel = [], []
    for i in range(N_POINTS):
        ref = _lal_modes(m1[i], m2[i], c1[i], c2[i], incl[i], phi[i])
        got = precessing_case["modes"][i]
        assert got.shape == ref.shape
        np.testing.assert_array_equal(np.abs(got) > 0, np.abs(ref) > 0)
        for k in range(5):
            if np.max(np.abs(ref[k])) == 0.0:
                continue
            # A (2,m) mode can be ~1e-6 of the dominant one; judge it on its own norm.
            mism.append(_mismatch_two_sided(ref[k], got[k]))
            rel.append(_max_rel(ref[k], got[k]))
    _summary("\nJ-frame mode mismatch", mism)
    _summary("J-frame mode max rel error", rel)
    assert max(mism) < MISMATCH_TOL
    assert max(rel) < AMP_REL_TOL


def test_polarizations_match_lal_xp_prec102(precessing_case):
    m1, m2, c1, c2, incl, phi = precessing_case["params"]
    mism, rel = [], []
    for i in range(N_POINTS):
        hp_ref, hc_ref = _lal_polarizations(m1[i], m2[i], c1[i], c2[i], incl[i], phi[i])
        for ref, got in ((hp_ref, precessing_case["hp"][i]), (hc_ref, precessing_case["hc"][i])):
            mism.append(_mismatch_one_sided(ref, got))
            rel.append(_max_rel(ref, got))
    _summary("\nh+/hx mismatch", mism)
    _summary("h+/hx max rel error", rel)
    assert max(mism) < MISMATCH_TOL
    assert max(rel) < AMP_REL_TOL


def test_aligned_spin_xas_matches_lal():
    m1, m2, c1, c2, incl, phi = _random_points(20, 7, precessing=False)
    got = X.imrphenomxas_h22(m1, m2, c1[:, 2], c2[:, 2], DIST, F_MIN, F_REF, phi,
                             DELTA_F, N_FREQ, f_max=F_MAX, xp=np)
    mism, rel = [], []
    for i in range(len(m1)):
        ref = lalsim.SimIMRPhenomXASGenerateFD(
            m1[i] * lal.MSUN_SI, m2[i] * lal.MSUN_SI, c1[i, 2], c2[i, 2], DIST,
            F_MIN, F_MAX, DELTA_F, phi[i], F_REF, None).data.data[:N_FREQ]
        mism.append(_mismatch_one_sided(ref, got[i]))
        rel.append(_max_rel(ref, got[i]))
    _summary("\nXAS h22 mismatch", mism)
    _summary("XAS h22 max rel error", rel)
    assert max(mism) < MISMATCH_TOL
    assert max(rel) < AMP_REL_TOL


def test_batch_equals_single_point(precessing_case):
    m1, m2, c1, c2, incl, phi = precessing_case["params"]
    for i in (0, 7, 31):
        one = X.imrphenomxp_nnlo_modes(m1[i:i + 1], m2[i:i + 1], c1[i:i + 1], c2[i:i + 1], DIST,
                                       phi[i:i + 1], F_MIN, F_REF, DELTA_F, N_FREQ, f_max=F_MAX,
                                       inclination=incl[i:i + 1], xp=np)[0]
        np.testing.assert_allclose(one, precessing_case["modes"][i], rtol=1e-13, atol=0)


def test_wigner_D_matches_lal():
    rng = np.random.default_rng(3)
    a, b, g = rng.uniform(-np.pi, np.pi, 3)
    D = X.wigner_D_l2(a, b, g, xp=np)[0]
    for i, mp in enumerate(range(-2, 3)):
        for j, m in enumerate(range(-2, 3)):
            assert abs(D[i, j] - lal.WignerDMatrix(2, mp, m, a, b, g)) < 1e-14


def test_lframe_rotation_matches_rift_rotate_hlm_static():
    lsu = pytest.importorskip("RIFT.lalsimutils")
    m1, m2, c1, c2, incl, phi = _random_points(1, 11)
    P = lsu.ChooseWaveformParams()
    P.m1, P.m2 = m1[0] * lal.MSUN_SI, m2[0] * lal.MSUN_SI
    P.s1x, P.s1y, P.s1z = c1[0]
    P.s2x, P.s2y, P.s2z = c2[0]
    P.incl, P.phiref, P.fref, P.fmin = incl[0], phi[0], F_REF, F_MIN
    angles = X.rift_jl_angles([P])
    modes = X.imrphenomxp_nnlo_modes(m1, m2, c1, c2, DIST, phi, F_MIN, F_REF, DELTA_F, 2049,
                                     f_max=F_MAX, inclination=incl, xp=np)
    got = X.jframe_to_rift_lframe(modes, angles, xp=np)[0]
    # rotate_hlm_static only accepts lal TimeSeries (its FrequencySeries/ndarray
    # branches reference a non-existent lal.Complex16FrequencySeries); the
    # container type does not affect the arithmetic.
    hlm = {}
    for k, m in enumerate(range(-2, 3)):
        ts = lal.CreateCOMPLEX16TimeSeries("h", lal.LIGOTimeGPS(0), 0.0, 1.0, lal.DimensionlessUnit,
                                           modes.shape[-1])
        ts.data.data[:] = modes[0, k]
        hlm[(2, m)] = ts
    al, be, ga = angles[0]
    ref = lsu.rotate_hlm_static(hlm, -ga - np.pi / 2, -be, -al, extra_polarization=np.pi / 2)
    for k, m in enumerate(range(-2, 3)):
        want = ref[(2, m)].data.data
        np.testing.assert_allclose(got[k], want, rtol=1e-12, atol=1e-12 * np.max(np.abs(want)))


def _usable_cupy():
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
        cp.arange(3).sum()
        return cp
    except Exception:
        return None


def test_cupy_matches_numpy(precessing_case):
    cp = _usable_cupy()
    if cp is None:
        pytest.skip("no usable CuPy/GPU on this host")
    m1, m2, c1, c2, incl, phi = precessing_case["params"]
    modes = X.imrphenomxp_nnlo_modes(cp.asarray(m1), cp.asarray(m2), cp.asarray(c1), cp.asarray(c2),
                                     DIST, cp.asarray(phi), F_MIN, F_REF, DELTA_F, N_FREQ,
                                     f_max=F_MAX, inclination=cp.asarray(incl), xp=cp)
    hp, hc = X.imrphenomxp_nnlo_polarizations(cp.asarray(m1), cp.asarray(m2), cp.asarray(c1),
                                              cp.asarray(c2), DIST, cp.asarray(incl), cp.asarray(phi),
                                              F_MIN, F_REF, DELTA_F, N_FREQ, f_max=F_MAX, xp=cp)
    ref = precessing_case["modes"]
    scale = np.max(np.abs(ref), axis=-1, keepdims=True)
    err = np.max(np.abs(cp.asnumpy(modes) - ref) / scale)
    print("\nCuPy vs NumPy modes: max |diff|/max|mode| = %.2e" % err)
    assert err < 1e-11
    for got, want in ((hp, precessing_case["hp"]), (hc, precessing_case["hc"])):
        s = np.max(np.abs(want), axis=-1, keepdims=True)
        assert np.max(np.abs(cp.asnumpy(got) - want) / s) < 1e-11
