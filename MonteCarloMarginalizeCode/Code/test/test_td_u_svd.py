"""
SVD-over-time compression of the time-indexed self term (td-gs code path).

compress_time_indexed_cross_terms(U3, V3, tol) stacks the pseudo-mode time series
of U(t), V(t) as M (npts_full x n_lm(n_lm+1)), takes its thin SVD and keeps the
components with s_k > tol s_0.  DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop
accepts the result (a TimeIndexedCrossTermsSVD) in place of the pair of 3-D
arrays.  These checks run on the CPU (xpy=np):

  (a) synthetic U(t), V(t) of exact low rank (1, 3, 8) and of full rank: the kept
      rank equals the true rank at tol 1e-13, and lnL / lnL(t) agree with the
      uncompressed NoLoop to ~1e-12 relative, for nearest and cubic time_interp,
      with and without phase marginalization, with mixed detectors (scalar,
      3-D, compressed), with a window past the buffer edge, and against the
      brute-force loop of test_td_time_indexed_U;
  (b) realistic U(t), V(t) from the td-gs Precompute (TDGSDetectorContext as in
      test_td_gs_route): full segment and a ringdown piece with l<=2
      IMRPhenomTPHM, and l<=4 NRSur7dq4 when the surrogate data are available;
      the singular-value spectra, r at tol 1e-10, 1e-12, 1e-13, 1e-14 and the
      lnL agreement at each are printed (run with -s) and checked;
  (c) refusals and edge cases: bad tol / shapes, all-zero cross terms (r = 0),
      mismatched objects, the calibration paths, the legacy array likelihoods,
      and the component-matrix identity.

A local, relative-only timing of the uncompressed vs compressed self term:
    python test/test_td_u_svd.py --profile

Run with:  pytest test/test_td_u_svd.py  (add -s to see the spectra)
"""
import os
import sys
import time

import numpy as np
import pytest
import lal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import RIFT.likelihood.factored_likelihood as fl
from test_td_time_indexed_U import _make_case, _brute_force, _PVec  # noqa: E402

try:
    from tdanalysis.forms.sliding import sliding_self_terms  # noqa: F401  (tdanalysis >= 1.1)
    from tdanalysis.covariance.psd import PSDToAutocorr  # noqa: F401
    HAVE_TDANALYSIS = True
except Exception:   # pragma: no cover
    HAVE_TDANALYSIS = False

TOLS = (1e-10, 1e-12, 1e-13, 1e-14)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _call(case, U, V, Lmax=2, **kw):
    return fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        case["tvals"], case["P"], case["lookupNKDict"], case["rholmsArrayDict"],
        U, V, case["epochDict"], Lmax=Lmax, xpy=np, **kw)


def _low_rank_case(seed, rank, modes=((2, 2), (2, -2), (2, 1)), npts_full=256, **kw):
    """_make_case with U(t) = sum_k a_k(t) A_k (A_k Hermitian), V(t) = sum_k a_k(t) B_k
    (B_k symmetric), REAL coefficient series a_k(t) shared by U and V, so U(t) is
    Hermitian, V(t) symmetric and the stacked pseudo-mode matrix has rank exactly
    `rank`.  (Complex coefficients would make the pair pseudo-modes U_ij + conj(U_ji)
    depend on conj(a_k) as well, doubling the rank.)  rank=None keeps the
    full-rank random U, V of _make_case."""
    case = _make_case(seed=seed, modes=modes, npts_full=npts_full, **kw)
    if rank is None:
        return case
    rng = np.random.default_rng(1000 + seed)
    n_lm = len(modes)
    for det in case["dets"]:
        a = np.abs(rng.standard_normal((npts_full, rank)))
        A = rng.standard_normal((rank, n_lm, n_lm)) + 1j*rng.standard_normal((rank, n_lm, n_lm))
        B = rng.standard_normal((rank, n_lm, n_lm)) + 1j*rng.standard_normal((rank, n_lm, n_lm))
        A = A + np.conj(np.swapaxes(A, 1, 2))
        B = B + np.swapaxes(B, 1, 2)
        floor = 2 * (np.sum(np.abs(A), axis=(1, 2)) + np.sum(np.abs(B), axis=(1, 2)))
        A = A + floor[:, None, None] * np.eye(n_lm)[None]
        case["U_t"][det] = np.einsum("tk,kij->tij", a, A)
        case["V_t"][det] = np.einsum("tk,kij->tij", a, B)
    return case


def _compress(case, tol, dets=None):
    comp = {}
    for det in (dets or case["dets"]):
        comp[det] = fl.compress_time_indexed_cross_terms(case["U_t"][det], case["V_t"][det], tol)
    return comp


def _max_rel(a, b):
    """max |a-b| / max(1, max|a|): a relative difference that stays meaningful
    when some lnL values are near zero.  Entries that are -inf on both sides
    (time marginalization underflow for a hopeless sample) count as equal; a
    mismatch in finiteness counts as infinite."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    fa = np.isfinite(a)
    fb = np.isfinite(b)
    if np.any(fa != fb) or np.any(a[~fa] != b[~fb]):
        return np.inf
    if not np.any(fa):
        return 0.0
    return float(np.max(np.abs(a[fa] - b[fb])) / max(1.0, np.max(np.abs(a[fa]))))


def _compare(case, comp, Lmax=2, rtol=1e-12, brute=True, **kw):
    U_ref, V_ref = case["U_t"], case["V_t"]
    lnL_ref = _call(case, U_ref, V_ref, Lmax=Lmax, **kw)
    lnL = _call(case, comp, comp, Lmax=Lmax, **kw)
    lnLt_ref = _call(case, U_ref, V_ref, Lmax=Lmax, return_lnLt=True, **kw)
    lnLt = _call(case, comp, comp, Lmax=Lmax, return_lnLt=True, **kw)
    assert lnLt.shape == (case["npts_extrinsic"], case["npts"])
    d_lnL = _max_rel(lnL_ref, lnL)
    d_lnLt = _max_rel(lnLt_ref, lnLt)
    assert d_lnL <= rtol, "lnL compressed vs uncompressed: %.3e" % d_lnL
    assert d_lnLt <= rtol, "lnL(t) compressed vs uncompressed: %.3e" % d_lnLt
    if brute:
        ref = _brute_force(case, U_ref, V_ref, **kw)
        np.testing.assert_allclose(lnL, ref, rtol=1e-10, atol=1e-10)
    return d_lnL, d_lnLt


# --------------------------------------------------------------------------
# (a) synthetic low-rank and full-rank U(t), V(t)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("time_interp", ['nearest', 'cubic'])
@pytest.mark.parametrize("rank", [1, 3, 8, None])
def test_rank_recovered_and_lnL_matches(rank, time_interp):
    case = _low_rank_case(seed=7 + (rank or 0), rank=rank)
    comp = _compress(case, 1e-13)
    n_cols = 3*4
    for det in case["dets"]:
        c = comp[det]
        assert c.n_cols == n_cols and c.npts_full == 256 and c.n_lm == 3
        if rank is None:
            assert c.r == min(256, n_cols)
            assert c.rel_err_fro == 0.0 and c.rel_err_max == 0.0
        else:
            assert c.r == rank, "kept %d components for rank %d (s/s0 = %s)" % (c.r, rank, c.singular_values/c.singular_values[0])
            # the dropped singular values are rounding noise
            assert c.rel_err_fro < 1e-14
            assert np.all(c.singular_values[rank:] < 1e-13*c.singular_values[0])
        assert c.c.shape == (256, c.r) and c.right.shape == (n_cols, c.r)
        assert c.c.flags.c_contiguous and c.right.flags.c_contiguous
    d = _compare(case, comp, time_interp=time_interp)
    print("[td-u-svd a] rank %s %s: lnL diff %.1e, lnL(t) diff %.1e" % (rank, time_interp, d[0], d[1]))


@pytest.mark.parametrize("time_interp", ['nearest', 'cubic'])
@pytest.mark.parametrize("rank", [1, 3, None])
def test_phase_marginalization(rank, time_interp):
    # Finite-piece norms depend on phase even for the quadrupole pair.
    case = _low_rank_case(seed=17 + (rank or 0), rank=rank, modes=((2, 2), (2, -2)))
    comp = _compress(case, 1e-13)
    with pytest.raises(NotImplementedError, match="explicit phase sampling"):
        _call(case, comp, comp, time_interp=time_interp, phase_marginalization=True)
    _compare(case, comp, time_interp=time_interp, phase_marginalization=False)


@pytest.mark.parametrize("time_interp", ['nearest', 'cubic'])
def test_mixed_detectors(time_interp):
    # H1 compressed, L1 scalar (2-D), V1 time-indexed (3-D, uncompressed)
    case = _low_rank_case(seed=37, rank=3, modes=((2, 2), (2, -2)))
    comp = _compress(case, 1e-13, dets=["H1"])
    U = dict(case["U_t"]); V = dict(case["V_t"])
    U["L1"] = case["U_s"]["L1"]; V["L1"] = case["V_s"]["L1"]
    U_c = dict(U); V_c = dict(V)
    U_c["H1"] = comp["H1"]; V_c["H1"] = comp["H1"]
    for kw in (dict(), dict(return_lnLt=True)):
        a = _call(case, U, V, time_interp=time_interp, **kw)
        b = _call(case, U_c, V_c, time_interp=time_interp, **kw)
        assert _max_rel(a, b) <= 1e-12
    ref = _brute_force(case, U, V, time_interp=time_interp)
    np.testing.assert_allclose(_call(case, U_c, V_c, time_interp=time_interp), ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("time_interp", ['nearest', 'cubic'])
def test_window_past_buffer_edge(time_interp):
    case = _low_rank_case(seed=41, rank=3, modes=((2, 2), (2, -2)), epoch_offset=-0.0005, npts_full=64)
    comp = _compress(case, 1e-13)
    _compare(case, comp, time_interp=time_interp)


def test_V_none_and_custom_loglikelihood():
    case = _low_rank_case(seed=31, rank=2, modes=((2, 2), (2, -2)))
    comp = _compress(case, 1e-13)
    V_none = {det: None for det in case["dets"]}
    a = _call(case, comp, comp, time_interp='cubic')
    b = _call(case, comp, V_none, time_interp='cubic')
    np.testing.assert_array_equal(a, b)
    seen = {}

    def loglike(kappa_sq, rho_sq):
        seen['shape'] = rho_sq.shape
        seen['varies'] = float(np.max(np.abs(np.diff(rho_sq, axis=-1))))
        return kappa_sq - 0.5*rho_sq + 0.1*np.log1p(np.abs(rho_sq))

    lnL = _call(case, comp, comp, time_interp='nearest', loglikelihood=loglike)
    assert seen['shape'] == (case["npts_extrinsic"], case["npts"])
    assert seen['varies'] > 1e-3
    ref = _brute_force(case, case["U_t"], case["V_t"], time_interp='nearest', loglikelihood=loglike)
    np.testing.assert_allclose(lnL, ref, rtol=1e-10, atol=1e-10)


def test_to_xpy_roundtrip_and_repr():
    case = _low_rank_case(seed=5, rank=2)
    c = _compress(case, 1e-13)["H1"]
    c2 = c.to_xpy(np)
    assert c2 is not c
    np.testing.assert_array_equal(c2.c, c.c)
    np.testing.assert_array_equal(c2.right, c.right)
    assert c2.r == c.r and c2.shape == (256, 3, 3) and c2.ndim == 3
    assert "r=2" in repr(c2)
    comp = _compress(case, 1e-13)
    comp2 = dict(comp); comp2["H1"] = c2
    a = _call(case, comp, comp)
    b = _call(case, comp2, {d: None for d in case["dets"]})
    np.testing.assert_array_equal(a, b)


def test_truncation_error_tracks_tol():
    # With a graded spectrum, loosening tol drops components and the lnL(t) error
    # grows with the reconstruction error; at rounding-level tol it vanishes.
    case = _make_case(seed=61, modes=((2, 2), (2, -2), (2, 1)), npts_full=512)
    rng = np.random.default_rng(61)
    n_lm = 3
    R = 10
    decay = 10.0**(-1.0*np.arange(R))           # graded coefficient amplitudes through 1e-9
    for det in case["dets"]:
        a = np.abs(rng.standard_normal((512, R))) * decay
        A = rng.standard_normal((R, n_lm, n_lm)) + 1j*rng.standard_normal((R, n_lm, n_lm))
        B = rng.standard_normal((R, n_lm, n_lm)) + 1j*rng.standard_normal((R, n_lm, n_lm))
        floor = 4 * (np.sum(np.abs(A), axis=(1, 2)) + np.sum(np.abs(B), axis=(1, 2)))
        A += floor[:, None, None] * np.eye(n_lm)[None]
        case["U_t"][det] = np.einsum("tk,kij->tij", a, A + np.conj(np.swapaxes(A, 1, 2)))
        case["V_t"][det] = np.einsum("tk,kij->tij", a, B + np.swapaxes(B, 1, 2))
    ref = _call(case, case["U_t"], case["V_t"], return_lnLt=True, loglikelihood=lambda k, r: r)
    last_err = np.inf
    ranks = []
    for tol in (1e-4, 1e-7, 1e-10, 1e-13):
        comp = _compress(case, tol)
        got = _call(case, comp, comp, return_lnLt=True, loglikelihood=lambda k, r: r)
        err = _max_rel(ref, got)
        ranks.append(comp["H1"].r)
        assert err <= 50*tol, "tol %g: self-term error %.2e" % (tol, err)
        assert err <= last_err*1.01
        last_err = err
    assert ranks[0] < ranks[-1] == R


# --------------------------------------------------------------------------
# (b) realistic U(t), V(t) from the td-gs Precompute
# --------------------------------------------------------------------------
if HAVE_TDANALYSIS:
    import lalsimulation as lalsim
    import RIFT.lalsimutils as lsu
    from RIFT.likelihood import td_gs_precompute as tdg
    from test_td_gs_route import make_psd, make_modes, make_data, peak_time, MREF, DT, FS  # noqa: E402


def _realistic_case(rholms, U, V, modes, npts=96, npts_extrinsic=6, seed=3, dets=("H1", "L1")):
    """Pack the td-gs Precompute output for one detector and present it on every
    detector in `dets` (same series, different antenna patterns and delays), with
    extrinsic samples whose windows land inside the storage grid."""
    lNK, lKN, lconj, U3, V3, rA, rI, epoch = fl.PackLikelihoodDataStructuresAsArrays(list(modes), None, rholms, U, V)
    N_window = rA.shape[1]
    deltaT = float(rholms[modes[0]].deltaT)
    npts = min(npts, N_window)
    rng = np.random.default_rng(seed)
    P = _PVec()
    P.phi = rng.uniform(0, 2*np.pi, npts_extrinsic)
    P.theta = rng.uniform(0.2, np.pi-0.2, npts_extrinsic)
    P.psi = rng.uniform(0, np.pi, npts_extrinsic)
    P.incl = rng.uniform(0.2, np.pi-0.2, npts_extrinsic)
    P.phiref = rng.uniform(0, 2*np.pi, npts_extrinsic)
    # distances around distMpcRef (1000 Mpc) keep lnL(t) within exp() range for the
    # time marginalization (the test templates are generated at 100 Mpc)
    P.dist = rng.uniform(700.0, 3000.0, npts_extrinsic) * (lal.PC_SI*1e6)
    P.tref = float(epoch) + 0.5*N_window*deltaT      # window centre at the middle of the storage grid
    P.deltaT = deltaT
    tvals = np.arange(npts)*deltaT - 0.5*npts*deltaT
    return dict(P=P, tvals=tvals, lookupNKDict={d: lNK for d in dets},
                rholmsArrayDict={d: rA for d in dets}, U_t={d: U3 for d in dets}, V_t={d: V3 for d in dets},
                epochDict={d: float(epoch) for d in dets}, npts=npts, npts_extrinsic=npts_extrinsic,
                deltaT=deltaT, dets=list(dets))


def _report_spectrum(name, case, Lmax, brute):
    """Print s_k/s_0, and r / reconstruction error / lnL agreement at each tol;
    return {tol: (r, d_lnL, d_lnLt)}."""
    det = case["dets"][0]
    full = fl.compress_time_indexed_cross_terms(case["U_t"][det], case["V_t"][det], 1e-15)
    s = full.singular_values/full.singular_values[0]
    print("[td-u-svd b] %s: U(t) %s, %d pair columns; s_k/s_0 = %s%s"
          % (name, case["U_t"][det].shape, full.n_cols,
             " ".join("%.1e" % x for x in s[:12]), " ..." if s.size > 12 else ""))
    out = {}
    for tol in TOLS:
        comp = _compress(case, tol)
        c = comp[det]
        d = {}
        for time_interp in ('nearest', 'cubic'):
            d[time_interp] = _compare(case, comp, Lmax=Lmax, rtol=np.inf, brute=False, time_interp=time_interp)
        d_lnL = max(v[0] for v in d.values())
        d_lnLt = max(v[1] for v in d.values())
        print("[td-u-svd b]   tol %.0e: r = %3d  rel err Frobenius %.1e  max %.1e  |  lnL diff %.1e  lnL(t) diff %.1e"
              % (tol, c.r, c.rel_err_fro, c.rel_err_max, d_lnL, d_lnLt))
        out[tol] = (c.r, d_lnL, d_lnLt)
    if brute:
        comp = _compress(case, 1e-13)
        for time_interp in ('nearest', 'cubic'):
            ref = _brute_force(case, case["U_t"], case["V_t"], time_interp=time_interp)
            np.testing.assert_allclose(_call(case, comp, comp, time_interp=time_interp), ref, rtol=1e-9, atol=1e-9)
    return out


def _check_realistic(out, n_cols, npts_full):
    # rounding-level tol: ~1e-12 relative agreement; r never exceeds the matrix rank bound
    for tol in TOLS:
        r, d_lnL, d_lnLt = out[tol]
        assert 0 < r <= min(n_cols, npts_full)
    assert out[1e-13][1] <= 1e-11 and out[1e-13][2] <= 1e-11
    assert out[1e-14][1] <= 1e-11 and out[1e-14][2] <= 1e-11
    # looser tol keeps no more components and errs no less than ~rounding
    assert out[1e-10][0] <= out[1e-12][0] <= out[1e-13][0] <= out[1e-14][0]
    assert out[1e-10][1] <= 1e-8 and out[1e-10][2] <= 1e-8


@pytest.fixture(scope="module")
def tphm_l2():
    if not HAVE_TDANALYSIS:
        pytest.skip("tdanalysis not importable")
    N = 4096
    psd = make_psd(N)
    hT = make_modes(N)
    k_inj = 1800
    dts = make_data(N, hT, k_inj)
    tpk = peak_time(hT, k_inj)
    msec = MREF*lal.MTSUN_SI
    return dict(N=N, psd=psd, hT=hT, dts=dts, tpk=tpk, k_inj=k_inj, modes=sorted(hT.keys()),
                pieces=[(tpk - 0.002, tpk + 120*msec)])


@pytest.mark.skipif(not HAVE_TDANALYSIS, reason="tdanalysis not importable")
@pytest.mark.parametrize("N_window", [1228, 122])
def test_realistic_full_segment_l2(tphm_l2, N_window):
    s = tphm_l2
    ctx = tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., ifo="H1", verbose=False)
    rho, U, V = ctx.precompute(s["hT"], s["k_inj"] - N_window//2, N_window)
    case = _realistic_case(rho, U, V, s["modes"])
    out = _report_spectrum("IMRPhenomTPHM l<=2, full segment, N_window=%d" % N_window, case, 2, brute=True)
    _check_realistic(out, 5*6, N_window)
    # the full segment has U(t) nearly constant (s_1/s_0 ~ 1e-5): its drift across the
    # storage grid has ~10 components above rounding, then a gap of 5 decades
    assert out[1e-10][0] <= 12


@pytest.mark.skipif(not HAVE_TDANALYSIS, reason="tdanalysis not importable")
@pytest.mark.parametrize("N_window", [1228, 122])
def test_realistic_ringdown_piece_l2(tphm_l2, N_window):
    s = tphm_l2
    ctx = tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., pieces=s["pieces"], ifo="H1", verbose=False)
    rho, U, V = ctx.precompute(s["hT"], s["k_inj"] - N_window//2, N_window)
    case = _realistic_case(rho, U, V, s["modes"])
    out = _report_spectrum("IMRPhenomTPHM l<=2, ringdown piece, N_window=%d" % N_window, case, 2, brute=True)
    _check_realistic(out, 5*6, N_window)


def _nrsur_available():
    if not HAVE_TDANALYSIS:
        return False
    try:
        P = lsu.ChooseWaveformParams(m1=35*lal.MSUN_SI, m2=30*lal.MSUN_SI, fmin=0., deltaT=DT, deltaF=FS/8192,
                                     dist=100e6*lal.PC_SI, approx=lalsim.NRSur7dq4)
        tdg.generate_td_modes(P, 2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not HAVE_TDANALYSIS, reason="tdanalysis not importable")
@pytest.mark.parametrize("piece", ["full", "ringdown"])
def test_realistic_nrsur_l4(piece):
    if not _nrsur_available():
        pytest.skip("NRSur7dq4 data not available (LAL_DATA_PATH)")
    N = 8192
    P = lsu.ChooseWaveformParams(m1=35*lal.MSUN_SI, m2=30*lal.MSUN_SI, fmin=0., deltaT=DT, deltaF=FS/N,
                                 dist=100e6*lal.PC_SI, approx=lalsim.NRSur7dq4)
    hT = tdg.generate_td_modes(P, 4)
    assert len(hT) == 21
    psd = make_psd(N)
    # the fmin=0 surrogate template is ~1.4 s long: place its peak 1024 samples before the data end
    k_inj = (N - 1024) + int(round(float(hT[(2, 2)].epoch)/DT))
    assert k_inj > 0
    dts = make_data(N, hT, k_inj)
    tpk = peak_time(hT, k_inj)
    msec = MREF*lal.MTSUN_SI
    pieces = None if piece == "full" else [(tpk - 0.002, tpk + 120*msec)]
    N_window = 1228 if piece == "full" else 122
    ctx = tdg.TDGSDetectorContext(dts, psd, 20., 1500., pieces=pieces, ifo="H1", verbose=False)
    rho, U, V = ctx.precompute(hT, k_inj - N_window//2, N_window)
    case = _realistic_case(rho, U, V, sorted(hT.keys()), npts=64, npts_extrinsic=4)
    out = _report_spectrum("NRSur7dq4 l<=4, %s, N_window=%d" % (piece, N_window), case, 4, brute=False)
    _check_realistic(out, 21*22, N_window)


# --------------------------------------------------------------------------
# (c) refusals and edge cases
# --------------------------------------------------------------------------
def test_refuse_bad_tol_and_shapes():
    case = _low_rank_case(seed=43, rank=2)
    U, V = case["U_t"]["H1"], case["V_t"]["H1"]
    for tol in (0.0, -1e-3, 1.0, 2.0):
        with pytest.raises(ValueError):
            fl.compress_time_indexed_cross_terms(U, V, tol)
    with pytest.raises(ValueError):            # 2-D input
        fl.compress_time_indexed_cross_terms(U[0], V[0], 1e-13)
    with pytest.raises(ValueError):            # U/V shape mismatch
        fl.compress_time_indexed_cross_terms(U, V[:-1], 1e-13)
    with pytest.raises(ValueError):            # not square
        fl.compress_time_indexed_cross_terms(U[:, :, :2], V[:, :, :2], 1e-13)


def test_all_zero_cross_terms_give_rank_zero():
    case = _low_rank_case(seed=47, rank=2, modes=((2, 2), (2, -2)))
    Z = {det: np.zeros_like(case["U_t"][det]) for det in case["dets"]}
    comp = {det: fl.compress_time_indexed_cross_terms(Z[det], Z[det], 1e-13) for det in case["dets"]}
    for det in case["dets"]:
        assert comp[det].r == 0 and comp[det].c.shape == (256, 0)
        assert comp[det].rel_err_fro == 0.0
    for time_interp in ('nearest', 'cubic'):
        a = _call(case, Z, Z, time_interp=time_interp, return_lnLt=True)
        b = _call(case, comp, comp, time_interp=time_interp, return_lnLt=True)
        np.testing.assert_array_equal(a, b)
        rho = _call(case, comp, comp, time_interp=time_interp, return_lnLt=True, loglikelihood=lambda k, r: r)
        assert np.all(rho == 0)


def test_refuse_mismatched_objects():
    case = _low_rank_case(seed=53, rank=2, modes=((2, 2), (2, -2)))
    comp = _compress(case, 1e-13)
    other = _compress(case, 1e-12)
    V_other = dict(comp); V_other["H1"] = other["H1"]
    with pytest.raises(ValueError):            # V a different object
        _call(case, comp, V_other)
    with pytest.raises(ValueError):            # V the 3-D array
        _call(case, comp, case["V_t"])
    short = {det: fl.compress_time_indexed_cross_terms(case["U_t"][det][:-1], case["V_t"][det][:-1], 1e-13) for det in case["dets"]}
    with pytest.raises(ValueError):            # time grid mismatch
        _call(case, short, short)
    case3 = _low_rank_case(seed=53, rank=2, modes=((2, 2), (2, -2), (2, 1)))
    wrong = _compress(case3, 1e-13)
    with pytest.raises(ValueError):            # n_lm mismatch
        _call(case, wrong, wrong)


def test_refuse_calibration_paths():
    case = _low_rank_case(seed=59, rank=2, modes=((2, 2), (2, -2)))
    comp = _compress(case, 1e-13)
    with pytest.raises(NotImplementedError):
        _call(case, comp, comp, n_cal=2)
    with pytest.raises(NotImplementedError):
        _call(case, comp, comp, n_cal=2, cal_method='fused')
    U_cal = {det: case["U_s"][det][None] for det in case["dets"]}
    V_cal = {det: case["V_s"][det][None] for det in case["dets"]}
    with pytest.raises(NotImplementedError):
        _call(case, comp, comp, ctUArrayDict_cal=U_cal, ctVArrayDict_cal=V_cal)


def test_refuse_legacy_array_likelihoods():
    case = _low_rank_case(seed=67, rank=2, modes=((2, 2), (2, -2)))
    comp = _compress(case, 1e-13)
    P = case["P"]
    with pytest.raises(NotImplementedError):
        fl.SingleDetectorLogLikelihoodModelViaArray(case["lookupNKDict"], comp, comp,
                                                    P.tref, P.phi[0], P.theta[0], P.incl[0], P.phiref[0],
                                                    P.psi[0], P.dist[0], "H1")
    Ps = _PVec()
    for name in ("phi", "theta", "psi", "incl", "phiref", "dist"):
        setattr(Ps, name, float(getattr(P, name)[0]))
    Ps.tref = P.tref
    Ps.deltaT = P.deltaT
    with pytest.raises(NotImplementedError):
        fl.DiscreteFactoredLogLikelihoodViaArray(case["tvals"], Ps, case["lookupNKDict"], case["rholmsArrayDict"],
                                                 comp, comp, case["epochDict"], Lmax=2)
    with pytest.raises(NotImplementedError):
        fl.DiscreteFactoredLogLikelihoodViaArrayVector(case["tvals"], P, case["lookupNKDict"], case["rholmsArrayDict"],
                                                       comp, comp, case["epochDict"], Lmax=2, xpy=np)
    with pytest.raises(NotImplementedError):
        fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopOrig(case["tvals"], P, case["lookupNKDict"], case["rholmsArrayDict"],
                                                                 comp, comp, case["epochDict"], Lmax=2)


def test_component_matrices_identity_and_self_term():
    # g_k = |F|^2 conj(Y)^T U_k Y + F^2 Y^T V_k Y with the upper-triangular component
    # matrices, exactly as complex numbers; and Re sum_k g_k c_k(t) is the self term.
    case = _low_rank_case(seed=71, rank=4)
    c = _compress(case, 1e-13)["H1"]
    rng = np.random.default_rng(71)
    ne = 5
    F = rng.standard_normal(ne) + 1j*rng.standard_normal(ne)
    Y = rng.standard_normal((ne, 3)) + 1j*rng.standard_normal((ne, 3))
    W = fl._time_indexed_self_term_weights(F, Y, c.iu, c.ju)
    g = W @ c.right
    for k in range(c.r):
        U_k, V_k = c.component_matrices(k)
        assert np.all(np.tril(U_k, -1) == 0) and np.all(np.tril(V_k, -1) == 0)
        g_k = ((F*np.conj(F)).real*np.einsum("ei,ej,ij->e", np.conj(Y), Y, U_k)
               + F**2*np.einsum("ei,ej,ij->e", Y, Y, V_k))
        np.testing.assert_allclose(g_k, g[:, k], rtol=1e-12, atol=1e-12)
    ifirst = np.array([3, 5, 0, 20, 100], dtype=np.int32)
    ref = fl._time_indexed_self_term(case["U_t"]["H1"], case["V_t"]["H1"], F, Y, ifirst, None, 8, 'nearest')
    got = fl._time_indexed_self_term_compressed(c, F, Y, ifirst, None, 8, 'nearest')
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12*np.max(np.abs(ref)))
    direct = np.zeros((ne, 8))
    for e in range(ne):
        for j in range(8):
            direct[e, j] = np.real(np.sum(g[e]*c.c[ifirst[e] + j]))
    np.testing.assert_allclose(got, direct, rtol=1e-12, atol=1e-12*np.max(np.abs(ref)))


# --------------------------------------------------------------------------
# local, relative-only timing of the self term
# --------------------------------------------------------------------------
def _profile(n_extrinsic=(500, 2000), npts=64, reps=3):
    """Uncompressed vs compressed self term (the only part that changes), numpy,
    this machine: relative numbers only.  Ranks are the realistic ones from (b)
    at tol 1e-13: 10 (full segment) and 19 (ringdown) for l<=2 (30 columns),
    40 (full segment, 1228 points) and 116 (ringdown, 122 points) for l<=4
    (462 columns)."""
    rng = np.random.default_rng(0)
    print("local numpy timing (relative only); self term for npts=%d window samples" % npts)
    print("%6s %9s %5s %7s | %12s %12s %7s" % ("n_lm", "npts_full", "r", "n_e", "uncompressed", "compressed", "ratio"))
    for n_lm, ranks in ((5, (10, 19)), (21, (40, 116))):
        for npts_full in (1228, 122):
            for r in ranks:
                r = min(r, npts_full)
                A = rng.standard_normal((r, n_lm, n_lm)) + 1j*rng.standard_normal((r, n_lm, n_lm))
                B = rng.standard_normal((r, n_lm, n_lm)) + 1j*rng.standard_normal((r, n_lm, n_lm))
                a = rng.standard_normal((npts_full, r))            # real: U(t) Hermitian, rank exactly r
                U = np.einsum("tk,kij->tij", a, A + np.conj(np.swapaxes(A, 1, 2)))
                V = np.einsum("tk,kij->tij", a, B + np.swapaxes(B, 1, 2))
                t0 = time.time()
                comp = fl.compress_time_indexed_cross_terms(U, V, 1e-13)
                t_svd = time.time() - t0
                assert comp.r == r
                for ne in n_extrinsic:
                    F = rng.standard_normal(ne) + 1j*rng.standard_normal(ne)
                    Y = rng.standard_normal((ne, n_lm)) + 1j*rng.standard_normal((ne, n_lm))
                    ifirst = rng.integers(-npts//2, npts_full - npts//2, ne).astype(np.int32)
                    frac = rng.uniform(0, 1, ne)
                    for time_interp in ('nearest', 'cubic'):
                        fr = None if time_interp == 'nearest' else frac
                        tu = []
                        tc = []
                        for _ in range(reps):
                            t0 = time.time()
                            ref = fl._time_indexed_self_term(U, V, F, Y, ifirst, fr, npts, time_interp)
                            tu.append(time.time() - t0)
                            t0 = time.time()
                            got = fl._time_indexed_self_term_compressed(comp, F, Y, ifirst, fr, npts, time_interp)
                            tc.append(time.time() - t0)
                        err = np.max(np.abs(got - ref))/np.max(np.abs(ref))
                        print("%6d %9d %5d %7d | %9.3f s %9.3f s %7.1fx  (%s, svd %.3f s, err %.1e)"
                              % (n_lm, npts_full, r, ne, min(tu), min(tc), min(tu)/min(tc), time_interp, t_svd, err))


if __name__ == "__main__":
    if "--profile" in sys.argv:
        _profile()
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
