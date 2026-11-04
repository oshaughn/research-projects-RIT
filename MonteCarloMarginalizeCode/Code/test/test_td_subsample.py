"""
Sub-sample template placement on the td-gs code path (--td-subsample K).

With K > 1, TDGSDetectorContext.precompute returns Q(t), U(t), V(t) on a grid of
spacing deltaT/K: phase j is the template moved by a fraction of a sample with
tdanalysis' Lanczos resampler (tdanalysis.utils.subsample_align.fractional_shift) and
run through the same exact methods; DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop
(td_subsample=K) locates each extrinsic sample's arrival time on that grid and gathers
its time bins with stride K.

  (a) K = 1 is the historical code path: outputs equal those of a context built without
      the argument, and phase 0 of a K > 1 run equals the K = 1 run bit for bit;
  (b) K = 4, 8: Q, U, V at every fractional phase equal a dense brute force with the
      template explicitly shifted by tdanalysis' kernel, to 1e-12 relative on a
      well-conditioned ACF, on the full segment, a ringdown piece and two spliced pieces,
      dense and GS solvers (prefix / recursion / chunked methods);
  (c) lnL(t_c) over 0.1-sample steps: K = 1 is a staircase (one plateau per sample),
      K = 8 has ~8 plateaus per sample, and every value matches the dense residual
      likelihood with the template placed at the same 1/K grid point;
  (d) Precompute + NoLoop for random extrinsic samples (two detectors, 12 time bins)
      against a dense reference that places the template with tdanalysis' kernel at
      each sample's fractional arrival time rounded to the 1/K grid: lnL(t) per bin,
      the time-marginalized lnL, and cubic lookup on the fine grid;
  (e) refusals: K < 1, the fused calmarg kernel with K > 1, missing tdanalysis helper.

Run with:  pytest -q test/test_td_subsample.py
"""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import lal
import lalsimulation as lalsim
import scipy.linalg

import RIFT.likelihood.factored_likelihood as fl

try:
    from tdanalysis.forms.sliding import sliding_self_terms  # noqa: F401
    from tdanalysis.covariance.psd import PSDToAutocorr  # noqa: F401
except Exception as _exc:  # pragma: no cover
    pytest.skip("tdanalysis >= 1.1 not importable (%r)" % (_exc,), allow_module_level=True)

from RIFT.likelihood import td_gs_precompute as tdg

try:
    _fractional_shift, SHIFT_STENCIL = tdg.import_fractional_shift()
except ImportError as _exc:  # pragma: no cover
    pytest.skip(str(_exc), allow_module_level=True)

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import test_td_gs_route as T  # noqa: E402  (shared fixtures: modes, data, brute force)

DT, FS, EPOCH0 = T.DT, T.FS, T.EPOCH0
ACF_COND = 1e4          # well-conditioned ACF: the dense brute force is itself good to ~1e-13


# --------------------------------------------------------------------------- setup

@pytest.fixture(scope="module")
def s():
    base = T.ringdown_setup.__wrapped__()
    j_first, j_last = T.template_support(base["hT"])
    base["j_first"], base["j_last"] = j_first, j_last
    # ringdown piece that the shifted template (support widened by SHIFT_STENCIL) still
    # ends inside for every grid shift: the GS 'prefix' method is used on every phase
    base["ringdown"] = T.ringdown_piece_ending_at(base, j_last + base["N_shift"] + base["N_window"] + SHIFT_STENCIL)
    tpk = base["tpk"]
    base["two_pieces"] = [(tpk - 0.15, tpk - 0.05), (tpk - 0.01, tpk + 0.04)]
    base["harr"] = {m: np.asarray(base["hT"][m].data.data, dtype=np.complex128) for m in base["hT"]}
    return base


def ctx_for(s, pieces, K, dense_max_n=0, u_method="auto", **kw):
    return tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., pieces=pieces, dense_max_n=dense_max_n,
                                   u_method=u_method, ifo="H1", verbose=False, subsample=K,
                                   acf_condition_number=ACF_COND, **kw)


def as_hT(harr):
    """Dict of objects with .data.data, the attribute test_td_gs_route.brute_force uses."""
    return {m: SimpleNamespace(data=SimpleNamespace(data=harr[m])) for m in harr}


def brute_fine(ctx, harr, N_shift, K, idx):
    """Dense brute force at fine entries K*i + j for i in idx and every phase j: the
    template explicitly moved by tdanalysis' kernel (s_j) and placed at N_shift + i + e_j."""
    out = {}
    for j, (s_j, e_j) in enumerate(tdg.subsample_phases(K)):
        hj = tdg.shifted_modes(harr, s_j, _fractional_shift)
        ref = T.brute_force(ctx.d, ctx.acf_lags, ctx.runs, as_hT(hj), N_shift + e_j, idx)
        for key, val in ref.items():
            out[key[:-1] + (K * key[-1] + j,)] = val
    return out


def compare_fine(ref, harr, rho, U, V, entries):
    modes = list(harr.keys())
    scale = np.max(np.abs(U[((2, 2), (2, 2))]))
    qscale = np.max(np.abs(rho[(2, 2)].data.data))
    worst = 0.0
    for r in entries:
        for ma in modes:
            worst = max(worst, abs(ref[("Q", ma, r)] - rho[ma].data.data[r]) / qscale)
            for mb in modes:
                worst = max(worst, abs(ref[("U", ma, mb, r)] - U[(ma, mb)][r]) / scale,
                            abs(ref[("V", ma, mb, r)] - V[(ma, mb)][r]) / scale)
    return worst


# --------------------------------------------------------------------------- (a) K = 1 unchanged

def test_subsample_phases():
    assert tdg.subsample_phases(1) == [(0.0, 0)]
    assert tdg.subsample_phases(2) == [(0.0, 0), (0.5, 0)]
    assert tdg.subsample_phases(4) == [(0.0, 0), (0.25, 0), (0.5, 0), (-0.25, 1)]
    for K in (3, 5, 8, 16):
        ph = tdg.subsample_phases(K)
        assert len(ph) == K
        for j, (s_j, e_j) in enumerate(ph):
            assert abs(s_j + e_j - j / K) < 1e-15 and -0.5 < s_j <= 0.5 and e_j in (0, 1)
    with pytest.raises(ValueError):
        tdg.subsample_phases(0)


def test_shifted_modes_is_tdanalysis_kernel(s):
    """shifted_modes applies tdanalysis' fractional_shift as is (same length, s = 0 is the
    identity), and a shift by a whole-sample-free fraction interpolates a band-limited
    series: a pure tone well below Nyquist is reproduced to the resampler's accuracy."""
    harr = s["harr"]
    h0 = tdg.shifted_modes(harr, 0.0, _fractional_shift)
    assert all(np.array_equal(h0[m], harr[m]) for m in harr)
    h1 = tdg.shifted_modes(harr, 0.3, _fractional_shift)
    for m in harr:
        assert h1[m].shape == harr[m].shape and h1[m].dtype == np.complex128
        np.testing.assert_array_equal(h1[m], _fractional_shift(harr[m], 0.3))
    n = np.arange(2048)
    tone = np.exp(2j * np.pi * 150.0 / FS * n)
    sh = _fractional_shift(tone, 0.37)
    inner = slice(SHIFT_STENCIL, 2048 - SHIFT_STENCIL)
    err = np.max(np.abs(sh[inner] - np.exp(2j * np.pi * 150.0 / FS * (n[inner] - 0.37))))
    print("[td-ss a] tdanalysis fractional_shift(0.37) on a 150 Hz tone: max error %.2e" % err)
    assert err < 1e-3        # tdanalysis quotes ~2e-4 relative L2 up to 500 Hz for this kernel


def test_k1_is_the_historical_path_and_phase0_of_k4(s):
    for pieces, dmax in ((None, 0), (s["ringdown"], 0), (s["ringdown"], 4096), (s["two_pieces"], 0)):
        plain = tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., pieces=pieces, dense_max_n=dmax,
                                        ifo="H1", verbose=False, acf_condition_number=ACF_COND)
        a = plain.precompute(s["hT"], s["N_shift"], s["N_window"])
        b = ctx_for(s, pieces, 1, dmax).precompute(s["hT"], s["N_shift"], s["N_window"])
        c = ctx_for(s, pieces, 4, dmax).precompute(s["hT"], s["N_shift"], s["N_window"])
        for m in a[0]:
            assert np.array_equal(a[0][m].data.data, b[0][m].data.data)
            assert np.array_equal(a[0][m].data.data, c[0][m].data.data[0::4])
            assert a[0][m].deltaT == b[0][m].deltaT == DT and c[0][m].deltaT == DT / 4
            assert float(a[0][m].epoch) == float(b[0][m].epoch) == float(c[0][m].epoch)
            assert c[0][m].data.length == 4 * s["N_window"]
        for k in a[1]:
            assert np.array_equal(a[1][k], b[1][k]) and np.array_equal(a[2][k], b[2][k])
            assert np.array_equal(a[1][k], c[1][k][0::4]) and np.array_equal(a[2][k], c[2][k][0::4])
            assert c[1][k].shape == (4 * s["N_window"],)
    print("[td-ss a] K=1 with and without subsample=, and phase 0 of K=4: bit-identical on 4 layouts")


# --------------------------------------------------------------------------- (b) Q, U, V at fractional shifts

@pytest.mark.parametrize("K", [4, 8])
def test_fine_grid_matches_brute_force_shifted_template(s, K):
    harr = s["harr"]
    idx = [0, 1, 128, 254, 255]
    layouts = {
        "full segment (gs)": (None, 0),
        "full segment (dense)": (None, 100000),
        "ringdown piece (gs)": (s["ringdown"], 0),
        "ringdown piece (dense)": (s["ringdown"], 4096),
        "two spliced pieces (gs)": (s["two_pieces"], 0),
        "two spliced pieces (dense)": (s["two_pieces"], 4096),
    }
    for label, (pieces, dmax) in layouts.items():
        ctx = ctx_for(s, pieces, K, dmax)
        rho, U, V = ctx.precompute(s["hT"], s["N_shift"], s["N_window"])
        assert rho[(2, 2)].data.length == K * s["N_window"]
        ref = brute_fine(ctx, harr, s["N_shift"], K, idx)
        entries = [K * i + j for i in idx for j in range(K)]
        worst = compare_fine(ref, harr, rho, U, V, entries)
        routes = [[r for (r, o) in rp] for rp in ctx.uv_routes_by_phase]
        print("[td-ss b] K=%d %-28s methods per phase %s: worst rel vs shifted-template brute force %.2e"
              % (K, label, routes[0] if all(r == routes[0] for r in routes) else routes, worst))
        assert worst < 1e-12
        for k in U:
            if k[0] != k[1]:
                assert np.array_equal(U[(k[1], k[0])], np.conj(U[k]))
                assert np.array_equal(V[(k[1], k[0])], V[k])
        if label == "ringdown piece (gs)":
            assert all(rp == [("prefix", "end")] for rp in ctx.uv_routes_by_phase)
        if label == "full segment (gs)":
            assert all(rp == [("prefix", "end")] for rp in ctx.uv_routes_by_phase)


def test_u_method_choices_and_stride_on_fine_grid(s):
    """Every explicit U/V method gives the same fine grid, and --td-u-stride interpolates
    each phase separately (phase 0 exact at the stride points, Q always exact)."""
    K = 4
    ref = ctx_for(s, s["two_pieces"], K, 0, "recursion").precompute(s["hT"], s["N_shift"], s["N_window"])
    scale = np.max(np.abs(ref[1][((2, 2), (2, 2))]))
    for method in ("batched", "chunked", "prefix"):
        pieces = s["ringdown"] if method == "prefix" else s["two_pieces"]
        if method == "prefix":
            ref = ctx_for(s, pieces, K, 0, "recursion").precompute(s["hT"], s["N_shift"], s["N_window"])
            scale = np.max(np.abs(ref[1][((2, 2), (2, 2))]))
        out = ctx_for(s, pieces, K, 0, method).precompute(s["hT"], s["N_shift"], s["N_window"])
        worst = max(max(np.max(np.abs(out[1][k] - ref[1][k])), np.max(np.abs(out[2][k] - ref[2][k])))
                    for k in ref[1]) / scale
        print("[td-ss b] K=4 %s vs recursion over the whole fine grid: %.2e" % (method, worst))
        assert worst < 1e-11
    exact = ctx_for(s, s["ringdown"], K, 4096).precompute(s["hT"], s["N_shift"], s["N_window"])
    strided = ctx_for(s, s["ringdown"], K, 4096, u_stride=4).precompute(s["hT"], s["N_shift"], s["N_window"])
    for m in exact[0]:
        assert np.array_equal(exact[0][m].data.data, strided[0][m].data.data)
    tsel = np.arange(0, s["N_window"], 4)
    for j in range(K):
        fine = K * tsel + j
        np.testing.assert_allclose(strided[1][((2, 2), (2, 2))][fine], exact[1][((2, 2), (2, 2))][fine], rtol=1e-13, atol=0)


# --------------------------------------------------------------------------- likelihood helpers

LMS = None


def _pack(rholms, U, V):
    (lookupNK, _kn, _knc, ctU, ctV, rhoArr, _intp, epoch) = fl.PackLikelihoodDataStructuresAsArrays(
        list(rholms.keys()), None, rholms, U, V)
    return lookupNK, ctU, ctV, rhoArr, epoch


def _pvec(ra, dec, psi, incl, phiref, dist_mpc, tref):
    P = SimpleNamespace()
    P.phi, P.theta, P.psi = np.atleast_1d(ra).astype(float), np.atleast_1d(dec).astype(float), np.atleast_1d(psi).astype(float)
    P.incl, P.phiref = np.atleast_1d(incl).astype(float), np.atleast_1d(phiref).astype(float)
    P.dist = np.atleast_1d(dist_mpc).astype(float) * 1e6 * lal.PC_SI
    P.tref = float(tref)
    P.deltaT = DT
    return P


def _dense_blocks(ctx):
    blocks = []
    for (i0, i1) in ctx.runs:
        Cinv = np.linalg.inv(scipy.linalg.toeplitz(ctx.acf_lags(i1 - i0)))
        blocks.append((i0, i1, Cinv, Cinv @ ctx.d[i0:i1]))
    return blocks


def _lnL_dense(ctx, blocks, harr, lms, fine_entry, K, N_shift, F, Y, amp):
    """Direct residual likelihood -1/2 (d-h)^T C^-1 (d-h) + 1/2 d^T C^-1 d on the pieces,
    for the template moved by tdanalysis' kernel and placed at fine entry r
    (shift N_shift + r/K): h = amp Re[F sum_lm Y_lm h_lm]."""
    N = ctx.d.size
    i, j = divmod(int(fine_entry), K)
    s_j, e_j = tdg.subsample_phases(K)[j]
    hj = tdg.shifted_modes(harr, s_j, _fractional_shift)
    k = N_shift + i + e_j
    H = np.zeros(N, dtype=complex)
    for lm_idx, lm in enumerate(lms):
        h = hj[tuple(lm)]
        x = np.zeros(N, dtype=complex)
        if k >= 0:
            x[k:] = h[:N - k]
        else:
            x[:N + k] = h[-k:]
        H += Y[lm_idx] * x
    h = amp * np.real(F * H)
    out = 0.0
    for (i0, i1, Cinv, w) in blocks:
        r = ctx.d[i0:i1] - h[i0:i1]
        out += -0.5 * (r @ Cinv @ r) + 0.5 * (ctx.d[i0:i1] @ w)
    return out


def _geometry(det, P, e, epoch):
    """F, Y, amp and the sample position (coarse units, from the rholm epoch) of the first
    time bin of extrinsic sample e, with lal (not RIFT's vectorized tools)."""
    d = lalsim.DetectorPrefixToLALDetector(det)
    gmst = lal.GreenwichMeanSiderealTime(P.tref)
    fp, fc = lal.ComputeDetAMResponse(d.response, P.phi[e], P.theta[e], P.psi[e], gmst)
    delay = lal.TimeDelayFromEarthCenter(d.location, P.phi[e], P.theta[e], lal.LIGOTimeGPS(P.tref))
    t_det = (P.tref - epoch) + delay
    amp = fl.distMpcRef / (P.dist[e] / (1e6 * lal.PC_SI))
    return fp + 1j * fc, t_det, amp


def _ylm(lms, incl, phiref):
    return np.array([lal.SpinWeightedSphericalHarmonic(incl, -phiref, -2, int(l), int(m)) for (l, m) in lms])


# --------------------------------------------------------------------------- (c) staircase in t_c

def test_lnL_of_tc_staircase_removed(s):
    """lnL at the injection's sky as a function of the geocenter time in 0.1-sample steps
    over 3 samples: K=1 changes only at sample boundaries; K=8 at every 1/8 sample; every
    value equals the dense residual likelihood at the 1/K grid point RIFT used."""
    harr = s["harr"]
    N_shift, N_window = s["N_shift"], s["N_window"]
    ra, dec, psi, incl, phiref, dist = 0.8, -0.4, 0.6, 0.0, 0.0, 300.0
    steps = np.arange(31) * 0.1 * DT
    results = {}
    for K in (1, 8):
        ctx = ctx_for(s, s["ringdown"], K, 4096)
        rho, U, V = ctx.precompute(s["hT"], N_shift, N_window)
        lookupNK, ctU, ctV, rhoArr, epoch = _pack(rho, U, V)
        lms = lookupNK
        blocks = _dense_blocks(ctx)
        # sweep 3 samples around the injected template's position on the storage grid
        # (k_inj - N_shift = 64), so the staircase is probed where lnL is largest
        d = lalsim.DetectorPrefixToLALDetector("H1")
        t_base = epoch + (64 - 1.5) * DT
        t_base -= lal.TimeDelayFromEarthCenter(d.location, ra, dec, lal.LIGOTimeGPS(t_base))
        lnL, lnL_ref = [], []
        for dt_c in steps:
            P = _pvec(ra, dec, psi, incl, phiref, dist, t_base + dt_c)
            val = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
                np.array([0.]), P, {"H1": lms}, {"H1": rhoArr}, {"H1": ctU}, {"H1": ctV}, {"H1": epoch},
                Lmax=2, xpy=np, return_lnLt=True, td_subsample=K)
            lnL.append(float(np.asarray(val)[0, 0]))
            F, t_det, amp = _geometry("H1", P, 0, epoch)
            r = int(np.rint(t_det / DT * K))
            lnL_ref.append(_lnL_dense(ctx, blocks, harr, lms, r, K, N_shift, F, _ylm(lms, incl, phiref), amp))
        lnL, lnL_ref = np.array(lnL), np.array(lnL_ref)
        # Plateaus: within one the gathered Q, U, V are the same and lnL only drifts with the
        # antenna pattern's sidereal-time dependence (~1e-5 nat here); a step is a jump of
        # more than 1e-6 of the sweep's range.
        jumps = np.abs(np.diff(lnL)) > 1e-6 * (lnL.max() - lnL.min())
        n_plateaus = 1 + int(np.sum(jumps))
        err = np.max(np.abs(lnL - lnL_ref))
        results[K] = (n_plateaus, err, lnL)
        print("[td-ss c] K=%d: lnL(t_c) over 31 points in 0.1-sample steps (3 samples): %d plateaus; "
              "max |RIFT - dense at the same 1/K point| %.2e nat (lnL range %.3f..%.3f)"
              % (K, n_plateaus, err, lnL.min(), lnL.max()))
        assert err < 1e-9 * max(1.0, np.max(np.abs(lnL)))
    assert results[1][0] <= 4                 # one plateau per sample
    assert results[8][0] >= 20                # ~8 per sample (31 points in 0.1-sample steps)
    # K=8 agrees with K=1 where the sweep sits on whole samples (steps 5, 15, 25: phase 0)
    l1, l8 = results[1][2], results[8][2]
    assert np.allclose(l1[5::10], l8[5::10], rtol=0, atol=1e-9 * max(1.0, np.max(np.abs(l1))))


# --------------------------------------------------------------------------- (d) end to end, random extrinsic samples

@pytest.mark.parametrize("K", [1, 8])
def test_noloop_vs_dense_random_extrinsic(s, K):
    harr = s["harr"]
    N_shift, N_window = s["N_shift"], s["N_window"]
    dets = ["H1", "L1"]
    rng = np.random.default_rng(11)
    ne, npts = 4, 12
    ctx = {d: ctx_for(s, s["ringdown"], K, 4096) for d in dets}
    out = {d: ctx[d].precompute(s["hT"], N_shift, N_window) for d in dets}
    packed = {d: _pack(*out[d]) for d in dets}
    lookupNK = {d: packed[d][0] for d in dets}
    ctU = {d: packed[d][1] for d in dets}
    ctV = {d: packed[d][2] for d in dets}
    rhoArr = {d: packed[d][3] for d in dets}
    epochs = {d: packed[d][4] for d in dets}
    blocks = {d: _dense_blocks(ctx[d]) for d in dets}
    tref = epochs["H1"] + (N_window // 2) * DT
    P = _pvec(rng.uniform(0, 2 * np.pi, ne), np.arcsin(rng.uniform(-1, 1, ne)), rng.uniform(0, np.pi, ne),
              np.arccos(rng.uniform(-1, 1, ne)), rng.uniform(0, 2 * np.pi, ne), rng.uniform(150, 600, ne), tref)
    tvals = (np.arange(npts) - npts // 2) * DT + 0.37 * DT
    kw = dict(Lmax=2, xpy=np, td_subsample=K)
    lnLt = np.asarray(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, P, lookupNK, rhoArr, ctU, ctV, epochs, return_lnLt=True, **kw))
    # NoLoop subtracts one global maximum over all samples before the time integral, so a
    # sample far below the loudest one underflows when evaluated together with it:
    # marginalize one extrinsic sample at a time.
    lnLm = np.zeros(ne)
    for e in range(ne):
        Pe = SimpleNamespace(**{k: (v[e:e + 1] if isinstance(v, np.ndarray) else v) for k, v in vars(P).items()})
        lnLm[e] = np.asarray(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            tvals, Pe, lookupNK, rhoArr, ctU, ctV, epochs, **kw))[0]
    ref = np.zeros((ne, npts))
    for det in dets:
        lms = lookupNK[det]
        for e in range(ne):
            F, t_det, amp = _geometry(det, P, e, epochs[det])
            Y = _ylm(lms, P.incl[e], P.phiref[e])
            r0 = int(np.rint((t_det + tvals[0]) / DT * K))
            for m in range(npts):
                ref[e, m] += _lnL_dense(ctx[det], blocks[det], harr, lms, r0 + m * K, K, N_shift, F, Y, amp)
    err_t = np.max(np.abs(lnLt - ref))
    w = fl.simpson_weights(npts, DT)
    ref_m = np.array([np.log(np.exp(ref[e] - ref[e].max()) @ w) + ref[e].max() for e in range(ne)])
    err_m = np.max(np.abs(lnLm - ref_m))
    print("[td-ss d] K=%d, %d samples x %d bins x 2 detectors: max |lnL(t) - dense| %.2e, "
          "max |marginalized - dense| %.2e (lnL(t) range %.2f..%.2f)"
          % (K, ne, npts, err_t, err_m, ref.min(), ref.max()))
    scale = max(1.0, np.max(np.abs(ref)))
    assert err_t < 1e-9 * scale and err_m < 1e-9 * scale
    if K > 1:
        # cubic lookup on the fine grid: four-point Lagrange stencil in fine entries
        lnLc = np.asarray(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            tvals, P, lookupNK, rhoArr, ctU, ctV, epochs, return_lnLt=True, time_interp='cubic', **kw))
        refc = np.zeros((ne, npts))
        for det in dets:
            lms = lookupNK[det]
            Q = rhoArr[det].T
            U3, V3 = ctU[det], ctV[det]
            n_time = Q.shape[0]
            for e in range(ne):
                F, t_det, amp = _geometry(det, P, e, epochs[det])
                Y = _ylm(lms, P.incl[e], P.phiref[e])
                f = (t_det + tvals[0]) / DT * K
                i0 = int(np.floor(f))
                u = f - np.floor(f)
                wts = (-u * (u - 1) * (u - 2) / 6, (u + 1) * (u - 1) * (u - 2) / 2, -(u + 1) * u * (u - 2) / 2, (u + 1) * u * (u - 1) / 6)
                for m in range(npts):
                    Qh = np.zeros(len(lms), complex)
                    Uh = np.zeros((len(lms), len(lms)), complex)
                    Vh = np.zeros((len(lms), len(lms)), complex)
                    for off, wt in zip((-1, 0, 1, 2), wts):
                        idx = i0 + m * K + off
                        if 0 <= idx < n_time:
                            Qh += wt * Q[idx]
                            Uh += wt * U3[idx]
                            Vh += wt * V3[idx]
                    FY = F * Y
                    refc[e, m] += amp * np.real(np.conj(FY) @ Qh) - 0.25 * amp ** 2 * np.real(
                        abs(F) ** 2 * (np.conj(Y) @ Uh @ Y) + F ** 2 * (Y @ Vh @ Y))
        errc = np.max(np.abs(lnLc - refc))
        print("[td-ss d] K=%d cubic on the fine grid vs 4-point stencil brute force: %.2e" % (K, errc))
        assert errc < 1e-9 * scale


def test_driver_interface_with_subsample(s):
    """PrecomputeLikelihoodTerms(td_gs_contexts=...) with subsample=4: K*N_window entries at
    deltaT/4, same epoch as K=1; Pack + NoLoop(td_subsample=4) run; K=1 driver output is
    unchanged."""
    N = s["N"]
    dets = ["H1", "L1"]
    data_dict = {d: s["dts"] for d in dets}
    psd_dict = {d: s["psd"] for d in dets}
    P = T.make_P(N)
    P.tref = s["tpk"]
    t_window = 0.03
    N_window = int(2 * t_window / DT)
    res = {}
    for K in (1, 4):
        contexts = {d: tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., ifo=d, verbose=False, subsample=K)
                    for d in dets}
        Pk = T.make_P(N)
        Pk.tref = s["tpk"]
        out = fl.PrecomputeLikelihoodTerms(s["tpk"], t_window, Pk, data_dict, psd_dict, 2, 1500.,
                                           False, False, 0., verbose=False, quiet=True,
                                           skip_interpolation=False, td_gs_contexts=contexts,
                                           return_calibration_crossterms=True)
        res[K] = out
        rholms_intp, crossTerms, crossTermsV, rholms, guess_snr = out[:5]
        for det in dets:
            for m in rholms[det]:
                assert rholms[det][m].data.length == K * N_window
                assert abs(rholms[det][m].deltaT - DT / K) < 1e-18
            for k in crossTerms[det]:
                assert crossTerms[det][k].shape == (K * N_window,)
            assert rholms_intp[det] is not None
    for det in dets:
        for m in res[1][3][det]:
            assert float(res[1][3][det][m].epoch) == float(res[4][3][det][m].epoch)
            np.testing.assert_array_equal(res[1][3][det][m].data.data, res[4][3][det][m].data.data[0::4])
    assert abs(res[1][4] - res[4][4]) < 1e-2 * res[1][4]       # SNR guess: max|Q| on the fine vs coarse grid
    rholms, crossTerms, crossTermsV = res[4][3], res[4][1], res[4][2]
    packed = {d: _pack(rholms[d], crossTerms[d], crossTermsV[d]) for d in dets}
    Pv = _pvec([0.8], [-0.4], [0.6], [0.9], [1.1], [200.0], s["tpk"])
    tvals = np.linspace(-0.01, 0.01, int(0.02 / DT))
    lnL = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, Pv, {d: packed[d][0] for d in dets}, {d: packed[d][3] for d in dets},
        {d: packed[d][1] for d in dets}, {d: packed[d][2] for d in dets}, {d: packed[d][4] for d in dets},
        Lmax=2, xpy=np, td_subsample=4)
    assert np.isfinite(lnL).all()
    print("[td-ss d] driver with subsample=4: %d entries per detector at deltaT/4; NoLoop lnL = %.3f"
          % (4 * N_window, float(np.asarray(lnL)[0])))


# --------------------------------------------------------------------------- (e) refusals

def test_refusals(s, monkeypatch):
    with pytest.raises(ValueError, match="positive integer"):
        tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., ifo="H1", verbose=False, subsample=0)
    ctx = ctx_for(s, s["ringdown"], 2, 4096)
    rho, U, V = ctx.precompute(s["hT"], s["N_shift"], s["N_window"])
    lookupNK, ctU, ctV, rhoArr, epoch = _pack(rho, U, V)
    P = _pvec([0.8], [-0.4], [0.6], [0.9], [1.1], [200.0], epoch + 0.03)
    args = (np.array([0., DT]), P, {"H1": lookupNK}, {"H1": rhoArr}, {"H1": ctU}, {"H1": ctV}, {"H1": epoch})
    with pytest.raises(ValueError, match="td_subsample"):
        fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(*args, Lmax=2, xpy=np, td_subsample=0)
    # calibration marginalization runs with K > 1 on the loop reduction (test_td_gs_calibration,
    # section g); the fused kernel has no stride and is refused
    with pytest.raises(NotImplementedError, match="fused"):
        fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(*args, Lmax=2, xpy=np, td_subsample=2, n_cal=2, cal_method="fused")
    # the 'sinc' stencil, a Q pregrid, explicit times and the band-limited quadratures have no
    # sub-sample grid form; even scalar U refuses them with K > 1
    U2 = {"H1": ctU[0]}
    V2 = {"H1": ctV[0]}
    for kw in (dict(time_interp="sinc"), dict(time_quadrature="bandlimited"),
               dict(explicit_time_values=True), dict(time_interp="cubic", q_deltaT=DT / 8)):
        with pytest.raises(NotImplementedError):
            fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(*args[:4], U2, V2, *args[6:], Lmax=2, xpy=np, td_subsample=2, **kw)
    # A piece ending just before the template's first sample at the earliest grid shift is
    # template-free at K=1 (accepted); the Lanczos stencil of a shifted phase reaches
    # SHIFT_STENCIL-1 samples before that sample and enters the piece: refused, margin named.
    # (Needs a template with leading zeros: the IMRPhenomTPHM modes start at array index 0,
    # where fractional_shift truncates the stencil exactly as GWTransientTD does.)
    hT_pad = {}
    for m, h in s["hT"].items():
        ts = lal.CreateCOMPLEX16TimeSeries("h", h.epoch, 0., h.deltaT, h.sampleUnits, h.data.length)
        x = np.zeros(h.data.length, dtype=complex)
        x[64:] = np.asarray(h.data.data)[:-64]
        ts.data.data = x
        hT_pad[m] = ts
    lo = s["N_shift"] + 64                               # turn-on at the earliest shift
    piece = [(EPOCH0 + (lo - 200) * DT, EPOCH0 + (lo - 5) * DT)]
    ctx_for(s, piece, 1, 4096).precompute(hT_pad, s["N_shift"], s["N_window"])     # accepted
    with pytest.raises(ValueError, match="Lanczos shift kernel"):
        ctx_for(s, piece, 4, 4096).precompute(hT_pad, s["N_shift"], s["N_window"])
    # tdanalysis without the helper: a clear ImportError naming what is needed
    import tdanalysis.utils
    monkeypatch.setitem(sys.modules, "tdanalysis.utils.subsample_align", None)
    with pytest.raises(ImportError, match="subsample_align"):
        tdg.import_fractional_shift()
    with pytest.raises(ImportError, match="tdanalysis"):
        tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., ifo="H1", verbose=False, subsample=4)
