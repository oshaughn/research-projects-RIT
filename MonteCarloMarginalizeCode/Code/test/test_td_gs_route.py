"""
test_td_gs_route.py: tests for the fully time-domain Precompute code path
(RIFT/likelihood/td_gs_precompute.py, ILE --internal-ile-inner-product td-gs).

Run with  python -m pytest test/test_td_gs_route.py  from MonteCarloMarginalizeCode/Code
(needs lal, lalsimulation with IMRPhenomTPHM, and tdanalysis >= 1.1 on PYTHONPATH).

  (a) test_q_matches_fd_diagonal_full_segment : Q_lm(t) vs the fd-diagonal
      ComputeModeIPTimeSeries on a 2 s segment, compact template, PSD-derived ACF:
      <= 1e-5 of max|Q|, same argmax sample, same epoch/length/deltaT.
  (b) test_u_matches_scalar_u_full_segment  : U(t), V(t) at the grid centre vs the scalar
      ComputeModeCrossTermIP (<= 1e-5 of |U_22|) and U(t) constant across the grid.
  (c) test_ringdown_piece_vs_brute_force    : ringdown-only piece, dense and GS solvers,
      recursion and batched U/V methods, vs a direct dense C^{-1} quadratic form at
      several shifts (<= 1e-8); U(t) varies strongly with t; --td-u-stride error reported.
  (d) test_two_pieces_vs_block_diagonal     : spliced two-piece segment vs a dense
      block-diagonal brute force (<= 1e-8); plus an inspiral piece the template overruns
      at both ends.
  (e) test_acf_file_loading_and_validation  : [tau, R] loader conventions and errors.
  (f) test_default_fd_route_unchanged       : PrecomputeLikelihoodTerms default path and
      PackLikelihoodDataStructuresAsArrays bit-identical to by-hand lsu constructions;
      hlmoft(no_condition=False) still tapers the SEOB branch.
  (g) test_prefix_route_* / test_all_routes_* : the prefix-sum method (windowed sums of one
      kernel correlation per mode) and the column-blocked "chunked" method against the
      recursion, the batched reference and the dense brute force (<= 1e-12 relative
      between methods sharing the GS kernels) on the full segment, a ringdown piece the
      template ends inside, a piece it overruns at both ends, two spliced pieces, and the
      mirror case (template starts inside, overruns the data end); a ringdown piece whose
      end is one sample too early for the prefix condition is sent to the recursion by
      "auto" and refused by u_method="prefix".
  Plus: whiteness diagnostic on noise drawn from the ACF, FD-native approximants refused,
      a piece before the template start refused, and the Precompute driver interface
      (two detectors, packed shapes).

The FD-vs-TD comparisons use a template whose start is smoothed FOR THE TEST ONLY (the
same series is fed to both code paths): the Toeplitz covariance and the circulant FD weight
agree only for signals with little power at the band edges, and an abruptly starting
chirp has plenty.  The code path itself never tapers.
"""

import numpy as np
import pytest

import lal
import lalsimulation as lalsim
import scipy.linalg

import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl

try:
    from tdanalysis.forms.toeplitz_solver import ToeplitzSolver  # noqa: F401
    from tdanalysis.forms.sliding import sliding_self_terms  # noqa: F401
    from tdanalysis.covariance.psd import PSDToAutocorr  # noqa: F401
except Exception as _exc:   # pragma: no cover
    pytest.skip("tdanalysis >= 1.1 not importable (%r)" % (_exc,), allow_module_level=True)

from RIFT.likelihood import td_gs_precompute as tdg

FS = 4096.0
DT = 1.0 / FS
FNYQ = FS / 2.0
EPOCH0 = 1000000000.0
MREF = 65.0
U_METHODS = ["recursion", "batched", "chunked"]      # valid for every cut pattern
ALL_METHODS = U_METHODS + ["prefix", "auto"]


def analytic_psd(f):
    f = np.atleast_1d(np.asarray(f, dtype=float))
    fsafe = np.maximum(np.abs(f), 1.0)
    out = 1e-46 * (1.0 + (20.0 / fsafe) ** 4 + (fsafe / 250.0) ** 2)
    return out if out.size > 1 else float(out[0])


def make_psd(N):
    df = FS / N
    psd = lal.CreateREAL8FrequencySeries("psd", lal.LIGOTimeGPS(0.), 0., df,
                                         lsu.lsu_HertzUnit, N // 2 + 1)
    psd.data.data = analytic_psd(np.arange(N // 2 + 1) * df)
    return psd


def make_P(N, approx=lalsim.IMRPhenomTPHM):
    return lsu.ChooseWaveformParams(m1=35 * lal.MSUN_SI, m2=30 * lal.MSUN_SI, fmin=30.,
                                    deltaT=DT, deltaF=FS / N, dist=100e6 * lal.PC_SI,
                                    approx=approx)


def make_modes(N, smooth_start=None):
    """IMRPhenomTPHM l=2 modes, untapered (td-gs code path).  smooth_start (fraction of the
    template's non-zero length) applies a test-only Hann rise so FD/TD comparisons are
    not dominated by band-edge leakage."""
    hT = tdg.generate_td_modes(make_P(N), 2)
    if smooth_start:
        last = int(np.flatnonzero(np.abs(hT[(2, 2)].data.data) > 0)[-1])
        ntap = int(smooth_start * last)
        win = np.ones(N)
        win[:ntap] = 0.5 - 0.5 * np.cos(np.pi * np.arange(ntap) / ntap)
        for m in hT:
            hT[m].data.data *= win
    return hT


def make_data(N, hT, k_inj, seed=3, amp=0.3, sigma=1e-23):
    rng = np.random.default_rng(seed)
    d = sigma * rng.standard_normal(N)
    h22 = np.asarray(hT[(2, 2)].data.data)
    d[k_inj:] += amp * np.real(h22[:N - k_inj])
    ts = lal.CreateREAL8TimeSeries("d", lal.LIGOTimeGPS(EPOCH0), 0., DT,
                                   lsu.lsu_DimensionlessUnit, N)
    ts.data.data = d
    return ts


def fd_versions(dts, hT):
    N = dts.data.length
    dC = lal.CreateCOMPLEX16TimeSeries("d", dts.epoch, 0., DT, lsu.lsu_DimensionlessUnit, N)
    dC.data.data = np.asarray(dts.data.data) + 0j
    dF = lsu.DataFourier(dC)
    hF, hFc = {}, {}
    for m in hT:
        c = lal.CutCOMPLEX16TimeSeries(hT[m], 0, N)
        hF[m] = lsu.DataFourier(c)
        c.data.data = np.conj(c.data.data)
        hFc[m] = lsu.DataFourier(c)
    return dF, hF, hFc


def peak_time(hT, k_inj):
    """GPS time of the injected template's t=0 (its peak reference)."""
    return EPOCH0 + k_inj * DT - float(hT[(2, 2)].epoch)


def brute_force(d, acf_fn, runs, hT, N_shift, idx):
    """Direct dense block-diagonal C^{-1} forms at grid indices idx."""
    N = d.size
    modes = list(hT.keys())
    out = {}
    blocks = []
    for (i0, i1) in runs:
        Cinv = np.linalg.inv(scipy.linalg.toeplitz(acf_fn(i1 - i0)))
        blocks.append((i0, i1, Cinv, Cinv @ d[i0:i1]))
    for i in idx:
        k = N_shift + i
        hk = {}
        for m in modes:
            h = np.asarray(hT[m].data.data)
            x = np.zeros(N, dtype=complex)
            if k >= 0:
                x[k:] = h[:N - k]
            else:
                x[:N + k] = h[-k:]
            hk[m] = x
        for ma in modes:
            q = sum(np.conj(hk[ma][i0:i1]) @ w for (i0, i1, _, w) in blocks)
            out[("Q", ma, i)] = q
            for mb in modes:
                u = sum(np.conj(hk[ma][i0:i1]) @ Cinv @ hk[mb][i0:i1] for (i0, i1, Cinv, _) in blocks)
                v = sum(hk[ma][i0:i1] @ Cinv @ hk[mb][i0:i1] for (i0, i1, Cinv, _) in blocks)
                out[("U", ma, mb, i)] = u
                out[("V", ma, mb, i)] = v
    return out


def compare_brute(ctx, hT, rho, U, V, N_shift, idx, ref=None):
    if ref is None:
        ref = brute_force(ctx.d, ctx.acf_lags, ctx.runs, hT, N_shift, idx)
    modes = list(hT.keys())
    scale = np.max(np.abs(U[((2, 2), (2, 2))]))
    qscale = np.max(np.abs(rho[(2, 2)].data.data))
    worst = 0.0
    for i in idx:
        for ma in modes:
            worst = max(worst, abs(ref[("Q", ma, i)] - rho[ma].data.data[i]) / qscale)
            for mb in modes:
                worst = max(worst, abs(ref[("U", ma, mb, i)] - U[(ma, mb)][i]) / scale,
                            abs(ref[("V", ma, mb, i)] - V[(ma, mb)][i]) / scale)
    return worst


# --------------------------------------------------------------------------- (a), (b): full segment

@pytest.fixture(scope="module")
def full_segment():
    N = 8192
    FLOW, FMAX = 20.0, 2000.0
    psd = make_psd(N)
    hT = make_modes(N, smooth_start=0.5)
    k_inj = N // 2 - 250
    dts = make_data(N, hT, k_inj)
    N_shift, N_window = k_inj - 64, 256
    dF, hF, hFc = fd_versions(dts, hT)
    rho_fd = fl.ComputeModeIPTimeSeries(hF, dF, psd, FLOW, FMAX, FNYQ, N_shift, N_window,
                                        False, False, 0.)
    U_fd = fl.ComputeModeCrossTermIP(hF, hF, psd, FLOW, FMAX, FNYQ, FS / N, False, False, 0.,
                                     verbose=False, same_waveform_Q=True)
    V_fd = fl.ComputeModeCrossTermIP(hFc, hF, psd, FLOW, FMAX, FNYQ, FS / N, False, False, 0.,
                                     verbose=False, same_waveform_Q=True, prefix="V")
    ctx = tdg.TDGSDetectorContext(dts, psd, FLOW, FMAX, ifo="H1", verbose=False)
    rho_td, U_td, V_td = ctx.precompute(hT, N_shift, N_window)
    return dict(hT=hT, rho_fd=rho_fd, U_fd=U_fd, V_fd=V_fd, rho_td=rho_td, U_td=U_td,
                V_td=V_td, N_window=N_window)


def test_q_matches_fd_diagonal_full_segment(full_segment):
    s = full_segment
    qmax = np.max(np.abs(s["rho_fd"][(2, 2)].data.data))
    worst = 0.0
    for m in s["hT"]:
        a, b = s["rho_fd"][m], s["rho_td"][m]
        assert b.data.length == s["N_window"] == a.data.length
        assert abs(b.deltaT - a.deltaT) < 1e-15
        assert float(b.epoch) == float(a.epoch)
        assert np.argmax(np.abs(a.data.data)) == np.argmax(np.abs(b.data.data))
        worst = max(worst, np.max(np.abs(a.data.data - b.data.data)) / qmax)
    print("[td-gs a] Q(t) td-gs vs fd-diagonal: max|diff|/max|Q22| = %.3e" % worst)
    assert worst < 1e-5


def test_u_matches_scalar_u_full_segment(full_segment):
    s = full_segment
    u22 = abs(s["U_fd"][((2, 2), (2, 2))])
    c = s["N_window"] // 2
    worst_u = max(abs(s["U_td"][k][c] - s["U_fd"][k]) / u22 for k in s["U_fd"])
    worst_v = max(abs(s["V_td"][k][c] - s["V_fd"][k]) / u22 for k in s["V_fd"])
    var = max((np.max(np.abs(s["U_td"][k])) - np.min(np.abs(s["U_td"][k]))) / u22 for k in s["U_td"])
    print("[td-gs b] U(t_c) vs scalar U: %.3e; V: %.3e; max U variation across grid: %.3e"
          % (worst_u, worst_v, var))
    assert worst_u < 1e-5 and worst_v < 1e-5
    assert var < 1e-7
    for k in s["U_td"]:
        assert s["U_td"][k].shape == (s["N_window"],)
        assert np.allclose(s["U_td"][(k[1], k[0])], np.conj(s["U_td"][k]))
        assert np.allclose(s["V_td"][(k[1], k[0])], s["V_td"][k])


# --------------------------------------------------------------------------- (c): ringdown piece

@pytest.fixture(scope="module")
def ringdown_setup():
    N = 4096
    psd = make_psd(N)
    hT = make_modes(N)
    k_inj = 1800
    dts = make_data(N, hT, k_inj)
    tpk = peak_time(hT, k_inj)
    msec = MREF * lal.MTSUN_SI
    pieces = [(tpk - 0.002, tpk + 120 * msec)]
    return dict(N=N, psd=psd, hT=hT, dts=dts, tpk=tpk, pieces=pieces, N_shift=k_inj - 64,
                N_window=256)


@pytest.mark.parametrize("u_method", U_METHODS)
@pytest.mark.parametrize("dense_max_n", [4096, 0])
def test_ringdown_piece_vs_brute_force(ringdown_setup, dense_max_n, u_method):
    s = ringdown_setup
    ctx = tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., pieces=s["pieces"],
                                  dense_max_n=dense_max_n, u_method=u_method, ifo="H1", verbose=False)
    assert (ctx.solvers[0].method == "dense") == (dense_max_n > 0)
    rho, U, V = ctx.precompute(s["hT"], s["N_shift"], s["N_window"])
    idx = [0, 37, 128, 200, 255]
    worst = compare_brute(ctx, s["hT"], rho, U, V, s["N_shift"], idx)
    u = np.abs(U[((2, 2), (2, 2))])
    print("[td-gs c] ringdown piece (%d samples, %s solver, %s U/V): brute-force worst rel %.3e; "
          "U22 max/min over the grid = %.1f" % (ctx.runs[0][1] - ctx.runs[0][0],
                                                "dense" if dense_max_n else "gs", u_method,
                                                worst, u.max() / u.min()))
    assert worst < 1e-8
    assert u.max() / u.min() > 100


def test_recursion_equals_batched_everywhere(ringdown_setup):
    """The two U/V methods agree at EVERY grid sample (brute force only probes a few)."""
    s = ringdown_setup
    out = {}
    for method in U_METHODS:
        ctx = tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., pieces=s["pieces"],
                                      u_method=method, ifo="H1", verbose=False)
        out[method] = ctx.precompute(s["hT"], s["N_shift"], s["N_window"])
    scale = np.max(np.abs(out["batched"][1][((2, 2), (2, 2))]))
    worst = max(max(np.max(np.abs(out["recursion"][i][k] - out["batched"][i][k])) / scale
                    for k in out["batched"][i]) for i in (1, 2))
    worst_q = max(np.max(np.abs(out["recursion"][0][m].data.data - out["batched"][0][m].data.data))
                  for m in s["hT"]) / np.max(np.abs(out["batched"][0][(2, 2)].data.data))
    print("[td-gs c] recursion vs batched over the whole grid: U/V %.3e, Q %.3e" % (worst, worst_q))
    assert worst < 1e-8 and worst_q < 1e-12     # both methods sit ~1e-9 from the dense reference


def test_u_stride_interpolation_error(ringdown_setup):
    s = ringdown_setup
    exact = tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., pieces=s["pieces"],
                                    ifo="H1", verbose=False).precompute(s["hT"], s["N_shift"], s["N_window"])
    scale = np.max(np.abs(exact[1][((2, 2), (2, 2))]))
    errs = {}
    for stride in (2, 4, 8):
        strided = tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., pieces=s["pieces"],
                                          u_stride=stride, ifo="H1", verbose=False).precompute(
            s["hT"], s["N_shift"], s["N_window"])
        err = max(np.max(np.abs(strided[1][k] - exact[1][k])) / scale for k in exact[1])
        errv = max(np.max(np.abs(strided[2][k] - exact[2][k])) / scale for k in exact[2])
        assert np.array_equal(strided[0][(2, 2)].data.data, exact[0][(2, 2)].data.data)  # Q exact
        errs[stride] = (err, errv)
        print("[td-gs c] --td-u-stride %d: max |U interp error| / max|U22| = %.3e (V: %.3e)"
              % (stride, err, errv))
    assert errs[2][0] < 1e-3 and errs[4][0] < 1e-2


# --------------------------------------------------------------------------- (d): spliced pieces

@pytest.mark.parametrize("u_method", U_METHODS)
def test_two_pieces_vs_block_diagonal(ringdown_setup, u_method):
    s = ringdown_setup
    tpk = s["tpk"]
    pieces = [(tpk - 0.15, tpk - 0.05), (tpk - 0.01, tpk + 0.04)]
    ctx = tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., pieces=pieces, u_method=u_method,
                                  ifo="H1", verbose=False)
    assert len(ctx.runs) == 2
    rho, U, V = ctx.precompute(s["hT"], s["N_shift"], s["N_window"])
    idx = [0, 50, 128, 255]
    worst = compare_brute(ctx, s["hT"], rho, U, V, s["N_shift"], idx)
    print("[td-gs d] two spliced pieces %s (%s) vs dense block-diagonal: worst rel %.3e"
          % (ctx.runs, u_method, worst))
    assert worst < 1e-8


@pytest.mark.parametrize("u_method", U_METHODS)
def test_inspiral_piece_template_overruns_both_ends(ringdown_setup, u_method):
    """A short inspiral piece: the template starts before it and ends after it for every
    shift, so both cuts are active."""
    s = ringdown_setup
    tpk = s["tpk"]
    pieces = [(tpk - 0.17, tpk - 0.10)]
    ctx = tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., pieces=pieces, u_method=u_method,
                                  ifo="H1", verbose=False)
    rho, U, V = ctx.precompute(s["hT"], s["N_shift"], s["N_window"])
    idx = [0, 1, 100, 254, 255]
    worst = compare_brute(ctx, s["hT"], rho, U, V, s["N_shift"], idx)
    print("[td-gs d] inspiral piece %s (%s), template overruns both ends: worst rel %.3e"
          % (ctx.runs, u_method, worst))
    assert worst < 1e-8


# --------------------------------------------------------------------------- (g): prefix / chunked methods

def template_support(hT):
    """(j_first, j_last) over all modes: first and last non-zero template sample."""
    nz = [np.flatnonzero(np.abs(np.asarray(h.data.data)) > 0) for h in hT.values()]
    return min(int(s[0]) for s in nz), max(int(s[-1]) for s in nz)


def route_outputs(s, pieces, methods, dense_max_n=0, N_shift=None, hT=None, dts=None, **kw):
    """precompute on the same data/pieces for every method; returns {method: (ctx, rho, U, V)}."""
    out = {}
    for method in methods:
        ctx = tdg.TDGSDetectorContext(dts or s["dts"], s["psd"], 20., 1500., pieces=pieces,
                                      dense_max_n=dense_max_n, u_method=method, ifo="H1",
                                      verbose=False, **kw)
        rho, U, V = ctx.precompute(hT or s["hT"], s["N_shift"] if N_shift is None else N_shift,
                                   s["N_window"])
        out[method] = (ctx, rho, U, V)
    return out


def route_diff(a, b):
    """Worst |diff| / max|U22| over all U, V pairs and all grid samples, and the same for Q."""
    _, rho_a, U_a, V_a = a
    _, rho_b, U_b, V_b = b
    scale = np.max(np.abs(U_b[((2, 2), (2, 2))]))
    worst = max(max(np.max(np.abs(U_a[k] - U_b[k])) for k in U_b),
                max(np.max(np.abs(V_a[k] - V_b[k])) for k in V_b)) / scale
    worst_q = max(np.max(np.abs(rho_a[m].data.data - rho_b[m].data.data)) for m in rho_b) \
        / np.max(np.abs(rho_b[(2, 2)].data.data))
    return worst, worst_q


def check_routes(label, s, out, idx, ref="recursion", tol_route=1e-12, tol_brute=1e-8,
                 hT=None, N_shift=None):
    """Every method vs ``ref`` over the whole grid and vs the dense brute force at ``idx``
    (one brute force per layout: all methods share data, ACF and pieces)."""
    hT = hT or s["hT"]
    N_shift = s["N_shift"] if N_shift is None else N_shift
    ctx0 = out[ref][0]
    brute = brute_force(ctx0.d, ctx0.acf_lags, ctx0.runs, hT, N_shift, idx)
    for method, (ctx, rho, U, V) in out.items():
        assert ctx.runs == ctx0.runs and np.array_equal(ctx.acf, ctx0.acf)
        worst_b = compare_brute(ctx, hT, rho, U, V, N_shift, idx, ref=brute)
        worst_r, worst_q = route_diff(out[method], out[ref])
        print("[td-gs g] %s: %-9s methods %s: vs %s U/V %.2e Q %.2e; vs brute force %.2e"
              % (label, method, ctx.uv_routes, ref, worst_r, worst_q, worst_b))
        assert worst_b < tol_brute
        assert worst_r < tol_route and worst_q < tol_route
        for k in U:
            if k[0] != k[1]:
                assert np.array_equal(U[(k[1], k[0])], np.conj(U[k]))
                assert np.array_equal(V[(k[1], k[0])], V[k])


def test_all_routes_full_segment_gs(ringdown_setup):
    """Full segment on a GS solver: the template ends inside for every shift, so auto
    picks the prefix method; all methods agree to rounding and match the brute force."""
    s = ringdown_setup
    j_first, j_last = template_support(s["hT"])
    assert j_last + s["N_shift"] + s["N_window"] - 1 <= s["N"] - 1
    out = route_outputs(s, None, ALL_METHODS, dense_max_n=0)
    assert out["auto"][0].uv_routes == [("prefix", "end")]
    assert out["prefix"][0].uv_routes == [("prefix", "end")]
    assert out["chunked"][0].uv_routes == [("chunked", None)]
    check_routes("full segment (gs, n=%d)" % s["N"], s, out, [0, 1, 77, 128, 254, 255])


def ringdown_piece_ending_at(s, i1):
    """[tpk - 2 ms, data sample i1) as GPS seconds."""
    return [(s["tpk"] - 0.002, EPOCH0 + i1 * DT)]


def test_prefix_route_ringdown_piece_and_one_shift_violation(ringdown_setup):
    """A ringdown piece whose end is exactly the last sample the template reaches at the
    largest grid shift satisfies the prefix condition; one sample shorter and it does not:
    auto then uses the recursion and u_method='prefix' refuses."""
    s = ringdown_setup
    j_first, j_last = template_support(s["hT"])
    i1_ok = j_last + s["N_shift"] + s["N_window"]        # last template sample at i1_ok - 1
    out = route_outputs(s, ringdown_piece_ending_at(s, i1_ok), ALL_METHODS, dense_max_n=0)
    ctx = out["auto"][0]
    assert ctx.runs[0][1] == i1_ok and ctx.solvers[0].method == "gs"
    assert ctx.uv_routes == [("prefix", "end")]
    check_routes("ringdown piece %s" % (ctx.runs,), s, out, [0, 1, 100, 254, 255])
    # one shift of the grid now puts the template's last sample past the piece end
    out2 = route_outputs(s, ringdown_piece_ending_at(s, i1_ok - 1), ["auto", "recursion", "chunked"],
                         dense_max_n=0)
    ctx2 = out2["auto"][0]
    assert ctx2.runs[0][1] == i1_ok - 1
    assert ctx2.uv_routes == [("recursion", None)]
    check_routes("ringdown piece %s (prefix condition fails by one shift)" % (ctx2.runs,),
                 s, out2, [0, 1, 100, 254, 255])
    with pytest.raises(ValueError, match="u_method='prefix'"):
        route_outputs(s, ringdown_piece_ending_at(s, i1_ok - 1), ["prefix"], dense_max_n=0)
    # the dense solver keeps the batched method on auto whatever the condition
    ctx3 = route_outputs(s, ringdown_piece_ending_at(s, i1_ok), ["auto"], dense_max_n=4096)["auto"][0]
    assert ctx3.uv_routes == [("chunked", None)]


def test_all_routes_inspiral_piece_overrun_both_ends_gs(ringdown_setup):
    """GS solver on a short inspiral piece the template overruns at both ends: no prefix
    orientation exists, auto uses the recursion, 'prefix' refuses."""
    s = ringdown_setup
    pieces = [(s["tpk"] - 0.17, s["tpk"] - 0.10)]
    out = route_outputs(s, pieces, U_METHODS + ["auto"], dense_max_n=0)
    ctx = out["auto"][0]
    j_first, j_last = template_support(s["hT"])
    i0, i1 = ctx.runs[0]
    assert j_first + s["N_shift"] + s["N_window"] - 1 < i0 and j_last + s["N_shift"] > i1 - 1
    assert ctx.uv_routes == [("recursion", None)]
    check_routes("inspiral piece %s (gs)" % (ctx.runs,), s, out, [0, 1, 100, 254, 255])
    with pytest.raises(ValueError, match="u_method='prefix'"):
        route_outputs(s, pieces, ["prefix"], dense_max_n=0)


def test_all_routes_two_spliced_pieces(ringdown_setup):
    """Two spliced GS pieces: auto picks per piece (prefix where the template ends inside,
    recursion elsewhere); every method matches the block-diagonal brute force."""
    s = ringdown_setup
    tpk = s["tpk"]
    j_first, j_last = template_support(s["hT"])
    i1_ok = j_last + s["N_shift"] + s["N_window"]
    # inspiral piece the template overruns at both ends + a ringdown piece it ends inside
    pieces = [(tpk - 0.15, tpk - 0.05), (tpk - 0.01, EPOCH0 + i1_ok * DT)]
    out = route_outputs(s, pieces, U_METHODS + ["auto"], dense_max_n=0)
    ctx = out["auto"][0]
    assert len(ctx.runs) == 2
    assert ctx.uv_routes == [("recursion", None), ("prefix", "end")]
    check_routes("two pieces %s (gs, mixed)" % (ctx.runs,), s, out, [0, 50, 128, 255])
    with pytest.raises(ValueError, match="u_method='prefix'"):
        route_outputs(s, pieces, ["prefix"], dense_max_n=0)
    # second piece one sample short: both on the recursion
    pieces = [(tpk - 0.15, tpk - 0.05), (tpk - 0.01, EPOCH0 + (i1_ok - 1) * DT)]
    out = route_outputs(s, pieces, U_METHODS + ["auto"], dense_max_n=0)
    assert out["auto"][0].uv_routes == [("recursion", None), ("recursion", None)]
    check_routes("two pieces %s (gs)" % (out["auto"][0].runs,), s, out, [0, 50, 128, 255])
    # dense first piece, GS second: chunked + prefix
    pieces = [(tpk - 0.15, tpk - 0.05), (tpk - 0.01, EPOCH0 + i1_ok * DT)]
    n1, n2 = [b - a for (a, b) in ctx.runs]
    assert n1 < n2
    out = route_outputs(s, pieces, ["auto", "recursion"], dense_max_n=n1)
    assert out["auto"][0].uv_routes == [("chunked", None), ("prefix", "end")]
    check_routes("two pieces %s (dense + gs)" % (out["auto"][0].runs,), s, out, [0, 50, 128, 255],
                 tol_route=1e-8)     # Cholesky solve vs GS kernels at cond 1e8: ~1e-9


def test_prefix_route_mirror_template_overruns_data_end(ringdown_setup):
    """Full segment with the injection late in the data: the template starts inside but
    its ringdown runs past the data end for the late shifts.  The mirror ("start")
    orientation of the prefix method applies (persymmetry of C^-1)."""
    s = ringdown_setup
    N = s["N"]
    k_inj = 3000
    dts = make_data(N, s["hT"], k_inj)
    N_shift = k_inj - 64
    j_first, j_last = template_support(s["hT"])
    assert j_last + N_shift + s["N_window"] - 1 > N - 1 and j_first + N_shift >= 0
    # The numerical GS form built from x, y is persymmetric only to ~cond(C) eps, and
    # the mirror evaluates it on the reversed vectors: at the default cond 1e8 it sits
    # ~1e-9 from the recursion (as far as every method sits from the brute force); at
    # cond 1e4 everything agrees to 1e-12, brute force included.
    for cond, tol, methods in ((1e8, 1e-8, ["recursion", "prefix", "auto"]), (1e4, 1e-12, ALL_METHODS)):
        out = route_outputs(s, None, methods, dense_max_n=0, N_shift=N_shift, dts=dts,
                            acf_condition_number=cond)
        assert out["auto"][0].uv_routes == [("prefix", "start")]
        check_routes("full segment, template overruns the data end (cond %.0e)" % cond, s, out,
                     [0, 1, 128, 254, 255], N_shift=N_shift, tol_route=tol, tol_brute=tol)


def test_all_routes_well_conditioned_acf_vs_brute_force(ringdown_setup):
    """With a well-conditioned ACF (cond 1e4) the dense brute force is itself good to
    ~1e-13, and every method on every layout reproduces it to 1e-12."""
    s = ringdown_setup
    tpk = s["tpk"]
    j_first, j_last = template_support(s["hT"])
    i1_ok = j_last + s["N_shift"] + s["N_window"]
    layouts = {
        "full segment": (None, ALL_METHODS),
        "ringdown piece": (ringdown_piece_ending_at(s, i1_ok), ALL_METHODS),
        "inspiral piece": ([(tpk - 0.17, tpk - 0.10)], U_METHODS + ["auto"]),
        "two pieces": ([(tpk - 0.15, tpk - 0.05), (tpk - 0.01, EPOCH0 + i1_ok * DT)], U_METHODS + ["auto"]),
    }
    for label, (pieces, methods) in layouts.items():
        out = route_outputs(s, pieces, methods, dense_max_n=0, acf_condition_number=1e4)
        check_routes("%s (cond 1e4)" % label, s, out, [0, 1, 128, 254, 255],
                     tol_route=1e-12, tol_brute=1e-12)


def test_chunked_block_size_independence(ringdown_setup):
    """The chunked method gives the batched result for any block size (down to one
    column per block) on dense and GS pieces.  On the Cholesky solver (dense) the block
    size changes the order of the triangular-solve sums, ~1e-13 of max|U| at cond 1e8."""
    s = ringdown_setup
    for dense_max_n in (4096, 0):
        ref = route_outputs(s, s["pieces"], ["batched"], dense_max_n=dense_max_n)["batched"]
        n = ref[0].runs[0][1] - ref[0].runs[0][0]
        for block_bytes in (1, 40 * 16 * n * 6, 10 ** 9):
            got = route_outputs(s, s["pieces"], ["chunked"], dense_max_n=dense_max_n,
                                uv_block_bytes=block_bytes)["chunked"]
            worst, worst_q = route_diff(got, ref)
            print("[td-gs g] chunked (%s, block %d bytes) vs batched: U/V %.2e Q %.2e"
                  % ("dense" if dense_max_n else "gs", block_bytes, worst, worst_q))
            assert worst < (1e-12 if dense_max_n else 1e-13) and worst_q == 0.0


def test_u_method_names(ringdown_setup):
    s = ringdown_setup
    assert set(tdg.U_METHODS) == set(ALL_METHODS)
    for bad in ("prefixsum", "", None):
        with pytest.raises(ValueError, match="u_method"):
            tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., pieces=s["pieces"],
                                    u_method=bad, ifo="H1", verbose=False)


# --------------------------------------------------------------------------- (e): ACF files

def test_acf_file_loading_and_validation(tmp_path, ringdown_setup):
    s = ringdown_setup
    ctx_psd = tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., pieces=s["pieces"],
                                      ifo="H1", verbose=False)
    acf = ctx_psd.acf_lags(2048)
    tau = np.arange(acf.size) * DT
    good = tmp_path / "acf.npy"
    np.save(good, np.vstack([tau, acf]))
    np.testing.assert_array_equal(tdg.load_acf_file(str(good), DT), acf)
    swapped = tmp_path / "acf_swapped.npy"
    np.save(swapped, np.vstack([acf, tau]))               # rows the other way round
    np.testing.assert_array_equal(tdg.load_acf_file(str(swapped), DT), acf)
    bad_fs = tmp_path / "acf_2048.npy"
    np.save(bad_fs, np.vstack([np.arange(acf.size) / 2048., acf]))
    with pytest.raises(ValueError, match="--srate 2048"):
        tdg.load_acf_file(str(bad_fs), DT)
    bad_shape = tmp_path / "acf_bad.npy"
    np.save(bad_shape, np.zeros((3, 10)))
    with pytest.raises(ValueError, match="expected shape"):
        tdg.load_acf_file(str(bad_shape), DT)
    # Too short for the piece: refused with a clear message.
    short = tdg.load_acf_file(str(good), DT)[:50]
    with pytest.raises(ValueError, match="ACF too short"):
        tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., pieces=s["pieces"], acf=short,
                                ifo="H1", verbose=False)
    # A file holding the PSD-derived ACF reproduces the PSD-derived one exactly.
    ctx_file = tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., pieces=s["pieces"],
                                       acf=tdg.load_acf_file(str(good), DT), ifo="H1", verbose=False)
    assert ctx_file.acf_supplied and not ctx_psd.acf_supplied
    a = ctx_psd.precompute(s["hT"], s["N_shift"], s["N_window"])
    b = ctx_file.precompute(s["hT"], s["N_shift"], s["N_window"])
    assert np.allclose(a[1][((2, 2), (2, 2))], b[1][((2, 2), (2, 2))], rtol=1e-12, atol=0)
    assert np.allclose(a[0][(2, 2)].data.data, b[0][(2, 2)].data.data, rtol=1e-12, atol=0)
    print("[td-gs e] ACF file loader: conventions and validation errors OK")


def test_td_piece_parsing():
    assert tdg.parse_td_pieces(None, None) == []
    msec = MREF * lal.MTSUN_SI
    out = tdg.parse_td_pieces(["0,100", "-300,-100"], MREF)
    assert np.allclose(out, [(-300 * msec, -100 * msec), (0., 100 * msec)])
    with pytest.raises(ValueError, match="td-mref"):
        tdg.parse_td_pieces(["0,100"], None)
    with pytest.raises(ValueError, match="overlap"):
        tdg.parse_td_pieces(["0,100", "50,150"], MREF)
    with pytest.raises(ValueError, match="END must exceed"):
        tdg.parse_td_pieces(["100,0"], MREF)


def test_whiteness_on_noise_drawn_from_acf(ringdown_setup):
    """Noise d = L z (C = L L^T from the ACF) gives d^T C^-1 d / n ~ 1 and unit whitened
    variance; white noise with the wrong scale does not."""
    s = ringdown_setup
    N = s["N"]
    acf = tdg.acf_from_psd(s["psd"], FS / N, FS, N, 20., 1500.)[0]
    L = np.linalg.cholesky(scipy.linalg.toeplitz(acf))
    z = np.random.default_rng(11).standard_normal(N)
    ts = lal.CreateREAL8TimeSeries("d", lal.LIGOTimeGPS(EPOCH0), 0., DT, lsu.lsu_DimensionlessUnit, N)
    ts.data.data = L @ z
    ctx = tdg.TDGSDetectorContext(ts, s["psd"], 20., 1500., whiteness_check=True, ifo="H1", verbose=False)
    e = ctx.whiteness[0]
    print("[td-gs] whiteness on ACF-coloured noise: chi2/n = %.3f, whitened var %.3f, lag-1 %.3f; "
          "PSD->ACF info: %s" % (e["chi2_per_n"], e["white_var"], e["white_lag1"],
                                {k: ctx.acf_info[k] for k in ("n_floored", "condition_number") if k in ctx.acf_info}))
    assert abs(e["chi2_per_n"] - 1) < 0.1 and abs(e["white_var"] - 1) < 0.1 and abs(e["white_lag1"]) < 0.1
    assert int(ctx.acf_info.get("n_floored", 0) or 0) == 0
    # the same piece on the GS solver: whitened by Levinson-Durbin, same diagnostic
    ctx_gs = tdg.TDGSDetectorContext(ts, s["psd"], 20., 1500., whiteness_check=True, dense_max_n=0,
                                     ifo="H1", verbose=False)
    assert ctx_gs.solvers[0].method == "gs"
    e_gs = ctx_gs.whiteness[0]
    print("[td-gs] whiteness, GS piece (Levinson-Durbin): whitened var %.6f (Cholesky %.6f), "
          "lag-1 %.6f (Cholesky %.6f)" % (e_gs["white_var"], e["white_var"], e_gs["white_lag1"], e["white_lag1"]))
    assert abs(e_gs["white_var"] - e["white_var"]) < 1e-5 * e["white_var"]
    assert abs(e_gs["white_lag1"] - e["white_lag1"]) < 1e-5
    ctx_bad = tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., ifo="H1", verbose=False)
    assert not (0.7 <= ctx_bad.whiteness[0]["chi2_per_n"] <= 1.4)


# --------------------------------------------------------------------------- refusals

def test_fd_native_approximants_refused():
    for approx in (lalsim.IMRPhenomD, lalsim.IMRPhenomXPHM, lalsim.IMRPhenomPv2):
        assert tdg.template_route_is_fd(approx)
        with pytest.raises(ValueError, match="frequency domain"):
            tdg.generate_td_modes(make_P(4096, approx), 2)
    for approx in (lalsim.IMRPhenomTPHM, lalsim.SEOBNRv4, lalsim.SEOBNRv4PHM, lalsim.TaylorT4):
        assert not tdg.template_route_is_fd(approx)


def test_piece_before_template_start_refused(ringdown_setup):
    s = ringdown_setup
    # The template switches on 0.2256 s before its peak; a piece starting 0.3 s before the
    # peak contains that turn-on for every grid shift.
    ctx = tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500.,
                                  pieces=[(s["tpk"] - 0.3, s["tpk"] + 0.03)], ifo="H1", verbose=False)
    with pytest.raises(ValueError, match="turn-on"):
        ctx.precompute(s["hT"], s["N_shift"], s["N_window"])
    # Outside the valid data: refused at construction.
    with pytest.raises(ValueError, match="does not lie inside"):
        tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500.,
                                pieces=[(EPOCH0 - 1., EPOCH0 + 0.1)], ifo="H1", verbose=False)


# --------------------------------------------------------------------------- driver interface

def test_precompute_driver_interface(ringdown_setup):
    s = ringdown_setup
    N = s["N"]
    dets = ["H1", "L1"]
    data_dict = {d: s["dts"] for d in dets}
    psd_dict = {d: s["psd"] for d in dets}
    contexts = {d: tdg.TDGSDetectorContext(s["dts"], s["psd"], 20., 1500., ifo=d, verbose=False)
                for d in dets}
    P = make_P(N)
    P.tref = s["tpk"]
    t_window = 0.03
    out = fl.PrecomputeLikelihoodTerms(s["tpk"], t_window, P, data_dict, psd_dict, 2, 1500.,
                                       False, False, 0., verbose=False, quiet=True,
                                       skip_interpolation=True, td_gs_contexts=contexts,
                                       return_calibration_crossterms=True)
    rholms_intp, crossTerms, crossTermsV, rholms, guess_snr, rest, ct_cal, ct_cal_V = out
    assert ct_cal is None and ct_cal_V is None and rest is None
    assert np.isfinite(guess_snr) and guess_snr > 0
    N_window = int(2 * t_window / DT)
    for det in dets:
        for m in rholms[det]:
            assert isinstance(rholms[det][m], lal.COMPLEX16TimeSeries)
            assert rholms[det][m].data.length == N_window
        for k in crossTerms[det]:
            assert crossTerms[det][k].shape == (N_window,)
            assert crossTermsV[det][k].shape == (N_window,)
        packed = fl.PackLikelihoodDataStructuresAsArrays(
            list(rholms[det].keys()), None, rholms[det], crossTerms[det], crossTermsV[det])
        _, lookupKN, _, ctU, ctV, rholmArray, _, epoch = packed
        n = len(rholms[det])
        assert ctU.shape == (N_window, n, n) and ctV.shape == (N_window, n, n)
        assert rholmArray.shape == (n, N_window)
        i22 = lookupKN[(2, 2)]
        np.testing.assert_array_equal(ctU[:, i22, i22], crossTerms[det][((2, 2), (2, 2))])
        assert epoch == float(rholms[det][(2, 2)].epoch)
        # The (2,2) data-term peak sits where the FD code path bookkeeping puts the injection:
        # grid index k_inj - N_shift with N_shift from t_det (the injection is at the
        # geocenter time; the detector arrival time differs by the light-travel delay).
        t_det = fl.ComputeArrivalTimeAtDetector(det, P.phi, P.theta, s["tpk"])
        rho_epoch = s["dts"].epoch - s["hT"][(2, 2)].epoch
        N_shift = int(float(float(t_det) - t_window - float(rho_epoch)) / DT + 0.5)
        i_pk = np.argmax(np.abs(rholms[det][(2, 2)].data.data))
        assert i_pk == 1800 - N_shift
    # Unsupported options are refused loudly.
    with pytest.raises(ValueError, match="not supported"):
        fl.PrecomputeLikelihoodTerms(s["tpk"], t_window, P, data_dict, psd_dict, 2, 1500.,
                                     verbose=False, quiet=True, skip_interpolation=True,
                                     td_gs_contexts=contexts, IP_factory=object())
    print("[td-gs] driver: 2 detectors, %d modes, N_window=%d, guess_snr=%.2f" % (n, N_window, guess_snr))


# --------------------------------------------------------------------------- (f): default code path unchanged

def test_default_fd_route_unchanged():
    N = 4096
    FLOW, FMAX = 20.0, 1500.0
    psd = make_psd(N)
    hT = make_modes(N)
    dts = make_data(N, hT, 1800)
    dF, _, _ = fd_versions(dts, hT)
    det = "H1"
    P = make_P(N)
    P.fmin = FLOW
    t_window = 0.03
    tref = EPOCH0 + 0.5
    out = fl.PrecomputeLikelihoodTerms(tref, t_window, P, {det: dF}, {det: psd}, 2, FMAX,
                                       False, False, 0., verbose=False, quiet=True,
                                       skip_interpolation=True, ignore_threshold=None)
    _, crossTerms, crossTermsV, rholms, guess_snr, _ = out
    # By-hand reconstruction of the default path from the (unchanged) generator and the
    # (unchanged) cross-term / overlap helpers.
    P2 = make_P(N)
    P2.fmin = FLOW
    P2.dist = fl.distMpcRef * 1e6 * lsu.lsu_PC
    P2.deltaF = dF.deltaF
    hlms, hlms_conj = fl.internal_hlm_generator(P2, 2, verbose=False, quiet=True,
                                                extra_waveform_kwargs={'fd_standoff_factor': 0.9})
    U_ref = fl.ComputeModeCrossTermIP(hlms, hlms, psd, FLOW, FMAX, FNYQ, dF.deltaF, False, False, 0.,
                                      verbose=False, same_waveform_Q=True)
    V_ref = fl.ComputeModeCrossTermIP(hlms_conj, hlms, psd, FLOW, FMAX, FNYQ, dF.deltaF, False, False, 0.,
                                      verbose=False, same_waveform_Q=True, prefix="V")
    assert set(crossTerms[det]) == set(U_ref)
    for k in U_ref:
        assert crossTerms[det][k] == U_ref[k]
        assert crossTermsV[det][k] == V_ref[k]
        assert np.ndim(crossTerms[det][k]) == 0
    t_det = fl.ComputeArrivalTimeAtDetector(det, P.phi, P.theta, tref)
    rho_epoch = dF.epoch - hlms[(2, 2)].epoch
    N_shift = int(float(float(t_det) - t_window - float(rho_epoch)) / DT + 0.5)
    N_window = int(2 * t_window / DT)
    CO = lsu.ComplexOverlap(FLOW, FMAX, FNYQ, dF.deltaF, psd, False, False, 0., full_output=True)
    for m in hlms:
        _, rhoTS, _, _ = CO.ip(hlms[m], dF)
        rhoTS.epoch = dF.epoch - hlms[m].epoch
        lsu.DataRollBins(rhoTS, N_shift)
        ref = lal.CutCOMPLEX16TimeSeries(rhoTS, 0, N_window)
        np.testing.assert_array_equal(rholms[det][m].data.data, ref.data.data)
        assert float(rholms[det][m].epoch) == float(ref.epoch)
    packed = fl.PackLikelihoodDataStructuresAsArrays(list(rholms[det].keys()), None, rholms[det],
                                                     crossTerms[det], crossTermsV[det])
    assert packed[3].shape == (len(hlms), len(hlms)) and packed[3].dtype == np.complex128
    i22 = packed[1][(2, 2)]
    assert packed[3][i22, i22] == U_ref[((2, 2), (2, 2))]
    print("[td-gs f] default fd-diagonal Precompute bit-identical to by-hand lsu construction; "
          "guess_snr=%.2f" % guess_snr)


def test_hlmoft_no_condition_seob_branch():
    """hlmoft(no_condition=False) still applies the SEOB-branch start taper exactly as
    before; no_condition=True returns the untapered modes."""
    N = 4096
    P = make_P(N, lalsim.SEOBNRv4)
    P.taper = lsu.lsu_TAPER_NONE
    raw = lsu.hlmoft(P, 2, no_condition=True)
    P = make_P(N, lalsim.SEOBNRv4)
    P.taper = lsu.lsu_TAPER_NONE
    cond = lsu.hlmoft(P, 2, no_condition=False)
    ntaper = int(0.01 * raw[(2, 2)].data.length)
    ntaper = np.max([ntaper, int(1. / (P.fmin * P.deltaT))])
    vectaper = 0.5 - 0.5 * np.cos(np.pi * np.arange(ntaper) / (1. * ntaper))
    for m in raw:
        expect = np.array(raw[m].data.data)
        expect[:ntaper] *= vectaper
        np.testing.assert_array_equal(cond[m].data.data, expect)
        assert np.abs(raw[m].data.data[0]) > 0          # abrupt start retained
        assert cond[m].data.data[0] == 0
    print("[td-gs] hlmoft SEOB branch: conditioning unchanged when no_condition=False")


def test_tdanalysis_api_check(monkeypatch):
    """The tdanalysis API check passes on the tdanalysis under test and names what an
    earlier 1.1 development tree (same version string, other names) lacks."""
    assert tdg._missing_tdanalysis_api() == []
    import types
    old_tree = types.ModuleType("tdanalysis.forms.sliding")
    for name in ("sliding_data_term", "DEFAULT_BLOCK_BYTES", "PrefixNotApplicable"):
        setattr(old_tree, name, getattr(tdg._sliding, name))
    old_tree.turn_on_inside_window = tdg.is_turn_on_inside_window
    old_tree.METHODS = tdg.SLIDING_METHODS

    def sliding_self_terms(H, solver, shifts, method="auto", pairs=None, block_bytes=None,
                           want_V=True, offset=0, return_method=False, support=None):
        raise AssertionError("not called")
    old_tree.sliding_self_terms = sliding_self_terms
    monkeypatch.setattr(tdg, "_sliding", old_tree)
    missing = tdg._missing_tdanalysis_api()
    assert "tdanalysis.forms.sliding.is_turn_on_inside_window" in missing
    assert "tdanalysis.forms.sliding.SLIDING_METHODS" in missing
    assert any("compute_V=" in m for m in missing)
    assert not any("offset=" in m for m in missing)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-s", "-q"]))


def test_highpass_zero_phase():
    """--td-highpass: the filter ILE applies to the data (tdanalysis' highpass_zero_phase,
    re-exported by td_gs_precompute) is flat in band, zero phase, and suppresses the
    sub-band rise."""
    from tdanalysis.utils.filters import highpass_zero_phase
    assert tdg.highpass_zero_phase is highpass_zero_phase
    fs, n = 4096.0, 4 * 4096
    t = np.arange(n) / fs
    rng = np.random.default_rng(4)
    band = np.zeros(n)
    for f0 in (20.0, 60.0, 300.0):
        band += np.cos(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
    low = 1e3 * np.cos(2 * np.pi * 2.0 * t + 0.3)
    y = tdg.highpass_zero_phase(band + low, fs, 10.0)
    core = slice(n // 4, 3 * n // 4)           # away from the edge transients
    err = np.max(np.abs(y[core] - band[core])) / np.max(np.abs(band[core]))
    assert err < 1e-3                          # 20 Hz passes at 1 - 2e-5, 2 Hz down by > 1e10
