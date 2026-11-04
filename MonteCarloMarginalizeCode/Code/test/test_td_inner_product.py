"""
test_td_inner_product.py:  Unit tests for the TD Toeplitz inner-product adapter
(RIFT/likelihood/td_inner_product.py, task T1 of TD_LIKELIHOOD.md).

Requires lal + RIFT to import.  Runnable two ways:
    pytest test_td_inner_product.py
    python  test_td_inner_product.py       (runs every test, prints the acceptance numbers)

Acceptance tests (plan Sec. 6, task T1):
  (i)   test_complexip_vs_lsu      : ToeplitzComplexIP.ip vs lsu.ComplexIP.ip, <= 0.5% rel.
  (ii)  test_complexoverlap_vs_lsu : ToeplitzComplexOverlap rhoTS vs lsu.ComplexOverlap,
                                     <= 1% in max|.|, argmax bin exact (0 samples).
  (iii) test_hermiticity_symmetry  : ip(a,b) = conj(ip(b,a)); U/V-style matrix symmetries.
  (iv)  test_dense_vs_iterative    : dense vs tdanalysis PCG covariance solve, 1e-8 rel
                                     (skips cleanly with a message if tdanalysis absent).
Plus:
  test_array_psd_input             : numpy-array PSD (post resample_psd_series) form works.

Which conda env ran these: recorded at the top of the T1 report -- `junior_rift`
(lal 7.7.0), with PYTHONPATH pointing at MonteCarloMarginalizeCode/Code (and, for test
(iv), at a tdanalysis checkout, e.g. an installed public tdanalysis wheel).
"""

import numpy as np

import lal
import RIFT.lalsimutils as lsu

try:
    import RIFT.likelihood.td_inner_product as tdip
except Exception:  # allow running from the test/ directory
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RIFT", "likelihood"))
    import td_inner_product as tdip


# --------------------------------------------------------------------------- config / helpers

FS = 4096.0
FNYQ = FS / 2.0
N = 4096                    # two-sided length (long segment for a good match)
DELTAF = FS / N            # = 1.0
DELTAT = 1.0 / FS
FLOW = 40.0
FMAX = 1000.0
LEN1 = N // 2 + 1


def analytic_psd(f):
    """Smooth, strictly positive analytic one-sided PSD (aLIGO-ish, arbitrary scale)."""
    f = np.atleast_1d(np.asarray(f, dtype=float))
    fsafe = np.maximum(np.abs(f), 1.0)
    out = 1e-46 * (1.0 + (20.0 / fsafe) ** 4 + (fsafe / 250.0) ** 2)
    return out if out.size > 1 else float(out[0])


def make_bandlimited_series(rng, flow=FLOW, fmax=FMAX):
    """A random band-limited two-sided COMPLEX16FrequencySeries in lal packing."""
    fvals_1s = np.arange(LEN1) * DELTAF
    spec1 = np.zeros(LEN1, dtype=complex)
    band = (fvals_1s >= flow) & (fvals_1s <= fmax)
    spec1[band] = rng.standard_normal(band.sum()) + 1j * rng.standard_normal(band.sum())
    # lal two-sided packing exactly as lsu.InnerProduct.weights2side does:
    h2 = np.zeros(N, dtype=complex)
    h2[:LEN1] = spec1[::-1]
    h2[LEN1 - 1:] = spec1[:-1]
    hf = lal.CreateCOMPLEX16FrequencySeries(
        "h", lal.LIGOTimeGPS(0.), 0., DELTAF, lsu.lsu_HertzUnit, N)
    hf.data.data = h2
    return hf


def make_context(analyticPSD_Q=True, psd=None):
    if psd is None:
        psd = analytic_psd
    return tdip.ToeplitzInnerProductContext(
        psd, DELTAF, FNYQ, FLOW, FMAX, analyticPSD_Q=analyticPSD_Q)


# --------------------------------------------------------------------------- (i) ComplexIP

def test_complexip_vs_lsu():
    rng = np.random.default_rng(3)
    IP = lsu.ComplexIP(FLOW, FMAX, FNYQ, DELTAF, analytic_psd, analyticPSD_Q=True)
    ctx = make_context()
    tip = tdip.ToeplitzComplexIP(ctx)

    print("[T1-i] derived ip_normalization = 2*deltaT = %.6e (2/fs = %.6e)"
          % (ctx.ip_normalization, 2.0 / FS))

    rels = []
    for _ in range(8):
        h1 = make_bandlimited_series(rng)
        h2 = make_bandlimited_series(rng)
        a = IP.ip(h1, h2)
        b = tip.ip(h1, h2)
        rels.append(abs(a - b) / abs(a))
    rmax = float(np.max(rels))
    rmed = float(np.median(rels))
    print("[T1-i] ToeplitzComplexIP vs lsu.ComplexIP: max rel = %.3e, median = %.3e"
          % (rmax, rmed))
    assert rmax < 5e-3     # plan AC (i): <= 0.5%


# --------------------------------------------------------------------------- (ii) ComplexOverlap

def test_complexoverlap_vs_lsu():
    rng = np.random.default_rng(11)
    CO = lsu.ComplexOverlap(FLOW, FMAX, FNYQ, DELTAF, analytic_psd,
                            analyticPSD_Q=True, full_output=True)
    ctx = make_context()
    tov = tdip.ToeplitzComplexOverlap(ctx)

    maxrels = []
    idx_mismatch = 0
    for _ in range(6):
        hlm = make_bandlimited_series(rng)
        data = make_bandlimited_series(rng)
        r0, ts0, i0, p0 = CO.ip(hlm, data)
        r1, ts1, i1, p1 = tov.ip(hlm, data)
        d0 = np.abs(ts0.data.data)
        d1 = np.abs(ts1.data.data)
        maxrels.append(np.max(np.abs(d1 - d0)) / np.max(d0))
        if i0 != i1:
            idx_mismatch += 1
        # epoch / length contract for the FL:920-929 drop-in
        assert ts1.data.length == ctx.len2side
        assert abs(ts1.deltaT - DELTAT) <= 1e-12
    mr = float(np.max(maxrels))
    print("[T1-ii] ToeplitzComplexOverlap rhoTS vs lsu: max|.| rel = %.3e ; "
          "argmax mismatches = %d/6" % (mr, idx_mismatch))
    assert mr < 1e-2       # plan AC (ii): <= 1% in max|.|
    assert idx_mismatch == 0   # argmax bin exact (0 samples)


# --------------------------------------------------------------------------- (iii) symmetry

def test_hermiticity_symmetry():
    rng = np.random.default_rng(21)
    IP = lsu.ComplexIP(FLOW, FMAX, FNYQ, DELTAF, analytic_psd, analyticPSD_Q=True)
    ctx = make_context()
    tip = tdip.ToeplitzComplexIP(ctx)

    # ip(a,b) = conj(ip(b,a)) for the Hermitian usage pattern (conj on first arg).
    rels = []
    for _ in range(8):
        a = make_bandlimited_series(rng)
        b = make_bandlimited_series(rng)
        ab = tip.ip(a, b)
        ba = tip.ip(b, a)
        rels.append(abs(ab - np.conj(ba)) / abs(ab))
    rmax = float(np.max(rels))
    print("[T1-iii] |ip(a,b) - conj(ip(b,a))| / |ip(a,b)| max = %.3e" % rmax)
    assert rmax < 1e-10

    # cross-consistency with lsu on the same identity (both are Hermitian in this sense).
    a = make_bandlimited_series(rng)
    b = make_bandlimited_series(rng)
    lsu_ab = IP.ip(a, b)
    print("[T1-iii] lsu.ComplexIP Hermiticity |ab - conj(ba)|/|ab| = %.3e"
          % (abs(lsu_ab - np.conj(IP.ip(b, a))) / abs(lsu_ab)))

    # U/V-style matrix symmetry: the cross-term matrix U_ij = ip(h_i, h_j) built from a
    # set of modes is Hermitian (U = U^dagger).
    modes = [make_bandlimited_series(rng) for _ in range(4)]
    U = np.array([[tip.ip(hi, hj) for hj in modes] for hi in modes])
    herm = np.max(np.abs(U - U.conj().T)) / np.max(np.abs(U))
    print("[T1-iii] U-matrix Hermiticity ||U - U^H|| / ||U|| = %.3e" % herm)
    assert herm < 1e-10


# --------------------------------------------------------------------------- (iv) dense vs iterative

def test_dense_vs_iterative():
    try:
        from tdanalysis.forms.pcg import PCGQuadraticForm  # noqa: F401
    except Exception as exc:
        import pytest
        pytest.skip("tdanalysis not importable ({0!r}); iterative-path test skipped. "
                    "Install RIFT[td-pcg] on Python >=3.11 to exercise it.".format(exc))
        return

    rng = np.random.default_rng(7)
    ctx = make_context()
    d = rng.standard_normal(ctx.N)
    x_dense = ctx.solve_covariance(d, method="dense")
    x_pcg = ctx.solve_covariance(d, method="pcg")
    rel = np.linalg.norm(x_pcg - x_dense) / np.linalg.norm(x_dense)
    print("[T1-iv] dense vs tdanalysis-PCG covariance solve rel = %.3e" % rel)
    assert rel < 1e-8      # plan AC (iv): dense vs iterative to 1e-8


# --------------------------------------------------------------------------- array-PSD input

def test_array_psd_input():
    rng = np.random.default_rng(5)
    # numpy-array PSD sampled on the one-sided rfft grid (what ILE holds post-resample).
    f1s = np.arange(LEN1) * DELTAF
    S_arr = analytic_psd(f1s)
    IP = lsu.ComplexIP(FLOW, FMAX, FNYQ, DELTAF, S_arr, analyticPSD_Q=False)
    ctx = make_context(analyticPSD_Q=False, psd=S_arr)
    tip = tdip.ToeplitzComplexIP(ctx)
    rels = []
    for _ in range(6):
        h1 = make_bandlimited_series(rng)
        h2 = make_bandlimited_series(rng)
        rels.append(abs(IP.ip(h1, h2) - tip.ip(h1, h2)) / abs(IP.ip(h1, h2)))
    rmax = float(np.max(rels))
    print("[T1-array] array-PSD ToeplitzComplexIP vs lsu: max rel = %.3e" % rmax)
    assert rmax < 5e-3


# --------------------------------------------------------------------------- solver="gs"
# solver="gs": every inner product uses C^{-1}, C = Toeplitz(ACF), solved by tdanalysis'
# ToeplitzSolver on its GS method (tdanalysis >= 1.1).  C is a finite-segment Toeplitz covariance, NOT the circulant the FD
# weights imply, so the reference is a dense inverse of the same C; FD agreement is only
# expected for signals away from the segment edges.

def _require_tdanalysis_gs():
    try:
        from tdanalysis.forms.toeplitz_solver import ToeplitzSolver  # noqa: F401
        from tdanalysis.covariance.psd import PSDToAutocorr  # noqa: F401
    except Exception as exc:
        import pytest
        pytest.skip("tdanalysis >= 1.1 not importable ({0!r}); GS tests skipped.".format(exc))


def _gs_context(**kw):
    return tdip.ToeplitzInnerProductContext(
        analytic_psd, DELTAF, FNYQ, FLOW, FMAX, analyticPSD_Q=True, solver="gs", **kw)


def _burst_fd(t0, f0, phi, n=N):
    """A compact complex Gaussian burst at t0 [s] as a lal two-sided FD series."""
    t = np.arange(n) * DELTAT
    x = np.exp(-0.5 * ((t - t0) / 0.03) ** 2) * np.exp(1j * (2 * np.pi * f0 * (t - t0) + phi))
    ts = lal.CreateCOMPLEX16TimeSeries("x", lal.LIGOTimeGPS(0.), 0., DELTAT,
                                       lsu.lsu_DimensionlessUnit, n)
    ts.data.data = x
    return lsu.DataFourier(ts)


def test_gs_acf_does_not_wrap():
    """The GS ACF is long enough that lag N-1 is not a copy of lag 1 (C is not circulant)."""
    _require_tdanalysis_gs()
    acf = _gs_context().gs_acf(N)
    print("[gs] acf[1] = %.3e, acf[N-1] = %.3e" % (acf[1], acf[N - 1]))
    assert abs(acf[N - 1]) < 1e-2 * abs(acf[1])


def test_gs_full_segment_vs_dense_inverse():
    """apply_K with solver='gs' equals a dense C^{-1} v on the full segment."""
    _require_tdanalysis_gs()
    import scipy.linalg
    ctx = _gs_context()
    Cinv = np.linalg.inv(scipy.linalg.toeplitz(ctx.gs_acf(N)))
    rng = np.random.default_rng(1)
    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    ref = Cinv @ v
    rel = np.linalg.norm(ctx.apply_K(v) - ref) / np.linalg.norm(ref)
    print("[gs] full-segment GS vs dense C^-1: rel = %.3e" % rel)
    assert rel < 1e-6


def test_gs_segment_vs_dense_inverse():
    """solver='gs' on a truncated segment: IP and Q(0) equal conj(u)^T C_seg^{-1} v (dense)."""
    _require_tdanalysis_gs()
    import scipy.linalg
    rng = np.random.default_rng(5)
    s0, dur = 1500, 0.25
    ctx = _gs_context(segment_duration=dur, segment_start_index=s0)
    n = ctx.N_seg
    Cinv = np.linalg.inv(scipy.linalg.toeplitz(ctx.gs_acf(n)))
    tip = tdip.ToeplitzComplexIP(ctx)
    tov = tdip.ToeplitzComplexOverlap(ctx)
    rels = []
    for _ in range(4):
        a = make_bandlimited_series(rng)
        b = make_bandlimited_series(rng)
        ua = tdip._ifft_to_td(a)[s0:s0 + n]
        ub = tdip._ifft_to_td(b)[s0:s0 + n]
        ref = np.conj(ua) @ Cinv @ ub
        rels.append(abs(tip.ip(a, b) - ref) / abs(ref))
        _, ts, _, _ = tov.ip(a, b)
        rels.append(abs(ts.data.data[0] - ref) / abs(ref))
    rmax = float(np.max(rels))
    print("[gs] truncated-segment GS vs dense C_seg^-1: max rel = %.3e" % rmax)
    assert rmax < 1e-6


def test_gs_vs_fd_for_compact_signals():
    """Away from the segment edges the Toeplitz and FD inner products agree (IP and Q(t))."""
    _require_tdanalysis_gs()
    ctx = _gs_context()
    IP = lsu.ComplexIP(FLOW, FMAX, FNYQ, DELTAF, analytic_psd, analyticPSD_Q=True)
    CO = lsu.ComplexOverlap(FLOW, FMAX, FNYQ, DELTAF, analytic_psd,
                            analyticPSD_Q=True, full_output=True)
    tip = tdip.ToeplitzComplexIP(ctx)
    tov = tdip.ToeplitzComplexOverlap(ctx)
    worst = 0.0
    for (f1, f2, dt) in [(150, 150, 0.0), (150, 170, 0.01), (300, 280, 0.02), (80, 90, 0.0)]:
        a = _burst_fd(0.5, f1, 0.0)
        b = _burst_fd(0.5 + dt, f2, 0.7)
        scale = np.sqrt(abs(IP.ip(a, a) * IP.ip(b, b)))
        worst = max(worst, abs(IP.ip(a, b) - tip.ip(a, b)) / scale)
        r0, ts0, i0, _ = CO.ip(a, b)
        r1, ts1, i1, _ = tov.ip(a, b)
        assert i0 == i1
        worst = max(worst, np.max(np.abs(ts1.data.data - ts0.data.data)) / scale)
    print("[gs] compact signals, GS vs FD: max |diff| / norm = %.3e" % worst)
    assert worst < 1e-5


def test_gs_supplied_acf_is_used():
    """An explicitly supplied ACF defines C (no PSD->ACF step)."""
    _require_tdanalysis_gs()
    acf = _gs_context().gs_acf(N) * 4.0
    ctx = _gs_context(acf=acf)
    np.testing.assert_array_equal(ctx.gs_acf(N), acf)
    v = np.random.default_rng(2).standard_normal(N)
    w_ref = _gs_context().apply_K(v)
    assert np.allclose(ctx.apply_K(v), w_ref / 4.0, rtol=1e-8, atol=0)


def test_gs_spliced_pieces_block_diagonal():
    """Gaps split the window into pieces; each piece gets its own Toeplitz C (block-diagonal)."""
    _require_tdanalysis_gs()
    import scipy.linalg
    rng = np.random.default_rng(9)
    s0, dur = 1000, 0.5
    gaps = [(300, 420), (900, 1000)]          # sample ranges within the window
    ctx = _gs_context(segment_duration=dur, segment_start_index=s0, gap_sample_ranges=gaps)
    runs = [(int(a), int(b)) for (a, b) in ctx._gs_runs]
    assert runs == [(0, 300), (420, 900), (1000, ctx.N_seg)]
    tip = tdip.ToeplitzComplexIP(ctx)
    tov = tdip.ToeplitzComplexOverlap(ctx)
    rels = []
    for _ in range(3):
        a = make_bandlimited_series(rng)
        b = make_bandlimited_series(rng)
        ua = tdip._ifft_to_td(a)[s0:s0 + ctx.N_seg]
        ub = tdip._ifft_to_td(b)[s0:s0 + ctx.N_seg]
        ref = 0.0
        for (i0, i1) in runs:
            Cinv = np.linalg.inv(scipy.linalg.toeplitz(ctx.gs_acf(i1 - i0)))
            ref = ref + np.conj(ua[i0:i1]) @ Cinv @ ub[i0:i1]
        rels.append(abs(tip.ip(a, b) - ref) / abs(ref))
        _, ts, _, _ = tov.ip(a, b)
        rels.append(abs(ts.data.data[0] - ref) / abs(ref))
    rmax = float(np.max(rels))
    print("[gs] spliced pieces vs dense block-diagonal C^-1: max rel = %.3e" % rmax)
    assert rmax < 1e-6


# --------------------------------------------------------------------------- script entry

if __name__ == "__main__":
    import time
    tests = [
        ("(i) ComplexIP vs lsu", test_complexip_vs_lsu),
        ("(ii) ComplexOverlap vs lsu", test_complexoverlap_vs_lsu),
        ("(iii) Hermiticity/symmetry", test_hermiticity_symmetry),
        ("(iv) dense vs iterative", test_dense_vs_iterative),
        ("array-PSD input", test_array_psd_input),
        ("gs ACF does not wrap", test_gs_acf_does_not_wrap),
        ("gs full segment vs dense", test_gs_full_segment_vs_dense_inverse),
        ("gs truncated segment vs dense", test_gs_segment_vs_dense_inverse),
        ("gs vs FD, compact signals", test_gs_vs_fd_for_compact_signals),
        ("gs supplied ACF", test_gs_supplied_acf_is_used),
        ("gs spliced pieces", test_gs_spliced_pieces_block_diagonal),
    ]
    t0 = time.time()
    failures = 0
    skipped = 0
    for name, fn in tests:
        print("\n=== %s ===" % name)
        try:
            fn()
            print("PASS: %s" % name)
        except AssertionError as e:
            failures += 1
            print("FAIL: %s : %s" % (name, e))
        except BaseException as e:
            # pytest.skip raises Skipped (a BaseException subclass, not Exception);
            # treat any skip-like outcome as a skip, re-raise everything else.
            if e.__class__.__name__ in ("Skipped", "OutcomeException"):
                skipped += 1
                print("SKIP: %s : %s" % (name, e))
            else:
                raise
    print("\nTotal runtime: %.2f s ; %d/%d passed (%d skipped)"
          % (time.time() - t0, len(tests) - failures - skipped, len(tests), skipped))
    raise SystemExit(1 if failures else 0)
