"""CHEAP smoke / mutation-bearing coverage of the angle-marginalization feature.

Exists because the full validation suite (test_angle_marg_exact.py) is too
expensive for a per-PR gate -- it exceeds 20 minutes on 2 cores and has killed
the CI runner three different ways -- but excluding it wholesale would leave the
ENTIRE production feature ungated: a mutation that returns a wrong marginal,
ignores --angle-marg-scheme, breaks the Laplace root enumeration, or drops the
suspect label would all be green.

So this file covers that surface deliberately at MINIMAL scale.  It is not a
substitute for the validation suite's error-law measurements; it is the
mutation-bearing floor that must stay in CI.  Keep it cheap: every test here
must run in seconds, or it belongs in the excluded file instead.
"""
import ast
import pathlib

import numpy as np
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import anglemarg as AM
from RIFT.likelihood.jax_ile import build_likelihood_data
from RIFT.likelihood.jax_ile.core import (
    _accumulate_unit, _time_marginalize, _logsumexp_grid_blocked,
    fused_log_likelihood_distphipsimarg, phi_ref_grid, psi_grid,
    make_distance_grid)

RA, DEC, INCL = 1.1, -0.35, 0.9
INTERP = "sinc"
S = 1



def test_scheme_selector_returns_both_schemes_and_reports():
    """auto must be able to choose EITHER scheme.  A regression here is not
    hypothetical: a previous head floored the amplitude before passing it to the
    selector, so `auto` could never return 'exact' at all."""
    lo, _ = AM.choose_angle_marg_scheme(AM.ANGLE_MARG_CROSSOVER_AMPLITUDE / 10.0)
    hi, _ = AM.choose_angle_marg_scheme(AM.ANGLE_MARG_CROSSOVER_AMPLITUDE * 10.0)
    assert lo == "exact", "sub-crossover amplitude must select the exact scheme"
    assert hi == "laplace", "high amplitude must select the laplace scheme"
    assert lo != hi


def test_dense_sizes_grow_with_amplitude_and_mode_content():
    """Both levers must be live: sqrt(A) AND m_max.  Either being ignored is a
    real bug that shipped once."""
    n_lo = AM._dense_grid_sizes(50.0, m_max=2)
    n_hi = AM._dense_grid_sizes(5000.0, m_max=2)
    n_m = AM._dense_grid_sizes(50.0, m_max=8)
    assert n_hi[0] > n_lo[0] and n_hi[1] > n_lo[1], "must grow with amplitude"
    assert n_m[0] > n_lo[0], "phi sizing must grow with m_max"


def test_amp_sizing_is_required_not_defaulted():
    """A silently-defaulted amplitude is how the SNR-guess bug got in."""
    try:
        AM._require_amp_sizing(None)
    except Exception as exc:
        assert "amp_sizing" in str(exc)
    else:
        raise AssertionError("a missing amp_sizing must raise, not default")


def test_failsafe_record_roundtrips_and_barriers():
    """The host record must reset, report, and barrier -- without it the driver
    cannot label an artifact and the condition dies with the log line."""
    import inspect
    AM.reset_amp_failsafe()
    st = AM.amp_failsafe_state()
    assert st["tripped"] is False
    for fn in (AM.amp_failsafe_state, AM.reset_amp_failsafe):
        assert "effects_barrier" in inspect.getsource(fn)
    src = inspect.getsource(AM._runtime_amp_failsafe)
    assert src.find("lax.cond") < src.find("debug.callback"), (
        "the callback must sit inside lax.cond; an unconditional callback fires "
        "on every likelihood evaluation and destroys throughput")


def _driver_src():
    return (pathlib.Path(__file__).resolve().parents[2] / "bin"
            / "integrate_likelihood_extrinsic_jax").read_text()


def test_driver_actually_passes_the_scheme_through():
    """AST guard on the VALUE node.  Checking only that some `angle_marg=`
    keyword is present is foolable by hardcoding angle_marg="grid" -- flag
    parsed, help present, print present, feature inert.  That is this repo's
    documented silent-no-op pattern."""
    tree = ast.parse(_driver_src())
    seen = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for k in node.keywords or []:
                if k.arg == "angle_marg":
                    seen.append(k.value)
    assert seen, "the driver must pass angle_marg to the wrapper"
    assert any(isinstance(v, ast.Name) and v.id == "angle_marg" for v in seen), (
        "angle_marg must be forwarded as the parsed option, not a constant")


def test_driver_labels_both_artifacts_and_never_implies_verification():
    src = _driver_src()
    assert "def angle_grid_suspect_note(" in src
    # The note is computed ONCE in analyze_one from the RESOLVED scheme and
    # threaded into both writers.  Recomputing it inside each writer with no
    # argument left `scheme` None, so the standing label never emitted -- an
    # inert guard.  So assert the THREADING, which is the real contract.
    wd = src[src.index("def write_dat("):]
    wd = wd[:wd.index("\ndef ")]
    assert "angle_note" in wd, (
        "the EVIDENCE row must carry the label independently of sample export -- "
        "write_samples early-returns without --save-samples")
    ao = src[src.index("def analyze_one("):]
    assert 'angle_grid_suspect_note(_scheme)' in ao, (
        "the note must be computed from the RESOLVED scheme; called with no "
        "argument it silently degrades to the empty string")
    assert ao.count("angle_note=_ev_note") >= 2, (
        "both writers must receive the same computed note")
    assert "BEST-EFFORT" in src, (
        "artifacts must state that no-detection is NOT verification: the "
        "detector is a droppable jax callback, so silence cannot be read as "
        "an adequate grid")
    assert "SUSPECT-ANGLE-GRID" in src


def make_synth(scale=1.0, seed=3, modes=((2, 2), (2, -2)), npts=32,
               deltaT=1.0 / 1024, kappa_boost=1.0):
    """Structurally-faithful synthetic packed data (cf. test_jax_likelihood).

    U is Hermitian positive definite and V complex symmetric, as the real
    precompute produces; ``scale`` sets the overall amplitude (lnL ~ scale^2),
    standing in for SNR.  ``kappa_boost`` multiplies the rholm timeseries
    ONLY (not U/V), producing a target with a large coherent (phi,psi)
    amplitude A -- the regime where an undersized dense grid measurably
    biases the marginal (used by the sizing regression tests).
    """
    rng = np.random.default_rng(seed)
    tw = npts * deltaT / 2.0
    tvals = np.linspace(-tw, tw, npts)
    tref = 1126259462.413
    K = len(modes)
    packed = {}
    for det in ("H1", "L1"):
        npts_full = 4096
        white = (rng.standard_normal((K, npts_full))
                 + 1j * rng.standard_normal((K, npts_full)))
        kx = np.arange(-40, 41)
        kern = np.exp(-0.5 * (kx / 12.0) ** 2)
        kern /= kern.sum()
        rho = np.stack([np.convolve(white[k].real, kern, "same")
                        + 1j * np.convolve(white[k].imag, kern, "same")
                        for k in range(K)]).astype(np.complex128)
        rho *= np.sqrt(len(kx)) * scale * kappa_boost
        M = rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K))
        U = (M @ M.conj().T + 3 * np.eye(K)) * scale ** 2
        B = rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K))
        V = (B @ B.T) * scale ** 2 * 0.3
        packed[det] = dict(lms=np.array(modes, dtype=int), rholmArray=rho,
                           U=U, V=V, epoch=tref - 0.5)
    return build_likelihood_data(packed, deltaT, tref, tvals)


RA, DEC, INCL = np.array([0.9]), np.array([0.4]), np.array([1.1])
S = 1


def _dist_grid(data, n=64):
    return make_distance_grid(30.0, 3000.0, n, distMpcRef=data.distMpcRef)


def brute_marginal(data, x_grid, log_w, nphi, npsi):
    """Brute-force dist+phi+psi marginal: dense product grid of DIRECT
    likelihood evaluations (no coefficient machinery shared with the schemes
    under test)."""
    ph = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    ps = np.linspace(0, np.pi, npsi, endpoint=False)
    m = jnp.full((S, data.npts), -jnp.inf)
    s = jnp.zeros((S, data.npts))
    for p in ph:
        rb = np.repeat(RA[None, :], npsi, 0).ravel()
        db = np.repeat(DEC[None, :], npsi, 0).ravel()
        ib = np.repeat(INCL[None, :], npsi, 0).ravel()
        pb = np.full(npsi * S, p)
        sb = np.repeat(ps[:, None], S, 1).ravel()
        ku, rs = _accumulate_unit(data, rb, db, sb, ib, pb, INTERP, False)
        lnL = _logsumexp_grid_blocked(ku.real, rs, x_grid,
                                      -0.5 * jnp.square(x_grid), log_w, 64)
        m, s = AM._lse_update(m, s, lnL.reshape(npsi, S, data.npts), axis=0)
    lnL_t = m + jnp.log(s) - np.log(nphi * npsi)
    return np.asarray(_time_marginalize(lnL_t, data.w_t))



# ---------------------------------------------------------------------------
# NUMERICAL execution.  Everything above this line checks selection, sizing and
# source wiring -- none of it would catch a mutation that returns a WRONG
# MARGINAL or breaks the Laplace stationary-point enumeration.  These two do, at
# the smallest scale that still discriminates.
#
# COVERAGE LIMIT, stated so this file is not mistaken for end-to-end coverage:
#   * the EXACT scheme is exercised END TO END, through
#     fused_log_likelihood_distphipsimarg_exact -- phi, psi, distance and time
#     marginalization included -- against a direct product-grid reference.
#   * the LAPLACE scheme is exercised only at its KERNEL, _laplace_psi_lnI.
#     fused_log_likelihood_distphipsimarg_laplace and its phi/distance/time
#     marginalization are NOT run here.
#
# That asymmetry is a deliberate CI-cost tradeoff, not an oversight: the fused
# Laplace path is covered in test_angle_marg_exact.py, which is EXCLUDED from
# the per-PR gate (it exceeds 20 minutes on 2 cores) and must be run by hand
# when touching anglemarg.py -- the command is in .travis/test-jax.sh next to
# the exclusion.  A cheap fused-Laplace finiteness smoke test would close the
# gap and is worth adding if someone finds a configuration that stays fast.
# ---------------------------------------------------------------------------

def test_exact_scheme_matches_a_direct_reference_small_scale():
    """Minimal exact-vs-reference: catches a wrong marginal.

    scale=1.5 keeps the dense grids tiny, so this is seconds -- the expensive
    amplitude ladder stays in the excluded validation suite.  The reference is a
    direct product-grid evaluation that shares no coefficient machinery with the
    scheme under test.
    """
    data = make_synth(scale=1.5)
    x_grid, log_w = _dist_grid(data)
    amp = AM.estimate_angle_amplitude(data, x_grid)
    got = AM.fused_log_likelihood_distphipsimarg_exact(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x_grid, log_w, interp=INTERP, amp_sizing=amp)
    ref = brute_marginal(data, x_grid, log_w, 256, 128)
    d = float(np.max(np.abs(np.asarray(got) - np.asarray(ref))))
    assert d < 1e-3, "exact scheme disagrees with a direct reference by %.3e nats" % d


def test_laplace_survives_first_harmonic_cancellation():
    """Regression for the missed-maxima defect (first external review).

    When the FIRST polarization harmonic cancels (c1 = 0) while the quadratic
    one survives (c2 = -d, d > 0.5), both of the original two Newton seeds land
    on MINIMA, every term was rejected, and the kernel returned -inf for an
    integral that is finite.  Enumerating the stationary points fixes it.  Pure
    kernel, no data, milliseconds.
    """
    for d in (0.6, 2.0, 25.0):
        # c1 = 0 (first harmonic cancels), c2 = -d (quadratic harmonic survives)
        val = float(np.asarray(AM._laplace_psi_lnI(
            jnp.asarray(0.0), jnp.asarray(0.0 + 0.0j),
            jnp.asarray(-float(d) + 0.0j))))
        assert np.isfinite(val), (
            "_laplace_psi_lnI returned non-finite at c1=0, c2=-%g -- the "
            "first-harmonic cancellation case, whose integral is finite" % d)
