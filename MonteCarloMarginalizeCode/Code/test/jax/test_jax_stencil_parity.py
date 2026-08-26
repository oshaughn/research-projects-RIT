"""The JAX 'sinc' gatherer is the same stencil the numpy/cupy/CUDA paths use.

WHY THIS FILE EXISTS.  RIFT now has the 2a-tap Lanczos stencil on four backends.  Three of them
(numpy, cupy, CUDA) consume ONE weight array from
``factored_likelihood._sinc_lanczos_weight_matrix``, so they cannot drift.  JAX cannot share it:
the weights depend on the traced sub-sample offset, so ``jax_ile.core._sinc_lanczos_weights_jax``
is a second, independent expression of the same formula.  That duplication is the long-term risk
in this feature, and these tests are what converts it from "trust the reviewer" into "CI fails".

Four things are checked:

(a) WEIGHT PARITY -- the JAX and numpy weight generators agree elementwise, including at u = 0
    (where the |x| >= a hard zero bites) and at u -> 1.

(b) GATHER PARITY -- the assembled JAX window equals the numpy ``_sinc_Q_window_numpy`` window on
    the same buffer, INCLUDING near the buffer edges, where the shared convention is that
    out-of-buffer taps are dropped WITHOUT renormalising the remaining weights.  A backend that
    renormalised after masking would pass (a) and fail here.

(c) ACCURACY, i.e. the regression that would have caught the defect this work was opened for.
    Against an exactly-known band-limited signal, 'sinc' must beat 'cubic' by a wide margin in a
    regime where Q is poorly oversampled.  Swap 'sinc' back to 'cubic' in _GATHERERS and this
    test fails -- verified by mutation, see test_mutation_cubic_fails_accuracy_gate.

(d) VECTORISED == UNROLLED -- the tap axis must stay an array.  Written as a Python loop the
    stencil compiles for >1 h inside a NUTS trace.  Bit-equivalence is asserted so that
    "simplifying" it back into a loop is a test failure and not a silent 4-orders-of-magnitude
    compile regression.

The GPU (cupy) leg SKIPS when cupy is unavailable -- it must not silently pass.

Run:
  PYTHONPATH=<...>/Code  python -m pytest -q test/jax/test_jax_stencil_parity.py
"""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)   # else the parity tolerances below are meaningless
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import core as JC
from RIFT.likelihood.time_interp_choice import SINC_HALFWIDTH_DEFAULT

A = SINC_HALFWIDTH_DEFAULT

# Parity tolerances.  These are STATED, not implicit (the brief for this work asked for exactly
# that).  Both generators evaluate the same closed-form expression in float64, differing only in
# the order of the sinc/normalise operations and in whose libm supplies sin(), so the expected
# disagreement is a few ulp of the O(1) weights.  1e-14 is ~50 ulp: tight enough that a genuine
# formula difference (a missing window, an unnormalised sum, an off-by-one tap) cannot hide under
# it, loose enough not to be a libm-version tripwire.
TOL_WEIGHTS = 1e-14
# The gather additionally sums 2a products against O(1) complex samples, so it carries the
# weight error plus ~2a rounding steps; 1e-12 relative to unit-scale data covers that with margin.
TOL_GATHER = 1e-12


def _numpy_weight_matrix(u, a=A):
    """The shared CPU/GPU generator.  Imported lazily: factored_likelihood costs numba+lal."""
    from RIFT.likelihood.factored_likelihood import _sinc_lanczos_weight_matrix
    return _sinc_lanczos_weight_matrix(np.atleast_1d(np.asarray(u, dtype=float)), a)


# ----------------------------------------------------------------------------- (a) weight parity

def test_weight_parity_against_numpy_generator():
    # Endpoints included on purpose: u = 0 is where the |x| >= a hard zero applies (tap k = a
    # sits exactly at x = -a) and where the stencil must collapse to the identity.
    u = np.concatenate([[0.0, 1e-15, 0.5, 1.0 - 1e-12],
                        np.random.default_rng(0).uniform(0.0, 1.0, 97)])
    k_np, w_np = _numpy_weight_matrix(u)
    k_jx, w_jx = JC._sinc_lanczos_weights_jax(jnp.asarray(u), A)
    np.testing.assert_array_equal(np.asarray(k_jx), np.asarray(k_np))
    assert w_np.shape == (len(u), 2 * A)
    err = np.max(np.abs(np.asarray(w_jx) - w_np))
    assert err < TOL_WEIGHTS, "JAX/numpy sinc weights disagree by %.3e" % err


def test_weight_parity_outside_the_unit_interval():
    """The ``|x| >= a`` hard zero is only reachable for u outside [0, 1), and it must still match.

    Added after a mutation run: deleting that clause from the JAX generator survived every other
    test here.  It is an EQUIVALENT mutation on the wired path -- for u in [0, 1) the guarded tap
    sits at x = -a, where sinc(x/a) = sinc(-1) = 3.9e-17 rather than exactly 0, so the guard moves
    the weight by 1.5e-33 -- but the two generators are library helpers, and off the wired path
    the clause is worth 2.2e-3 in the weights.  So the choice is to pin it, not to call it dead.
    """
    u = np.array([-2.0, -0.5, -1e-12, 1.0, 1.5, 2.0, 3.25])
    k_np, w_np = _numpy_weight_matrix(u)
    k_jx, w_jx = JC._sinc_lanczos_weights_jax(jnp.asarray(u), A)
    np.testing.assert_array_equal(np.asarray(k_jx), np.asarray(k_np))
    err = np.max(np.abs(np.asarray(w_jx) - w_np))
    assert err < TOL_WEIGHTS, "out-of-range-u weights disagree by %.3e" % err
    # The guard must actually be exercised, or this test proves nothing about it.
    x = u[:, None] - np.asarray(k_np)
    assert np.any(np.abs(x) >= A), "no tap reached |x| >= a; the guard is untested"


def test_weights_sum_to_one_and_are_identity_at_zero_offset():
    """Both properties are relied on downstream: unit sum makes constants exact, and the u=0
    identity is what lets 'sinc' reproduce the original samples where no shift is needed."""
    _, w = JC._sinc_lanczos_weights_jax(jnp.linspace(0.0, 1.0, 41), A)
    np.testing.assert_allclose(np.asarray(jnp.sum(w, axis=-1)), 1.0, atol=1e-14)
    _, w0 = JC._sinc_lanczos_weights_jax(jnp.zeros(1), A)
    expect = np.zeros(2 * A)
    expect[A - 1] = 1.0                       # k = 0 is at index a-1 in arange(-a+1, a+1)
    np.testing.assert_allclose(np.asarray(w0)[0], expect, atol=1e-14)


# ----------------------------------------------------------------------------- (b) gather parity

def _numpy_window(Q_col, ifirst, frac, npts):
    """One-mode wrapper around the production CPU window builder."""
    from RIFT.likelihood.factored_likelihood import _sinc_Q_window_numpy
    Q_block = np.asarray(Q_col)[:, None]                     # (n_time, n_lm=1)
    out = _sinc_Q_window_numpy(Q_block, np.asarray(ifirst, dtype=int),
                               np.asarray(frac, dtype=float), npts, a=A)
    return out[:, :, 0]                                      # (n_ext, npts)


@pytest.mark.parametrize("place", ["interior", "left_edge", "right_edge"])
def test_gather_parity_against_numpy_window(place):
    """Interior AND both edges: the edge cases pin the zero-extension convention, which is the
    one place the four backends could agree on weights and still disagree on output."""
    rng = np.random.default_rng(7)
    n_time, npts, n_ext = 512, 24, 40
    Q = rng.normal(size=n_time) + 1j * rng.normal(size=n_time)
    frac = rng.uniform(0.0, 1.0, n_ext)
    if place == "interior":
        ifirst = rng.integers(2 * A, n_time - npts - 2 * A, n_ext)
    elif place == "left_edge":
        # Windows that start before the buffer, so the leading taps fall off the front.
        ifirst = rng.integers(-A - 3, A, n_ext)
    else:
        ifirst = rng.integers(n_time - npts - A, n_time - npts + A + 3, n_ext)

    ref = _numpy_window(Q, ifirst, frac, npts)

    # JAX takes a continuous position; pos = ifirst + frac + t reproduces the same window.
    pos = (ifirst[:, None] + frac[:, None] + np.arange(npts)[None, :])
    got = np.asarray(JC._GATHERERS["sinc"](jnp.asarray(Q), jnp.asarray(pos)))

    err = np.max(np.abs(got - ref))
    assert err < TOL_GATHER, "%s: JAX/numpy sinc windows disagree by %.3e" % (place, err)
    if place != "interior":
        # Guard against the test passing because every tap happened to land in-bounds: the
        # comparison must actually exercise the zero-extension branch.
        assert np.any(np.asarray(ifirst) < A) or np.any(np.asarray(ifirst) + npts + A > n_time)


def test_gpu_gather_parity_against_numpy_window():
    """cupy leg.  SKIPS without a GPU -- it must not silently pass (cf. the same fix made to
    test_noloop_gpu_stencils)."""
    # Plain import, not importorskip: on a CPU head node cupy is INSTALLED but raises ImportError
    # on libcuda.so.1, which importorskip will treat as an error from pytest 9.1 onward.
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:
        pytest.skip("no usable cupy/CUDA device, GPU stencil parity NOT exercised here: %s"
                    % type(exc).__name__)
    from RIFT.likelihood import Q_inner_product

    rng = np.random.default_rng(11)
    n_time, npts, n_ext, n_lm = 512, 16, 32, 2
    Q = (rng.normal(size=(n_time, n_lm)) + 1j * rng.normal(size=(n_time, n_lm)))
    Amat = (rng.normal(size=(n_ext, n_lm)) + 1j * rng.normal(size=(n_ext, n_lm)))
    ifirst = rng.integers(2 * A, n_time - npts - 2 * A, n_ext).astype(np.int32)
    frac = rng.uniform(0.0, 1.0, n_ext)

    gpu = cupy.asnumpy(Q_inner_product.Q_inner_product_sinc_cupy(
        cupy.asarray(Q), cupy.asarray(Amat), cupy.asarray(ifirst),
        cupy.asarray(frac), npts, halfwidth=A))

    # JAX: gather each mode, then contract on lm exactly as the kernel does.
    pos = (ifirst[:, None] + frac[:, None] + np.arange(npts)[None, :])
    jx = np.zeros((n_ext, npts), dtype=complex)
    for lm in range(n_lm):
        jx += Amat[:, lm][:, None] * np.asarray(
            JC._GATHERERS["sinc"](jnp.asarray(Q[:, lm]), jnp.asarray(pos)))

    err = np.max(np.abs(gpu - jx))
    assert err < TOL_GATHER, "JAX/CUDA sinc disagree by %.3e" % err


# ---------------------------------------------------------------------------- (c) accuracy gate

def _bandlimited_series(n, f_over_nyq, rng, n_tones=24):
    """A sum of tones strictly below f_over_nyq * Nyquist, evaluatable at ANY real sample index.

    Deliberately NOT an FFT zero-pad of a stored buffer.  A pad of a truncated slice is not a
    valid reference here: lnL(t) is band-limited but the slice is not, so truncation destroys the
    band-limitation and the pad wraps -- measured to disagree with a converged Lanczos ladder at
    the 1.4e-2 level during the sky-offset diagnosis.  Summing known tones sidesteps that: the
    exact value at a fractional index is the closed form below, with no reference error at all.
    """
    f = rng.uniform(0.02, f_over_nyq, n_tones) * np.pi     # rad/sample; Nyquist = pi
    amp = rng.normal(size=n_tones) + 1j * rng.normal(size=n_tones)
    phase = rng.uniform(0, 2 * np.pi, n_tones)

    def evaluate(t):
        t = np.asarray(t, dtype=float)
        return np.sum(amp * np.exp(1j * (f * t[..., None] + phase)), axis=-1)

    return evaluate(np.arange(n)), evaluate


# Required margin of 'sinc' over 'cubic' at f/fNyq = 0.7.  MEASURED, not chosen: the ratio over
# seeds 3-10 on this generator is 37.1, 39.0, 40.4, 42.9, 44.6, 47.7, 47.7, 49.0 (ldas-grid,
# igwn python 3.11, jax 0.7.1, 2026-08-26).  The gate is set at 20x -- comfortably under the
# observed floor of 37, so it fires on a change in stencil ORDER rather than on an unlucky draw,
# while still being ~10x above the 2.3x that separates cubic from linear here.
REQUIRED_SINC_OVER_CUBIC = 20.0
# Floor the observed ratios must clear, so an erosion of the margin is caught long before the
# gate itself starts failing intermittently.  Set below the measured minimum of 37.1, not at a
# multiple of the gate.
MARGIN_WATCHDOG = 25.0
BANDWIDTH_FRACTION = 0.7      # poorly oversampled: the regime the 3G demo and O4 high-fmin sit in


def _stencil_errors(seed=3):
    rng = np.random.default_rng(seed)
    n = 4096
    series, exact = _bandlimited_series(n, BANDWIDTH_FRACTION, rng)
    # Interior only: this measures interpolation error, not the (deliberate) edge truncation.
    pos = rng.uniform(4 * A, n - 4 * A, 2000)
    truth = exact(pos)
    errs = {}
    for name in ("nearest", "linear", "cubic", "sinc"):
        got = np.asarray(JC._GATHERERS[name](jnp.asarray(series), jnp.asarray(pos)))
        errs[name] = float(np.max(np.abs(got - truth)) / np.max(np.abs(truth)))
    return errs


def test_sinc_beats_cubic_on_a_poorly_oversampled_band_limited_signal():
    """THE REGRESSION GATE.  This is the test that fails if the stencil reverts to cubic."""
    errs = _stencil_errors()
    ratio = errs["cubic"] / errs["sinc"]
    assert ratio > REQUIRED_SINC_OVER_CUBIC, (
        "sinc no longer beats cubic at f/fNyq=%.2f: errors %r, ratio %.1f < %.1f"
        % (BANDWIDTH_FRACTION, errs, ratio, REQUIRED_SINC_OVER_CUBIC))
    # Ordering of the whole ladder, so a stencil that regresses to a lower order is caught even
    # if it is not literally cubic.
    assert errs["sinc"] < errs["cubic"] < errs["linear"]
    assert errs["nearest"] > errs["cubic"]


def test_mutation_cubic_fails_accuracy_gate():
    """MUTATION TEST.  Reintroduce the cubic stencil under the 'sinc' key and confirm the gate
    above fails.  A gate that passes both ways is worthless, so this is not optional decoration:
    it is the evidence that test_sinc_beats_cubic... has any power at all."""
    saved = JC._GATHERERS["sinc"]
    JC._GATHERERS["sinc"] = JC._gather_cubic
    try:
        with pytest.raises(AssertionError):
            test_sinc_beats_cubic_on_a_poorly_oversampled_band_limited_signal()
    finally:
        JC._GATHERERS["sinc"] = saved
    # And the restore worked, so later tests in this file are not running against the mutant.
    assert JC._GATHERERS["sinc"] is saved


def test_accuracy_margin_is_not_marginal():
    """Record how much room the gate has, over several seeds, so a future tightening is an
    informed edit rather than a guess.  Fails only if some seed lands within 2x of the gate."""
    ratios = [_stencil_errors(s)["cubic"] / _stencil_errors(s)["sinc"] for s in (3, 4, 5, 6, 7)]
    assert min(ratios) > MARGIN_WATCHDOG, \
        "gate margin has eroded: ratios %r, floor %.0f, gate %.0fx" % (
            ratios, MARGIN_WATCHDOG, REQUIRED_SINC_OVER_CUBIC)


# ------------------------------------------------------------------- (d) vectorised == unrolled

# The two forms are the same arithmetic in a different association order -- jnp.sum over the tap
# axis against a sequential accumulation -- so XLA is free to reassociate and the difference is
# NOT exactly zero: measured max|diff| 1.8e-15 on O(1)-O(10) complex data, i.e. a few ulp.  (The
# sky-offset diagnostic's unnormalised prototype did land on exact 0.0; adding the weight
# normalisation is what moved it off.)  1e-12 is ~1000x that and still orders of magnitude below
# any semantic change: a dropped tap, a wrong offset or a lost window all move this by O(0.1).
TOL_UNROLL = 1e-12


@pytest.mark.parametrize("a", [4, 8, 16])
def test_vectorised_matches_unrolled(a):
    """See _make_gather_sinc for why the tap axis must stay an array; this test is what makes a
    revert to the loop form fail loudly."""
    rng = np.random.default_rng(0)
    n = 4000
    Q = jnp.asarray(rng.normal(size=n) + 1j * rng.normal(size=n))
    pos = jnp.asarray(rng.uniform(-5, n + 5, 3000))          # includes out-of-buffer positions
    fast = JC._make_gather_sinc(a)(Q, pos)
    slow = JC._make_gather_sinc_unrolled(a)(Q, pos)
    err = float(jnp.max(jnp.abs(fast - slow)))
    assert err < TOL_UNROLL, "a=%d: vectorised and unrolled forms differ by %.3e" % (a, err)


# --------------------------------------------------------------------------------- wiring checks

def test_sinc_is_reachable_from_the_registry_and_the_cli():
    """The stencil existing is not the same as a user being able to ask for it: 'cubic' was
    implemented in _GATHERERS for a release while --interp's choices list still refused it."""
    assert "sinc" in JC._GATHERERS
    import ast, io, os
    driver = os.path.join(os.path.dirname(__file__), "..", "..", "bin",
                          "integrate_likelihood_extrinsic_jax")
    src = io.open(driver, encoding="utf-8").read()
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_option"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--interp"):
            found = [kw for kw in node.keywords if kw.arg == "choices"]
    assert found, "--interp option not found in the JAX driver"
    # The choices expression must be derived from the registry, not a literal list that can rot.
    assert "_JAX_GATHERER_NAMES" in ast.unparse(found[0].value), \
        "--interp choices is a literal list again; it will drift from _GATHERERS"


def test_likelihood_runs_and_differentiates_with_sinc():
    """Wire-level check: the stencil must work THROUGH the likelihood, not just as a helper.
    Unit-testing the gatherer proves nothing about whether _accumulate_unit can call it."""
    from test_jax_likelihood import make_synthetic, make_Pvec             # noqa: E402
    from RIFT.likelihood.jax_ile import build_likelihood_data, fused_log_likelihood

    packed, _ref, tvals, deltaT, tref = make_synthetic()
    P, distMpc = make_Pvec(9, tref, deltaT)
    data = build_likelihood_data(packed, deltaT, tref, tvals)

    def run(nm):
        return np.asarray(fused_log_likelihood(
            data, P.phi, P.theta, P.psi, P.incl, P.phiref, distMpc, interp=nm))

    vals = {nm: run(nm) for nm in ("nearest", "cubic", "sinc")}
    assert all(np.all(np.isfinite(v)) for v in vals.values()), vals
    # sinc must actually change the answer -- otherwise the interp argument is being ignored,
    # which is exactly the silent no-op a helper-only test cannot see.
    assert not np.array_equal(vals["sinc"], vals["cubic"])
    # ... and the two interpolating stencils must sit far closer to each other than either does
    # to 'nearest', since both approximate the same band-limited value.
    d_interp = np.max(np.abs(vals["sinc"] - vals["cubic"]))
    d_nearest = np.max(np.abs(vals["cubic"] - vals["nearest"]))
    assert d_interp < 0.5 * d_nearest, (d_interp, d_nearest)

    # Differentiable through the stencil, which is the whole reason the JAX path exists.
    def scalar(ra):
        return fused_log_likelihood(
            data, jnp.array([ra]), P.theta[:1], P.psi[:1], P.incl[:1], P.phiref[:1],
            distMpc[:1], interp="sinc")[0]

    g = float(jax.grad(scalar)(float(P.phi[0])))
    assert np.isfinite(g) and g != 0.0
