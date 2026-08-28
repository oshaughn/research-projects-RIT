"""
Gate for the COMPILE-COST structure of the anglemarg laplace path.

WHY THIS EXISTS (2026-08-28): a single SNR-40 production run with
--angle-marg-scheme auto sat in XLA compilation for >88 minutes and reached
22.2 GiB RSS before being killed by hand (~/rift_costbakeoff_20260826/
time_arms.log) -- against a 25 GiB per-user cgroup on the interactive hosts.
The cause was STRUCTURAL, not mathematical: Python-level loops in
_laplace_psi_lnI (24-cell bracket scan x 4 root slots, 4 x 20 bisection
steps, 320-point u-quadrature) unrolled into the traced graph, and the
distance-block Python loop in fused_log_likelihood_distphipsimarg_laplace
then instantiated that whole unrolled kernel once per distance block --
64 copies at the production n_grid=256 -- inside a jax.checkpoint'ed scan
body that reverse-mode AD retraces.  XLA compile time and memory are
superlinear in graph size, hence the wall.

The fix rolls those loops into lax.scan / lax.fori_loop, so the traced graph
is CONSTANT in the distance-grid size and in the loop trip counts.  These
tests pin exactly that structural property, plus the one numerical seam the
restructure introduced (tail padding of the distance grid).  They are
trace-only where possible and run in seconds; the numerical VALIDATION of the
laplace scheme itself lives in test_angle_marg_exact.py (excluded from the
per-PR gate on cost grounds) and test_angle_marg_smoke.py.

Each test here fails under a deliberate mutation (verified by hand before
landing; the mutations and observed failures are recorded in the PR):
  * re-unrolling the distance scan into a Python loop -> graph-growth test
    fails (equation count scales with n_grid again);
  * re-unrolling any kernel loop -> kernel-size ceiling fails;
  * breaking the tail padding (0.0 instead of -inf pad weights) -> the
    padding-exactness test fails.
"""

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import build_likelihood_data
from RIFT.likelihood.jax_ile import anglemarg as AM
from RIFT.likelihood.jax_ile.core import make_distance_grid


def _total_eqns(closed_jaxpr):
    """Equation count including nested jaxprs, each nested body counted ONCE
    (scan/cond/checkpoint bodies are nested; an unrolled Python loop's
    equations are all at one level, so unrolling inflates this count)."""
    def walk(jaxpr):
        n = len(jaxpr.eqns)
        for eqn in jaxpr.eqns:
            for val in eqn.params.values():
                vals = val if isinstance(val, (tuple, list)) else (val,)
                for v in vals:
                    if hasattr(v, "jaxpr"):          # ClosedJaxpr
                        n += walk(v.jaxpr)
                    elif hasattr(v, "eqns"):         # raw Jaxpr
                        n += walk(v)
        return n
    return walk(closed_jaxpr.jaxpr)


def make_synth(scale=1.0, seed=3, modes=((2, 2), (2, -2)), npts=16,
               deltaT=1.0 / 1024, kappa_boost=1.0):
    """Small structurally-faithful packed data (as test_angle_marg_exact)."""
    rng = np.random.default_rng(seed)
    tw = npts * deltaT / 2.0
    tvals = np.linspace(-tw, tw, npts)
    tref = 1126259462.413
    K = len(modes)
    packed = {}
    for det in ("H1", "L1"):
        npts_full = 1024
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


def _fused_jaxpr(data, n_grid):
    xg, lwg = make_distance_grid(30.0, 3000.0, n_grid,
                                 distMpcRef=data.distMpcRef)
    def f(ra, dec, incl):
        return AM.fused_log_likelihood_distphipsimarg_laplace(
            data, ra, dec, incl, xg, lwg, amp_sizing=900.0)
    return jax.make_jaxpr(f)(jnp.asarray([0.9]), jnp.asarray([0.4]),
                             jnp.asarray([1.1]))


def test_laplace_graph_size_independent_of_distance_grid():
    """The traced graph must NOT grow with the distance-grid size.

    Pre-fix, each extra dist_block=4 block of distance nodes re-instantiated
    the full _laplace_psi_lnI kernel in the scan body (measured: hundreds of
    extra equations per block; at the production n_grid=256 the resulting
    graph took >88 min and >20 GiB to compile).  With the distance nodes
    scanned, the equation count is IDENTICAL for any n_grid: only the
    scanned xs shapes change.  Trace-only: no XLA compile, no execution.
    """
    data = make_synth()
    n8 = _total_eqns(_fused_jaxpr(data, 8))
    n64 = _total_eqns(_fused_jaxpr(data, 64))
    assert n64 == n8, (
        "laplace traced graph grew with the distance grid (%d -> %d eqns "
        "for n_grid 8 -> 64): a distance-block loop is unrolling into the "
        "graph again, which is the >88-minute XLA compile of 2026-08-28."
        % (n8, n64))


def test_laplace_kernel_graph_is_rolled():
    """_laplace_psi_lnI's traced size must stay near its rolled size.

    Measured at the commit that introduces this test (jax 0.9.2; jax 0.7.1
    within 1%): 400 equations rolled, 6819 with all three pre-fix Python
    loops inlined, and PER-LOOP mutants of 835 (20-step bisection
    unrolled -- the smallest), 1042 (24-cell bracket walk unrolled), 1410
    (320-point quadrature unrolled).  The ceiling of 600 sits 1.5x above
    the rolled size to absorb jax-version drift in how primitives are
    counted, and 1.4x below the smallest single-loop mutant, so ANY one
    loop unrolling again trips it.  If a jax upgrade legitimately inflates
    the rolled count past 600, re-measure all four numbers above before
    touching the ceiling.
    """
    shape = (4, 3)
    a = jnp.zeros(shape)
    c1 = jnp.full(shape, 30.0 + 10.0j)
    c2 = jnp.full(shape, 40.0 - 5.0j)
    n = _total_eqns(jax.make_jaxpr(AM._laplace_psi_lnI)(a, c1, c2))
    assert n <= 600, (
        "_laplace_psi_lnI traces to %d equations (ceiling 600): a bracket/"
        "bisection/quadrature loop has unrolled into the graph again.  That "
        "size is multiplied by every distance block and by AD; see module "
        "docstring." % n)


def test_laplace_dist_tail_padding_exact():
    """A distance grid NOT divisible by dist_block must give the same
    marginal as one evaluated without tail padding.

    The scan packs G nodes into blocks of dist_block, edge-padding the tail
    with -inf log-weights (exactly-zero contribution to the running
    log-sum-exp).  Mutating the pad weights to 0.0 double-counts the last
    node and shifts the marginal by ~log-weight amounts; this test fails
    under that mutation and under any off-by-one in the packing.
    """
    data = make_synth(kappa_boost=4.0)
    xg, lwg = make_distance_grid(30.0, 3000.0, 10, distMpcRef=data.distMpcRef)
    ra, dec, incl = jnp.asarray([0.9]), jnp.asarray([0.4]), jnp.asarray([1.1])
    # dist_block=4 -> 3 blocks, 2 padded tail nodes; dist_block=5 and =1
    # divide 10 exactly -> no padding.  All three must agree to roundoff.
    v4 = AM.fused_log_likelihood_distphipsimarg_laplace(
        data, ra, dec, incl, xg, lwg, amp_sizing=900.0, dist_block=4)
    v5 = AM.fused_log_likelihood_distphipsimarg_laplace(
        data, ra, dec, incl, xg, lwg, amp_sizing=900.0, dist_block=5)
    v1 = AM.fused_log_likelihood_distphipsimarg_laplace(
        data, ra, dec, incl, xg, lwg, amp_sizing=900.0, dist_block=1)
    assert np.allclose(np.asarray(v4), np.asarray(v5), rtol=0, atol=1e-12), \
        (np.asarray(v4), np.asarray(v5))
    assert np.allclose(np.asarray(v4), np.asarray(v1), rtol=0, atol=1e-12), \
        (np.asarray(v4), np.asarray(v1))
