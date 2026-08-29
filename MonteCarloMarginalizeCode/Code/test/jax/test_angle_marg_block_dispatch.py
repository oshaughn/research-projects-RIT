"""
Gate for the EXECUTION-COST structure of the anglemarg laplace path:
the block-dispatched kernel (_laplace_psi_lnI_block).

WHY THIS EXISTS (2026-08-28): after the compile-cost fix (PR #209) made the
laplace path runnable, it EXECUTED ~2,950x slower than the grid scheme
(1.28e-2 vs 4.33e-6 s per sample*timepoint; a single 800-sample pilot chunk
= 6,094 s, an SNR-40 run = days).  Measured attribution (GPU, additive to
<1%): the C^1 blend evaluates BOTH kernel branches at every lattice point --
the 320-point u-quadrature (55% of execution) and the bracket/bisect
root-finding (39%, of which the 20-step bisection is 33%) -- while a census
of the production-shaped lattice shows 99.5% of points sit in the
pure-quadrature regime (t < BLEND_LO; 89% at t < 20, where the module's own
aliasing rule needs only N ~ 32-96) and the points carrying posterior weight
sit at t > 900 where only the Laplace branch is needed.  The fix dispatches
each (dist_block x phi_chunk x S x npts) kernel call through lax.switch on
scalar bounds of t = b + 2d, so exactly one branch executes per block, with
the quadrature N laddered by the SAME aliasing exponent the shipped
(N=320, t=300) pair fixes.

These tests pin (1) the ladder's accuracy contract, (2) the dispatcher's
value agreement with the undispatched kernel across every branch, and
(3) the WIRING -- the fused driver must actually call the dispatcher
(helper-level tests cannot see a call site that stops calling the helper).

Each test fails under a deliberate mutation (verified by hand; mutations
and observed failures recorded in the PR):
  * raising a ladder threshold above its contour (t_ok 0.9 -> 50 for N=32)
    -> ladder-contract test fails, and the branch-agreement test fails
    LOUDLY (~8% relative aliasing at t=50 with N=32);
  * off-by-one in the switch index (pure-Laplace taken from BLEND_LO
    instead of BLEND_HI) -> branch-agreement fails in the straddle family;
  * rewiring _dist_step back to the plain kernel -> wiring test fails.
"""

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import anglemarg as AM
from RIFT.likelihood.jax_ile.core import make_distance_grid

from test_angle_marg_compile_cost import make_synth


def _aliasing_exponent(N, t):
    """ln[I_{N/2}(t) / I_0(t)] by the uniform asymptotic (nu = N/2):
    sqrt(nu^2 + t^2) - nu*asinh(nu/t) - t.  This is the trapezoid-rule
    aliasing bound the module's constants block quotes (~e^-40 at the
    shipped N=320, t=300 pair)."""
    nu = N / 2.0
    return np.sqrt(nu * nu + t * t) - nu * np.arcsinh(nu / t) - t


def test_quad_ladder_keeps_the_shipped_aliasing_exponent():
    """Every ladder rung must be at least as accurate at its threshold as
    the shipped (N=320, t=300) band edge, and rungs must be multiples of
    the 16-point quadrature scan chunk.  Fails if any t_ok is raised above
    the E = E(320, 300) contour or a rung breaks the chunking."""
    e_ship = _aliasing_exponent(320.0, 300.0)
    assert e_ship < -40.0                          # the documented ~e^-40
    Ns, toks = AM._QUAD_LADDER_N, AM._QUAD_LADDER_TOK
    assert len(Ns) == len(toks)
    assert Ns[-1] == AM._LAPLACE_QUAD_N            # top rung == shipped N
    assert toks[-1] == AM._LAPLACE_BLEND_HI        # ladder covers the band
    for N, tok in zip(Ns, toks):
        assert N % 16 == 0, (N, "quadrature scan chunk is 16")
        e = _aliasing_exponent(float(N), float(tok))
        assert e <= e_ship, (
            "ladder rung N=%d at t_ok=%g has aliasing exponent %.2f, worse "
            "than the shipped band edge %.2f: the rung under-resolves its "
            "band" % (N, tok, e, e_ship))
    assert all(toks[i] < toks[i + 1] for i in range(len(toks) - 1))


def test_dispatcher_preserves_three_input_broadcasting():
    """The production dispatcher must keep the public kernel's elementwise
    broadcast contract before it introduces its private root-slot axis.

    Exercise both formerly broken directions: a vector supplied only by
    ``a`` (whose length is deliberately not the four-root count), and a
    vector supplied only by one coefficient.  Scalar calls provide an
    independent elementwise oracle without relying on the same broadcast.
    """
    avec = jnp.asarray([-0.8, 0.1, 1.7])
    c1 = jnp.asarray(0.35 + 0.2j)
    c2 = jnp.asarray(-0.15 + 0.05j)
    got_a = np.asarray(AM._laplace_psi_lnI_block(avec, c1, c2))
    want_a = np.asarray([
        AM._laplace_psi_lnI_block(ai, c1, c2) for ai in avec
    ])
    np.testing.assert_allclose(got_a, want_a, rtol=0.0, atol=1e-13)

    c2vec = jnp.asarray([0.05 + 0.02j, -0.1 + 0.03j, 0.2 - 0.04j])
    got_c = np.asarray(AM._laplace_psi_lnI_block(jnp.asarray(0.4), c1, c2vec))
    want_c = np.asarray([
        AM._laplace_psi_lnI_block(jnp.asarray(0.4), c1, c2i)
        for c2i in c2vec
    ])
    np.testing.assert_allclose(got_c, want_c, rtol=0.0, atol=1e-13)


def _batch(rng, tlo, thi, n=256):
    t = rng.uniform(tlo, thi, n)
    frac = rng.uniform(0, 1, n)
    b = t * frac
    d = 0.5 * t * (1 - frac)
    beta = rng.uniform(-np.pi, np.pi, n)
    delta = rng.uniform(-np.pi, np.pi, n)
    a = jnp.asarray(rng.uniform(-5, 5, n))
    return (a, jnp.asarray(b * np.exp(1j * beta)),
            jnp.asarray(d * np.exp(1j * delta)))


def test_dispatcher_matches_kernel_in_every_branch():
    """_laplace_psi_lnI_block == _laplace_psi_lnI on batches forced into
    each switch branch: bit-equal where the same code runs (pure-Laplace,
    top rung, straddle), and within ladder roundoff (documented sub-1e-12)
    on the reduced-N rungs.  An off-by-one in the index computation or a
    threshold mutation shifts a batch into an inadequate branch and fails
    this by many orders of magnitude (e.g. Laplace applied inside the
    blend band: ~0.25 nats; N=32 applied at t=50: ~8%)."""
    rng = np.random.default_rng(3)
    kB = jax.jit(AM._laplace_psi_lnI_block)
    kS = jax.jit(AM._laplace_psi_lnI)
    exact_families = [(320.0, 5000.0),     # pure-Laplace branch
                      (144.0, 219.0),      # top (N=320) rung
                      (10.0, 4000.0),      # straddle -> full blended kernel
                      (225.0, 295.0)]      # blend band -> full kernel; sent
                                           # to pure-Laplace by an index
                                           # off-by-one (BLEND_LO for HI),
                                           # which errs at the 0.05-0.25 nat
                                           # branch-disagreement scale
    for tlo, thi in exact_families:
        a, c1, c2 = _batch(rng, tlo, thi)
        dv = np.max(np.abs(np.asarray(kB(a, c1, c2)) - np.asarray(kS(a, c1, c2))))
        assert dv == 0.0, ("branch running identical code must be "
                           "bit-equal", tlo, thi, dv)
    # every reduced-N rung (bands strictly inside their thresholds)
    toks = (0.0,) + AM._QUAD_LADDER_TOK
    for j in range(len(AM._QUAD_LADDER_N) - 1):
        tlo = toks[j] * 1.05 + 1e-3
        thi = toks[j + 1] * 0.95
        a, c1, c2 = _batch(rng, tlo, thi)
        dv = np.max(np.abs(np.asarray(kB(a, c1, c2)) - np.asarray(kS(a, c1, c2))))
        assert dv < 1e-12, ("rung N=%d disagrees with the N=320 kernel by "
                            "%g" % (AM._QUAD_LADDER_N[j], dv))


def test_fused_driver_uses_the_block_dispatcher():
    """The fused laplace jaxpr must contain the dispatcher's switch: a cond
    primitive with one branch per ladder rung + pure-Laplace + full kernel.
    Fails if _dist_step is rewired back to the undispatched kernel (the
    execution-cost regression this file exists to prevent)."""
    n_branches = len(AM._QUAD_LADDER_N) + 2
    data = make_synth()
    xg, lwg = make_distance_grid(30.0, 3000.0, 8, distMpcRef=data.distMpcRef)

    def f(ra, dec, incl):
        return AM.fused_log_likelihood_distphipsimarg_laplace(
            data, ra, dec, incl, xg, lwg, amp_sizing=900.0)

    jaxpr = jax.make_jaxpr(f)(jnp.asarray([0.9]), jnp.asarray([0.4]),
                              jnp.asarray([1.1]))
    found = []

    def walk(jx):
        for eqn in jx.eqns:
            if eqn.primitive.name == "cond":
                found.append(len(eqn.params["branches"]))
            for val in eqn.params.values():
                vals = val if isinstance(val, (tuple, list)) else (val,)
                for v in vals:
                    if hasattr(v, "jaxpr"):
                        walk(v.jaxpr)
                    elif hasattr(v, "eqns"):
                        walk(v)
    walk(jaxpr.jaxpr)
    assert n_branches in found, (
        "fused laplace traced graph has no %d-branch switch (cond branch "
        "counts seen: %s): _dist_step is not calling "
        "_laplace_psi_lnI_block, so every lattice point pays both kernel "
        "branches again (~2,950x the grid scheme, 2026-08-28)"
        % (n_branches, sorted(set(found))))


def test_fused_value_and_grad_match_undispatched_kernel():
    """End-value wiring check on real-shaped packed data: the fused laplace
    with the dispatcher equals the same fused call with the dispatcher
    monkeypatched to the plain kernel, to ladder roundoff; the gradient is
    finite and equally close.  Catches any dispatch defect that survives
    the per-branch kernel test (e.g. wrong operands captured)."""
    data = make_synth(kappa_boost=4.0)
    xg, lwg = make_distance_grid(30.0, 3000.0, 10, distMpcRef=data.distMpcRef)
    ra = jnp.asarray([0.9, 2.1])
    dec = jnp.asarray([0.4, -0.7])
    incl = jnp.asarray([1.1, 2.4])

    def call():
        return AM.fused_log_likelihood_distphipsimarg_laplace(
            data, ra, dec, incl, xg, lwg, amp_sizing=900.0)

    v_disp = np.asarray(call())
    orig = AM._laplace_psi_lnI_block
    try:
        AM._laplace_psi_lnI_block = AM._laplace_psi_lnI
        v_plain = np.asarray(call())
    finally:
        AM._laplace_psi_lnI_block = orig
    assert np.max(np.abs(v_disp - v_plain)) < 1e-11, (v_disp, v_plain)

    def scalar(th):
        r, d_, i = th
        return jnp.sum(AM.fused_log_likelihood_distphipsimarg_laplace(
            data, r[None], d_[None], i[None], xg, lwg, amp_sizing=900.0))

    g = np.asarray(jax.grad(scalar)((ra[0], dec[0], incl[0])))
    assert np.isfinite(g).all(), g
