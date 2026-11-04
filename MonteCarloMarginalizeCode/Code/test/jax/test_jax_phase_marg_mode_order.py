#!/usr/bin/env python
"""Phase marginalization must accept either packed order of the (2,+-2) pair.

WHAT WENT WRONG.  ``_accumulate_unit``'s phase-marginalized branch is
position-dependent: it conjugates column 1 of ``Y`` and of ``Q`` and pairs
column 1 with ``conj(F)``.  It enforced that position by REFUSING any ``lms``
other than the literal list ``[(2,2), (2,-2)]``::

    NotImplementedError: phase marginalization currently requires modes
    [(2,2),(2,-2)]; got [(2, -2), (2, 2)]

The packed column order is not the caller's to choose -- it comes from a python
dict's iteration order in the precompute upstream -- so a correctly configured
``--phase-marginalization`` run could arrive with complete, valid data and simply
die.  Found 2026-09-06 driving PR #266 configurations; the campaign worked around
it by permuting ``lms``, the ``Q`` columns and ``U``/``V`` on BOTH indices itself,
and measured that permutation neutral to 4.5e-13 nats.  The fix moves that
permutation into the library.

WHY THESE TESTS LOOK LIKE THIS.

  * The equality test drives BOTH orders through the same synthetic likelihood.
    It is not a test of the permutation helper: a helper-level assertion cannot
    see a call site that stops calling the helper, and this module's recurring
    defect class is guards that look like coverage and are not.

  * ``U`` and ``V`` are (K,K) with the mode index on BOTH axes.  Permuting one
    axis returns a WRONG likelihood with no error, so
    ``test_one_axis_relabelling_is_detectable`` pins that the fixture can see
    that mistake -- without it the equality test would pass under a half-fixed
    implementation.

  * The bitwise tests protect the ordering that already works.  Nothing about
    the numbers a working run produces may change, so the canonical order must
    not merely agree to tolerance: it must take the untouched code path.

FLOATING POINT.  x64 is requested below; several assertions here are bitwise and
would be meaningless (or spuriously loose) in float32.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                        # noqa: E402

from RIFT.likelihood.jax_ile import build_likelihood_data       # noqa: E402
from RIFT.likelihood.jax_ile import core as _core               # noqa: E402
from RIFT.likelihood.jax_ile.core import (                      # noqa: E402
    _accumulate_unit, _permute_modes, _phase_marg_permutation,
    make_log_likelihood)

TREF = 1126259462.413
CANONICAL = ((2, 2), (2, -2))
SWAPPED = ((2, -2), (2, 2))
INTERPS = ("nearest", "linear", "cubic", "sinc")


# ---------------------------------------------------------------------------
# Fixture.  Structurally faithful packed data (U Hermitian PD, V complex
# symmetric), the construction test_angle_marg_smoke / test_distance_grid use.
# ---------------------------------------------------------------------------

def _packed(seed=3, npts=32, deltaT=1.0 / 1024, modes=CANONICAL):
    rng = np.random.default_rng(seed)
    K = len(modes)
    out = {}
    for det in ("H1", "L1"):
        white = (rng.standard_normal((K, 4096))
                 + 1j * rng.standard_normal((K, 4096)))
        kx = np.arange(-40, 41)
        kern = np.exp(-0.5 * (kx / 12.0) ** 2)
        kern /= kern.sum()
        rho = np.stack([np.convolve(white[k].real, kern, "same")
                        + 1j * np.convolve(white[k].imag, kern, "same")
                        for k in range(K)]).astype(np.complex128)
        rho *= np.sqrt(len(kx))
        M = rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K))
        U = M @ M.conj().T + 3 * np.eye(K)
        B = rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K))
        V = (B @ B.T) * 0.3
        out[det] = dict(lms=np.array(modes, dtype=int), rholmArray=rho,
                        U=U, V=V, epoch=TREF - 0.5)
    return out


def _relabel(pk, perm, u_axes=(0, 1), v_axes=(0, 1)):
    """Repack the SAME physics with the mode axis reordered by ``perm``.

    ``u_axes`` / ``v_axes`` exist only so a test can build a deliberately
    HALF-relabelled bank; the honest relabelling permutes both axes of each.
    """
    p = np.asarray(perm, dtype=int)
    out = {}
    for det, d in pk.items():
        U = np.asarray(d["U"])
        V = np.asarray(d["V"])
        for ax in u_axes:
            U = np.take(U, p, axis=ax)
        for ax in v_axes:
            V = np.take(V, p, axis=ax)
        out[det] = dict(lms=np.asarray(d["lms"])[p],
                        rholmArray=np.asarray(d["rholmArray"])[p],
                        U=U, V=V, epoch=d["epoch"])
    return out


def _data(pk, npts=32, deltaT=1.0 / 1024):
    tw = npts * deltaT / 2.0
    return build_likelihood_data(pk, deltaT, TREF, np.linspace(-tw, tw, npts))


def _angles(S=5, seed=11):
    rng = np.random.default_rng(seed)
    return [jnp.asarray(x) for x in (rng.uniform(0.0, 2 * np.pi, S),
                                     rng.uniform(-1.2, 1.2, S),
                                     rng.uniform(0.0, np.pi, S),
                                     rng.uniform(0.0, np.pi, S),
                                     rng.uniform(0.0, 2 * np.pi, S))]


def _acc(pk, interp="cubic", guard=0, th=None):
    k, r = _accumulate_unit(_data(pk), *(th or _angles()), interp, True,
                            guard=guard)
    return np.asarray(k), np.asarray(r)


# ---------------------------------------------------------------------------
# 1. The defect: both orders must be accepted, and must agree.
# ---------------------------------------------------------------------------

def test_swapped_order_is_accepted_at_all():
    """The bug was an outright refusal, so pin the refusal's absence first --
    an equality test alone would report an ERROR, not a diagnosis."""
    _accumulate_unit(_data(_relabel(_packed(), [1, 0])), *_angles(),
                     "cubic", True)


def test_both_orders_give_the_same_accumulation():
    """Same physics, two packings: the accumulation must not know the
    difference.  Every stencil, guarded and unguarded."""
    pk = _packed()
    th = _angles()
    for interp in INTERPS:
        for guard in (0, 3):
            k0, r0 = _acc(pk, interp, guard, th)
            k1, r1 = _acc(_relabel(pk, [1, 0]), interp, guard, th)
            scale = max(np.abs(k0).max(), 1.0)
            assert np.abs(k0 - k1).max() <= 1e-12 * scale, (
                "interp=%s guard=%d: kappa differs by %.3g between mode orders"
                % (interp, guard, np.abs(k0 - k1).max()))
            assert np.abs(r0 - r1).max() <= 1e-12 * max(np.abs(r0).max(), 1.0), (
                "interp=%s guard=%d: rho^2 differs by %.3g between mode orders"
                % (interp, guard, np.abs(r0 - r1).max()))


def test_both_orders_give_the_same_lnL_through_the_public_seam():
    """Through ``make_log_likelihood`` -- the seam a driver actually calls --
    not only through the private accumulator."""
    pk = _packed()
    ra, dec, psi, incl, phiref = _angles()
    dist = jnp.full(ra.shape, 400.0)
    f0 = make_log_likelihood(_data(pk), interp="cubic",
                             phase_marginalization=True)
    f1 = make_log_likelihood(_data(_relabel(pk, [1, 0])), interp="cubic",
                             phase_marginalization=True)
    l0 = np.asarray(f0(ra, dec, psi, incl, phiref, dist))
    l1 = np.asarray(f1(ra, dec, psi, incl, phiref, dist))
    assert np.isfinite(l0).all(), "fixture produced a non-finite lnL"
    assert np.abs(l0 - l1).max() <= 1e-10, (
        "lnL differs by %.3g nats between packed mode orders" % np.abs(l0 - l1).max())


# ---------------------------------------------------------------------------
# 2. U and V carry the mode index on BOTH axes.
# ---------------------------------------------------------------------------

def test_one_axis_relabelling_is_detectable():
    """The equality test above is only a test of "permute BOTH axes" if the
    fixture can tell a half-permutation apart.  Assert that it can, per matrix
    and per axis, in the units the assertion is made in.

    Without this, an implementation that permuted only ``U[perm]`` would pass
    every other test in this file on a fixture whose U happened to be
    symmetric, and would return a silently wrong likelihood in production.
    """
    pk = _packed()
    th = _angles()
    _, r0 = _acc(pk, th=th)
    tol = 1e-12 * max(np.abs(r0).max(), 1.0)
    for name, kw in (("U axis 0 only", dict(u_axes=(0,))),
                     ("U axis 1 only", dict(u_axes=(1,))),
                     ("V axis 0 only", dict(v_axes=(0,))),
                     ("V axis 1 only", dict(v_axes=(1,)))):
        _, rb = _acc(_relabel(pk, [1, 0], **kw), th=th)
        assert np.abs(r0 - rb).max() > 1e6 * tol, (
            "%s is invisible in rho^2 (max diff %.3g <= %.3g): this fixture "
            "cannot detect a half-permuted U/V, so the equality tests do not "
            "gate it" % (name, np.abs(r0 - rb).max(), 1e6 * tol))


def test_permute_modes_permutes_both_axes_of_U_and_V():
    """Element-level contract of the helper, independent of any contraction:
    ``out[i, j] == in[perm[i], perm[j]]``."""
    rng = np.random.default_rng(5)
    K = 2
    U = rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K))
    V = rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K))
    Q = rng.standard_normal((7, K)) + 1j * rng.standard_normal((7, K))
    perm = [1, 0]
    lms2, Q2, U2, V2 = _permute_modes(list(SWAPPED), jnp.asarray(Q),
                                      jnp.asarray(U), jnp.asarray(V), perm)
    assert lms2 == list(CANONICAL)
    for i in range(K):
        assert np.array_equal(np.asarray(Q2)[:, i], Q[:, perm[i]])
        for j in range(K):
            assert np.asarray(U2)[i, j] == U[perm[i], perm[j]], "U axis pair"
            assert np.asarray(V2)[i, j] == V[perm[i], perm[j]], "V axis pair"


# ---------------------------------------------------------------------------
# 3. The ordering that already works must be untouched -- bitwise.
# ---------------------------------------------------------------------------

def test_canonical_order_never_enters_the_permutation_path():
    """The strongest available statement of "no numerical change for data that
    already works": the canonical order does not merely agree to tolerance, it
    executes the SAME operations it executed before this fix, because the
    permutation is skipped entirely rather than applied as an identity.

    Asserted by making the permutation helper fatal for the duration -- a
    counter that is merely checked at the end can be defeated by a call that
    happens somewhere the counter does not look.
    """
    assert _phase_marg_permutation(list(CANONICAL)) is None, (
        "canonical order must yield None (skip), not an identity permutation: "
        "an identity `take` is still a new node in the XLA graph")

    def _fatal(*a, **kw):
        raise AssertionError(
            "_permute_modes was called for the canonical mode order; the "
            "already-working path is no longer bit-for-bit what it was")

    saved = _core._permute_modes
    _core._permute_modes = _fatal
    try:
        for interp in INTERPS:
            _accumulate_unit(_data(_packed()), *_angles(), interp, True)
    finally:
        _core._permute_modes = saved


def test_relabelling_is_bitwise_exact_not_merely_close():
    """A permutation moves bytes; it does not arithmetic on them.  After
    canonicalization the swapped bank is bit-identical to the canonical one, so
    every downstream op sees identical inputs and the outputs must match to the
    last bit.

    Pinned bitwise on purpose.  A drift to ~1e-16 here would mean the
    canonicalization stopped being exact data movement (a cast, a reassociating
    fusion) -- worth a red build and a look, not a widened tolerance.
    """
    pk = _packed()
    th = _angles()
    for interp in INTERPS:
        k0, r0 = _acc(pk, interp, th=th)
        k1, r1 = _acc(_relabel(pk, [1, 0]), interp, th=th)
        assert k0.tobytes() == k1.tobytes(), (
            "interp=%s: kappa not bitwise identical across mode orders "
            "(max diff %.3g)" % (interp, np.abs(k0 - k1).max()))
        assert r0.tobytes() == r1.tobytes(), (
            "interp=%s: rho^2 not bitwise identical across mode orders "
            "(max diff %.3g)" % (interp, np.abs(r0 - r1).max()))


def test_non_phase_marginalized_path_is_untouched():
    """``phase_marginalization=False`` must not canonicalize anything: it never
    refused an order, and its contractions are order-symmetric, so touching it
    would change working numbers for no benefit."""
    def _fatal(*a, **kw):
        raise AssertionError("_permute_modes called with phase_marginalization=False")

    saved = _core._permute_modes
    _core._permute_modes = _fatal
    try:
        for pk in (_packed(), _relabel(_packed(), [1, 0])):
            _accumulate_unit(_data(pk), *_angles(), "cubic", False)
    finally:
        _core._permute_modes = saved


# ---------------------------------------------------------------------------
# 4. Only the ORDER is free.  Every other mode set is still a real gap.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("modes", [
    ((2, 2),),                                   # one mode
    ((2, 2), (2, -2), (3, 3)),                   # a third mode
    ((2, 2), (2, 1)),                            # right count, wrong pair
    ((2, 1), (2, -1)),                           # a different m pair entirely
    ((2, 2), (2, 2)),                            # duplicated
])
def test_other_mode_sets_still_raise(modes):
    """Widening the ORDER must not have widened the SET.  The conjugation the
    accumulator applies is specific to one m=+2/m=-2 pair; silently accepting a
    third mode would drop it from the likelihood instead of failing."""
    with pytest.raises(NotImplementedError):
        _phase_marg_permutation([tuple(m) for m in modes])


def test_the_refusal_is_reachable_from_the_accumulator():
    """...and the accumulator still surfaces it, rather than the helper being
    correct in isolation while nothing calls it."""
    with pytest.raises(NotImplementedError):
        _accumulate_unit(_data(_packed(modes=((2, 2), (2, 1)))), *_angles(),
                         "cubic", True)
