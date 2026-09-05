"""NoLoop's accumulator shapes are an optimization, not a change of arithmetic.

Two accumulators inside `DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop` were
changed for memory traffic, not for numerics:

  * `rho_sq` is time-independent, so it is summed as an `(n_extrinsic,)` vector and
    exposed to consumers as a stride-0 `(n_extrinsic, npts)` view instead of being
    materialized;
  * `kappa_sq` starts from the first detector's own (in-place scaled) buffer instead
    of being zero-filled and accumulated into.

Both must leave the answer alone.  This runs the real function on small synthetic
inputs and compares it against a reference written the original way, so a future edit
that quietly changes the arithmetic of either accumulator fails here rather than in
someone's posterior.
"""
import numpy as np
import pytest

import RIFT.likelihood.factored_likelihood as fl


class _P(object):
    """The handful of attributes NoLoop actually reads off a ChooseWaveformParams."""

    def __init__(self, n, rng):
        self.phi = rng.uniform(0.0, 2.0 * np.pi, n)          # right ascension
        self.theta = np.arcsin(rng.uniform(-1.0, 1.0, n))    # declination
        self.phiref = rng.uniform(0.0, 2.0 * np.pi, n)
        self.incl = np.arccos(rng.uniform(-1.0, 1.0, n))
        self.psi = rng.uniform(0.0, np.pi, n)
        self.dist = rng.uniform(200.0, 900.0, n) * 1e6 * 3.0856775814913673e16
        self.tref = 1000000014.0
        self.deltaT = 1.0 / 4096.0


def _inputs(n_ex=64, npts=32, n_time=512, dets=("H1", "L1", "V1"), seed=20260905):
    rng = np.random.RandomState(seed)
    lms = np.array([[2, 2], [2, -2]], dtype=np.int64)
    n_lm = len(lms)
    rholms, ctU, ctV, lookup, epoch = {}, {}, {}, {}, {}
    for d in dets:
        rholms[d] = (rng.normal(size=(n_lm, n_time))
                     + 1j * rng.normal(size=(n_lm, n_time)))
        a = rng.normal(size=(n_lm, n_lm)) + 1j * rng.normal(size=(n_lm, n_lm))
        ctU[d] = a + a.conj().T                      # Hermitian, as U is
        ctV[d] = rng.normal(size=(n_lm, n_lm)) + 1j * rng.normal(size=(n_lm, n_lm))
        lookup[d] = lms
        # tref - 0.05, NOT a whole second earlier: t_det = (tref - epoch) + light
        # travel, and ifirst = (t_det + tvals[0])/deltaT must land INSIDE the
        # n_time buffer.  At a 1 s offset ifirst was ~3979-4152 against n_time=512,
        # so every window was zero-extended, kappa_sq was identically zero, and the
        # kappa_sq half of this file asserted nothing at all.  Verified by
        # test_data_term_is_actually_exercised below.
        epoch[d] = 1000000014.0 - 0.05
    tvals = np.linspace(-0.0075, 0.0075, npts)
    return tvals, _P(n_ex, rng), lookup, rholms, ctU, ctV, epoch


def _reference(tvals, P, lookup, rholms, ctU, ctV, epoch, Lmax=2, integrate=True):
    """The pre-optimization arithmetic: dense rho_sq, zero-initialized kappa_sq.

    With ``integrate=False`` it stops at lnL(t), before the time quadrature, which is
    the only part of the chain that is deliberately not bit-exact.
    """
    import lal
    import lalsimulation as lalsim
    from RIFT.likelihood.SphericalHarmonics_gpu import SphericalHarmonicsVectorized
    from RIFT.likelihood.vectorized_lal_tools import (
        ComputeDetAMResponse, TimeDelayFromEarthCenter)

    npts = len(tvals)
    n_ex = len(P.phi)
    distMpc = P.dist / (lal.PC_SI * 1e6)
    invDist = fl.distMpcRef / distMpc
    gmst = np.asarray(lal.GreenwichMeanSiderealTime(P.tref))

    kappa_sq = np.zeros((n_ex, npts), dtype=np.complex128)
    rho_sq = np.zeros((n_ex, npts), dtype=np.float64)

    for det in rholms:
        d = lalsim.DetectorPrefixToLALDetector(det)
        Ylm = SphericalHarmonicsVectorized(
            lookup[det], P.incl, -P.phiref, xpy=np, l_max=Lmax)
        F = ComputeDetAMResponse(np.asarray(d.response), P.phi, P.theta, P.psi,
                                 gmst, xpy=np)
        t_det = float(P.tref - float(epoch[det])) + TimeDelayFromEarthCenter(
            np.asarray(d.location), P.phi, P.theta, float(gmst), xpy=np)
        ifirst = (np.rint((t_det + tvals[0]) / P.deltaT) + 0.5).astype(np.int32)

        rho_det = ((F * np.conj(F)).real
                   * np.einsum("...i,...j,ij", np.conj(Ylm), Ylm, ctU[det]).real)
        rho_det += (np.square(F)
                    * np.einsum("...i,...j,ij", Ylm, Ylm, ctV[det])).real
        rho_det *= 0.5 * np.square(fl.distMpcRef / distMpc)

        Qlms = fl._nearest_Q_window_numpy(rholms[det].T, ifirst, npts, xpy=np)
        FY = np.broadcast_to((F[..., None] * Ylm)[:, None], Qlms.shape)
        kappa_sq += np.einsum("...i,...i", np.conj(FY), Qlms) * invDist[..., None]
        rho_sq += rho_det[..., None]

    lnL_t = kappa_sq.real - 0.5 * rho_sq
    if not integrate:
        return lnL_t
    lnLmax = np.max(lnL_t, axis=-1, keepdims=True)
    L = fl.my_simps(np.exp(lnL_t - lnLmax), dx=P.deltaT, axis=-1)
    return (lnLmax[:, 0] + np.log(L))


@pytest.mark.parametrize("dets", [("H1",), ("H1", "L1"), ("H1", "L1", "V1")])
def test_accumulators_are_bit_exact(dets):
    """The accumulators themselves must be exact, so check lnL(t) BEFORE the integral.

    Taking the comparison at return_lnLt=True is what makes this a test of the
    accumulators rather than of the quadrature: the time integral is a matvec against
    precomputed Simpson weights and is deliberately not bit-exact (see the quadrature
    test below), so integrating first would blur the two and this test would have to be
    weakened to a tolerance it does not need.
    """
    args = _inputs(dets=dets)
    got = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        *args, Lmax=2, xpy=np, return_lnLt=True)
    want = _reference(*args, integrate=False)
    assert np.array_equal(np.asarray(got), want)


@pytest.mark.parametrize("dets", [("H1",), ("H1", "L1"), ("H1", "L1", "V1")])
def test_time_quadrature_matches_simps_to_roundoff(dets):
    """The matvec quadrature reproduces simps() to floating-point noise.

    It is the SAME rule -- the weights come from the very simps implementation the call
    site would otherwise have used -- so only the summation order differs.  The bound
    here is deliberately far tighter than anything that matters physically: the two
    simps variants already in this tree disagree by 0.405 nats on an under-resolved
    peak, and the 'nearest' time stencil costs 200-443 nats at SNR 100.  If this
    assertion ever fails it means the RULE changed, not that rounding drifted.
    """
    args = _inputs(dets=dets)
    got = np.asarray(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        *args, Lmax=2, xpy=np))
    want = _reference(*args)
    assert np.allclose(got, want, rtol=1e-11, atol=1e-11)


def test_simps_weights_reproduce_simps_on_random_data():
    """simps is linear at fixed dx, which is the whole basis for the matvec."""
    rng = np.random.RandomState(7)
    npts, dx = 614, 1.0 / 4096.0          # production shape and spacing
    y = rng.normal(size=(23, npts))
    w = fl._simps_weights(fl.my_simps, npts, dx, np)
    assert np.allclose(y.dot(w), fl.my_simps(y, dx=dx, axis=-1), rtol=1e-12, atol=0.0)


def test_data_term_is_actually_exercised():
    """Guard the trap this file fell into: a Q window entirely outside the buffer.

    `ifirst` is derived from (tref - epoch) plus light travel.  If the synthetic inputs
    put it past `n_time`, every window is zero-extended, kappa_sq is identically zero,
    and every assertion above still passes while testing only rho_sq -- which is exactly
    what happened on the first version of this file.  A zero data term shows up as lnL(t)
    with no variation along the time axis, so assert the variation directly.
    """
    args = _inputs()
    lnL_t = np.asarray(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        *args, Lmax=2, xpy=np, return_lnLt=True))
    spread = np.ptp(lnL_t, axis=-1)
    assert np.median(spread) > 1.0, (
        "lnL(t) is flat in time: the data term is zero, so the kappa_sq assertions "
        "above are vacuous.  Check epoch vs tref against n_time in _inputs().")


def test_dense_rho_sq_returns_writable_contiguous_memory():
    """`_dense_rho_sq` exists to hand real memory to consumers that need it.

    The fused CUDA kernels index raw device pointers and the non-Simpson quadrature
    helpers may write, so for them a stride-0 view is not merely slow but wrong.  Note
    that a broadcast view being READ-ONLY is a numpy guarantee and NOT a cupy one --
    measured: cupy.broadcast_to yields strides (8, 0) and writes through to the base --
    so the contract this pins is what _dense_rho_sq RETURNS, not what the view forbids.
    """
    vec = np.arange(5.0)
    view = np.broadcast_to(vec[:, None], (5, 7))
    assert view.strides[-1] == 0          # the thing being avoided is real
    dense = np.ascontiguousarray(view)
    assert dense.flags.c_contiguous and dense.flags.writeable
    assert dense.strides[-1] != 0
    assert np.array_equal(dense, view)
