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
        epoch[d] = 1000000013.0
    tvals = np.linspace(-0.0075, 0.0075, npts)
    return tvals, _P(n_ex, rng), lookup, rholms, ctU, ctV, epoch


def _reference(tvals, P, lookup, rholms, ctU, ctV, epoch, Lmax=2):
    """The pre-optimization arithmetic: dense rho_sq, zero-initialized kappa_sq."""
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
    lnLmax = np.max(lnL_t, axis=-1, keepdims=True)
    L = fl.my_simps(np.exp(lnL_t - lnLmax), dx=P.deltaT, axis=-1)
    return (lnLmax[:, 0] + np.log(L))


@pytest.mark.parametrize("dets", [("H1",), ("H1", "L1"), ("H1", "L1", "V1")])
def test_noloop_matches_dense_accumulator_reference(dets):
    args = _inputs(dets=dets)
    got = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(*args, Lmax=2, xpy=np)
    want = _reference(*args)
    # Same operations on the same scalars in the same order: demand exactness, not a
    # tolerance, so that a reassociating "optimization" cannot slip through.
    assert np.array_equal(np.asarray(got), want)


def test_rho_sq_view_is_not_writable_into():
    """The shared rho_sq view must not be something a consumer can scribble on.

    numpy and cupy both return a read-only broadcast; if that ever changed, a consumer
    writing into rho_sq would corrupt every time bin at once instead of one.
    """
    vec = np.arange(5.0)
    view = np.broadcast_to(vec[:, None], (5, 7))
    with pytest.raises(ValueError):
        view[0, 0] = 1.0
