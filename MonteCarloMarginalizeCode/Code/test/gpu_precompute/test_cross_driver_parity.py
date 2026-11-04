"""Pin classic/JAX compound-likelihood algebra at production response orders."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import lal
import numpy as np

from RIFT.likelihood import factored_likelihood_rotating_freqresponse as compound
from RIFT.likelihood import slowrot_freqresponse
from RIFT.likelihood.jax_ile.banded import build_rotating_freqresponse_data
from RIFT.likelihood.jax_ile.core import fused_log_likelihood

jax.config.update("jax_enable_x64", True)


def test_p1_q1_cubic_fixed_point_matches_classic_likelihood():
    """Same Q/U/V, geometry and parameters must agree before any sampler runs."""
    rng = np.random.default_rng(12291)
    qmax, pmax = 1, 1
    modes = [(2, 2), (2, -2)]
    a_list = compound.compound_index_set(qmax, pmax)
    A, K, N = len(a_list), len(modes), 256
    delta_t = 1.0/1024
    tref = 1000000000.0
    epoch = tref - 128*delta_t
    det = "H1"

    q = rng.normal(size=(A, K, N)) + 1j*rng.normal(size=(A, K, N))
    u = rng.normal(size=(A, A, K, K)) + 1j*rng.normal(size=(A, A,K,K))
    v = rng.normal(size=(A, A, K, K)) + 1j*rng.normal(size=(A,A,K,K))
    meta = dict(
        feature="rotation_freqresponse", post_phase_required=True,
        event_time_geo=tref, modes=modes, a_list=a_list,
        Qmax=qmax, p_max=pmax, f_sidereal=compound.flwr.F_SIDEREAL,
        L_arm=4000.0)
    lookup = {det: np.asarray(modes, dtype=int)}
    rho = {det: {a: q[i] for i, a in enumerate(a_list)}}
    U, V, epochs = {det: u}, {det: v}, {det: epoch}
    geometry = {det: slowrot_freqresponse.detector_geometry(det, L_arm=4000.0)}
    tvals = np.arange(-4, 5)*delta_t
    data = build_rotating_freqresponse_data(
        meta, lookup, rho, U, V, epochs, delta_t, tvals, geometry)

    ra = np.asarray([0.7, 5.8])
    dec = np.asarray([-0.2, 0.85])
    psi = np.asarray([0.3, 2.7])
    incl = np.asarray([0.6, 2.5])
    phiref = np.asarray([0.4, 5.0])
    dist_mpc = np.asarray([100.0, 900.0])
    params = SimpleNamespace(
        phi=ra, theta=dec, psi=psi, incl=incl, phiref=phiref,
        dist=dist_mpc*1.0e6*lal.PC_SI, tref=tref, deltaT=delta_t)

    classic_t = compound.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
        tvals, params, meta, lookup, rho, U, V, epochs, Lmax=2,
        array_output=True, time_interp="cubic")
    jax_t = np.asarray(fused_log_likelihood(
        data, *(jnp.asarray(x) for x in
                (ra, dec, psi, incl, phiref, dist_mpc)),
        interp="cubic", return_lnLt=True))
    np.testing.assert_allclose(jax_t, classic_t, rtol=2e-13, atol=2e-13)

    classic_marg = compound.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
        tvals, params, meta, lookup, rho, U, V, epochs, Lmax=2,
        array_output=False, time_interp="cubic")
    jax_marg = np.asarray(fused_log_likelihood(
        data, *(jnp.asarray(x) for x in
                (ra, dec, psi, incl, phiref, dist_mpc)), interp="cubic"))
    np.testing.assert_allclose(jax_marg, classic_marg, rtol=2e-13, atol=2e-13)
