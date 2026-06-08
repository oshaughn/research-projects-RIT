"""
High-level glue: build a JAX extrinsic likelihood from the *same* inputs the
production ILE uses (a waveform-template ``ChooseWaveformParams`` ``P``, a
``data_dict`` of frequency-domain strain, and a ``psd_dict``).

The expensive, data-touching steps are delegated verbatim to the production
code: ``factored_likelihood.PrecomputeLikelihoodTerms`` (waveform generation +
the ``<h_lm(t)|d>`` and ``<h_lm|h_l'm'>`` inner products) and
``factored_likelihood.PackLikelihoodDataStructuresAsArrays`` (array packing).
We only wrap their outputs into the JAX :class:`JAXLikelihoodData` and expose
the AD niceties (value, gradient, Hessian/Fisher).

This keeps frame reading, PSD handling, epoch bookkeeping and waveform
generation bit-identical to ILE, while the extrinsic likelihood becomes a
differentiable JAX function.
"""

import numpy as np

import jax
import jax.numpy as jnp

import RIFT.likelihood.factored_likelihood as factored_likelihood

import os

from .core import (build_likelihood_data, fused_log_likelihood,
                   fused_log_likelihood_distmarg, fused_log_likelihood_distphimarg,
                   make_distance_grid, make_distance_grid_adaptive,
                   estimate_distance_peak, phi_ref_grid, phi_ref_conditional_lnL,
                   DIST_MPC_REF)

# Parameter order used throughout the wrapper's vectorized interface.
EXTRINSIC_PARAM_ORDER = ("ra", "dec", "psi", "incl", "phiref", "distMpc")


def build_data_from_precompute(P, data_dict, psd_dict, fiducial_epoch,
                               storage_window_half, integration_window_half,
                               Lmax, fMax,
                               analyticPSD_Q=False, inv_spec_trunc_Q=False,
                               T_spec=0.0, tvals=None, verbose=False,
                               skip_interpolation=True,
                               **precompute_kwargs):
    """Run the production precompute + packing, return a JAXLikelihoodData.

    Parameters mirror the production driver, which uses *two distinct* time
    windows (mirrored here):

    * ``storage_window_half`` (``--internal-data-storage-window-half``, default
      0.15 s) -- the half-width of the rholm timeseries *buffer* built by
      ``PrecomputeLikelihoodTerms``.  Must exceed the integration half-width by
      enough to absorb the per-detector time-delay excursion as the sky
      location roams, or the analysis window slides off the buffer.
    * ``integration_window_half`` (``--data-integration-window-half``, default
      0.075 s) -- the half-width of the time-*marginalization* window; the
      ``tvals`` grid is ``linspace(-iwh, iwh, int(2*iwh/deltaT))``, exactly as
      the driver constructs it.

    Returns
    -------
    (data, extras) where ``data`` is a :class:`JAXLikelihoodData` and
    ``extras`` is a dict with the raw precompute products (rholms, cross terms,
    guessed SNR) for callers that want them.
    """
    rholms_intp, cross_terms, cross_terms_V, rholms, guess_snr, _ = \
        factored_likelihood.PrecomputeLikelihoodTerms(
            fiducial_epoch, storage_window_half, P, data_dict, psd_dict,
            Lmax, fMax, analyticPSD_Q, inv_spec_trunc_Q, T_spec,
            verbose=verbose, skip_interpolation=skip_interpolation,
            **precompute_kwargs)

    packed = {}
    for det in data_dict.keys():
        (lookupNK, lookupKN, lookupKNconj, ctU, ctV,
         rholmArray, rholm_intpArray, epoch) = \
            factored_likelihood.PackLikelihoodDataStructuresAsArrays(
                list(rholms[det].keys()), rholms_intp[det], rholms[det],
                cross_terms[det], cross_terms_V[det])
        packed[det] = dict(lms=lookupNK, rholmArray=rholmArray,
                           U=ctU, V=ctV, epoch=epoch)

    deltaT = float(P.deltaT)
    if tvals is None:
        npts = int(2 * integration_window_half / deltaT)
        tvals = np.linspace(-integration_window_half,
                            integration_window_half, npts)

    data = build_likelihood_data(packed, deltaT, float(fiducial_epoch), tvals)
    extras = dict(rholms=rholms, cross_terms=cross_terms,
                  cross_terms_V=cross_terms_V, guess_snr=guess_snr,
                  rholms_intp=rholms_intp)
    return data, extras


class JAXExtrinsicLikelihood:
    """Differentiable extrinsic log-likelihood with AD conveniences.

    Wraps a :class:`JAXLikelihoodData` and offers value / gradient /
    value-and-grad / Hessian over the 6 extrinsic parameters
    ``(ra, dec, psi, incl, phiref, distMpc)``.  The single-point methods take
    scalars (for optimizers / Fisher); the batched ``log_likelihood`` takes
    arrays of shape (S,).
    """

    def __init__(self, data, interp="linear", phase_marginalization=False):
        self.data = data
        self.interp = interp
        self.phase_marginalization = phase_marginalization

        def _batched(ra, dec, psi, incl, phiref, distMpc):
            return fused_log_likelihood(
                data, ra, dec, psi, incl, phiref, distMpc,
                interp=interp, phase_marginalization=phase_marginalization)

        self._batched = jax.jit(_batched)

        # single-point scalar version, used for grad/hessian over a 6-vector
        def _scalar(theta6):
            v = fused_log_likelihood(
                data,
                theta6[0:1], theta6[1:2], theta6[2:3],
                theta6[3:4], theta6[4:5], theta6[5:6],
                interp=interp, phase_marginalization=phase_marginalization)
            return v[0]

        self._scalar = _scalar
        self._value_and_grad = jax.jit(jax.value_and_grad(_scalar))
        self._hessian = jax.jit(jax.hessian(_scalar))

    # -- batched ---------------------------------------------------------
    def log_likelihood(self, ra, dec, psi, incl, phiref, distMpc):
        """lnL for arrays of extrinsic params, shape (S,)."""
        return self._batched(
            jnp.asarray(ra), jnp.asarray(dec), jnp.asarray(psi),
            jnp.asarray(incl), jnp.asarray(phiref), jnp.asarray(distMpc))

    # -- single-point AD -------------------------------------------------
    def value(self, theta6):
        return float(self._scalar(jnp.asarray(theta6, dtype=jnp.float64)))

    def value_and_grad(self, theta6):
        v, g = self._value_and_grad(jnp.asarray(theta6, dtype=jnp.float64))
        return float(v), np.asarray(g)

    def fisher(self, theta6):
        """Observed Fisher matrix ``-Hessian(lnL)`` at ``theta6`` (6x6)."""
        H = np.asarray(self._hessian(jnp.asarray(theta6, dtype=jnp.float64)))
        return -H


class JAXDistanceMarginalizedLikelihood:
    """Distance- and time-marginalized lnL over the 5 angular parameters.

    ``theta5 = (ra, dec, psi, incl, phiref)``.  Distance is integrated out with
    the chosen prior (default volumetric), which regulates the amplitude/distance
    degeneracy of the bare factored likelihood -- giving a smooth, bounded
    objective suitable for gradient ascent, Fisher forecasting and a
    well-conditioned evidence integral.  This mirrors the production
    distance-marginalized ILE path.
    """

    ANGULAR_PARAM_ORDER = ("ra", "dec", "psi", "incl", "phiref")

    def __init__(self, data, d_min, d_max, n_grid=256, d_prior="euclidean",
                 interp="linear", phase_marginalization=False):
        self.data = data
        self.x_grid, self.log_w_grid = make_distance_grid(
            d_min, d_max, n_grid, d_prior, distMpcRef=data.distMpcRef)

        def _batched(ra, dec, psi, incl, phiref):
            return fused_log_likelihood_distmarg(
                data, ra, dec, psi, incl, phiref,
                self.x_grid, self.log_w_grid,
                interp=interp, phase_marginalization=phase_marginalization)

        self._batched = jax.jit(_batched)

        def _scalar(theta5):
            v = fused_log_likelihood_distmarg(
                data, theta5[0:1], theta5[1:2], theta5[2:3],
                theta5[3:4], theta5[4:5], self.x_grid, self.log_w_grid,
                interp=interp, phase_marginalization=phase_marginalization)
            return v[0]

        self._scalar = _scalar
        self._value_and_grad = jax.jit(jax.value_and_grad(_scalar))
        self._hessian = jax.jit(jax.hessian(_scalar))

    def log_likelihood(self, ra, dec, psi, incl, phiref):
        return self._batched(
            jnp.asarray(ra), jnp.asarray(dec), jnp.asarray(psi),
            jnp.asarray(incl), jnp.asarray(phiref))

    def value(self, theta5):
        return float(self._scalar(jnp.asarray(theta5, dtype=jnp.float64)))

    def value_and_grad(self, theta5):
        v, g = self._value_and_grad(jnp.asarray(theta5, dtype=jnp.float64))
        return float(v), np.asarray(g)

    def fisher(self, theta5):
        H = np.asarray(self._hessian(jnp.asarray(theta5, dtype=jnp.float64)))
        return -H


class JAXDistPhiMargLikelihood:
    """Distance- and φ_ref-marginalised lnL over 4 angular parameters.

    Both the luminosity distance and the orbital phase φ_ref are integrated
    out; the result is a smooth function of
    ``theta4 = (ra, dec, psi, incl)`` only.

    This removes the curved φ_ref–psi degeneracy ridge that causes flowMC /
    NUTS to collapse at high SNR, while handling all l_max correctly (grid
    sum, no QAS approximation).

    Parameters
    ----------
    nphi : int
        φ_ref grid size.  32 is exact and fast for l_max = 2; use 64–128
        for l_max ≥ 4 or for production-quality runs.
    """

    ANGULAR_PARAM_ORDER = ("ra", "dec", "psi", "incl")

    def __init__(self, data, d_min, d_max, nphi=32, n_grid=256,
                 d_prior="euclidean", interp="linear", guess_snr=None):
        self.data = data
        self.nphi = int(nphi)
        self._phi_grid = phi_ref_grid(self.nphi)
        # Adaptive distance quadrature: concentrate grid resolution on the
        # distance posterior, whose peak/width are set from guess_snr + the
        # precompute's max rho_sq_unit -- avoiding the uniform grid's high-SNR
        # under-resolution (the ~1% evidence bias).  Static grid -> feeds the
        # same stable logsumexp kernel and is gradient-stable.  Enable with env
        # JAX_ILE_DISTGRID_ADAPTIVE=1; falls back to uniform otherwise.
        if int(os.environ.get("JAX_ILE_DISTGRID_ADAPTIVE", "0")) and guess_snr:
            d_peak, sigma_d = estimate_distance_peak(data, guess_snr)
            self.x_grid, self.log_w_grid = make_distance_grid_adaptive(
                d_min, d_max, d_peak, sigma_d, d_prior, distMpcRef=data.distMpcRef)
            self.dist_grid_info = dict(mode="adaptive", d_peak=float(d_peak),
                                       sigma_d=float(sigma_d),
                                       n=int(self.x_grid.shape[0]))
        else:
            self.x_grid, self.log_w_grid = make_distance_grid(
                d_min, d_max, n_grid, d_prior, distMpcRef=data.distMpcRef)
            self.dist_grid_info = dict(mode="uniform", n=int(self.x_grid.shape[0]))

        xg, lwg, pg = self.x_grid, self.log_w_grid, self._phi_grid

        def _batched(ra, dec, psi, incl):
            return fused_log_likelihood_distphimarg(
                data, ra, dec, psi, incl, xg, lwg, pg, interp=interp)

        self._batched = jax.jit(_batched)

        def _scalar(theta4):
            v = fused_log_likelihood_distphimarg(
                data, theta4[0:1], theta4[1:2], theta4[2:3], theta4[3:4],
                xg, lwg, pg, interp=interp)
            return v[0]

        self._scalar = _scalar
        self._value_and_grad = jax.jit(jax.value_and_grad(_scalar))
        self._hessian = jax.jit(jax.hessian(_scalar))

    def log_likelihood(self, ra, dec, psi, incl):
        """lnL for arrays of 4 angular parameters, shape (S,)."""
        return self._batched(
            jnp.asarray(ra), jnp.asarray(dec),
            jnp.asarray(psi), jnp.asarray(incl))

    def value(self, theta4):
        return float(self._scalar(jnp.asarray(theta4, dtype=jnp.float64)))

    def value_and_grad(self, theta4):
        v, g = self._value_and_grad(jnp.asarray(theta4, dtype=jnp.float64))
        return float(v), np.asarray(g)

    def fisher(self, theta4):
        """Observed Fisher matrix ``-Hessian(lnL)`` at ``theta4`` (4×4)."""
        H = np.asarray(self._hessian(jnp.asarray(theta4, dtype=jnp.float64)))
        return -H

    def sample_phi_ref(self, ra, dec, psi, incl, distMpc, rng=None,
                       n_samples=1, interp="linear"):
        """Draw φ_ref from its conditional posterior given the other params.

        Evaluates ``phi_ref_conditional_lnL`` on the grid, normalises, draws
        a grid index, then adds a uniform sub-bin jitter so the sample is not
        pinned to a grid point.

        Parameters
        ----------
        ra, dec, psi, incl, distMpc : float scalars or (S,) arrays
        rng : numpy.random.Generator (optional)
        n_samples : int  — draws per input sample

        Returns
        -------
        phi_ref : (S,) float array (or (S, n_samples) when n_samples > 1)
        """
        rng = rng or np.random.default_rng()
        ra_ = np.atleast_1d(np.asarray(ra, float))
        dec_ = np.atleast_1d(np.asarray(dec, float))
        psi_ = np.atleast_1d(np.asarray(psi, float))
        incl_ = np.atleast_1d(np.asarray(incl, float))
        dist_ = np.atleast_1d(np.asarray(distMpc, float))

        lnL_phi = np.asarray(phi_ref_conditional_lnL(
            self.data,
            jnp.asarray(ra_), jnp.asarray(dec_),
            jnp.asarray(psi_), jnp.asarray(incl_),
            jnp.asarray(dist_), self._phi_grid, interp=interp))  # (nphi, S)

        phi_vals = np.asarray(self._phi_grid)
        dphi = float(phi_vals[1] - phi_vals[0])
        S = ra_.shape[0]

        out = np.empty((S, n_samples))
        for i in range(S):
            lw = lnL_phi[:, i]
            lw = lw - lw.max()
            w = np.exp(lw); w /= w.sum()
            idxs = rng.choice(self.nphi, size=n_samples, p=w)
            jitter = rng.uniform(-0.5 * dphi, 0.5 * dphi, size=n_samples)
            out[i] = (phi_vals[idxs] + jitter) % (2.0 * np.pi)

        return out[:, 0] if n_samples == 1 else out
