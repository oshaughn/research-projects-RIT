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

from . import core as _core

import os

from .core import (build_likelihood_data, fused_log_likelihood,
                   fused_log_likelihood_distmarg, fused_log_likelihood_distphimarg,
                   fused_log_likelihood_distphipsimarg,
                   fused_log_likelihood_distpsimarg,
                   make_distance_grid, make_distance_grid_adaptive,
                   make_distance_grid_loguniform, loguniform_grid_size,
                   DIST_GRID_TOL_DEFAULT, DIST_GRID_SCHEMES,
                   estimate_distance_peak, phi_ref_grid, psi_grid,
                   phi_ref_conditional_lnL, DIST_MPC_REF, JAX_INTERP_DEFAULT,
                   TIME_QUAD_DEFAULT, _TIME_QUAD_CHOICES, default_time_guard)
# Generic probe direction for the build-time identity check.  The A0==0/B1==0
# identity is a property of the spin-2 detector response, so it does not depend
# on where we probe; a single generic (ra, dec, incl) away from any pole or
# face-on/edge-on special case is enough, and keeps the check O(1).
_ANGLE_MARG_PROBE_RA = [1.0]
_ANGLE_MARG_PROBE_DEC = [0.3]
_ANGLE_MARG_PROBE_INCL = [1.0]
from . import core as _core
from .anglemarg import (ANGLE_MARG_DEFAULT, ANGLE_MARG_LEGACY,  # noqa: F401
                        ANGLE_MARG_CHOICES)

# Parameter order used throughout the wrapper's vectorized interface.
EXTRINSIC_PARAM_ORDER = ("ra", "dec", "psi", "incl", "phiref", "distMpc")
_TIME_SUPPORT_DELAY_MARGIN = 0.05


def bandlimited_storage_requirement(deltaT, integration_window_half):
    """Return ``(storage_half, g0, g_certificate)`` for adaptive time support."""
    tvals = factored_likelihood.marginalization_time_grid(
        integration_window_half, deltaT, xpy=np)
    g_default = default_time_guard(len(tvals))
    g0 = 1 << int(np.ceil(np.log2(g_default)))
    g_certificate = 2 * g0
    # Fifty milliseconds exceeds the Earth-diameter light time (~42.6 ms), so
    # this support guarantee does not encode an HLV-only network assumption.
    storage_half = (float(integration_window_half) + g_certificate * float(deltaT)
                    + _TIME_SUPPORT_DELAY_MARGIN + 16 * float(deltaT))
    return storage_half, g0, g_certificate


def _validate_nonlinear_time_quadrature(time_quadrature, endpoint):
    if time_quadrature not in _TIME_QUAD_CHOICES:
        raise ValueError("time_quadrature must be one of %r" % (_TIME_QUAD_CHOICES,))
    if time_quadrature == "bandlimited":
        raise ValueError(
            "time_quadrature='bandlimited' is not valid for %s: the primitive "
            "time fields must be refined before its nonlinear marginalization; "
            "use 'simpson'" % endpoint)


def build_rotation_data_from_precompute(P, data_dict, psd_dict, fiducial_epoch,
                                        integration_window_half, Lmax, fMax,
                                        t_window=0.1, harmonics=(-2, -1, 0, 1, 2),
                                        p_max=0, analyticPSD_Q=False,
                                        inv_spec_trunc_Q=False, T_spec=0.0,
                                        tvals=None, verbose=False,
                                        **precompute_kwargs):
    """One-call builder for the slow-rotation (Path A/B) banded JAX likelihood.

    Runs the production ``PrecomputeLikelihoodTermsWithRotation`` +
    ``pack_rotation_arrays`` (heavy, data-touching -- reused verbatim) and wraps
    the packed banks into a banded :class:`JAXLikelihoodData`.  The returned
    object flows through every ``fused_log_likelihood*`` / marginalization
    variant and the samplers exactly like the baseline data.

    ``t_window`` is the rholm-buffer half width for the rotation precompute (it
    builds its own buffer, unlike the baseline two-window driver); ``tvals`` is
    the marginalization grid, defaulting to
    ``factored_likelihood.marginalization_time_grid(iwh, deltaT)`` -- spacing
    exactly ``deltaT``, and the SAME grid batchmode builds (issue #146).

    ``harmonics`` defaults to the ``p_max=0`` width; at ``p_max>=1`` the precompute
    widens it to ``2 + p_max`` (issue #142) and warns, because the JAX packer would
    otherwise drop the response coefficients that have no band.
    """
    import RIFT.likelihood.factored_likelihood_with_rotation as flwr
    from .banded import build_rotation_data

    ri, ct, ctV, rho, meta = flwr.PrecomputeLikelihoodTermsWithRotation(
        fiducial_epoch, t_window, P, data_dict, psd_dict, Lmax, fMax,
        harmonics=harmonics, p_max=p_max, f_sidereal=flwr.F_SIDEREAL,
        analyticPSD_Q=analyticPSD_Q, inv_spec_trunc_Q=inv_spec_trunc_Q,
        T_spec=T_spec, verbose=verbose, quiet=not verbose,
        skip_interpolation=True, **precompute_kwargs)
    lk, rbn, ubn, vbn, ep = flwr.pack_rotation_arrays(meta, rho, ct, ctV)

    deltaT = float(P.deltaT)
    if tvals is None:
        # THE one window-grid constructor, shared with
        # bin/integrate_likelihood_extrinsic_batchmode (issue #146).  Spacing is
        # exactly deltaT, matching the pos<->sample mapping and Simpson weights
        # the likelihood assumes; a linspace(-iwh,iwh,npts) grid is spaced
        # 2*iwh/(npts-1) instead, which shifts the time reference by a fraction
        # of a sample -> a sky bias that only shows up at high SNR, where cubic
        # interpolation resolves the razor-sharp peak.
        tvals = factored_likelihood.marginalization_time_grid(
            integration_window_half, deltaT, xpy=np)
    data = build_rotation_data(meta, lk, rbn, ubn, vbn, ep, deltaT, tvals)
    extras = dict(meta=meta, rho_by_a=rbn, U_by_aa=ubn, V_by_aa=vbn,
                  epochDict=ep, lookupNKDict=lk)
    return data, extras


def build_freqresponse_data_from_precompute(P, data_dict, psd_dict, fiducial_epoch,
                                            integration_window_half, Lmax, fMax,
                                            t_window=0.1, Qmax=4, L_arm=None,
                                            analyticPSD_Q=False,
                                            inv_spec_trunc_Q=False, T_spec=0.0,
                                            tvals=None, verbose=False,
                                            **precompute_kwargs):
    """One-call builder for the finite-size (Path D) banded JAX likelihood.

    Runs ``PrecomputeLikelihoodTermsFreqResponse`` + ``pack_freqresponse_arrays``
    (reused verbatim) and wraps the packed banks into a banded
    :class:`JAXLikelihoodData`.  ``L_arm`` overrides the arm length (e.g. 40000.
    for a 40-km CE arm; ``None`` = native LAL arm lengths).  The finite-size
    coefficients need the arm *unit vectors*, so the detector geometry is
    recomputed via ``slowrot_freqresponse.detector_geometry`` and attached.
    """
    import RIFT.likelihood.factored_likelihood_freqresponse as flfr
    import RIFT.likelihood.slowrot_freqresponse as sfr
    from .banded import build_freqresponse_data

    bk = flfr.PrecomputeLikelihoodTermsFreqResponse(
        fiducial_epoch, t_window, P, data_dict, psd_dict, Lmax, fMax,
        Qmax=Qmax, L_arm=L_arm, analyticPSD_Q=analyticPSD_Q,
        inv_spec_trunc_Q=inv_spec_trunc_Q, T_spec=T_spec, verbose=verbose,
        quiet=not verbose, skip_interpolation=True, **precompute_kwargs)
    meta = bk[4]
    lk, rbp, ubp, vbp, ep = flfr.pack_freqresponse_arrays(bk[4], bk[3], bk[1], bk[2])

    def _L_of(det):
        return L_arm.get(det, None) if isinstance(L_arm, dict) else L_arm
    det_geom = {det: sfr.detector_geometry(det, L_arm=_L_of(det))
                for det in data_dict.keys()}

    deltaT = float(P.deltaT)
    if tvals is None:
        # THE one window-grid constructor, shared with
        # bin/integrate_likelihood_extrinsic_batchmode (issue #146).  Spacing is
        # exactly deltaT, matching the pos<->sample mapping and Simpson weights
        # the likelihood assumes; a linspace(-iwh,iwh,npts) grid is spaced
        # 2*iwh/(npts-1) instead, which shifts the time reference by a fraction
        # of a sample -> a sky bias that only shows up at high SNR, where cubic
        # interpolation resolves the razor-sharp peak.
        tvals = factored_likelihood.marginalization_time_grid(
            integration_window_half, deltaT, xpy=np)
    data = build_freqresponse_data(meta, lk, rbp, ubp, vbp, ep, deltaT, tvals,
                                   det_geom)
    extras = dict(meta=meta, rho_by_p=rbp, U_by_pp=ubp, V_by_pp=vbp,
                  epochDict=ep, lookupNKDict=lk, det_geom=det_geom)
    return data, extras


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
      ``tvals`` grid comes from
      ``factored_likelihood.marginalization_time_grid(iwh, deltaT)``, i.e.
      ``(arange(npts) - npts//2)*deltaT`` with ``npts = int(2*iwh/deltaT)`` --
      spacing exactly ``deltaT`` (see the ``if tvals is None`` branch below).
      ``bin/integrate_likelihood_extrinsic_batchmode`` calls the same helper at
      all ten of its window-grid sites, so the two drivers agree by value
      (issue #146; it formerly built ``linspace(-iwh, iwh, int(2*iwh/deltaT))``,
      spaced ``2*iwh/(npts-1)``).  Anything that compares this data object
      against the numpy reference should still pass ``data.tvals`` to the
      reference rather than rebuild a grid.

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
        # THE one window-grid constructor, shared with
        # bin/integrate_likelihood_extrinsic_batchmode (issue #146): spacing
        # exactly deltaT, which is the convention both likelihoods EVALUATE in
        # (each steps by deltaT from tvals[0] and integrates with dx=deltaT).
        # All ten of batchmode's window-grid sites now call this same helper, so
        # the two drivers build identical grids at every sample rate.
        tvals = factored_likelihood.marginalization_time_grid(
            integration_window_half, deltaT, xpy=np)

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

    def __init__(self, data, interp=JAX_INTERP_DEFAULT, phase_marginalization=False,
                 *, time_quadrature=TIME_QUAD_DEFAULT):
        self.data = data
        self.interp = interp
        self.phase_marginalization = phase_marginalization
        if time_quadrature not in _TIME_QUAD_CHOICES:
            raise ValueError("time_quadrature must be one of %r" % (_TIME_QUAD_CHOICES,))
        self.time_quadrature = time_quadrature
        if time_quadrature == "bandlimited":
            g_default = default_time_guard(data.npts)
            self.time_guard_initial = 1 << int(np.ceil(np.log2(g_default)))
            self.time_guard_certified = 2 * self.time_guard_initial

        def _batched(ra, dec, psi, incl, phiref, distMpc):
            return fused_log_likelihood(
                data, ra, dec, psi, incl, phiref, distMpc,
                interp=interp, phase_marginalization=phase_marginalization,
                time_quadrature=time_quadrature)

        self._batched = jax.jit(_batched)

        # single-point scalar version, used for grad/hessian over a 6-vector
        def _scalar(theta6):
            v = fused_log_likelihood(
                data,
                theta6[0:1], theta6[1:2], theta6[2:3],
                theta6[3:4], theta6[4:5], theta6[5:6],
                interp=interp, phase_marginalization=phase_marginalization,
                time_quadrature=time_quadrature)
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
                 interp=JAX_INTERP_DEFAULT, phase_marginalization=False,
                 *, time_quadrature=TIME_QUAD_DEFAULT, d_prior_range=None):
        self.data = data
        self.interp = interp   # the instance's stencil; sample_phi_ref defaults to it
        self.phase_marginalization = phase_marginalization
        _validate_nonlinear_time_quadrature(
            time_quadrature, "distance marginalization")
        self.time_quadrature = time_quadrature
        self.x_grid, self.log_w_grid = make_distance_grid(
            d_min, d_max, n_grid, d_prior, distMpcRef=data.distMpcRef,
            d_prior_range=d_prior_range)

        def _batched(ra, dec, psi, incl, phiref):
            return fused_log_likelihood_distmarg(
                data, ra, dec, psi, incl, phiref,
                self.x_grid, self.log_w_grid,
                interp=interp, phase_marginalization=phase_marginalization,
                time_quadrature=time_quadrature)

        self._batched = jax.jit(_batched)

        def _scalar(theta5):
            v = fused_log_likelihood_distmarg(
                data, theta5[0:1], theta5[1:2], theta5[2:3],
                theta5[3:4], theta5[4:5], self.x_grid, self.log_w_grid,
                interp=interp, phase_marginalization=phase_marginalization,
                time_quadrature=time_quadrature)
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
                 d_prior="euclidean", interp=JAX_INTERP_DEFAULT, guess_snr=None,
                 *, time_quadrature=TIME_QUAD_DEFAULT, d_prior_range=None):
        self.data = data
        self.interp = interp   # the instance's stencil; sample_phi_ref defaults to it
        _validate_nonlinear_time_quadrature(
            time_quadrature, "distance/phase marginalization")
        self.time_quadrature = time_quadrature
        self.nphi = int(nphi)
        self._phi_grid = phi_ref_grid(self.nphi)
        # Adaptive distance quadrature: concentrate grid resolution on the
        # distance posterior, whose peak/width are set from guess_snr + the
        # precompute's max rho_sq_unit -- avoiding the uniform grid's high-SNR
        # under-resolution (the ~1% evidence bias).  Static grid -> feeds the
        # same stable logsumexp kernel and is gradient-stable.  Enable with env
        # JAX_ILE_DISTGRID_ADAPTIVE=1; falls back to uniform otherwise.
        if int(os.environ.get("JAX_ILE_DISTGRID_ADAPTIVE", "0")) and guess_snr:
            # interp= must be forwarded: this sizes the distance grid the likelihood then
            # integrates on, so leaving it at the module default silently mixes stencils --
            # and would break the documented 'pass interp="linear" to reproduce a
            # pre-2026-08-26 run' recipe, which is the whole mitigation for that default move.
            d_peak, sigma_d = estimate_distance_peak(data, guess_snr, interp=interp)
            self.x_grid, self.log_w_grid = make_distance_grid_adaptive(
                d_min, d_max, d_peak, sigma_d, d_prior, distMpcRef=data.distMpcRef,
                d_prior_range=d_prior_range)
            self.dist_grid_info = dict(mode="adaptive", d_peak=float(d_peak),
                                       sigma_d=float(sigma_d),
                                       n=int(self.x_grid.shape[0]))
        else:
            self.x_grid, self.log_w_grid = make_distance_grid(
                d_min, d_max, n_grid, d_prior, distMpcRef=data.distMpcRef,
                d_prior_range=d_prior_range)
            self.dist_grid_info = dict(mode="uniform", n=int(self.x_grid.shape[0]))

        xg, lwg, pg = self.x_grid, self.log_w_grid, self._phi_grid

        def _batched(ra, dec, psi, incl):
            return fused_log_likelihood_distphimarg(
                data, ra, dec, psi, incl, xg, lwg, pg, interp=interp,
                time_quadrature=time_quadrature)

        self._batched = jax.jit(_batched)

        def _scalar(theta4):
            v = fused_log_likelihood_distphimarg(
                data, theta4[0:1], theta4[1:2], theta4[2:3], theta4[3:4],
                xg, lwg, pg, interp=interp, time_quadrature=time_quadrature)
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
                       n_samples=1, interp=None):
        """Draw φ_ref from its conditional posterior given the other params.

        Evaluates ``phi_ref_conditional_lnL`` on the grid, normalises, draws
        a grid index, then adds a uniform sub-bin jitter so the sample is not
        pinned to a grid point.

        Parameters
        ----------
        ra, dec, psi, incl, distMpc : float scalars or (S,) arrays
        rng : numpy.random.Generator (optional)
        n_samples : int  — draws per input sample
        interp : str or None — stencil to evaluate the conditional with.  None (the default)
            means **this instance's** stencil, not the module default.

        Notes
        -----
        This argument carried its own module-level default until 2026-08-26, which was harmless
        only while that string happened to equal the constructor's: an instance built with any
        other stencil drew its phases from a DIFFERENT likelihood than the one it reports lnL and
        evidence from, and nothing raised.  Moving the module default to 'sinc' made it bite the
        documented backward-compatibility recipe -- constructing with interp="linear" gave a
        linear evidence and sinc phase draws.  Pass interp= only to override deliberately.

        Returns
        -------
        phi_ref : (S,) float array (or (S, n_samples) when n_samples > 1)
        """
        interp = self.interp if interp is None else interp
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
            jnp.asarray(dist_), self._phi_grid, interp=interp,
            time_quadrature=self.time_quadrature))  # (nphi, S)

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


class JAXDistPhiPsiMargLikelihood:
    """Distance-, phi_ref- AND psi-marginalised lnL over 3 angles (ra, dec, incl).

    Integrates out luminosity distance, orbital phase phi_ref and polarization psi,
    leaving a smooth 3-D target.  Removing psi (spin-2, the dimension most entangled
    with distance/inclination) lowers the sampler dimension and stabilises the
    distance integral relative to the 4-D phi-marginalised likelihood.
    """

    ANGULAR_PARAM_ORDER = ("ra", "dec", "incl")

    def __init__(self, data, d_min, d_max, nphi=32, npsi=16, n_grid=256,
                 d_prior="euclidean", interp=JAX_INTERP_DEFAULT, guess_snr=None,
                 angle_marg=ANGLE_MARG_DEFAULT, *,
                 time_quadrature=TIME_QUAD_DEFAULT, d_prior_range=None,
                 dist_grid="uniform", dist_grid_tol=DIST_GRID_TOL_DEFAULT):
        self.data = data
        self.interp = interp   # the instance's stencil; sample_phi_ref defaults to it
        _validate_nonlinear_time_quadrature(
            time_quadrature, "distance/phase/polarization marginalization")
        self.time_quadrature = time_quadrature
        self.nphi = int(nphi)
        self.npsi = int(npsi)
        self._phi_grid = phi_ref_grid(self.nphi)
        self._psi_grid = psi_grid(self.npsi)
        # (phi_ref, psi) marginalization scheme.  "grid" is the historical
        # nphi x npsi quadrature, kept as the DEFAULT so existing command
        # lines reproduce existing runs; "exact" / "laplace" are the
        # exact-coefficient schemes of RIFT.likelihood.jax_ile.anglemarg
        # (which fix the grid path's SNR-unbounded quadrature error and its
        # nphi=8 Nyquist aliasing); "auto" selects between them.  Both the
        # selection and the dense-grid sizing key on a DATA-DERIVED amplitude
        # bound (estimate_angle_amplitude, computed below on the FULL prior
        # distance support) -- never on guess_snr: an absent or underestimated SNR
        # must not be able to silently under-resolve the quadrature
        # (external-review defect 2).  self.angle_marg_info records what
        # actually ran -- callers must surface it in the run log.
        if angle_marg not in ANGLE_MARG_CHOICES:
            raise ValueError("angle_marg must be one of grid/exact/laplace/"
                             "auto, got %r" % (angle_marg,))
        if dist_grid not in DIST_GRID_SCHEMES:
            # An unrecognised value must NEVER fall through to the default: a
            # typo that silently returns the old answer is precisely the
            # silently-inert-flag failure this module keeps being bitten by.
            raise ValueError("dist_grid must be one of %r, got %r"
                             % (DIST_GRID_SCHEMES, dist_grid))
        if dist_grid != "uniform" and _core._DISTMARG_GH_N > 0:
            # core._distmarg_gh_logL places its own per-sample nodes and reads
            # ONLY min(x_grid)/max(x_grid); the node positions and the whole
            # log_w_grid are unused.  Both schemes span the same support, so the
            # arms would be bit-identical while dist_grid_info still reported
            # mode='loguniform'.  That is the silently-inert-flag class the
            # other refusals here exist to prevent, and it is reachable
            # without the user naming a dense scheme: under GH
            # choose_angle_marg_scheme resolves to one regardless.  Which one
            # is irrelevant to this refusal -- the per-sample quadrature reads
            # only the support on every dense path -- so do not re-tie this
            # comment to a particular selector outcome.
            raise ValueError(
                "dist_grid=%r cannot be combined with JAX_ILE_DISTMARG_GH=%d: "
                "the per-sample Gauss-Hermite distance quadrature places its "
                "own nodes and uses only the SUPPORT of x_grid, so this option "
                "would be bit-identically inert while still being reported as "
                "active.  Unset JAX_ILE_DISTMARG_GH, or use "
                "dist_grid='uniform'." % (dist_grid, _core._DISTMARG_GH_N))
        if dist_grid != "uniform" and d_prior_range is not None and (
                float(d_prior_range[0]) != float(d_min)
                or float(d_prior_range[1]) != float(d_max)):
            raise ValueError(
                "dist_grid=%r cannot be combined with a narrowed distance range "
                "(--limit-distance): make_distance_grid_loguniform takes no "
                "d_prior_range, so it would normalize the distance prior onto the "
                "BOX -- the exact renormalization --limit-distance exists to "
                "prevent, and worth several nats of evidence silently.  The two "
                "features address the same problem from opposite sides (narrow the "
                "range vs resolve the peak anywhere in it); composing them needs a "
                "prior-range split in the log-uniform builder and its own "
                "validation.  Use one or the other." % (dist_grid,))
        from . import anglemarg as _anglemarg
        # THE FULL-SUPPORT distance grid.  Two distinct roles are deliberately
        # separated here (see DESIGN_jax_distance_quadrature.md, "decoupling"):
        #
        #   x_grid_full  sizes the ANGLE lattice.  It always spans the whole
        #                prior range [d_min, d_max], whatever grid the
        #                likelihood ends up integrating on.
        #   self.x_grid  is what the fused kernel integrates over.
        #
        # estimate_angle_amplitude reads only min/max of the grid it is given
        # (the per-angle distance maximum is closed form at
        # clip(A/B, x_min, x_max)), so a narrowed distance grid that still
        # contains A/B leaves the amplitude untouched -- but one that does NOT
        # contain it silently SHRINKS the angle lattice (measured: a
        # [0.8 d, 1.25 d] window drops the amplitude 12.6% and the lattice from
        # (624, 320) to (592, 304)).  Sizing from the full support costs one
        # build-time scalar and removes that coupling by construction instead
        # of bounding it.  For dist_grid="uniform" this IS self.x_grid, so the
        # default path is unchanged, node for node.
        # --limit-distance is EXACTLY the "narrowed distance grid" the note above
        # measures (12.6% amplitude, (624,320) -> (592,304) lattice).  d_min/d_max
        # are then the SAMPLED box, so the full-support grid is built over
        # d_prior_range instead -- which is what "the whole prior range" means once
        # the two roles are split.  With no box the two are equal and this is the
        # same call, node for node.
        _full_lo, _full_hi = ((d_min, d_max) if d_prior_range is None
                              else (d_prior_range[0], d_prior_range[1]))
        x_grid_full, log_w_full = make_distance_grid(
            _full_lo, _full_hi, n_grid, d_prior, distMpcRef=data.distMpcRef)
        if int(os.environ.get("JAX_ILE_DISTGRID_ADAPTIVE", "0")) and guess_snr:
            if dist_grid != "uniform":
                raise ValueError(
                    "JAX_ILE_DISTGRID_ADAPTIVE=1 and dist_grid=%r both ask to "
                    "replace the distance grid.  Unset the environment "
                    "variable (it is deprecated; see "
                    "DESIGN_jax_distance_quadrature.md)." % (dist_grid,))
            # DEPRECATED.  Kept reachable so nothing that sets this variable
            # today changes behaviour, but it is measurably unsafe: its window
            # is centred on estimate_distance_peak, a 300-step gradient ascent
            # that is NOT converged (measured rho 39.97 against the amplitude
            # bound's 55.07), and its trapezoid gives the last node a full
            # rather than half interval, misplacing ~3% of the volumetric prior
            # mass onto d_max.  Measured 9.4 nats of lnL error at SNR 40.
            print("WARNING: JAX_ILE_DISTGRID_ADAPTIVE is DEPRECATED and "
                  "measurably unsafe (9.4 nats at SNR 40 on the reference "
                  "configuration).  Use dist_grid='loguniform' "
                  "(--distance-grid-scheme loguniform); see "
                  "DESIGN_jax_distance_quadrature.md.")
            # interp= must be forwarded: this sizes the distance grid the likelihood then
            # integrates on, so leaving it at the module default silently mixes stencils --
            # and would break the documented 'pass interp="linear" to reproduce a
            # pre-2026-08-26 run' recipe, which is the whole mitigation for that default move.
            d_peak, sigma_d = estimate_distance_peak(data, guess_snr, interp=interp)
            self.x_grid, self.log_w_grid = make_distance_grid_adaptive(
                d_min, d_max, d_peak, sigma_d, d_prior, distMpcRef=data.distMpcRef,
                d_prior_range=d_prior_range)
            self.dist_grid_info = dict(mode="adaptive", d_peak=float(d_peak),
                                       sigma_d=float(sigma_d),
                                       n=int(self.x_grid.shape[0]))
        else:
            if d_prior_range is None:
                # Default path, unchanged node for node: the integration grid IS
                # the full-support grid.
                self.x_grid, self.log_w_grid = x_grid_full, log_w_full
            else:
                # --limit-distance: integrate on the box, normalize on the prior.
                self.x_grid, self.log_w_grid = make_distance_grid(
                    d_min, d_max, n_grid, d_prior, distMpcRef=data.distMpcRef,
                    d_prior_range=d_prior_range)
            self.dist_grid_info = dict(mode="uniform", n=int(self.x_grid.shape[0]))

        if angle_marg == "grid":
            if dist_grid != "uniform":
                # Fail closed.  The log-uniform grid is sized from the
                # data-derived angle amplitude, which the grid scheme neither
                # computes nor rechecks at runtime; applying it there would be
                # an unvalidated path, and silently ignoring the request would
                # be a silent no-op.
                raise ValueError(
                    "dist_grid=%r requires angle_marg in "
                    "('exact', 'laplace', 'auto'): the log-uniform grid is "
                    "sized from the data-derived angle amplitude, which the "
                    "'grid' scheme does not compute." % (dist_grid,))
            scheme, sel_info = "grid", dict(reason="default grid quadrature")
            amp_sizing = None
        else:
            # Eager, build-time (grid sizes must be static under jit): bound
            # the exponent amplitude from the coefficient tables themselves,
            # over a sky sample and the FULL prior distance support.
            amp_data, amp_diag = _anglemarg.estimate_angle_amplitude(
                data, x_grid_full, interp=interp, return_diagnostics=True)
            # sizing is FLOORED at the crossover (never below the calibration
            # point); the SELECTION below uses the UNfloored bound, so quiet
            # targets stay on the exact branch.  Computed here, before the
            # distance grid, because the distance grid is sized from this same
            # floored number -- see the rho_max note directly below.
            amp_sizing = max(amp_data,
                             _anglemarg.ANGLE_MARG_CROSSOVER_AMPLITUDE)
            if dist_grid == "loguniform":
                # REFUSE the truncated regime.  The spacing contract assumes the
                # integrand is a Gaussian PEAK inside the support, whose relative
                # width 1/rho is what c(tol)/rho_max resolves.  When the
                # maximizing distance x* = A/B lies OUTSIDE [x_min, x_max] the
                # integrand is monotone on the support instead -- a boundary
                # layer at one prior edge -- and a log-uniform grid is the wrong
                # instrument for it twice over: its ABSOLUTE spacing is coarsest
                # exactly at d_max where the layer sits, and refining it adds
                # nodes proportionally everywhere so the layer never resolves
                # (measured: tol 0.5 -> 1e-9 moves the error only 5.23 -> 3.92
                # nats, while uniform 256 -> 4096 moves 2.52 -> 0.36).  Worse,
                # the clip makes the amplitude UNDER-read, so the derived node
                # count moves the wrong way -- in the extreme it reads 0, the
                # crossover floor pins rho_max = 30, and the grid collapses to
                # 145 nodes.  We refuse rather than fall back to uniform: a
                # fallback would make this flag silently produce the other
                # scheme's grid, and this regime is a physics signal (the
                # posterior rails against a prior edge) that the caller should
                # see rather than have papered over.  Neither grid is good here
                # -- uniform 256 is itself 2.5 nats out.
                if amp_diag["clip_excess"] > 1.0 + 1e-3:
                    raise ValueError(
                        "dist_grid='loguniform' refuses this event: the "
                        "likelihood's maximizing distance lies OUTSIDE "
                        "[d_min, d_max] = [%g, %g] Mpc, so the distance "
                        "integrand is a boundary layer at a prior edge rather "
                        "than an interior peak, and the log-uniform spacing "
                        "contract does not apply (measured 1.9-4.6 nats of "
                        "error there, worse than the uniform default).  "
                        "Diagnostic: unclipped amplitude %.6g against clipped "
                        "%.6g (excess %.4g).  Recourse: widen --d-max (or "
                        "narrow --d-min) so the posterior is interior, or stay "
                        "on --distance-grid-scheme uniform and raise "
                        "--distance-grid-points.  See "
                        "DESIGN_jax_distance_quadrature.md section 1a."
                        % (float(d_min), float(d_max),
                           amp_diag["amp_unclipped"], amp_diag["amp_clipped"],
                           amp_diag["clip_excess"]))
                # rho_max = sqrt(2 A): A is the max over angles of the
                # closed-form distance maximum A_ang^2/(2 B_ang) = rho^2/2.  NOT
                # an identity and NOT a proven bound -- A carries
                # ANGLE_AMP_MARGIN and the max is over a SAMPLED sky, so this is
                # sqrt(margin) * rho_sampled_max.  No NEW estimator is
                # introduced and no peak is located, which is the point.
                #
                # A is amp_SIZING, not amp_data, and that choice is what makes
                # the runtime fail-safe cover this grid.  _runtime_amp_failsafe
                # compares the per-call amplitude against amp_sizing; sizing the
                # distance spacing from the unfloored amp_data instead would
                # leave a silent gap for quiet targets (amp_data < crossover),
                # where a runtime amplitude between amp_data and amp_sizing
                # under-resolves the distance peak WITHOUT tripping anything.
                # Flooring costs a minimum of ~144 nodes on a quiet event, whose
                # run is cheap anyway.
                #
                # ...and the sizing amplitude is the one the guard ADMITS, not
                # the one it was built from.  _runtime_amp_failsafe stays silent
                # until amp_call > AMP_FAILSAFE_TRIP_FACTOR * amp_sizing, so a
                # call just under that threshold carries an interior peak at
                # rho = sqrt(TRIP) * sqrt(2*amp_sizing) -- sqrt(2) above the
                # spacing's design point at the shipped factor -- and is neither
                # printed nor recorded.  Through the Gaussian alias law
                # (2 exp(-2 pi^2/c^2), c fixed by tol) that turns the advertised
                # tol into ~sqrt(2*tol): 0.01 becomes 0.14, unlabelled.  Sizing
                # from TRIP*amp_sizing closes the window by construction and
                # costs sqrt(TRIP) = 1.41x the nodes; the alternative -- a
                # second, distance-specific runtime guard at a tighter threshold
                # -- adds a mechanism where a constant will do, and the two
                # would then have to be kept consistent by hand.
                rho_max = float(np.sqrt(
                    2.0 * _anglemarg.AMP_FAILSAFE_TRIP_FACTOR
                    * max(float(amp_sizing), 0.0)))
                # SECOND precondition (the first is clip_excess, above): a peak
                # that is interior but sits ~1 width inside an edge breaks the
                # spacing law while clip_excess reads exactly 1.  The alias law
                # is Poisson summation on an UNTRUNCATED Gaussian; a truncated
                # support adds an Euler-Maclaurin endpoint term proportional to
                # the integrand's derivative there, worth ~11% at the shipped
                # tol (c = 1.93) against a promised 1%.  Refuse rather than
                # widen silently: the option's whole claim is the stated
                # fractional error, and both alternatives -- shipping the error
                # or moving the user's distance prior for them -- are worse than
                # saying so.  Evaluated at the CONTRACT spacing c/rho_max, which
                # is the coarsest the built grid can be (the node count ceils).
                dlnd_contract = (_core.loguniform_spacing_for_tolerance(
                    dist_grid_tol) / rho_max)
                eps_end = float(_core.ENDPOINT_ERROR_MARGIN
                                * _core.loguniform_endpoint_error(
                                    dlnd_contract,
                                    amp_diag["endpoint_scale"]))
                # ...and a NON-POSITIVE clearance is not a small correction.
                # _endpoint_bell clamps k <= 0 to zero on the argument that an
                # endpoint sitting ON the peak is a stationary point with
                # nothing to correct -- true, and MEASURED true at exactly
                # k = 0 (1.2e-9).  But the clamp cannot tell k = 0 from k < 0,
                # where the peak has left the support entirely and the error
                # climbs steeply, so the term scores zero exactly where it is
                # worst.  Between that and the clip_excess trip there is a
                # window neither guard sees: measured on the reference synthetic
                # at rho_pk 51.4, clearance -0.66 builds with a TRUE error of
                # 0.0668 -- 6.7x the promised tol -- while clip_excess is still
                # ~1.  Refuse it here.  This also refuses the benign k = 0
                # point, which costs nothing: the error is 2e-2 by half a width
                # either side of it.  (External review of the endpoint guard.)
                # A loud EXTERIOR entry that no other diagnostic can see is a
                # REAL structural hole and is deliberately NOT refused here.
                # External review (P2) is right that clip_excess is a ratio of
                # GLOBAL maxima, that peak_clearance describes the global
                # argmax, and that the endpoint term cannot represent an
                # exterior entry (its bell is clamped at k <= 0, and
                # Euler-Maclaurin is the wrong expansion for a peak outside the
                # support).  So an interior dominant entry with a near-equal
                # exterior secondary reads clean on all three.
                #
                # It is not refused because no threshold on the available
                # quantity survives contact with the fixtures.  The obvious
                # one -- refuse when the exterior entry's weight exp(-gap)
                # exceeds tol, i.e. gap < ln(1/tol) = 4.6 nats -- refuses every
                # QUIET event: measured, _synth() has amp_clipped 3.30 nats in
                # TOTAL and an exterior gap of 3.29, and a quieter one 0.0083
                # and 0.0082.  Their whole exponent range is smaller than the
                # threshold, and nothing is wrong with them: at low amplitude
                # the distance integrand is smooth and the grid over-resolves
                # it, so the exterior entry carries weight but not error.  The
                # honest condition is weight TIMES that entry's own resolution
                # error, which needs a model of the latter that cannot be
                # validated against any configuration reachable here.
                #
                # What IS established: on every loud fixture and every prior
                # this code is run with, the loudest exterior entry sits 32-5273
                # nats below the maximum, against a 15-nat contribution band --
                # so the hole is not reachable there.  That premise is pinned by
                # test_exterior_entries_stay_far_below_the_dominant_one, which
                # fails if it ever stops holding, and exterior_gap is reported
                # so the condition is visible rather than silent.
                if amp_diag["peak_clearance"] <= 0.0:
                    raise ValueError(
                        "dist_grid='loguniform' refuses this event: the "
                        "dominant peak sits AT or OUTSIDE a prior edge of "
                        "[d_min, d_max] = [%g, %g] Mpc (clearance %.4g peak "
                        "widths, rho = %.4g), so the spacing contract's "
                        "precondition -- a Gaussian peak INSIDE the support -- "
                        "does not hold.  This is the same physics signal as "
                        "the exterior refusal, caught earlier: the truncated "
                        "endpoint term is what dominates here, and it is not "
                        "bounded by the alias law the node count is derived "
                        "from.  Recourse: widen the distance prior so the peak "
                        "is interior with clearance >= %.2f widths, or stay on "
                        "--distance-grid-scheme uniform.  See "
                        "DESIGN_jax_distance_quadrature.md section 1a."
                        % (float(d_min), float(d_max),
                           amp_diag["peak_clearance"], amp_diag["peak_rho"],
                           _core.loguniform_min_clearance(dist_grid_tol)))
                if eps_end > float(dist_grid_tol):
                    raise ValueError(
                        "dist_grid='loguniform' refuses this event: the "
                        "likelihood's maximizing distance is INTERIOR to "
                        "[d_min, d_max] = [%g, %g] Mpc but too close to an "
                        "edge for the spacing contract, which assumes an "
                        "effectively untruncated Gaussian peak.  The dominant "
                        "peak sits %.3g peak widths (1/rho, rho = %.4g) from "
                        "the nearer edge; at this grid's spacing dlnd = %.4g "
                        "the truncated-endpoint term alone is ~%.3g of the "
                        "distance integral, against the requested tol=%g.  A "
                        "peak at the sizing SNR needs >= %.2f widths.  "
                        "Recourse: widen the distance prior so the peak has "
                        "clearance (this is the same physics signal as the "
                        "exterior refusal -- the posterior is close to a prior "
                        "edge), or stay on --distance-grid-scheme uniform and "
                        "raise --distance-grid-points.  Note that TIGHTENING "
                        "--distance-grid-tol does not help: the endpoint term "
                        "falls as c(tol)^2 while the budget falls faster.  See "
                        "DESIGN_jax_distance_quadrature.md section 1a."
                        % (float(d_min), float(d_max),
                           amp_diag["peak_clearance"], amp_diag["peak_rho"],
                           dlnd_contract, eps_end, float(dist_grid_tol),
                           _core.loguniform_min_clearance(dist_grid_tol)))
                self.x_grid, self.log_w_grid = make_distance_grid_loguniform(
                    d_min, d_max, rho_max, d_prior,
                    distMpcRef=data.distMpcRef, tol=dist_grid_tol)
                self.dist_grid_info = dict(
                    mode="loguniform", n=int(self.x_grid.shape[0]),
                    tol=float(dist_grid_tol), rho_max=rho_max,
                    dlnd=float(np.log(float(d_max) / float(d_min))
                               / (int(self.x_grid.shape[0]) - 1)),
                    n_uniform_requested=int(n_grid))
            # The A0==0/B1==0 identity that the GH psi-marginal node placement
            # is DERIVED from is measured ONCE here, on concrete tables, and
            # gates EVERY route to that placement -- not just 'auto'.  An
            # earlier revision checked it only in the 'auto' branch, so an
            # explicit --angle-marg-scheme laplace walked past it and an
            # m_max == 2 dataset whose identity fails was evaluated with a
            # placement whose premise was absent (external review).  It cannot
            # be checked inside the kernel: that runs under jit/grad, where the
            # coefficient tables are tracers.
            gh_ok, gh_info = None, {}
            if _core._DISTMARG_GH_N > 0 and angle_marg in ("auto", "laplace"):
                gh_ok, gh_info = _anglemarg.gh_laplace_supported(
                    *_anglemarg.angle_coefficient_tables(
                        data,
                        jnp.asarray(_ANGLE_MARG_PROBE_RA),
                        jnp.asarray(_ANGLE_MARG_PROBE_DEC),
                        jnp.asarray(_ANGLE_MARG_PROBE_INCL),
                        interp)[:2],
                    _anglemarg._data_m_max(data),
                    feature=getattr(data, "feature", None))
            if angle_marg == "auto":
                scheme, sel_info = _anglemarg.choose_angle_marg_scheme(
                    amp_data, gh_laplace_ok=gh_ok)
                sel_info.update(gh_info)
            else:
                if angle_marg == "laplace" and gh_ok is False:
                    raise ValueError(
                        "--angle-marg-scheme laplace was requested with "
                        "JAX_ILE_DISTMARG_GH set, but its psi-marginal "
                        "distance-node placement is not valid for this data: "
                        "%s.  The placement is DERIVED from A0 == 0 and "
                        "B1 == 0 (that is what reduces stationarity to "
                        "z^2 w = conj(w)), so it must not be used where they "
                        "do not hold.  Use --angle-marg-scheme exact, or unset "
                        "JAX_ILE_DISTMARG_GH."
                        % gh_info.get("gh_laplace_reason", "identity absent"))
                scheme, sel_info = angle_marg, dict(
                    reason="forced by caller", amplitude=amp_data,
                    crossover=_anglemarg.ANGLE_MARG_CROSSOVER_AMPLITUDE)
                sel_info.update(gh_info)
        self.angle_marg_scheme = scheme
        self.angle_marg_info = dict(sel_info, requested=angle_marg,
                                    scheme=scheme)
        # Bound AFTER the distance grid is final: dist_grid="loguniform"
        # replaces it inside the block above.
        xg, lwg, pg, sg = (self.x_grid, self.log_w_grid,
                           self._phi_grid, self._psi_grid)
        if scheme in ("exact", "laplace"):
            self.angle_marg_info["amp_sizing"] = amp_sizing
            self.angle_marg_info["sample_grid"] = tuple(
                _anglemarg.angle_sample_grid_sizes(
                    _anglemarg._data_m_max(data)))

        if scheme == "grid":
            def _fused(data_, ra, dec, incl, return_lnLt=False):
                return fused_log_likelihood_distphipsimarg(
                    data_, ra, dec, incl, xg, lwg, pg, sg, interp=interp,
                    time_quadrature=time_quadrature, return_lnLt=return_lnLt)
        elif scheme == "exact":
            def _fused(data_, ra, dec, incl, return_lnLt=False):
                return _anglemarg.fused_log_likelihood_distphipsimarg_exact(
                    data_, ra, dec, incl, xg, lwg, interp=interp,
                    amp_sizing=amp_sizing, time_quadrature=time_quadrature,
                    return_lnLt=return_lnLt)
        else:   # laplace
            def _fused(data_, ra, dec, incl, return_lnLt=False):
                return _anglemarg.fused_log_likelihood_distphipsimarg_laplace(
                    data_, ra, dec, incl, xg, lwg, interp=interp,
                    amp_sizing=amp_sizing, time_quadrature=time_quadrature,
                    return_lnLt=return_lnLt)

        def _batched(ra, dec, incl):
            return _fused(data, ra, dec, incl)
        self._batched = jax.jit(_batched)

        def _scalar(theta3):
            v = _fused(data, theta3[0:1], theta3[1:2], theta3[2:3])
            return v[0]
        self._scalar = _scalar
        self._value_and_grad = jax.jit(jax.value_and_grad(_scalar))
        self._hessian = jax.jit(jax.hessian(_scalar))

    def log_likelihood(self, ra, dec, incl):
        """lnL for arrays of 3 angular parameters (ra, dec, incl), shape (S,)."""
        return self._batched(jnp.asarray(ra), jnp.asarray(dec), jnp.asarray(incl))

    def value(self, theta3):
        return float(self._scalar(jnp.asarray(theta3, dtype=jnp.float64)))

    def value_and_grad(self, theta3):
        v, g = self._value_and_grad(jnp.asarray(theta3, dtype=jnp.float64))
        return float(v), np.asarray(g)

    def fisher(self, theta3):
        """Observed Fisher matrix ``-Hessian(lnL)`` at ``theta3`` (3×3)."""
        H = np.asarray(self._hessian(jnp.asarray(theta3, dtype=jnp.float64)))
        return -H


class JAXDistPsiMargLikelihood:
    """Distance- AND psi-marginalised lnL over 4 angles (ra, dec, phi_ref, incl).

    Integrates out luminosity distance and polarization psi (a short spin-2 grid
    scan, ~8 points) while KEEPING phi_ref as a sampled parameter, leaving a
    smooth 4-D target ``theta4 = (ra, dec, phiref, incl)``.

    This is the "d+psi" variant of the 4-D phi-marginalised likelihood
    (:class:`JAXDistPhiMargLikelihood`): psi marginalization still breaks the
    curved phi_ref-psi degeneracy ridge that collapses flowMC at high SNR, but
    the psi scan (~8 steps) is much cheaper than the phi_ref scan (~32 steps),
    so the kernel is faster.
    """

    ANGULAR_PARAM_ORDER = ("ra", "dec", "phiref", "incl")

    def __init__(self, data, d_min, d_max, npsi=8, n_grid=256,
                 d_prior="euclidean", interp=JAX_INTERP_DEFAULT, guess_snr=None,
                 *, time_quadrature=TIME_QUAD_DEFAULT, d_prior_range=None):
        self.data = data
        self.interp = interp   # the instance's stencil; sample_phi_ref defaults to it
        _validate_nonlinear_time_quadrature(
            time_quadrature, "distance/polarization marginalization")
        self.time_quadrature = time_quadrature
        self.npsi = int(npsi)
        self._psi_grid = psi_grid(self.npsi)
        if int(os.environ.get("JAX_ILE_DISTGRID_ADAPTIVE", "0")) and guess_snr:
            # interp= must be forwarded: this sizes the distance grid the likelihood then
            # integrates on, so leaving it at the module default silently mixes stencils --
            # and would break the documented 'pass interp="linear" to reproduce a
            # pre-2026-08-26 run' recipe, which is the whole mitigation for that default move.
            d_peak, sigma_d = estimate_distance_peak(data, guess_snr, interp=interp)
            self.x_grid, self.log_w_grid = make_distance_grid_adaptive(
                d_min, d_max, d_peak, sigma_d, d_prior, distMpcRef=data.distMpcRef,
                d_prior_range=d_prior_range)
            self.dist_grid_info = dict(mode="adaptive", d_peak=float(d_peak),
                                       sigma_d=float(sigma_d),
                                       n=int(self.x_grid.shape[0]))
        else:
            self.x_grid, self.log_w_grid = make_distance_grid(
                d_min, d_max, n_grid, d_prior, distMpcRef=data.distMpcRef,
                d_prior_range=d_prior_range)
            self.dist_grid_info = dict(mode="uniform", n=int(self.x_grid.shape[0]))

        xg, lwg, sg = self.x_grid, self.log_w_grid, self._psi_grid

        def _batched(ra, dec, phiref, incl):
            return fused_log_likelihood_distpsimarg(
                data, ra, dec, phiref, incl, xg, lwg, sg, interp=interp,
                time_quadrature=time_quadrature)
        self._batched = jax.jit(_batched)

        def _scalar(theta4):
            v = fused_log_likelihood_distpsimarg(
                data, theta4[0:1], theta4[1:2], theta4[2:3], theta4[3:4],
                xg, lwg, sg, interp=interp, time_quadrature=time_quadrature)
            return v[0]
        self._scalar = _scalar
        self._value_and_grad = jax.jit(jax.value_and_grad(_scalar))
        self._hessian = jax.jit(jax.hessian(_scalar))

    def log_likelihood(self, ra, dec, phiref, incl):
        """lnL for arrays of 4 angular params (ra, dec, phiref, incl), shape (S,)."""
        return self._batched(
            jnp.asarray(ra), jnp.asarray(dec),
            jnp.asarray(phiref), jnp.asarray(incl))

    def value(self, theta4):
        return float(self._scalar(jnp.asarray(theta4, dtype=jnp.float64)))

    def value_and_grad(self, theta4):
        v, g = self._value_and_grad(jnp.asarray(theta4, dtype=jnp.float64))
        return float(v), np.asarray(g)

    def fisher(self, theta4):
        """Observed Fisher matrix ``-Hessian(lnL)`` at ``theta4`` (4×4)."""
        H = np.asarray(self._hessian(jnp.asarray(theta4, dtype=jnp.float64)))
        return -H
