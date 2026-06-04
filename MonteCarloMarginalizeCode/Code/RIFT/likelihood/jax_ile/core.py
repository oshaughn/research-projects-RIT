"""
The fused, AD-compatible factored log-likelihood in JAX.

This re-expresses the production
``factored_likelihood.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop``
(``n_cal == 1`` path) as a pure ``jax.numpy`` function of the extrinsic
parameters.  The likelihood model is the standard RIFT factored form

    lnL_t(theta) = Re[kappa(theta, t)] - 1/2 rho^2(theta, t)

    kappa(theta, t)  = (d_ref/d) * sum_det sum_lm conj(F_det Y_lm) rho_lm^det(t + tau_det)
    rho^2(theta, t)  = (d_ref/d)^2 * sum_det 0.5 [ |F|^2 Re(Y* U Y) + Re(F^2 Y V Y) ]

    lnL(theta) = log integral_t exp(lnL_t) dt   (Simpson rule, time marginalized)

The per-detector inputs ``rho_lm(t)`` (the rholm timeseries), the cross-term
matrices ``U`` (``<h_lm|h_l'm'>``) and ``V`` (``<h_lm*|h_l'm'>``), the (l,m)
list and the timeseries epoch are produced unchanged by
``PrecomputeLikelihoodTerms`` / ``PackLikelihoodDataStructuresAsArrays`` and are
passed in here as plain arrays.

Two time-interpolation modes are provided:

* ``interp="nearest"`` -- reproduces the production discrete-shift behaviour
  (round the per-detector arrival to the nearest sample) bit-for-bit, used to
  *validate* the JAX path against the numpy reference.
* ``interp="linear"`` (default) -- evaluates the rholm timeseries at the
  *continuous* arrival time, so the likelihood is differentiable with respect
  to sky location (through the geometric time delay) and the other extrinsic
  parameters.  This is the AD-friendly path used for gradient-based exploration.

Time marginalization uses a precomputed Simpson quadrature weight vector
(``scipy.integrate.simpson`` applied to the identity, exactly as the production
fused kernel does), so the time integral matches the reference numerically while
remaining a constant-weight (hence trivially differentiable) sum.

Conventions / "epoch" handling
-------------------------------
``rho_lm^det`` is a discrete timeseries whose sample ``k`` corresponds to GPS
time ``epoch_det + k * deltaT``.  The window time-bin ``t`` (with
``tvals = linspace(-t_window, +t_window, npts)`` about the fiducial geocenter
epoch) maps to the *fractional* sample position

    pos_det(theta, t) = ( (tref - epoch_det) + tau_det(RA,DEC) + tvals[0] ) / deltaT + t

matching the reference ``ifirst`` definition (which is ``round(pos)`` at
``t=0``).  Keeping ``pos`` continuous is exactly what makes the sky-location
dependence differentiable.
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy import integrate as _scipy_integrate

import lal
import lalsimulation as lalsim

from .detector import compute_detamresponse, time_delay_from_earth_center
from .spherical import spherical_harmonics_vectorized

# Fiducial template distance (Mpc); identical to factored_likelihood.distMpcRef.
DIST_MPC_REF = 1000.0

try:
    _simpson = _scipy_integrate.simpson
except AttributeError:  # older scipy
    _simpson = _scipy_integrate.simps


def _simpson_weights(npts, deltaT):
    """Simpson quadrature weights w_t such that sum_t w_t f_t == simps(f, dx=deltaT).

    Obtained (as in the production fused kernel) by applying the linear
    ``simps`` operator to the identity matrix, guaranteeing the JAX time
    integral matches ``factored_likelihood``'s ``my_simps`` exactly.
    """
    return _simpson(np.eye(npts), dx=deltaT, axis=-1)


class JAXLikelihoodData:
    """Immutable container of the precomputed, device-resident likelihood data.

    All heavy arrays are stored once as ``jnp`` arrays; the extrinsic ->
    lnL map (``fused_log_likelihood``) closes over an instance of this class.
    """

    def __init__(self, detectors, deltaT, gmst, tvals, tref,
                 distMpcRef=DIST_MPC_REF):
        self.detector_names = list(detectors.keys())
        self.detectors = detectors  # name -> dict (see build_likelihood_data)
        self.deltaT = float(deltaT)
        self.gmst = float(gmst)
        self._tref = float(tref)
        self.tvals = jnp.asarray(tvals, dtype=jnp.float64)
        self.npts = int(len(tvals))
        self.distMpcRef = float(distMpcRef)
        # Simpson weights (host-computed constant), pushed to device.
        self.w_t = jnp.asarray(_simpson_weights(self.npts, self.deltaT),
                               dtype=jnp.float64)
        # tvals[0] enters pos; cache as a python float.
        self.tval0 = float(tvals[0])

    @property
    def lms(self):
        """(l,m) list of the first detector (all detectors share modes)."""
        return self.detectors[self.detector_names[0]]["lms"]


def build_likelihood_data(packed_per_detector, deltaT, tref, tvals,
                          distMpcRef=DIST_MPC_REF):
    """Assemble a :class:`JAXLikelihoodData` from packed numpy arrays.

    Parameters
    ----------
    packed_per_detector : dict
        ``det -> dict`` with keys:
          * ``lms``       : (K,2) int array of (l,m) (``lookupNumberToKeys``)
          * ``rholmArray``: (K, npts_full) complex (``rholmArray``)
          * ``U``, ``V``  : (K,K) complex cross-term matrices
          * ``epoch``     : float GPS epoch of the rholm timeseries
        i.e. exactly the outputs of
        ``PackLikelihoodDataStructuresAsArrays`` for each detector.
    deltaT : float
        Sample spacing (``P.deltaT``).
    tref : float
        Fiducial geocenter epoch (used only to fix GMST and the per-detector
        ``tref - epoch`` offset; time itself is marginalized).
    tvals : array_like, shape (npts,)
        Time-window grid, ``linspace(-t_window, t_window, npts)``.
    """
    gmst = float(lal.GreenwichMeanSiderealTime(tref))
    detectors = {}
    for det, d in packed_per_detector.items():
        lms = [(int(l), int(m)) for (l, m) in np.asarray(d["lms"])]
        rho = np.asarray(d["rholmArray"], dtype=np.complex128)  # (K, npts_full)
        Q = jnp.asarray(np.ascontiguousarray(rho.T))            # (npts_full, K)
        D = lalsim.DetectorPrefixToLALDetector(det)
        detectors[det] = {
            "lms": lms,
            "Q": Q,
            "U": jnp.asarray(np.asarray(d["U"], dtype=np.complex128)),
            "V": jnp.asarray(np.asarray(d["V"], dtype=np.complex128)),
            "epoch": float(d["epoch"]),
            "location": jnp.asarray(np.asarray(D.location, dtype=np.float64)),
            "response": jnp.asarray(np.asarray(D.response, dtype=np.float64)),
            "npts_full": int(Q.shape[0]),
            "l_max": max(l for (l, m) in lms),
        }
    return JAXLikelihoodData(detectors, deltaT, gmst, tvals, tref, distMpcRef)


def _gather_nearest(Q_col, pos):
    """Q_col[(round(pos))] with the reference's (rint(.)+0.5)->int32 rounding.

    ``pos`` has shape (S, npts); ``Q_col`` shape (npts_full,).  Positions that
    fall outside the rholm buffer contribute ZERO (the rholm timeseries is zero
    beyond its computed support).  This matches the production "over-running
    window zeros" semantics and, crucially, prevents a sliding window that runs
    off the buffer edge from producing a *spurious* likelihood peak (which a
    flat edge-clamp would).  For in-bounds windows this reduces exactly to the
    reference's discrete slice.
    """
    n = Q_col.shape[0]
    idx = (jnp.rint(pos) + 0.5).astype(jnp.int32)
    valid = (idx >= 0) & (idx < n)
    idx = jnp.clip(idx, 0, n - 1)
    return jnp.where(valid, Q_col[idx], 0.0 + 0.0j)


def _gather_linear(Q_col, pos):
    """Linear interpolation of Q_col at continuous positions ``pos``.

    Differentiable with respect to ``pos`` (the sub-sample arrival time).
    Positions outside the buffer contribute ZERO (see :func:`_gather_nearest`);
    the validity mask is applied on the continuous ``pos`` so the likelihood
    falls off smoothly to zero as the window leaves the buffer instead of
    latching onto the (flat-extrapolated) edge sample.
    """
    n = Q_col.shape[0]
    i0f = jnp.floor(pos)
    frac = pos - i0f
    i0 = jnp.clip(i0f.astype(jnp.int32), 0, n - 1)
    i1 = jnp.clip(i0 + 1, 0, n - 1)
    val = Q_col[i0] * (1.0 - frac) + Q_col[i1] * frac
    valid = (pos >= 0.0) & (pos <= n - 1.0)
    return jnp.where(valid, val, 0.0 + 0.0j)


_GATHERERS = {"nearest": _gather_nearest, "linear": _gather_linear}


def _accumulate_unit(data, ra, dec, psi, incl, phiref, interp,
                     phase_marginalization):
    """Network kappa and rho^2 at the *fiducial* distance (invDist == 1).

    Returns ``(kappa_unit, rho_sq_unit)`` each shape (S, npts).  ``kappa_unit``
    is complex; the distance factor and the Re/abs reduction are applied by the
    caller (so the same accumulation feeds both the fixed-distance and the
    distance-marginalized paths).
    """
    ra = jnp.asarray(ra, dtype=jnp.float64)
    dec = jnp.asarray(dec, dtype=jnp.float64)
    psi = jnp.asarray(psi, dtype=jnp.float64)
    incl = jnp.asarray(incl, dtype=jnp.float64)
    phiref = jnp.asarray(phiref, dtype=jnp.float64)

    gather = _GATHERERS[interp]
    gmst = data.gmst
    inv_deltaT = 1.0 / data.deltaT
    S = ra.shape[0]
    npts = data.npts
    t_offsets = jnp.arange(npts, dtype=jnp.float64)

    kappa_unit = jnp.zeros((S, npts), dtype=jnp.complex128)
    rho_sq_unit = jnp.zeros((S, npts), dtype=jnp.float64)

    for det in data.detector_names:
        dd = data.detectors[det]
        lms = dd["lms"]
        Q = dd["Q"]
        U = dd["U"]
        V = dd["V"]
        K = len(lms)

        F = compute_detamresponse(dd["response"], ra, dec, psi, gmst)
        Y = spherical_harmonics_vectorized(lms, incl, -phiref, l_max=dd["l_max"])

        if phase_marginalization:
            if [tuple(x) for x in lms] != [(2, 2), (2, -2)]:
                raise NotImplementedError(
                    "phase marginalization currently requires modes "
                    "[(2,2),(2,-2)]; got %r" % (lms,))
            Y = Y.at[:, 1].set(jnp.conj(Y[:, 1]))
            F_lm = jnp.stack([F, jnp.conj(F)], axis=-1)
            Q = jnp.concatenate([Q[:, 0:1], jnp.conj(Q[:, 1:2])], axis=1)
        else:
            F_lm = F[:, None]

        YUY = jnp.einsum("si,sj,ij->s", jnp.conj(Y), Y, U)
        YVY = jnp.einsum("si,sj,ij->s", Y, Y, V)
        rho_sq_det = 0.5 * ((F * jnp.conj(F)).real * YUY.real
                            + (F * F * YVY).real)             # at invDist == 1

        FY_conj = jnp.conj(F_lm * Y)
        t_det = (data.tref_minus_epoch(det)
                 + time_delay_from_earth_center(dd["location"], ra, dec, gmst))
        p0 = (t_det + data.tval0) * inv_deltaT
        pos = p0[:, None] + t_offsets[None, :]

        kappa_det = jnp.zeros((S, npts), dtype=jnp.complex128)
        for k in range(K):
            Qi = gather(Q[:, k], pos)
            kappa_det = kappa_det + FY_conj[:, k][:, None] * Qi
        kappa_unit = kappa_unit + kappa_det
        rho_sq_unit = rho_sq_unit + rho_sq_det[:, None]

    return kappa_unit, rho_sq_unit


def _time_marginalize(lnL_t, w_t):
    """log integral_t exp(lnL_t) dt via constant Simpson weights, log-sum-exp stable."""
    m = jnp.max(lnL_t, axis=-1, keepdims=True)
    L = jnp.sum(w_t[None, :] * jnp.exp(lnL_t - m), axis=-1)
    return m[:, 0] + jnp.log(L)


def fused_log_likelihood(data, ra, dec, psi, incl, phiref, distMpc,
                         interp="linear", phase_marginalization=False):
    """Time-marginalized factored log-likelihood at a fixed distance, lnL(theta).

    Parameters
    ----------
    data : JAXLikelihoodData
    ra, dec, psi, incl, phiref, distMpc : array_like, shape (S,)
        Extrinsic parameters.  ``distMpc`` is luminosity distance in Mpc.
    interp : {"linear", "nearest"}
        Time-interpolation of the rholm timeseries (see module docstring).
    phase_marginalization : bool
        Marginalize the coalescence phase via ``|kappa|``.

    Returns
    -------
    lnL : array_like, shape (S,)
    """
    distMpc = jnp.asarray(distMpc, dtype=jnp.float64)
    invDist = data.distMpcRef / distMpc
    kappa_unit, rho_sq_unit = _accumulate_unit(
        data, ra, dec, psi, incl, phiref, interp, phase_marginalization)
    kappa_sq = kappa_unit * invDist[:, None]
    rho_sq = rho_sq_unit * jnp.square(invDist)[:, None]
    if phase_marginalization:
        lnL_t = jnp.abs(kappa_sq) - 0.5 * rho_sq
    else:
        lnL_t = kappa_sq.real - 0.5 * rho_sq
    return _time_marginalize(lnL_t, data.w_t)


def fused_log_likelihood_distmarg(data, ra, dec, psi, incl, phiref,
                                  x_grid, log_w_grid,
                                  interp="linear", phase_marginalization=False):
    """Distance- AND time-marginalized factored log-likelihood, lnL(angles).

    Marginalizes the luminosity distance analytically (numerical quadrature over
    a grid) BEFORE the time integral, exactly mirroring the production
    ``distmarg_loglikelihood`` ordering -- this regulates the well-known
    amplitude/distance degeneracy of the bare factored likelihood (lnL diverges
    on slivers where the template power rho^2 -> 0).  The result is a smooth
    function of the five angular/sky parameters only, suitable for
    gradient-based exploration and a well-conditioned evidence integral.

    The distance enters through ``x = distMpcRef / d``; with template ~ 1/d the
    per-time-bin exponent is ``x*Re(kappa_unit) - 0.5*x^2*rho_sq_unit``.  We
    integrate ``exp(exponent) * p(x)`` over the supplied grid:

        lnL_t = log sum_g exp( x_g Re(kappa) - 0.5 x_g^2 rho^2 + log_w_g )

    where ``log_w_g = log( p(d_g) |dd/dx|_g  Delta )`` are the (normalized)
    quadrature log-weights for the chosen distance prior (built by
    :func:`make_distance_grid`).  A ``lax.scan`` over the grid keeps the working
    set at (S, npts) regardless of grid size.

    Parameters
    ----------
    x_grid : array_like, shape (G,)
        Grid of ``x = distMpcRef / d`` values.
    log_w_grid : array_like, shape (G,)
        Log quadrature weights (including the distance prior) for each grid point.
    """
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)
    kappa_unit, rho_sq_unit = _accumulate_unit(
        data, ra, dec, psi, incl, phiref, interp, phase_marginalization)
    K = jnp.abs(kappa_unit) if phase_marginalization else kappa_unit.real
    R = rho_sq_unit

    # running log-sum-exp over the distance grid (carry: m, s) -> (S, npts)
    a = x_grid                    # (G,)
    b = -0.5 * jnp.square(x_grid)  # (G,)

    def step(carry, g):
        m, s = carry
        e = K * a[g] + R * b[g] + log_w_grid[g]      # (S, npts)
        m_new = jnp.maximum(m, e)
        s_new = s * jnp.exp(m - m_new) + jnp.exp(e - m_new)
        return (m_new, s_new), None

    S, npts = K.shape
    init = (jnp.full((S, npts), -jnp.inf), jnp.zeros((S, npts)))
    (m_f, s_f), _ = jax.lax.scan(step, init, jnp.arange(x_grid.shape[0]))
    lnL_t = m_f + jnp.log(s_f)
    return _time_marginalize(lnL_t, data.w_t)


def make_distance_grid(d_min, d_max, n_grid=256, d_prior="euclidean",
                       distMpcRef=DIST_MPC_REF):
    """Build (x_grid, log_w_grid) for distance marginalization.

    Uniform grid in distance ``d``; ``x = distMpcRef/d``.  Returns the log
    quadrature weights ``log( p(d) * Delta_d )`` for the requested prior,
    normalized so ``sum_g exp(log_w_g) == 1`` (a proper distance average).
    ``d_prior='euclidean'`` is the volumetric ``p(d) ∝ d^2`` prior.
    """
    d = np.linspace(d_min, d_max, n_grid)
    dd = d[1] - d[0]
    if d_prior in ("euclidean", "volumetric"):
        pd = d ** 2
    elif d_prior == "uniform":
        pd = np.ones_like(d)
    else:
        raise NotImplementedError("d_prior=%r" % d_prior)
    w = pd * dd
    w = w / np.sum(w)               # normalize the distance average
    x = distMpcRef / d
    log_w = np.log(w)
    return jnp.asarray(x), jnp.asarray(log_w)


# Small accessor used above; attached here to keep JAXLikelihoodData lean and
# to make the (tref - epoch_det) offset explicit per detector.
def _tref_minus_epoch(self, det):
    return self._tref - self.detectors[det]["epoch"]


JAXLikelihoodData.tref_minus_epoch = _tref_minus_epoch


def make_log_likelihood(data, interp="linear", phase_marginalization=False,
                        jit=True):
    """Return a closure ``f(ra, dec, psi, incl, phiref, distMpc) -> lnL``.

    The returned function closes over ``data`` (treated as constant) and is, by
    default, ``jax.jit``-compiled.  It is differentiable with respect to all six
    extrinsic arguments when ``interp="linear"``; combine with ``jax.grad`` /
    ``jax.value_and_grad`` / ``jax.vmap`` as needed.
    """
    def f(ra, dec, psi, incl, phiref, distMpc):
        return fused_log_likelihood(
            data, ra, dec, psi, incl, phiref, distMpc,
            interp=interp, phase_marginalization=phase_marginalization)
    return jax.jit(f) if jit else f
