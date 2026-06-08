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

import os
import numpy as np
import jax
import jax.numpy as jnp
from scipy import integrate as _scipy_integrate

# Adaptive (Gauss-Hermite) distance marginalization.  The distance integrand is
# exp(K x - 0.5 R x^2) with x = d_ref/d -- a Gaussian in x (peak x*=K/R, width
# 1/sqrt(R)) times the d^2 prior (∝ x^-4).  A uniform-in-d grid under-resolves
# the peak (width ~ d0/SNR) at high SNR, biasing the distance average ~1% low.
# GH nodes centred per-sample on x* with scale 1/sqrt(R) integrate it exactly at
# any SNR with few nodes.  Enable with env JAX_ILE_DISTMARG_GH=<n_nodes> (e.g. 48);
# 0 (default) keeps the legacy uniform grid.  See make_distance_grid.
_DISTMARG_GH_N = int(os.environ.get("JAX_ILE_DISTMARG_GH", "0"))

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
                                  interp="linear", phase_marginalization=False,
                                  grid_block=64):
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

    # log-sum-exp over the distance grid -> (S, npts), done in a few *vectorized*
    # blocks combined by a running log-sum-exp.  The block loop is a plain Python
    # loop (unrolled at trace time), NOT a lax.scan: this removes the
    # G-step sequential dependency that made reverse-mode AD (NUTS gradients)
    # slow, while a finite block size keeps the (S, npts, block) working set
    # bounded.  Mathematically identical to the previous scan.
    a = x_grid                     # (G,)
    b = -0.5 * jnp.square(x_grid)  # (G,)
    lnL_t = _logsumexp_grid_blocked(K, R, a, b, log_w_grid, grid_block)
    return _time_marginalize(lnL_t, data.w_t)


def _logsumexp_grid_blocked(K, R, a, b, log_w, block):
    """Stable log sum_g exp(K*a_g + R*b_g + log_w_g) over the grid axis.

    K, R: (S, npts).  a, b, log_w: (G,).  Returns (S, npts).  Processed in
    vectorized blocks of ``block`` grid points combined by a running
    log-sum-exp (Python loop -> unrolled, AD-fast).
    """
    G = a.shape[0]
    block = int(block) if block else G
    S, npts = K.shape
    m = jnp.full((S, npts), -jnp.inf)
    s = jnp.zeros((S, npts))
    for start in range(0, G, block):
        sl = slice(start, min(start + block, G))
        # (S, npts, blk)
        e = (K[:, :, None] * a[None, None, sl]
             + R[:, :, None] * b[None, None, sl]
             + log_w[None, None, sl])
        m_blk = jnp.max(e, axis=-1)                              # (S, npts)
        s_blk = jnp.sum(jnp.exp(e - m_blk[:, :, None]), axis=-1)  # (S, npts)
        m_new = jnp.maximum(m, m_blk)
        s = s * jnp.exp(m - m_new) + s_blk * jnp.exp(m_blk - m_new)
        m = m_new
    return m + jnp.log(s)


def make_distance_gh(n_nodes):
    """Physicists' Gauss-Hermite nodes/log-weights for ∫ e^{-t^2} f(t) dt."""
    xi, w = np.polynomial.hermite.hermgauss(int(n_nodes))
    return jnp.asarray(xi, jnp.float64), jnp.asarray(np.log(w), jnp.float64)


def _distmarg_gh_logL(K, R, gh_xi, gh_logw, x_min, x_max):
    """Per-(S,npts) distance-marginalized lnL via per-sample Gauss-Hermite.

    STATUS (EXPERIMENTAL, env-gated, default OFF): forward values look correct,
    but MALA / MAP-polish gradients still go nan after several anneal stages.
    Root cause: the analytic Gaussian prefactor ``0.5*K^2/R`` has gradient
    ~K^2/R^2 that blows up for small-R (noise) time-bins, corrupting the flow
    production draws -> nan lnL.  FIX DIRECTION (not yet done): drop the K^2/R
    factorization; feed the *adaptive* nodes x_k (centred on x*=K/R, scaled
    1/sqrt(R)) into the SAME stable form the uniform grid uses --
    logsumexp_k[ K*x_k - 0.5*R*x_k^2 + log_w_quad_k ] with trapezoidal weights
    log_w_quad_k = log(Delta x_k * prior(x_k)) -- gradient-stable (no 1/R).
    Until then use the uniform grid (DIST_GRID).

    Computes ``log E_{p(d)}[exp(K x - 0.5 R x^2)]`` with x = dref/d and the
    volumetric prior p(d) ∝ d^2 normalized over [d_min, d_max] (= the same
    "proper distance average" as :func:`make_distance_grid`), but with quadrature
    nodes centred per-sample on the analytic Gaussian peak x*=K/R, scale
    1/sqrt(R) -- so the narrow high-SNR peak is resolved at every SNR.

    In x the prior measure p(d)|dd/dx| ∝ x^{-4}; the Gaussian is absorbed into the
    GH weights (nodes x_k = x* + sqrt(2)/sqrt(R) * xi_k).  Nodes outside the
    physical support [x_min, x_max] are masked.
    """
    R = jnp.maximum(R, 1e-30)
    xstar = K / R                                    # (S, npts)
    inv_sqrtR = 1.0 / jnp.sqrt(R)
    x_k = xstar[..., None] + jnp.sqrt(2.0) * inv_sqrtR[..., None] * gh_xi  # (S,npts,G)
    in_supp = (x_k > x_min) & (x_k < x_max)
    safe_x = jnp.where(in_supp, x_k, 1.0)            # avoid log(<=0) in fwd AND bwd
    log_term = gh_logw - 4.0 * jnp.log(safe_x)       # GH weight * x^{-4} prior
    log_term = jnp.where(in_supp, log_term, -jnp.inf)
    # GRADIENT SAFETY: a fully out-of-support time-bin gives an all -inf row,
    # whose logsumexp has a nan gradient (0/0 softmax) -- which MALA / MAP-polish
    # then propagate into nan samples.  Dummy such rows to a finite value so the
    # backward pass is finite, then override the result to -inf via the outer
    # where (whose selected-against branch is now also finite -> finite grad).
    any_supp = jnp.any(in_supp, axis=-1)             # (S, npts)
    log_term = jnp.where(any_supp[..., None], log_term, 0.0)
    lse = jax.scipy.special.logsumexp(log_term, axis=-1)        # (S, npts), finite
    # ln E[L] = 0.5 K^2/R + log(sqrt(2)/sqrt(R)) + 3 ln dref - ln Zprior + lse,
    #   Zprior = ∫_{d_min}^{d_max} d^2 dd = dref^3 (x_min^{-3} - x_max^{-3})/3, so
    #   3 ln dref - ln Zprior = ln 3 - ln(x_min^{-3} - x_max^{-3})  (dref cancels).
    # x_min/x_max are tracers under jit -> use jnp throughout.
    C0 = 0.5 * jnp.log(2.0) + jnp.log(3.0) - jnp.log(x_min ** (-3.0) - x_max ** (-3.0))
    val = 0.5 * K * K / R - 0.5 * jnp.log(R) + C0 + lse
    return jnp.where(any_supp, val, -jnp.inf)


# ─────────────────────────────────────────────────────────────────────────────
# φ_ref grid-sum marginalisation
# ─────────────────────────────────────────────────────────────────────────────

def phi_ref_grid(nphi: int) -> np.ndarray:
    """Uniform grid of φ_ref values over [0, 2π), shape (nphi,).

    Returns a **numpy** array (not JAX) so that Python ``for`` loops over it
    inside ``jax.jit``-compiled functions produce concrete scalars rather than
    abstract tracers.  Pass directly to the ``fused_log_likelihood_*phimarg``
    functions; they handle conversion internally.

    32 points is exact and fast for l_max = 2 (m_max = 2 needs ≥ 4);
    use 64–128 for l_max ≥ 4 or production quality.  cogwheel uses 128.
    """
    return np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)


def fused_log_likelihood_phimarg(data, ra, dec, psi, incl, distMpc,
                                  phi_grid, interp="linear"):
    """Time-marginalized factored lnL with φ_ref marginalized via uniform grid sum.

    Evaluates the standard factored lnL at each φ_ref in ``phi_grid`` and
    integrates via logsumexp.  Works for **any l_max**; no QAS approximation.
    rho² is re-evaluated at each grid point, so all φ_ref-dependent cross-terms
    in the V matrix are handled correctly.

    Parameters
    ----------
    phi_grid : (nphi,) float array from :func:`phi_ref_grid`.
    """
    distMpc = jnp.asarray(distMpc, dtype=jnp.float64)
    invDist = data.distMpcRef / distMpc
    S = ra.shape[0]
    # lax.scan traces the body ONCE regardless of nphi, so only one copy of
    # _accumulate_unit lives in the XLA graph.  Memory is O(body), not O(nphi×body).
    phi_grid_jax = jnp.asarray(phi_grid, dtype=jnp.float64)   # scan sequence
    nphi = phi_grid_jax.shape[0]

    def _phi_step(carry, phi_val):
        m, s = carry
        phi_arr = jnp.broadcast_to(phi_val, (S,)).astype(jnp.float64)
        kappa_unit, rho_sq_unit = _accumulate_unit(
            data, ra, dec, psi, incl, phi_arr, interp, False)
        kappa = kappa_unit * invDist[:, None]
        rho_sq = rho_sq_unit * jnp.square(invDist)[:, None]
        lnL_t = kappa.real - 0.5 * rho_sq                      # (S, npts)
        m_new = jnp.maximum(m, lnL_t)
        s_new = s * jnp.exp(m - m_new) + jnp.exp(lnL_t - m_new)
        return (m_new, s_new), None

    m0 = jnp.full((S, data.npts), -jnp.inf, dtype=jnp.float64)
    s0 = jnp.zeros((S, data.npts), dtype=jnp.float64)
    (m, s), _ = jax.lax.scan(_phi_step, (m0, s0), phi_grid_jax)

    lnL_t_marg = m + jnp.log(s) - jnp.log(nphi)
    return _time_marginalize(lnL_t_marg, data.w_t)


def fused_log_likelihood_distphimarg(data, ra, dec, psi, incl,
                                      x_grid, log_w_grid,
                                      phi_grid, interp="linear",
                                      grid_block=64):
    """Distance- AND φ_ref-marginalized factored lnL over (ra, dec, psi, incl).

    Marginalises over both luminosity distance (via quadrature grid, as in
    :func:`fused_log_likelihood_distmarg`) and orbital phase φ_ref (via
    uniform grid sum).  The result is a smooth 4-D function of
    ``(ra, dec, psi, incl)`` only, with the curved φ_ref–psi degeneracy ridge
    removed.

    Both integrations are performed *before* time marginalization so the
    per-bin lnL_t collapses cleanly.

    Parameters
    ----------
    phi_grid : (nphi,) float array from :func:`phi_ref_grid`.
    x_grid, log_w_grid : from :func:`make_distance_grid`.
    """
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)
    phi_grid_jax = jnp.asarray(phi_grid, dtype=jnp.float64)   # scan sequence
    nphi = phi_grid_jax.shape[0]
    S = ra.shape[0]
    a = x_grid                              # (G,)
    b = -0.5 * jnp.square(x_grid)          # (G,)

    # Adaptive Gauss-Hermite distance marginalization (env JAX_ILE_DISTMARG_GH>0):
    # resolves the narrowing high-SNR distance peak that the uniform-in-d grid
    # under-samples.  Falls back to the legacy grid otherwise.
    _use_gh = _DISTMARG_GH_N > 0
    if _use_gh:
        _gh_xi, _gh_logw = make_distance_gh(_DISTMARG_GH_N)
        _x_min = jnp.min(x_grid)            # = dref/d_max  (tracer under jit)
        _x_max = jnp.max(x_grid)            # = dref/d_min

    # lax.scan: body traced once; XLA executes sequentially — O(body) memory,
    # not O(nphi × body) as a Python loop unrolled into the graph would be.
    def _phi_step(carry, phi_val):
        m, s = carry
        phi_arr = jnp.broadcast_to(phi_val, (S,)).astype(jnp.float64)
        kappa_unit, rho_sq_unit = _accumulate_unit(
            data, ra, dec, psi, incl, phi_arr, interp, False)
        if _use_gh:
            lnL_t = _distmarg_gh_logL(
                kappa_unit.real, rho_sq_unit, _gh_xi, _gh_logw,
                _x_min, _x_max)
        else:
            lnL_t = _logsumexp_grid_blocked(
                kappa_unit.real, rho_sq_unit, a, b, log_w_grid, grid_block)
        m_new = jnp.maximum(m, lnL_t)
        s_new = s * jnp.exp(m - m_new) + jnp.exp(lnL_t - m_new)
        return (m_new, s_new), None

    m0 = jnp.full((S, data.npts), -jnp.inf, dtype=jnp.float64)
    s0 = jnp.zeros((S, data.npts), dtype=jnp.float64)
    (m, s), _ = jax.lax.scan(_phi_step, (m0, s0), phi_grid_jax)

    lnL_t_marg = m + jnp.log(s) - jnp.log(nphi)
    return _time_marginalize(lnL_t_marg, data.w_t)


def phi_ref_conditional_lnL(data, ra, dec, psi, incl, distMpc,
                              phi_grid, interp="linear"):
    """Log-likelihood vs φ_ref given the other extrinsic parameters.

    Returns a ``(nphi, S)`` array of time-marginalized lnL values, one per
    φ_ref grid point.  Used to draw φ_ref from the conditional posterior after
    the main (phi-marginalized) sampling step; the caller normalises and samples.
    """
    distMpc = jnp.asarray(distMpc, dtype=jnp.float64)
    invDist = data.distMpcRef / distMpc
    S = ra.shape[0]
    phi_grid_jax = jnp.asarray(phi_grid, dtype=jnp.float64)

    # lax.scan: body traced once; outputs stacked automatically → (nphi, S).
    def _phi_step(_, phi_val):
        phi_arr = jnp.broadcast_to(phi_val, (S,)).astype(jnp.float64)
        kappa_unit, rho_sq_unit = _accumulate_unit(
            data, ra, dec, psi, incl, phi_arr, interp, False)
        kappa = kappa_unit * invDist[:, None]
        rho_sq = rho_sq_unit * jnp.square(invDist)[:, None]
        lnL_t = kappa.real - 0.5 * rho_sq
        return None, _time_marginalize(lnL_t, data.w_t)   # carry=None, out=(S,)

    _, lnL_per_phi = jax.lax.scan(_phi_step, None, phi_grid_jax)
    return lnL_per_phi   # (nphi, S)


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


def estimate_distance_peak(data, guess_snr=None, n_sky=4000, seed=0, interp="linear"):
    """Characteristic distance peak/width directly from the precompute.

    The distance integrand per (sky, time-bin) is exp(K x - 0.5 R x^2) with
    x = d_ref/d, K = Re(kappa_unit), R = rho_sq_unit (= <h|h> at d_ref).  The
    matched-filter peak sits at x* = K/R (for K>0) with ln-peak 0.5 K^2/R =
    0.5 rho_mf^2.  We sweep the sky and read off the best (largest K^2/R, K>0)
    sample/time-bin: x* = K/R -> d_peak = d_ref/x*, and the fractional width is
    sigma_d/d_peak = 1/rho_mf with rho_mf = sqrt(K^2/R).

    This reads the *effective* SNR straight from the (PSD-scaled) data, so it is
    robust to ``guess_snr`` being the unscaled template SNR.  ``guess_snr`` is
    accepted only as a fallback if the sweep finds no K>0 sample.  Cheap: one
    random sky sweep, no gradients.  Returns (d_peak, sigma_d) in Mpc.
    """
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 2 * np.pi, n_sky)
    dec = np.arcsin(rng.uniform(-1.0, 1.0, n_sky))
    psi = rng.uniform(0.0, np.pi, n_sky)
    incl = np.arccos(rng.uniform(-1.0, 1.0, n_sky))
    phiref = rng.uniform(0.0, 2 * np.pi, n_sky)
    kappa_unit, rho_sq_unit = _accumulate_unit(
        data, ra, dec, psi, incl, phiref, interp, False)
    K = np.asarray(kappa_unit.real)
    R = np.maximum(np.asarray(rho_sq_unit), 1e-30)
    dref = float(data.distMpcRef)
    snr2 = np.where(K > 0.0, K * K / R, -np.inf)    # matched SNR^2, peak on x>0 only
    if not np.any(np.isfinite(snr2)):               # fallback to the SNR hint
        snr = max(float(guess_snr or 1.0), 1.0)
        R_max = max(float(np.max(R)), 1e-30)
        x_star = snr / np.sqrt(R_max)
        return dref / x_star, (dref / x_star) / snr
    idx = np.unravel_index(int(np.argmax(snr2)), snr2.shape)
    Kb, Rb = float(K[idx]), float(R[idx])
    x_star = Kb / Rb
    rho_mf = np.sqrt(Kb * Kb / Rb)
    d_peak = dref / x_star
    sigma_d = d_peak / max(rho_mf, 1.0)
    return d_peak, sigma_d


def make_distance_grid_adaptive(d_min, d_max, d_peak, sigma_d, d_prior="euclidean",
                                distMpcRef=DIST_MPC_REF, n_fine=128, n_coarse=64,
                                n_sigma=8.0):
    """Non-uniform distance grid: fine near the (SNR-set) peak, coarse on the tail.

    Concentrates resolution where the distance posterior lives (width ~sigma_d,
    so the high-SNR peak is resolved with few points) while keeping a coarse
    full-range backbone so off-peak / broad-prior samples still get support.
    Trapezoidal weights normalized to a *proper distance average* (same
    convention as :func:`make_distance_grid`), so it is drop-in for the existing
    stable logsumexp kernel -- and being a *static* grid it is gradient-stable
    (unlike the per-sample Gauss-Hermite path).
    """
    d_lo = max(float(d_min), d_peak - n_sigma * sigma_d)
    d_hi = min(float(d_max), d_peak + n_sigma * sigma_d)
    if not (d_hi > d_lo):                         # degenerate -> fall back to uniform
        return make_distance_grid(d_min, d_max, n_fine + n_coarse, d_prior, distMpcRef)
    fine = np.linspace(d_lo, d_hi, int(n_fine))
    coarse = np.linspace(float(d_min), float(d_max), int(n_coarse))
    d = np.unique(np.concatenate([coarse, fine]))           # sorted, deduped
    if d_prior in ("euclidean", "volumetric"):
        pd = d ** 2
    elif d_prior == "uniform":
        pd = np.ones_like(d)
    else:
        raise NotImplementedError("d_prior=%r" % d_prior)
    dd = np.empty_like(d)                                    # trapezoidal spacing
    dd[1:-1] = 0.5 * (d[2:] - d[:-2])
    dd[0] = d[1] - d[0]
    dd[-1] = d[-1] - d[-2]
    w = pd * dd
    w = w / np.sum(w)
    return jnp.asarray(distMpcRef / d), jnp.asarray(np.log(w))


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
