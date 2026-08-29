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

Time-interpolation modes (``interp=``, ``--interp``).  ``nearest`` and ``linear`` are
strictly the crudest; between ``cubic`` and ``sinc`` there is no universal ordering --
see the note under ``sinc``:

* ``interp="nearest"`` -- reproduces the production discrete-shift behaviour
  (round the per-detector arrival to the nearest sample) bit-for-bit, used to
  *validate* the JAX path against the numpy reference.
* ``interp="linear"`` -- evaluates the rholm timeseries at the
  *continuous* arrival time, so the likelihood is differentiable with respect
  to sky location (through the geometric time delay) and the other extrinsic
  parameters.  This is the AD-friendly path used for gradient-based exploration.
  It WAS the default until 2026-08-26 and is no longer, because at high SNR it is
  the *worst* option here, worse than ``nearest``: it undershoots the sharp rholm
  peak and so biases the recovered arrival time and hence the sky location.
* ``interp="cubic"`` -- the 4-point cubic-Lagrange stencil the numpy/cupy/CUDA
  paths spell ``time_interp='cubic'``.
* ``interp="sinc"`` (**default** since 2026-08-26) -- the 2a-tap Lanczos windowed
  sinc (a = ``SINC_HALFWIDTH_DEFAULT``), matching ``time_interp='sinc'`` on those
  paths.  Chosen as the default because its error is BOUNDED across the measured
  sweep rather than lowest on average; see ``JAX_INTERP_DEFAULT``.
  Which of ``cubic`` and ``sinc`` is more accurate depends on how oversampled Q
  is -- on fmin and srate as well as on mass -- and there is no automatic rule;
  see ``RIFT/likelihood/DESIGN_q_window_stencil.md`` and
  ``RIFT.likelihood.time_interp_choice.CROSSOVER_GUIDANCE``.  Both are
  differentiable in ``pos``.

Time marginalization uses a precomputed Simpson quadrature weight vector
(``scipy.integrate.simpson`` applied to the identity, exactly as the production
fused kernel does), so the time integral matches the reference numerically while
remaining a constant-weight (hence trivially differentiable) sum.

Conventions / "epoch" handling
-------------------------------
``rho_lm^det`` is a discrete timeseries whose sample ``k`` corresponds to GPS
time ``epoch_det + k * deltaT``.  The window time-bin ``t`` (with
``tvals = (arange(npts) - npts//2)*deltaT`` about the fiducial geocenter
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

# The 'sinc' stencil half-width, shared with the numpy/cupy/CUDA backends.  Imported from the
# leaf module rather than from factored_likelihood so this stays free of numba and lal.
from RIFT.likelihood.time_interp_choice import SINC_HALFWIDTH_DEFAULT

# Adaptive (per-sample) distance marginalization.  The distance integrand is
# exp(K x - 0.5 R x^2) with x = d_ref/d -- a Gaussian in x (peak x*=K/R, width
# 1/sqrt(R)) times the d^2 prior (∝ x^-4).  A uniform-in-d grid under-resolves
# the peak (width ~ d0/SNR) at high SNR, biasing the distance average ~1% low.
# Nodes centred per-sample on x* with scale 1/sqrt(R) (trapezoid, gradient-stable
# placement via stop_gradient) integrate it to machine precision at any SNR with
# a few dozen nodes -- removing the evidence bias.  Enable with env
# JAX_ILE_DISTMARG_GH=<n_nodes> (e.g. 64); 0 keeps the legacy uniform grid.
# See make_distance_gh / _distmarg_gh_logL.
_DISTMARG_GH_N = int(os.environ.get("JAX_ILE_DISTMARG_GH", "0"))

import lal
import lalsimulation as lalsim

from .detector import compute_detamresponse, time_delay_from_earth_center
from .spherical import spherical_harmonics_vectorized
from . import response_slowrot as _rs
from . import response_freqresponse as _rf

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
        Time-window grid.  Only ``tvals[0]`` and ``len(tvals)`` are consumed --
        evaluation steps by ``deltaT`` and integrates with ``dx=deltaT`` regardless of
        the grid's own spacing -- so a grid whose spacing is not ``deltaT`` mislabels
        its own samples.  The builders default to
        ``factored_likelihood.marginalization_time_grid(iwh, deltaT)``, the same
        helper ``bin/integrate_likelihood_extrinsic_batchmode`` uses (issue #146).
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


def _gather_nearest(Q_col, pos, u=None):
    """Q_col[(round(pos))] with the reference's (rint(.)+0.5)->int32 rounding.

    ``u`` is accepted and ignored: every gatherer takes the same signature so the call sites
    can pass the separable fractional offset unconditionally (see :func:`_separable_u`), and a
    discrete gather has no use for it.

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


def _gather_linear(Q_col, pos, u=None):
    """Linear interpolation of Q_col at continuous positions ``pos``.

    Differentiable with respect to ``pos`` (the sub-sample arrival time).
    Positions outside the buffer contribute ZERO (see :func:`_gather_nearest`);
    the validity mask is applied on the continuous ``pos`` so the likelihood
    falls off smoothly to zero as the window leaves the buffer instead of
    latching onto the (flat-extrapolated) edge sample.
    """
    n = Q_col.shape[0]
    i0f = jnp.floor(pos)
    frac = (pos - i0f) if u is None else u
    i0 = jnp.clip(i0f.astype(jnp.int32), 0, n - 1)
    i1 = jnp.clip(i0 + 1, 0, n - 1)
    val = Q_col[i0] * (1.0 - frac) + Q_col[i1] * frac
    valid = (pos >= 0.0) & (pos <= n - 1.0)
    return jnp.where(valid, val, 0.0 + 0.0j)


def _gather_cubic(Q_col, pos, u=None):
    """Four-point cubic-Lagrange interpolation of Q_col at continuous ``pos``.

    Mirrors the production ``factored_likelihood._cubic_Q_window_numpy`` /
    ``Q_inner_product_cubic`` stencil EXACTLY: with ``i0 = floor(pos)`` and
    ``u = pos - i0`` the value is ``sum_{k=-1}^{2} w_k(u) Q[i0+k]`` with the cubic
    Lagrange weights below (at integer ``pos`` it reproduces the sample).  Unlike
    linear, cubic captures the curvature of the razor-sharp high-frequency rholm
    peak; for 3G/high-SNR signals linear *undershoots* that peak (worse than
    nearest) and biases the recovered arrival time -- hence the sky -- so this is
    the interpolation the maintained likelihood uses.  Still differentiable in
    ``pos`` (a polynomial in ``u``), so it drives gradient sampling.  Stencil
    points outside the buffer contribute ZERO (per-point zero extension, matching
    the reference), so an over-running window falls off to zero.
    """
    n = Q_col.shape[0]
    fl = jnp.floor(pos)
    i0 = fl.astype(jnp.int32)
    u = (pos - fl) if u is None else u
    w = (-u * (u - 1.0) * (u - 2.0) / 6.0,
         (u + 1.0) * (u - 1.0) * (u - 2.0) / 2.0,
         -(u + 1.0) * u * (u - 2.0) / 2.0,
         (u + 1.0) * u * (u - 1.0) / 6.0)
    out = jnp.zeros(pos.shape, dtype=jnp.complex128)
    for off, wk in zip((-1, 0, 1, 2), w):
        idx = i0 + off
        valid = (idx >= 0) & (idx < n)
        out = out + wk * jnp.where(valid, Q_col[jnp.clip(idx, 0, n - 1)], 0.0 + 0.0j)
    return out


def _sinc_lanczos_weights_jax(u, a):
    """Lanczos tap weights, mirroring ``factored_likelihood._sinc_lanczos_weight_matrix``.

    The numpy/cupy/CUDA paths all consume ONE weight array built by that function, so they
    cannot drift.  JAX cannot: the weights depend on ``u``, which is a traced function of the
    sky location, so they must be built inside the trace for ``jax.grad`` to see through them.
    This is therefore a deliberate SECOND definition of the same formula, and the thing that
    keeps it honest is ``test/jax/test_jax_stencil_parity.py``, which compares this against the
    numpy generator element-by-element -- including the two details that are easy to get wrong:

      * the ``|x| >= a`` hard zero.  On the wired path (u in [0,1)) it reaches only u == 0,
        where tap k = a sits exactly at x = -a and is worth 1.5e-33 -- but these are library
        helpers, and for u outside [0,1) the clause is worth 2.2e-3, so it is pinned there; and
      * the renormalisation to unit sum, which is applied over the FULL stencil and is NOT
        redone after out-of-buffer taps are dropped.  The CUDA kernel does the same, so the
        three backends agree in the zero-extension region as well as the interior.

    ``u`` has the shape of ``pos``; the return has that shape plus a trailing (2a,) tap axis.
    """
    k = jnp.arange(-a + 1, a + 1)
    x = u[..., None] - k
    w = jnp.sinc(x) * jnp.sinc(x / float(a))
    w = jnp.where(jnp.abs(x) >= a, 0.0, w)
    total = jnp.sum(w, axis=-1)
    total = jnp.where(total == 0.0, 1.0, total)
    return k, w / total[..., None]


def _make_gather_sinc(a):
    """Build the 2a-tap Lanczos gatherer used by ``interp="sinc"``.

    VECTORISED OVER TAPS, and that is load-bearing rather than tidiness.  Written as a Python
    loop over taps -- the shape the 4-tap :func:`_gather_cubic` above can afford -- a 16-tap
    stencil unrolls into 16 separate gathers, and inside a numpyro NUTS trace the resulting
    graph is large enough that XLA compilation dominates: measured >1 h of compile at 0-1% GPU
    against seconds for cubic.  Building the tap axis as an array gives one gather and one
    reduction, so graph size is independent of ``a`` (0.11 s against 0.84 s for the unrolled
    form at a=8).  The two forms agree to a few ulp -- not bit-for-bit, since ``jnp.sum`` over
    the tap axis and a sequential accumulation are free to associate differently -- and
    test_jax_stencil_parity.test_vectorised_matches_unrolled asserts that, to stop a later
    "simplification" back into a loop.

    Accuracy against ``cubic`` is NOT universal: it depends on how oversampled Q is, hence on
    fmin and srate as well as mass.  See RIFT/likelihood/DESIGN_q_window_stencil.md and
    RIFT.likelihood.time_interp_choice.CROSSOVER_GUIDANCE for the measured crossover.

    MEMORY.  XLA does not fuse the tap axis away on its own -- it materialises the
    ``(..., 2a)`` weight array -- so **pass ``u``**; see :func:`_separable_u`, which is what makes
    this stencil affordable.  Without it, GPU whole-likelihood scratch at S=20000/npts=614 is
    6583 MB against 101 MB for ``cubic``; with it, 1279 MB, and the gather itself drops
    1719.2 -> 2.7 MB.  Runtime halves as well, because the general form recomputes 16 sinc pairs
    per (sample, time-bin) when only ``S`` distinct weight rows exist.  Measured figures and the
    rejected ``lax.scan`` alternative are in DESIGN_q_window_stencil.md §9.5.
    """
    def _gather(Q_col, pos, u=None):
        n = Q_col.shape[0]
        fl = jnp.floor(pos)
        i0 = fl.astype(jnp.int32)
        k, w = _sinc_lanczos_weights_jax((pos - fl) if u is None else u, a)
        idx = i0[..., None] + k
        valid = (idx >= 0) & (idx < n)
        vals = Q_col[jnp.clip(idx, 0, n - 1)]
        return jnp.sum(w * jnp.where(valid, vals, 0.0 + 0.0j), axis=-1)
    return _gather


def _make_gather_sinc_unrolled(a):
    """Tap-by-tap form of :func:`_make_gather_sinc`.  Reference for the equivalence test ONLY.

    Do not wire this into ``_GATHERERS``; see the compile-time note there.
    """
    def _gather(Q_col, pos, u=None):
        n = Q_col.shape[0]
        fl = jnp.floor(pos)
        i0 = fl.astype(jnp.int32)
        k, w = _sinc_lanczos_weights_jax((pos - fl) if u is None else u, a)
        out = jnp.zeros(pos.shape, dtype=jnp.complex128)
        for j in range(2 * a):
            idx = i0 + int(k[j])
            valid = (idx >= 0) & (idx < n)
            out = out + w[..., j] * jnp.where(valid, Q_col[jnp.clip(idx, 0, n - 1)],
                                              0.0 + 0.0j)
        return out
    return _gather


def _separable_u(p0):
    """Fractional sample offset for a window built as ``pos = p0[:, None] + arange(npts)``.

    THIS IS A MEMORY FIX, and a large one.  Both accumulators build their window that way, with
    INTEGER time offsets, so ``frac(pos)`` does not vary along the time axis -- only ``S``
    distinct values exist, not ``S * npts``.  Letting a gatherer rediscover ``u`` from the full
    ``pos`` makes it build an ``(S, npts, 2a)`` weight array that XLA then has to materialise:
    at S = 20000, npts = 614 that is 1719 MB of scratch on GPU, against 0 MB for the 4-tap cubic,
    whose weights are cheap enough to stay fused.  Passing ``u`` with shape ``(S, 1)`` instead
    makes the weight array ``(S, 1, 2a)`` -- **2.7 MB, measured, a 637x reduction** -- and small
    enough that the surrounding product and reduction fuse, exactly as cubic's already do.

    It is also MORE accurate, not a trade.  ``p0`` is a sample index of order 1e5-1e6, so
    ``p0 + t`` can cross a binade and drop a low mantissa bit; ``frac(p0 + t)`` then differs from
    ``frac(p0)`` by up to an ulp of the position (~1.5e-11 at 65536, measured).  The numpy, cupy
    and CUDA backends all compute one fractional offset per sample from the sample position --
    i.e. this form -- so using it here IMPROVES cross-backend agreement rather than costing it.

    Callers that do not have a separable window simply omit ``u`` and every gatherer falls back
    to ``pos - floor(pos)``.  Getting it wrong is silent, so
    test_separable_u_matches_the_general_path pins the two against each other at production
    magnitudes, and test_accumulators_pass_separable_u pins that the call sites actually pass it.
    """
    return (p0 - jnp.floor(p0))[:, None]


_GATHERERS = {"nearest": _gather_nearest, "linear": _gather_linear,
              "cubic": _gather_cubic,
              "sinc": _make_gather_sinc(SINC_HALFWIDTH_DEFAULT)}

# The default stencil for every entry point in this package AND for the --interp flag of
# bin/integrate_likelihood_extrinsic_jax, which imports it from here so the two cannot drift.
#
# CHANGED 2026-08-26: 'linear' -> 'sinc'.  This CHANGES RESULTS for any caller that did not pass
# interp= explicitly; pass interp="linear" to reproduce a pre-2026-08-26 run.  Rationale, and the
# concern that goes with it, are recorded in DESIGN_q_window_stencil.md §9.4 -- in one line:
# linear is the worst stencil here at high SNR (worse than 'nearest'), this path is used
# exclusively at high SNR, and 'sinc' is the option whose error is BOUNDED (measured flat at
# 2.3-7.9 nats across the whole mass/fmin sweep) rather than the one with the best best-case.
JAX_INTERP_DEFAULT = "sinc"


def _guarded_window(data, guard):
    """``(npts, t_offsets)`` for the accumulation window widened by ``guard`` samples.

    ``guard == 0`` is the production window -- ``npts == data.npts`` and
    ``t_offsets == arange(data.npts)``, unchanged.  A positive ``guard`` adds that
    many samples at EACH end, so the window runs from ``-guard`` to
    ``data.npts + guard - 1`` and the caller gets ``data.npts + 2*guard`` columns.

    The extra columns are RECONSTRUCTION SUPPORT, not window: only
    :func:`_time_marginalize_bandlimited` asks for them, and it integrates the
    original ``data.npts`` columns after using the guard samples to keep the
    FFT's periodic seam out of the integrated region.  One definition here so
    the two accumulators cannot drift on the offset convention -- an off-by-one
    between them would misplace the arrival time of every guarded evaluation.
    """
    guard = int(guard)
    if guard < 0:
        raise ValueError("guard must be >= 0 samples, got %r" % (guard,))
    return (data.npts + 2 * guard,
            jnp.arange(-guard, data.npts + guard, dtype=jnp.float64))


def _accumulate_unit(data, ra, dec, psi, incl, phiref, interp,
                     phase_marginalization, guard=0):
    """Network kappa and rho^2 at the *fiducial* distance (invDist == 1).

    Returns ``(kappa_unit, rho_sq_unit)`` each shape (S, npts).  ``kappa_unit``
    is complex; the distance factor and the Re/abs reduction are applied by the
    caller (so the same accumulation feeds both the fixed-distance and the
    distance-marginalized paths).

    When ``data`` carries a slow-rotation / finite-size ``feature`` (built by
    :func:`banded.build_rotation_data` / :func:`banded.build_freqresponse_data`),
    the multi-band accumulator is used instead.  It returns the *identical*
    ``(kappa_unit, rho_sq_unit)`` contract, so every downstream marginalization
    variant (distance, phi_ref, psi, ...) inherits the feature for free.

    ``guard`` widens the evaluated window by that many samples at each end (see
    :func:`_guarded_window`), giving ``(S, data.npts + 2*guard)``.  It defaults
    to 0, i.e. every existing caller gets the production window unchanged; the
    band-limited time quadrature is the one caller that asks for more, and it
    pays for them in gathers (cost is linear in the number of columns).
    """
    if getattr(data, "feature", None) is not None:
        return _accumulate_unit_banded(data, ra, dec, psi, incl, phiref, interp,
                                       phase_marginalization, guard=guard)
    ra = jnp.asarray(ra, dtype=jnp.float64)
    dec = jnp.asarray(dec, dtype=jnp.float64)
    psi = jnp.asarray(psi, dtype=jnp.float64)
    incl = jnp.asarray(incl, dtype=jnp.float64)
    phiref = jnp.asarray(phiref, dtype=jnp.float64)

    gather = _GATHERERS[interp]
    gmst = data.gmst
    inv_deltaT = 1.0 / data.deltaT
    S = ra.shape[0]
    npts, t_offsets = _guarded_window(data, guard)

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
        # None for 'nearest': it ignores u, and feeding an unused value into this trace
        # is NOT free -- it cost >60% wall on the banded slow-rotation path (measured:
        # test_rotation_path_a 69.8 s -> >113 s), which is compile-bound, not arithmetic-
        # bound.  Only the weight-building stencils get it.  See _separable_u.
        u_sep = None if interp == "nearest" else _separable_u(p0)

        kappa_det = jnp.zeros((S, npts), dtype=jnp.complex128)
        for k in range(K):
            Qi = gather(Q[:, k], pos, u_sep)
            kappa_det = kappa_det + FY_conj[:, k][:, None] * Qi
        kappa_unit = kappa_unit + kappa_det
        # NOT a gap, and worth saying so because an earlier revision wrongly marked it as one:
        # this is the BASELINE (non-banded) accumulator, unreachable for slow rotation, since
        # _accumulate_unit delegates to _accumulate_unit_banded whenever data.feature is set.
        # Its response coefficient is the static scalar F, evaluated once at tref and carrying
        # no sidereal harmonic index, so there is no arrival-time post-phase to apply and <h|h>
        # genuinely does not depend on where in the window the template is placed.  The
        # slow-rotation model does have that dependence -- see _accumulate_unit_banded.
        rho_sq_unit = rho_sq_unit + rho_sq_det[:, None]

    return kappa_unit, rho_sq_unit


def _norm_is_arrival_time_dependent(data):
    """True when the model norm ``<h|h>`` depends on the template's arrival time.

    Only the slow-rotation bank has that dependence: its post-phase
    ``C~_a(t) = C_a exp(i n_a Omega (t - tref))`` multiplies the data term AND the
    norm, so :func:`_accumulate_unit_banded` returns a genuinely ``(S, npts)``
    ``rho_sq`` there.  The baseline accumulator (static ``F``) and the finite-size
    ``freqresponse`` bank (no sidereal modulation) both return a norm that is
    constant along the time axis, broadcast into the ``(S, npts)`` contract.

    One definition on purpose: the quadratures that hold the norm fixed refuse
    exactly the data this predicate flags, so the two must not drift apart.
    """
    return getattr(data, "feature", None) == "rotation"


def _banded_coefficients(data, det, ra, dec, psi):
    """Per-sample response coefficients ``C`` of shape (A, S) for detector ``det``.

    Dispatches on ``data.feature``: ``"rotation"`` -> the sidereal-harmonic
    coefficients ``C_{(p,n)}`` (Path A/B), ``"freqresponse"`` -> the finite-size
    basis coefficients ``b_p`` (Path D).  ``data.gmst`` (= GMST(tref), a host
    constant) is the sidereal reference, exactly as in the numpy NoLoop path.
    """
    dd = data.detectors[det]
    b = data.band
    if data.feature == "rotation":
        return _rs.rotation_coefficients_packed(
            dd["response"], dd["location"], ra, dec, psi, data.gmst,
            b["p_max"], b["a_list"])
    if data.feature == "freqresponse":
        return _rf.response_coefficients_packed(
            dd["response"], dd["x_arm"], dd["y_arm"], ra, dec, psi, data.gmst,
            b["Qmax"], b["p_list"])
    raise ValueError("unknown banded feature %r" % (data.feature,))


def _accumulate_unit_banded(data, ra, dec, psi, incl, phiref, interp,
                            phase_marginalization, guard=0):
    """Multi-band (slow-rotation / finite-size) network kappa and rho^2.

    Generalizes :func:`_accumulate_unit` by an extra summed "band" index
    ``a`` (sidereal harmonic ``(p,n)`` / finite-size basis weight ``p``), sized
    ``A``, contracted with the per-sample coefficient vector ``C_a`` from
    :func:`_banded_coefficients`.  The baseline is the ``A==1``, ``C==[F]`` case.

    Mirrors the numpy NoLoop references
    ``DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation`` and
    ``DiscreteFactoredLogLikelihoodFreqResponseNoLoop``:

        kappa_unit = sum_a conj(C_a) sum_lm conj(Y_lm) Q^a_lm(t)
        rho_sq_unit = 0.5 Re[ sum_{a,a'} conj(C_a) C_a' (Ybar U^{a,a'} Y)
                                       +  C_{aR}  C_a' (Y    V^{a,a'} Y) ]

    with the caller applying ``invDist`` / ``invDist^2`` and the ``Re / -1/2``
    reduction (so ``-0.5*rho_sq_unit == -0.25 Re[...]`` matches the reference
    ``term2``).  ``aR`` is the V-term reflection (``(p,-n)`` for rotation, the
    identity for finite-size), supplied as ``data.band['refl_idx']``.

    ARRIVAL-TIME POST-PHASE (``feature == "rotation"`` only).
    The bank's elementary templates ``chi_a(u) = e^{i n_a Omega u} h^{(p_a)}(u)`` live on
    the template's INTRINSIC time ``u``, while the physical response modulation lives on
    absolute time.  Placing the template at arrival time ``t`` (``t' = u + t``) factorizes
    it and leaves a residual factor that belongs to the coefficient,

        C~_a(t) = C_a * exp(i n_a Omega (t - tref))

    (``factored_likelihood_with_rotation.rotation_post_phase``), which must be applied to
    the data term AND the model norm -- using it in only one evaluates ``<d|h>`` and
    ``<h|h>`` for different ``h`` and breaks ``lnL <= (1/2)<d|d>``.  It makes ``rho_sq``
    arrival-time DEPENDENT, hence ``(S, npts)`` rather than a broadcast ``(S,)`` scalar.

    No ``(S, npts)`` phase array is materialized per band: with the gather positions
    ``pos_ij = p0_i + j`` the offset separates,

        delta_ij = pos_ij * deltaT - (tref - epoch) = delta0_i + jgrid_j,

    so ``exp(i m omega delta_ij) = pe[m, i] * pt[m, j]`` is rank-1, and the phase enters
    both terms only through the integer ``m`` (``-n_a`` for the data term, ``n_a' - n_a``
    for BOTH the U and V contractions).  One ``(M, S)`` and one ``(M, npts)`` table cover
    everything; ``M`` is the number of distinct ``m``, ``4*n_harmonics + 1`` at the default
    width whatever ``p_max`` is (several ``p`` share a harmonic once ``p_max >= 1``, so the
    ``(a, a')`` pairs genuinely collide in a bucket and the scatter-add accumulates them).

    This mirrors ``DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation``,
    including its choice of arrival sample: ``interp="nearest"`` phases each output bin at
    the sample the gather actually read.  The one exception is a position at ``rint(pos)
    == -1`` -- one bin off the FRONT of the rholm buffer -- where ``_gather_nearest``'s
    ``trunc(. + 0.5)`` index rounds to sample 0; see the note at the ``samp0`` assignment.

    ``freqresponse`` (Path D) has NO post-phase -- its basis is not a sidereal modulation
    -- and keeps the arrival-time-independent ``rho_sq``.

    ``guard`` widens the window as in :func:`_accumulate_unit` / :func:`_guarded_window`.
    The post-phase follows it: ``jgrid`` is built from the same (now partly negative)
    integer offsets, so a guarded sample is phased at the arrival time it was gathered
    from, exactly as an unguarded one is.

    ``phase_marginalization`` is not supported for banded features.
    """
    if phase_marginalization:
        raise NotImplementedError(
            "phase marginalization is not supported for slow-rotation / "
            "finite-size (banded) likelihoods")

    ra = jnp.asarray(ra, dtype=jnp.float64)
    dec = jnp.asarray(dec, dtype=jnp.float64)
    psi = jnp.asarray(psi, dtype=jnp.float64)
    incl = jnp.asarray(incl, dtype=jnp.float64)
    phiref = jnp.asarray(phiref, dtype=jnp.float64)

    gather = _GATHERERS[interp]
    gmst = data.gmst
    inv_deltaT = 1.0 / data.deltaT
    S = ra.shape[0]
    npts, t_offsets = _guarded_window(data, guard)
    refl_idx = data.band["refl_idx"]           # (A,) int, static

    # Arrival-time post-phase: rotation only (see the docstring).  Honour the bank
    # convention flag rather than assuming it, so a future change fails loudly.
    band = data.band
    post_phase = _norm_is_arrival_time_dependent(data)
    if post_phase:
        if not bool(band.get("post_phase_required", False)):
            raise ValueError(
                "rotation likelihood data does not declare post_phase_required; this "
                "evaluator applies the arrival-time post-phase (rotation_post_phase) to "
                "both the data term and the model norm and is only correct for a bank "
                "built in that convention.  meta['post_phase_required'] is set by "
                "PrecomputeLikelihoodTermsWithRotation as of PR #117 -- if this tree does "
                "not have #117, it does not have the corrected precompute either and the "
                "JAX rotation path MUST NOT be used on it.  Otherwise rebuild the bank "
                "with banded.build_rotation_data.")
        omega_sid = 2.0 * np.pi * float(band["f_sidereal"])
        pp_m = jnp.asarray(np.asarray(band["pp_m_values"], dtype=np.float64))  # (M,)
        pp_t1 = np.asarray(band["pp_term1_idx"], dtype=np.int64)               # (A,) static
        pp_t2 = jnp.asarray(np.asarray(band["pp_term2_idx"], dtype=np.int64))  # (A,A)
        M = int(pp_m.shape[0])

    kappa_unit = jnp.zeros((S, npts), dtype=jnp.complex128)
    rho_sq_unit = jnp.zeros((S, npts), dtype=jnp.float64)

    for det in data.detector_names:
        dd = data.detectors[det]
        lms = dd["lms"]
        Q_bank = dd["Q_bank"]                   # (A, npts_full, K)
        U_bank = dd["U_bank"]                   # (A, A, K, K)
        V_bank = dd["V_bank"]                   # (A, A, K, K)
        A = Q_bank.shape[0]
        K = len(lms)

        Y = spherical_harmonics_vectorized(lms, incl, -phiref, l_max=dd["l_max"])
        conjY = jnp.conj(Y)                     # (S, K)

        C = _banded_coefficients(data, det, ra, dec, psi)   # (A, S) complex
        C_refl = C[refl_idx]                                 # (A, S)

        t_det = (data.tref_minus_epoch(det)
                 + time_delay_from_earth_center(dd["location"], ra, dec, gmst))
        p0 = (t_det + data.tval0) * inv_deltaT
        pos = p0[:, None] + t_offsets[None, :]              # (S, npts)
        # None for 'nearest': it ignores u, and feeding an unused value into this trace
        # is NOT free -- it cost >60% wall on the banded slow-rotation path (measured:
        # test_rotation_path_a 69.8 s -> >113 s), which is compile-bound, not arithmetic-
        # bound.  Only the weight-building stencils get it.  See _separable_u.
        u_sep = None if interp == "nearest" else _separable_u(p0)

        if post_phase:
            # delta_ij = (arrival time of output bin j for sample i) - tref, in seconds.
            # ``pos`` is in samples from the rholm epoch, so delta = pos*deltaT - off with
            # off = tref - epoch.  It must be the arrival the GATHER actually uses, or the
            # data term and the model norm drift apart again: for interp="nearest" that is
            # the rounded position, for the interpolating stencils the continuous one.
            #
            # ``jnp.rint(p0) + j == jnp.rint(p0 + j)`` exactly (j is an integer and the sum
            # is well inside float64's exact-integer range), so this IS the gathered
            # position, and it stays separable in (i, j).  _gather_nearest's index is
            # ``trunc(rint(pos) + 0.5)``, which equals rint(pos) for every non-negative
            # position; the one place the two differ is rint(pos) == -1, where that
            # truncation reads sample 0 for a position one bin off the FRONT of the buffer.
            # That is a pre-existing quirk of the gather (the numpy NoLoop, which slices
            # ``ifirst:ilast``, is no better there) and not something the post-phase can or
            # should paper over; every position the gather treats as in-bounds and
            # non-negative is phased at exactly the sample it read.
            off = float(data.tref_minus_epoch(det))
            samp0 = jnp.rint(p0) if interp == "nearest" else p0
            delta0 = samp0 * data.deltaT - off              # (S,)
            jgrid = t_offsets * data.deltaT                 # (npts,)
            pe = jnp.exp(1j * omega_sid * pp_m[:, None] * delta0[None, :])   # (M, S)
            pt = jnp.exp(1j * omega_sid * pp_m[:, None] * jgrid[None, :])    # (M, npts)

        # --- term1: sum_a conj(C~_a) * ( sum_lm conj(Y_lm) Q^a_lm(t) ) ---
        # conj(C~_a) = conj(C_a) exp(-i n_a omega delta), i.e. the m = -n_a bucket.
        kappa_det = jnp.zeros((S, npts), dtype=jnp.complex128)
        for a in range(A):
            inner_a = jnp.zeros((S, npts), dtype=jnp.complex128)
            Qa = Q_bank[a]                                   # (npts_full, K)
            for k in range(K):
                inner_a = inner_a + conjY[:, k][:, None] * gather(Qa[:, k], pos, u_sep)
            if post_phase:
                i1 = int(pp_t1[a])
                kappa_det = kappa_det + ((jnp.conj(C[a]) * pe[i1])[:, None]
                                         * (pt[i1][None, :] * inner_a))
            else:
                kappa_det = kappa_det + jnp.conj(C[a])[:, None] * inner_a
        kappa_unit = kappa_unit + kappa_det

        # --- term2: 0.5 Re[ sum_{a,a'} conj(C~_a)C~_a' YbarUY + C~_aR C~_a' YVY ] ---
        # YUY[a,a'] = einsum(conjY, Y, U_bank[a,a']); YVY[a,a'] = einsum(Y, Y, V)
        YUY = jnp.einsum("si,sj,abij->abs", conjY, Y, U_bank)   # (A,A,S)
        YVY = jnp.einsum("si,sj,abij->abs", Y, Y, V_bank)       # (A,A,S)
        # conj(C_a) C_a'  and  C_aR C_a'  contracted over (a,a') -- the post-phase is
        # applied below, since it depends only on m = n_a' - n_a for both contractions.
        CC_U = jnp.einsum("as,bs->abs", jnp.conj(C), C)          # (A,A,S)
        CC_V = jnp.einsum("as,bs->abs", C_refl, C)              # (A,A,S)
        pair = CC_U * YUY + CC_V * YVY                           # (A,A,S) complex
        if post_phase:
            # BOTH contractions carry exp(i (n_a' - n_a) omega delta), so bucket the pairs
            # by m and pay one rank-1 phase per distinct m (M of them) instead of A^2.
            val_m = jnp.zeros((M, S), dtype=jnp.complex128).at[pp_t2].add(pair)
            # rho_sq becomes arrival-time dependent: (S, npts), not a broadcast scalar.
            rho_sq_det = 0.5 * jnp.einsum("ms,mt->st", val_m * pe, pt).real
        else:
            term2_c = jnp.sum(pair, axis=(0, 1))                 # (S,) complex
            rho_sq_det = 0.5 * term2_c.real                      # (S,)
        rho_sq_unit = rho_sq_unit + (rho_sq_det if post_phase
                                     else rho_sq_det[:, None])

    return kappa_unit, rho_sq_unit


TIME_QUAD_DEFAULT = "simpson"        # unchanged behaviour; "bandlimited" is opt-in
_TIME_QUAD_CHOICES = ("simpson", "bandlimited")
_TIME_UPSAMPLE_DEFAULT = 8
_TIME_ADAPTIVE_FACTOR_MAX = 1024
_TIME_ADAPTIVE_SAFETY = 2.0
_TIME_ADAPTIVE_RTOL = 1e-3
_TIME_ENDPOINT_LOG_GAP_MIN = 15.0


def default_time_guard(npts):
    """Guard width (samples per end) used by ``time_quad="bandlimited"`` by default.

    Half the window, floored at 32 samples: the reconstruction's seam error falls
    off only like 1/distance (see :func:`_time_marginalize_bandlimited`), so the
    guard has to be a FRACTION of the window rather than a small fixed pad, and
    the cost is linear -- half a window at each end doubles the gathers of an
    opt-in quadrature that exists to buy accuracy.  It is a convergence knob, not
    a constant of nature: pass ``time_guard=`` and raise it until the answer stops
    moving, exactly as with ``time_upsample``.
    """
    return max(32, int(npts) // 2)


def _upsample_bandlimited(x, factor, axis=-1):
    """Band-limited resampling of ``x`` by an integer ``factor``, ASSUMING PERIODICITY.

    Zero-pads the spectrum, which is interpolation only in the sense that the
    sampling theorem is: for a signal whose Fourier content the grid already
    resolves AND WHOSE PERIOD IS THE ARRAY, the padded inverse transform
    reproduces the underlying continuous function at the finer spacing, not an
    approximation of it.

    THE PERIODICITY CAVEAT IS NOT DECORATIVE, and it is not something the caller
    can check on the output.  The array is a period, so this treats ``x[-1]`` and
    ``x[0]`` as adjacent; when they are not -- as for any window CROPPED out of a
    longer timeseries, which is what the accumulators produce -- the seam's jump
    is spectral content as far as the FFT is concerned, and the reconstruction
    rings.  The retained samples stay exact (the interpolant passes through every
    input sample), so a test that only checks ``fine[::factor] == x`` cannot see
    it; the INSERTED samples carry the error, largest at the ends and falling off
    only like 1/(distance from the seam).  Callers reconstructing a crop must
    supply guard samples and discard the contaminated region --
    :func:`_time_marginalize_bandlimited` does, and explains the arithmetic.

    Nyquist handling: for even ``n`` the +n/2 bin is split evenly between the
    +n/2 and -n/2 positions.  Dumping it entirely into one of them biases the
    result by a term that oscillates at Nyquist -- small, but exactly the kind
    of grid-phase-dependent error this whole change exists to remove.
    """
    x = jnp.asarray(x)
    n = x.shape[axis]
    if factor == 1:
        return x
    X = jnp.fft.fft(x, axis=axis)
    X = jnp.moveaxis(X, axis, -1)
    n_out = n * factor
    half = n // 2
    pad_shape = X.shape[:-1] + (n_out - n,)
    if n % 2 == 0:
        lo = X[..., :half]
        hi = X[..., half + 1:]
        nyq = X[..., half:half + 1] * 0.5
        Y = jnp.concatenate(
            [lo, nyq, jnp.zeros(X.shape[:-1] + (n_out - n - 1,), X.dtype),
             nyq, hi], axis=-1)
    else:
        Y = jnp.concatenate(
            [X[..., :half + 1], jnp.zeros(pad_shape, X.dtype),
             X[..., half + 1:]], axis=-1)
    y = jnp.fft.ifft(Y, axis=-1) * factor
    return jnp.moveaxis(y, -1, axis)


def _time_marginalize_bandlimited(kappa_t, rho_sq, deltaT, factor, guard,
                                  phase_marginalization=False):
    """Time marginal evaluated on a band-limited RECONSTRUCTION of kappa(t).

    Why this exists.  ``_time_marginalize`` integrates exp(lnL_t) with fixed
    Simpson weights at the DATA sample spacing, but the integrand's width is
    sigma_t = 1/(2 pi rho sigma_f) -- it SHRINKS as the signal gets louder while
    the grid does not.  Measured on a 35+30 Msun HLV injection at rho=40:
    sigma_t = 61.2 us against grid spacings of 244/122/61 us at srate
    4096/8192/16384, i.e. under-resolved at the sample rates people actually
    use, and worse at higher SNR.  The required condition is
    srate >~ 2 pi sigma_f rho (~16 kHz at rho=40 here, growing linearly with
    SNR).  Simpson is not a safeguard here: Simpson = (4 T_h - T_2h)/3 carries
    the coarser T_2h alias, so when under-resolved it is WORSE than trapezoid --
    the measured lnL span over a rigid grid-phase scan was 1.649 / 0.385 /
    0.0095 nats at those three sample rates, dominated by the period-2h term.

    The fix costs no new likelihood evaluations.  kappa(t) is band-limited (it
    is a cross-correlation of band-limited data with a band-limited template),
    and rho_sq is time-independent (the PRECONDITION below), so the samples ALREADY
    COMPUTED determine the continuous integrand exactly.  One zero-padded FFT
    per row recovers it; the quadrature then runs on the reconstruction.

    Measured against a converged window-shift reference: -0.007 nats, versus
    +0.745 nats for stock Simpson at the same grid phase.  (That comparison was
    taken with ``guard == 0``, i.e. before the guard argument below existed, so
    it is quoted for the SIMPSON contrast and not as a bound on the seam error.)

    GUARD SAMPLES, and why this argument has no default.  ``kappa_t`` is a window
    CROPPED out of a longer correlation buffer, so its two ends are not
    neighbours -- while :func:`_upsample_bandlimited`, being an FFT, necessarily
    treats them as though they were.  The fictitious jump across that seam is
    spectral content as far as the transform is concerned, and it rings into the
    INSERTED samples, changing the integrand before any quadrature touches it.
    The rings are invisible at the input samples (the interpolant reproduces
    those exactly), which is why this has to be handled here rather than caught
    downstream.

    The cure is support, not smoothing: ``kappa_t`` carries ``guard`` extra
    samples at EACH end (:func:`_accumulate_unit` gathers them), the seam moves
    out to those ends, and only the middle
    ``npts = kappa_t.shape[-1] - 2*guard`` samples -- the window the stock
    quadrature integrates -- are kept.  The residual is then set by the seam's
    distance: writing the reconstruction as a Whittaker sum over the
    PERIODICALLY REPEATED window, the error at a point ``d`` samples inside the
    kept region is the tail

        sum_{j outside} (xtilde_j - x_j) sinc(t - j)  ~  |seam jump| / (pi d),

    which falls off like 1/d and NOT exponentially.  So ``guard`` is a
    convergence knob of the same standing as ``factor``: raise it until the
    answer stops moving (:func:`fused_log_likelihood` starts from
    :func:`default_time_guard`).  Passing zero on real cropped data is the defect
    itself, so there is no default to fall into; ``guard=0`` stays legitimate
    only for a window whose ends genuinely join -- a constant, or an integrand
    that has decayed to its pedestal at both ends.

    PRECONDITION on ``rho_sq``: the model norm must NOT depend on arrival time.
    Only ``rho_sq[..., :1]`` is used here, because the reconstruction is of kappa
    alone.  That is the complete norm for the baseline and finite-size
    accumulators, whose ``rho_sq`` is a constant column broadcast along time, but
    the slow-rotation post-phase makes ``rho_sq`` a genuine function of the
    arrival bin (:func:`_accumulate_unit_banded`), and taking its first bin there
    would evaluate a DIFFERENT likelihood.  Callers must screen such data with
    :func:`_norm_is_arrival_time_dependent`; :func:`fused_log_likelihood` does.
    """
    guard = int(guard)
    if guard < 0:
        raise ValueError("guard must be >= 0 samples, got %r" % (guard,))
    npts = kappa_t.shape[-1] - 2 * guard
    if npts < 2:
        raise ValueError(
            "guard=%d leaves %d sample(s) of a %d-sample array to integrate; the "
            "guard samples are reconstruction support, not window"
            % (guard, npts, kappa_t.shape[-1]))
    kappa_f = _upsample_bandlimited(kappa_t, factor, axis=-1)
    # Integrate the ORIGINAL interval (npts-1)*deltaT, and nothing else.  Two
    # distinct pieces of the fine grid are NOT part of it:
    #   * the leading and trailing `guard*factor` samples, whose coarse samples
    #     were gathered only to hold the periodic seam away from the window; and
    #   * the last factor-1 samples of the array, which lie PAST the final coarse
    #     sample and are the periodic continuation wrapping back toward sample 0.
    # Integrating either rescales the result -- for a constant integrand by
    # exactly log(kept length / ((npts-1)*deltaT)) -- against every other
    # quadrature in the module.  Starting at guard*factor and taking
    # (npts-1)*factor+1 points lands on the original two endpoints exactly.
    start = guard * factor
    kappa_f = kappa_f[..., start:start + (npts - 1) * factor + 1]
    # The norm is taken at the first bin OF THE KEPT WINDOW, not of the guarded
    # array: identical under the precondition (the column is constant), and the
    # one that stays right if a future caller ever relaxes it.
    rho_sq_0 = rho_sq[..., guard:guard + 1]
    if phase_marginalization:
        lnL_f = jnp.abs(kappa_f) - 0.5 * rho_sq_0
    else:
        lnL_f = kappa_f.real - 0.5 * rho_sq_0
    n_f = lnL_f.shape[-1]
    w_f = jnp.asarray(_simpson_weights(n_f, deltaT / factor))
    m = jnp.max(lnL_f, axis=-1, keepdims=True)
    L = jnp.sum(w_f[None, :] * jnp.exp(lnL_f - m), axis=-1)
    return m[:, 0] + jnp.log(L)


def _time_marginalize(lnL_t, w_t):
    """log integral_t exp(lnL_t) dt via constant Simpson weights, log-sum-exp stable."""
    m = jnp.max(lnL_t, axis=-1, keepdims=True)
    L = jnp.sum(w_t[None, :] * jnp.exp(lnL_t - m), axis=-1)
    return m[:, 0] + jnp.log(L)


def _reflected_fft_upsample(x, factor):
    """FFT-interpolate a finite row with the standard even extension.

    Periodize ``[x[0], ..., x[-1], x[-2], ..., x[1]]``.  Omitting duplicate
    turning samples is mathematically essential at Nyquist: duplicating them
    inserts an artificial flat pair, so ``(-1)**j`` no longer reconstructs
    ``cos(pi*t)`` and phase marginalization can converge to the wrong integral.
    Only the original closed forward interval is returned.
    """
    x = jnp.asarray(x)
    factor = int(factor)
    if factor == 1:
        return x
    n = x.shape[-1]
    reflected = jnp.concatenate((x, jnp.flip(x[..., 1:-1], axis=-1)), axis=-1)
    dense = _upsample_bandlimited(reflected, factor, axis=-1)
    return dense[..., :(n - 1) * factor + 1]


def _peak_width_from_lnL_jax(lnL_t, dx):
    """Measure per-row Gaussian peak width from finite centred differences."""
    n = lnL_t.shape[-1]
    finite = jnp.isfinite(lnL_t)
    safe = jnp.where(finite, lnL_t, -jnp.inf)
    jmax = jnp.argmax(safe, axis=-1)
    sigma = jnp.full(jmax.shape, jnp.inf, dtype=jnp.float64)
    measurable = jnp.zeros(jmax.shape, dtype=bool)

    def take(j):
        return jnp.take_along_axis(lnL_t, j[..., None], axis=-1)[..., 0]

    for d in (1, 2, 4, 8):
        if 2 * d >= n:
            break
        jc = jnp.clip(jmax, d, n - 1 - d)
        d2 = (take(jc - d) - 2.0 * take(jc) + take(jc + d)) / (d * dx) ** 2
        fresh = jnp.isfinite(d2) & (~measurable)
        neg = fresh & (d2 < 0)
        sigma = jnp.where(neg, 1.0 / jnp.sqrt(jnp.where(neg, -d2, 1.0)), sigma)
        measurable = measurable | fresh
    return sigma, measurable


def _log_trapezoid(lnL_t, dx):
    """Stable row-wise log trapezoid over exactly the represented interval."""
    m = jnp.max(lnL_t, axis=-1, keepdims=True)
    m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
    y = jnp.exp(lnL_t - m_safe)
    total = dx * (0.5 * y[..., 0] + jnp.sum(y[..., 1:-1], axis=-1)
                  + 0.5 * y[..., -1])
    return m_safe[..., 0] + jnp.log(total)


def _terminal_reflected_fft_at_factor(lnL_t, deltaT, factor):
    dense = _reflected_fft_upsample(lnL_t, factor).real
    value = _log_trapezoid(dense, deltaT / float(factor))
    sigma, measurable = _peak_width_from_lnL_jax(dense, deltaT / float(factor))
    resolved = (~measurable) | (~jnp.isfinite(sigma)) | (
        deltaT / float(factor) <= sigma / _TIME_ADAPTIVE_SAFETY)
    return value, resolved


def _time_marginalize_reflected_fft(lnL_t, deltaT, w_t):
    """Adaptive reflected-FFT terminal time marginalization.

    A per-row power-of-two factor is selected from the coarse-row curvature.
    ``lax.map`` keeps the switched FFT scratch row-local, so one sharp sample
    neither changes its batchmates' result nor materializes its fine grid for
    the whole sampler batch.  The selected grid is remeasured and doubled until
    both the width criterion and a 1e-3-nat convergence check pass.  Rows with
    any non-finite coarse bin retain the historical Simpson value; their
    sanitized values still enter traced FFT branches because JAX evaluates both
    sides of ``where``.
    """
    lnL_t = jnp.asarray(lnL_t, dtype=jnp.float64)
    simpson = _time_marginalize(lnL_t, w_t)
    finite_rows = jnp.all(jnp.isfinite(lnL_t), axis=-1)
    clean = jnp.where(finite_rows[:, None], lnL_t, 0.0)

    sigma, measurable = _peak_width_from_lnL_jax(clean, deltaT)
    need = jnp.where(measurable & jnp.isfinite(sigma) & (sigma > 0),
                     _TIME_ADAPTIVE_SAFETY * deltaT / sigma, 1.0)
    need = jnp.maximum(need, 1.0)
    factor_float = jnp.exp2(jnp.ceil(jnp.log2(need)))
    factor_float = jnp.where(factor_float < need, factor_float * 2.0, factor_float)
    too_sharp = (~jnp.isfinite(factor_float)) | (
        factor_float > _TIME_ADAPTIVE_FACTOR_MAX)
    # Clamp before the integer cast: inf or an out-of-range float can wrap to a
    # negative integer and otherwise masquerade as factor 1.
    factor = jnp.minimum(factor_float, float(_TIME_ADAPTIVE_FACTOR_MAX)).astype(
        jnp.int32)

    powers = tuple(1 << k for k in range(11))  # 1 .. 1024; two rechecks reach 4096

    def make_branch(base):
        def branch(x):
            v0, r0 = _terminal_reflected_fft_at_factor(x, deltaT, base)
            v1, r1 = _terminal_reflected_fft_at_factor(x, deltaT, 2 * base)
            v2, r2 = _terminal_reflected_fft_at_factor(x, deltaT, 4 * base)
            c1 = r1 & (jnp.abs(v1 - v0) <= _TIME_ADAPTIVE_RTOL)
            c2 = r2 & (jnp.abs(v2 - v1) <= _TIME_ADAPTIVE_RTOL)
            # A non-converged final doubling is a fail-closed NaN, not a
            # plausible-looking under-resolved likelihood.
            return jnp.where(c1, v1, jnp.where(c2, v2, jnp.nan))
        return branch

    def refine_one(args):
        row, row_factor = args
        index = jnp.clip(
            jnp.ceil(jnp.log2(row_factor.astype(jnp.float64))).astype(jnp.int32),
            0, len(powers) - 1)
        return jax.lax.switch(index, tuple(make_branch(f) for f in powers), row)

    refined = jax.lax.map(jax.checkpoint(refine_one), (clean, factor))
    refined = jnp.where(too_sharp, jnp.nan, refined)
    return jnp.where(finite_rows, refined, simpson)


def _time_marginalize_reflected_primitive(kappa_t, rho_sq, deltaT,
                                           phase_marginalization=False):
    """Adaptive integral after refining the band-limited complex primitive.

    This is required for phase marginalization: interpolating ``abs(kappa)``
    cannot recover intersample structure lost to that nonlinear operation.
    Arrival-time-dependent norms remain unsupported by the bandlimited mode.
    A refined row whose endpoint is within 15 nats of its peak fails closed:
    the even-extension boundary condition is not trustworthy when the finite
    window carries appreciable posterior mass at either turn.
    """
    kappa_t = jnp.asarray(kappa_t, dtype=jnp.complex128)
    rho_sq = jnp.asarray(rho_sq, dtype=jnp.float64)
    coarse = ((jnp.abs(kappa_t) if phase_marginalization else kappa_t.real)
              - 0.5 * rho_sq)
    finite_rows = jnp.all(jnp.isfinite(coarse), axis=-1)
    clean_kappa = jnp.where(finite_rows[:, None], kappa_t, 0.0)
    clean_rho = jnp.where(finite_rows[:, None], rho_sq, 0.0)
    # Probe the primitive at half a sample before deriving curvature.  A
    # near-Nyquist real kappa can alternate +/-A, making coarse ``abs(kappa)``
    # exactly constant even though the continuous phase-marginalized field has
    # a zero between every pair of samples.  No statistic of the coarse
    # nonlinear field can detect that alias.
    probe_kappa = _reflected_fft_upsample(clean_kappa, 2)
    probe_rho = jnp.broadcast_to(clean_rho[:, :1], probe_kappa.shape)
    probe = ((jnp.abs(probe_kappa) if phase_marginalization
              else probe_kappa.real) - 0.5 * probe_rho)
    sigma, measurable = _peak_width_from_lnL_jax(probe, deltaT / 2.0)
    need = jnp.where(measurable & jnp.isfinite(sigma) & (sigma > 0),
                     _TIME_ADAPTIVE_SAFETY * deltaT / sigma, 1.0)
    need = jnp.maximum(need, 1.0)
    factor_float = jnp.exp2(jnp.ceil(jnp.log2(need)))
    factor_float = jnp.where(factor_float < need, factor_float * 2.0, factor_float)
    too_sharp = (~jnp.isfinite(factor_float)) | (
        factor_float > _TIME_ADAPTIVE_FACTOR_MAX)
    factor = jnp.minimum(factor_float, float(_TIME_ADAPTIVE_FACTOR_MAX)).astype(
        jnp.int32)
    powers = tuple(1 << k for k in range(11))

    def make_branch(base):
        def at_factor(kappa, rho, f):
            dense_kappa = _reflected_fft_upsample(kappa, f)
            # Conventional baseline data have a time-independent model norm.
            # Keeping the first value avoids inventing high-frequency structure
            # in a constant primitive through roundoff.
            dense_rho = jnp.broadcast_to(rho[0], dense_kappa.shape)
            dense = ((jnp.abs(dense_kappa) if phase_marginalization
                      else dense_kappa.real) - 0.5 * dense_rho)
            value = _log_trapezoid(dense, deltaT / float(f))
            width, measured = _peak_width_from_lnL_jax(dense, deltaT / float(f))
            resolved = ((~measured) | (~jnp.isfinite(width))
                        | (deltaT / float(f) <= width / _TIME_ADAPTIVE_SAFETY))
            peak = jnp.max(dense)
            endpoint = jnp.maximum(dense[0], dense[-1])
            boundary_ok = endpoint <= peak - _TIME_ENDPOINT_LOG_GAP_MIN
            return value, resolved, boundary_ok

        def branch(args):
            kappa, rho = args
            v0, r0, b0 = at_factor(kappa, rho, base)
            v1, r1, b1 = at_factor(kappa, rho, 2 * base)
            v2, r2, b2 = at_factor(kappa, rho, 4 * base)
            c1 = r1 & b1 & (jnp.abs(v1 - v0) <= _TIME_ADAPTIVE_RTOL)
            c2 = r2 & b2 & (jnp.abs(v2 - v1) <= _TIME_ADAPTIVE_RTOL)
            return jnp.where(c1, v1, jnp.where(c2, v2, jnp.nan))
        return branch

    branches = tuple(make_branch(f) for f in powers)

    def refine_one(args):
        kappa, rho, row_factor = args
        index = jnp.clip(
            jnp.ceil(jnp.log2(row_factor.astype(jnp.float64))).astype(jnp.int32),
            0, len(powers) - 1)
        return jax.lax.switch(index, branches, (kappa, rho))

    # Rematerialize a row's selected branch during reverse mode instead of
    # retaining every dense abs/exp/FFT residual across the sampler batch.
    refined = jax.lax.map(
        jax.checkpoint(refine_one), (clean_kappa, clean_rho, factor))
    refined = jnp.where(too_sharp, jnp.nan, refined)
    return jnp.where(finite_rows, refined, jnp.nan)


def _time_marginalize_terminal(lnL_t, data, time_quadrature=TIME_QUAD_DEFAULT,
                               bandlimited_safe=False):
    """Common terminal selector used by every JAX time-marginalized endpoint."""
    if time_quadrature not in _TIME_QUAD_CHOICES:
        raise ValueError("time_quadrature must be one of %r, got %r"
                         % (_TIME_QUAD_CHOICES, time_quadrature))
    if time_quadrature == "simpson":
        return _time_marginalize(lnL_t, data.w_t)
    if not bandlimited_safe:
        raise ValueError(
            "bandlimited terminal interpolation is invalid after nonlinear "
            "distance/phase/polarization marginalization; use 'simpson'")
    return _time_marginalize_reflected_fft(lnL_t, data.deltaT, data.w_t)


def fused_log_likelihood(data, ra, dec, psi, incl, phiref, distMpc,
                         interp=JAX_INTERP_DEFAULT, phase_marginalization=False,
                         time_quad=TIME_QUAD_DEFAULT,
                         time_upsample=None, time_guard=None,
                         time_quadrature=None, return_lnLt=False):
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
    time_quad : {"simpson", "bandlimited"}
        Time quadrature.  ``"simpson"`` (default) is the unchanged behaviour.
        ``"bandlimited"`` integrates a band-limited reconstruction of kappa(t)
        over the SAME interval; it holds the model norm fixed in time and is
        therefore refused for slow-rotation data, whose ``<h|h>`` depends on the
        template arrival time (see :func:`_time_marginalize_bandlimited`).
    time_upsample : int
        Reconstruction factor for ``time_quad="bandlimited"``; costs no
        likelihood evaluations, so raise it until the answer stops moving.
    time_guard : int or None
        Guard samples per end for ``time_quad="bandlimited"``.  The window is a
        crop, the reconstruction is periodic, and the seam between the two ends
        rings into the integrand; these samples move that seam outside the
        integrated window (see :func:`_time_marginalize_bandlimited`).  Unlike
        ``time_upsample`` they DO cost gathers -- the accumulation runs on
        ``npts + 2*time_guard`` bins -- and the seam error falls off only like
        1/distance, so this is the second knob to raise when checking that the
        answer has stopped moving.  ``None`` takes :func:`default_time_guard`.
        Ignored (and forced to 0) by ``time_quad="simpson"``, which integrates
        the sampled window itself and has no seam.

    Returns
    -------
    lnL : array_like, shape (S,)
    """
    canonical_time_api = time_quadrature is not None
    if canonical_time_api:
        if time_quad != TIME_QUAD_DEFAULT and time_quad != time_quadrature:
            raise ValueError("time_quad and time_quadrature disagree")
        time_quad = time_quadrature
    if time_quad not in _TIME_QUAD_CHOICES:
        # Fail on an unrecognised value rather than silently falling through to
        # the default: a typo'd quadrature name that quietly gives you the OLD
        # behaviour is exactly the silent-no-op pattern this module keeps
        # getting bitten by.
        raise ValueError("time_quad must be one of %r, got %r"
                         % (_TIME_QUAD_CHOICES, time_quad))
    # ``time_quad`` / ``time_upsample`` is the PR-208 low-level API.  Preserve
    # its primitive-kappa behavior for direct callers; wrappers and drivers use
    # the conventional ILE ``time_quadrature`` spelling and the adaptive
    # terminal implementation.
    legacy_primitive_refinement = (time_quad == "bandlimited"
                                   and not canonical_time_api)
    if time_quad == "bandlimited" and _norm_is_arrival_time_dependent(data):
        # Same reason, other direction: the band-limited quadrature reconstructs
        # kappa(t) and holds the model norm at one time bin, so on data whose
        # <h|h> depends on the arrival time it would quietly return a DIFFERENT
        # likelihood rather than a better-integrated one.  Refuse it here instead.
        raise ValueError(
            "time_quad='bandlimited' holds the model norm fixed in time, but this "
            "likelihood data carries the slow-rotation post-phase, whose <h|h> "
            "depends on the template arrival time; use time_quad='simpson' for "
            "rotation data.")
    # Only the band-limited path widens the window; "simpson" integrates the
    # sampled window itself, so it must keep gathering exactly data.npts bins.
    if legacy_primitive_refinement:
        guard = (default_time_guard(data.npts) if time_guard is None
                 else int(time_guard))
    else:
        guard = 0
    distMpc = jnp.asarray(distMpc, dtype=jnp.float64)
    invDist = data.distMpcRef / distMpc
    kappa_unit, rho_sq_unit = _accumulate_unit(
        data, ra, dec, psi, incl, phiref, interp, phase_marginalization,
        guard=guard)
    kappa_sq = kappa_unit * invDist[:, None]
    rho_sq = rho_sq_unit * jnp.square(invDist)[:, None]
    if legacy_primitive_refinement:
        return _time_marginalize_bandlimited(
            kappa_sq, rho_sq, data.deltaT,
            int(_TIME_UPSAMPLE_DEFAULT if time_upsample is None else time_upsample), guard,
            phase_marginalization=phase_marginalization)
    if phase_marginalization:
        lnL_t = jnp.abs(kappa_sq) - 0.5 * rho_sq
    else:
        lnL_t = kappa_sq.real - 0.5 * rho_sq
    if return_lnLt:
        return lnL_t
    if canonical_time_api and time_quad == "bandlimited":
        return _time_marginalize_reflected_primitive(
            kappa_sq, rho_sq, data.deltaT,
            phase_marginalization=phase_marginalization)
    return _time_marginalize_terminal(
        lnL_t, data, time_quad, bandlimited_safe=not phase_marginalization)


def fused_log_likelihood_distmarg(data, ra, dec, psi, incl, phiref,
                                  x_grid, log_w_grid,
                                  interp=JAX_INTERP_DEFAULT, phase_marginalization=False,
                                  grid_block=64,
                                  time_quadrature=TIME_QUAD_DEFAULT,
                                  return_lnLt=False):
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
    if return_lnLt:
        return lnL_t
    return _time_marginalize_terminal(lnL_t, data, time_quadrature)


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


def make_distance_gh(n_nodes, n_sigma=7.0):
    """Per-sample distance-quadrature node OFFSETS (in units of the Gaussian
    width 1/sqrt(R)) and their spacing, for :func:`_distmarg_gh_logL`.

    Nodes are placed at ``x_k = x* + z_k / sqrt(R)`` with ``x* = K/R`` the
    Gaussian peak, so the (1/SNR)-narrow high-SNR distance peak is resolved at
    any SNR.  Uniform ``z_k`` over ``[-n_sigma, n_sigma]`` (±7σ captures the
    Gaussian to ~1e-11) -- the trapezoid rule converges *exponentially* on a
    Gaussian (Euler-Maclaurin), so ~5-10 nodes/σ is effectively exact.  Returns
    ``(z, dz)``; ``dz`` is unused by the (clip-aware) weight computation but kept
    for call-site signature stability.

    (Historically Gauss-Hermite abscissae -- hence the name -- but the GH form
    needed an explicit 0.5*K^2/R prefactor whose gradient blows up for small-R
    noise time-bins.  The uniform-offset + full-exponent form below is
    gradient-stable; see :func:`_distmarg_gh_logL`.)
    """
    z = np.linspace(-float(n_sigma), float(n_sigma), int(n_nodes))
    dz = float(z[1] - z[0]) if len(z) > 1 else 1.0
    return jnp.asarray(z, jnp.float64), jnp.asarray(dz, jnp.float64)


def _distmarg_gh_logL(K, R, z_off, _dz_unused, x_min, x_max):
    """Per-(S,npts) distance-marginalized lnL via a per-sample ADAPTIVE trapezoid.

    Computes ``log E_{p(d)}[exp(K x - 0.5 R x^2)]`` with x = dref/d and the
    volumetric prior p(d) ∝ d^2 normalized over [d_min, d_max] (the "proper
    distance average", identical to :func:`make_distance_grid`), but with
    quadrature nodes centred per-sample on the Gaussian peak x*=K/R, scale
    1/sqrt(R) -- so the (1/SNR)-narrow high-SNR distance peak that a fixed
    uniform-in-d grid under-resolves (biasing the average ~1% low) is resolved at
    EVERY SNR.

    GRADIENT-STABLE FORM (replaces the earlier analytic-prefactor Gauss-Hermite
    version, whose 0.5*K^2/R term had a ~K^2/R^2 gradient that blew up for
    small-R noise time-bins -> nan MALA/MAP-polish draws).  The node positions are
    placed under ``stop_gradient`` and the integrand is evaluated in the SAME
    stable form the uniform grid uses:

        log E[L] = C0 + logsumexp_k[ K x_k - 0.5 R x_k^2 - 4 ln x_k + ln(dx_k) ]
        C0 = ln 3 - ln(x_min^{-3} - x_max^{-3})        (dref-independent prior norm)

    so the gradient flows ONLY through the bounded ``K*x_k`` and ``-0.5*R*x_k^2``
    terms (x_k frozen) -- no 1/R, no nan.  Nodes are clipped to the physical
    support [x_min, x_max]; clipped (zero-width) nodes drop out via their dx=0
    weight, and fully-out-of-support rows are made gradient-safe and overridden
    to -inf.

    Parameters
    ----------
    z_off : (G,)  standard node offsets in units of 1/sqrt(R) (from make_distance_gh)
    _dz_unused : kept for call-site signature compatibility (weights use clipped dx)
    """
    R = jnp.maximum(R, 1e-30)
    # Node CENTRE = Gaussian peak x*=K/R, but clipped into the physical support so
    # bins whose peak lies outside [x_min,x_max] (template rails to d_min/d_max:
    # the amplitude/distance-degeneracy slivers) still get their boundary-dominated
    # integral resolved (matching the uniform grid) instead of an all-clipped -inf.
    # Width = 1/sqrt(R).  Both FROZEN: the node placement carries no gradient.
    center = jax.lax.stop_gradient(jnp.clip(K / R, x_min, x_max))   # (S,npts)
    sigma = jax.lax.stop_gradient(1.0 / jnp.sqrt(R))               # (S,npts)
    x_k = center[..., None] + sigma[..., None] * z_off            # (S,npts,G), monotone in k
    x_k = jax.lax.stop_gradient(jnp.clip(x_k, x_min, x_max))
    # composite-trapezoid node weights from adjacent spacing (nodes increasing in k)
    dx = jnp.diff(x_k, axis=-1)                                # (S,npts,G-1)
    w = jnp.concatenate([0.5 * dx[..., :1],
                         0.5 * (dx[..., 1:] + dx[..., :-1]),
                         0.5 * dx[..., -1:]], axis=-1)          # (S,npts,G)
    pos = w > 0                                                # live (non-clipped) nodes
    log_w = jnp.where(pos, jnp.log(jnp.where(pos, w, 1.0))
                      - 4.0 * jnp.log(x_k), -jnp.inf)          # trapz dx * x^{-4} prior
    expo = K[..., None] * x_k - 0.5 * R[..., None] * jnp.square(x_k) + log_w
    any_w = jnp.any(pos, axis=-1)                              # (S,npts)
    # all-clipped rows: dummy expo to finite so the backward pass is finite, then
    # override to -inf below (both where-branches finite -> finite gradient).
    expo = jnp.where(any_w[..., None], expo, 0.0)
    lse = jax.scipy.special.logsumexp(expo, axis=-1)           # (S,npts)
    C0 = jnp.log(3.0) - jnp.log(x_min ** (-3.0) - x_max ** (-3.0))
    return jnp.where(any_w, C0 + lse, -jnp.inf)


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
                                  phi_grid, interp=JAX_INTERP_DEFAULT,
                                  time_quadrature=TIME_QUAD_DEFAULT,
                                  return_lnLt=False):
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
    if return_lnLt:
        return lnL_t_marg
    return _time_marginalize_terminal(lnL_t_marg, data, time_quadrature)


def fused_log_likelihood_distphimarg(data, ra, dec, psi, incl,
                                      x_grid, log_w_grid,
                                      phi_grid, interp=JAX_INTERP_DEFAULT,
                                      grid_block=64,
                                      time_quadrature=TIME_QUAD_DEFAULT,
                                      return_lnLt=False,
                                      return_phi_lnLt=False):
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
        return (m_new, s_new), (lnL_t if return_phi_lnLt else None)

    m0 = jnp.full((S, data.npts), -jnp.inf, dtype=jnp.float64)
    s0 = jnp.zeros((S, data.npts), dtype=jnp.float64)
    (m, s), lnL_phi_t = jax.lax.scan(_phi_step, (m0, s0), phi_grid_jax)

    lnL_t_marg = m + jnp.log(s) - jnp.log(nphi)
    if return_phi_lnLt:
        return lnL_phi_t
    if return_lnLt:
        return lnL_t_marg
    return _time_marginalize_terminal(lnL_t_marg, data, time_quadrature)


def psi_grid(npsi: int) -> np.ndarray:
    """Uniform grid of polarization angle psi over [0, pi), shape (npsi,).

    psi enters at spin-weight 2 (the antenna patterns rotate as cos2psi, sin2psi),
    so the likelihood has period pi in psi and a low trig order; the grid average
    of exp(lnL) converges exponentially.  16 points is ample for l_max=2.
    """
    return np.linspace(0.0, np.pi, npsi, endpoint=False)


def fused_log_likelihood_distphipsimarg(data, ra, dec, incl,
                                        x_grid, log_w_grid, phi_grid, psi_grid_,
                                        interp=JAX_INTERP_DEFAULT, grid_block=64,
                                        time_quadrature=TIME_QUAD_DEFAULT,
                                        return_lnLt=False):
    """Distance-, phi_ref- AND psi-marginalized factored lnL over (ra, dec, incl).

    Marginalizes luminosity distance (quadrature grid), orbital phase phi_ref and
    polarization psi (uniform grid sums) -> a smooth 3-D function of (ra, dec, incl).
    Removing psi (the spin-2 polarization) integrates out the dimension most
    entangled with distance/inclination, stabilizing the distance integral and
    leaving a lower-dimensional, better-conditioned target for the flow.
    """
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)
    phi_g = jnp.asarray(phi_grid, dtype=jnp.float64)
    psi_g = jnp.asarray(psi_grid_, dtype=jnp.float64)
    S = ra.shape[0]
    a = x_grid
    b = -0.5 * jnp.square(x_grid)
    # flatten the (phi, psi) grid into one scan sequence (sequential -> O(body) mem)
    PHI, PSI = jnp.meshgrid(phi_g, psi_g, indexing="ij")
    pairs = jnp.stack([PHI.reshape(-1), PSI.reshape(-1)], axis=-1)   # (nphi*npsi, 2)
    npair = pairs.shape[0]
    _use_gh = _DISTMARG_GH_N > 0
    if _use_gh:
        gh_xi, gh_logw = make_distance_gh(_DISTMARG_GH_N)
        x_min = jnp.min(x_grid); x_max = jnp.max(x_grid)

    def _step(carry, pair):
        m, s = carry
        phi_arr = jnp.broadcast_to(pair[0], (S,)).astype(jnp.float64)
        psi_arr = jnp.broadcast_to(pair[1], (S,)).astype(jnp.float64)
        kappa_unit, rho_sq_unit = _accumulate_unit(
            data, ra, dec, psi_arr, incl, phi_arr, interp, False)
        if _use_gh:
            lnL_t = _distmarg_gh_logL(kappa_unit.real, rho_sq_unit,
                                      gh_xi, gh_logw, x_min, x_max)
        else:
            lnL_t = _logsumexp_grid_blocked(
                kappa_unit.real, rho_sq_unit, a, b, log_w_grid, grid_block)
        m_new = jnp.maximum(m, lnL_t)
        s_new = s * jnp.exp(m - m_new) + jnp.exp(lnL_t - m_new)
        return (m_new, s_new), None

    m0 = jnp.full((S, data.npts), -jnp.inf, dtype=jnp.float64)
    s0 = jnp.zeros((S, data.npts), dtype=jnp.float64)
    # Gradient checkpoint the scan body: the (phi,psi) grid is nphi*npsi steps
    # (e.g. 512); without remat, reverse-mode (MALA / value_and_grad) stores every
    # step's intermediates -> O(nstep) memory -> OOM.  remat recomputes in the
    # backward pass -> memory O(1) in the grid size.
    (m, s), _ = jax.lax.scan(jax.checkpoint(_step), (m0, s0), pairs)
    lnL_t_marg = m + jnp.log(s) - jnp.log(npair)
    if return_lnLt:
        return lnL_t_marg
    return _time_marginalize_terminal(lnL_t_marg, data, time_quadrature)


def fused_log_likelihood_distpsimarg(data, ra, dec, phiref, incl,
                                     x_grid, log_w_grid, psi_grid_,
                                     interp=JAX_INTERP_DEFAULT, grid_block=64,
                                     time_quadrature=TIME_QUAD_DEFAULT,
                                     return_lnLt=False):
    """Distance- AND psi-marginalized factored lnL over (ra, dec, phi_ref, incl).

    Marginalizes luminosity distance (quadrature grid) and polarization psi
    (uniform spin-2 grid sum) while keeping phi_ref a SAMPLED parameter -> a 4-D
    target (ra, dec, phi_ref, incl).  Cheaper than phi_ref marginalization (psi
    needs ~8 grid points vs phi's ~32, so the scan is short), the psi integral
    stabilizes the distance integral, and integrating out psi still breaks the
    phi_ref-psi degeneracy ridge.
    """
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)
    psi_g = jnp.asarray(psi_grid_, dtype=jnp.float64)
    S = ra.shape[0]
    a = x_grid
    b = -0.5 * jnp.square(x_grid)
    npsi = psi_g.shape[0]
    _use_gh = _DISTMARG_GH_N > 0
    if _use_gh:
        gh_xi, gh_logw = make_distance_gh(_DISTMARG_GH_N)
        x_min = jnp.min(x_grid); x_max = jnp.max(x_grid)

    def _psi_step(carry, psi_val):
        m, s = carry
        psi_arr = jnp.broadcast_to(psi_val, (S,)).astype(jnp.float64)
        kappa_unit, rho_sq_unit = _accumulate_unit(
            data, ra, dec, psi_arr, incl, phiref, interp, False)
        if _use_gh:
            lnL_t = _distmarg_gh_logL(kappa_unit.real, rho_sq_unit,
                                      gh_xi, gh_logw, x_min, x_max)
        else:
            lnL_t = _logsumexp_grid_blocked(
                kappa_unit.real, rho_sq_unit, a, b, log_w_grid, grid_block)
        m_new = jnp.maximum(m, lnL_t)
        s_new = s * jnp.exp(m - m_new) + jnp.exp(lnL_t - m_new)
        return (m_new, s_new), None

    m0 = jnp.full((S, data.npts), -jnp.inf, dtype=jnp.float64)
    s0 = jnp.zeros((S, data.npts), dtype=jnp.float64)
    # remat: cheap insurance (psi grid is small, but keeps gradient memory O(1)).
    (m, s), _ = jax.lax.scan(jax.checkpoint(_psi_step), (m0, s0), psi_g)
    lnL_t_marg = m + jnp.log(s) - jnp.log(npsi)
    if return_lnLt:
        return lnL_t_marg
    return _time_marginalize_terminal(lnL_t_marg, data, time_quadrature)


def phi_ref_conditional_lnL(data, ra, dec, psi, incl, distMpc,
                              phi_grid, interp=JAX_INTERP_DEFAULT,
                              time_quadrature=TIME_QUAD_DEFAULT):
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
        return None, _time_marginalize_terminal(lnL_t, data, time_quadrature)

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


def estimate_distance_peak(data, guess_snr=None, n_sky=4000, seed=0, interp=JAX_INTERP_DEFAULT):
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
    s_best = int(np.unravel_index(int(np.argmax(snr2)), snr2.shape)[0])
    # Refine: the best RANDOM sky point has wrong incl/psi -> wrong d_peak.
    # Gradient-ascend the peak matched-amplitude max_t 0.5 K^2/R over the 5 angles
    # to the true MAP (correct incl/psi/sky), so x*=K/R there gives the right peak.
    lo5 = np.array([0.0, -np.pi / 2 + 1e-3, 0.0, 1e-3, 0.0])
    hi5 = np.array([2 * np.pi, np.pi / 2 - 1e-3, np.pi - 1e-3, np.pi - 1e-3, 2 * np.pi])

    def _peak(th5):
        k, r = _accumulate_unit(data, th5[0:1], th5[1:2], th5[2:3],
                                th5[3:4], th5[4:5], interp, False)
        return jnp.max(0.5 * (k.real ** 2) / jnp.maximum(r, 1e-30))

    gfun = jax.jit(jax.grad(_peak))
    th = np.clip(np.array([ra[s_best], dec[s_best], psi[s_best],
                           incl[s_best], phiref[s_best]]), lo5, hi5)
    step = 1e-3
    v_prev = float(_peak(jnp.asarray(th)))
    for _ in range(300):
        g = np.asarray(gfun(jnp.asarray(th)))
        if not np.all(np.isfinite(g)):
            break
        th_new = np.clip(th + step * g, lo5, hi5)
        v_new = float(_peak(jnp.asarray(th_new)))
        if v_new >= v_prev:
            th, v_prev, step = th_new, v_new, step * 1.2
        else:
            step *= 0.5
            if step < 1e-9:
                break
    kk, rr = _accumulate_unit(data, th[0:1], th[1:2], th[2:3], th[3:4],
                              th[4:5], interp, False)
    Kt = np.asarray(kk.real)[0]
    Rt = np.maximum(np.asarray(rr)[0], 1e-30)
    snr2_t = np.where(Kt > 0.0, Kt * Kt / Rt, -np.inf)
    tb = int(np.argmax(snr2_t))
    Kb, Rb = float(Kt[tb]), float(Rt[tb])
    x_star = Kb / Rb
    rho_mf = np.sqrt(Kb * Kb / Rb)
    d_peak = dref / x_star
    sigma_d = d_peak / max(rho_mf, 1.0)
    return d_peak, sigma_d


def make_distance_grid_adaptive(d_min, d_max, d_peak, sigma_d, d_prior="euclidean",
                                distMpcRef=DIST_MPC_REF, n_fine_max=160, n_coarse=48,
                                n_sigma=12.0, oversample=4.0):
    """Non-uniform distance grid: fine near the (SNR-set) peak, coarse on the tail.

    Concentrates resolution where the distance posterior lives while staying
    robust to a mis-located ``d_peak`` estimate: the fine region spans a
    *multiplicative* range ``[d_peak/range_factor, d_peak*range_factor]`` (so a
    peak estimate off by a few x is still covered), with spacing ~ sigma_d/oversample
    so the true peak is resolved; the point count is sized to that and capped at
    ``n_fine_max``.  A coarse full-range backbone keeps off-peak samples supported.
    Trapezoidal weights normalized to a *proper distance average* (same convention
    as :func:`make_distance_grid`) -> drop-in for the stable logsumexp kernel, and
    being a *static* grid it is gradient-stable (unlike per-sample Gauss-Hermite).

    LIMITATION (found in validation): a single static window is INSUFFICIENT here
    because of the distance-inclination degeneracy -- the matched SNR rho_mf is
    nearly constant across many (incl, psi) at DIFFERENT best-fit distances, so the
    marginal distance posterior is broad (~factor 2-3), not one narrow peak.  A
    window centred on one d_peak (e.g. 80 Mpc) misses the face-on tail (~200 Mpc);
    covering the whole degeneracy at sigma_d resolution needs ~3*SNR points -> OOM
    on an 11GB card.  CORRECT FIX = PER-SAMPLE adaptive nodes in the stable
    logsumexp(K x - 0.5 R x^2 + log_w_quad) form, with node positions x_k =
    stop_gradient(K/R + z_k/sqrt(R)) so each sample resolves its OWN peak with ~32
    nodes (8x LESS memory than a 256 static grid) and the 1/R only enters through
    stop_gradient -> gradient-stable.  That is a kernel change (TODO).
    """
    half = n_sigma * sigma_d                       # additive: peak is well-located
    d_lo = max(float(d_min), d_peak - half)
    d_hi = min(float(d_max), d_peak + half)
    if not (d_hi > d_lo) or not (sigma_d > 0):    # degenerate -> uniform fallback
        return make_distance_grid(d_min, d_max, n_fine_max + n_coarse, d_prior, distMpcRef)
    n_fine = int(np.clip((d_hi - d_lo) / (sigma_d / float(oversample)),
                         32, int(n_fine_max)))
    fine = np.linspace(d_lo, d_hi, n_fine)
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


def make_log_likelihood(data, interp=JAX_INTERP_DEFAULT, phase_marginalization=False,
                        jit=True):
    """Return a closure ``f(ra, dec, psi, incl, phiref, distMpc) -> lnL``.

    The returned function closes over ``data`` (treated as constant) and is, by
    default, ``jax.jit``-compiled.  It is differentiable with respect to all six
    extrinsic arguments for any INTERPOLATING stencil -- ``linear``, ``cubic`` or
    ``sinc`` -- but NOT for ``nearest``, whose gather is piecewise constant in the
    arrival time and therefore has zero gradient through the sky.  Combine with
    ``jax.grad`` / ``jax.value_and_grad`` / ``jax.vmap`` as needed.
    """
    def f(ra, dec, psi, incl, phiref, distMpc):
        return fused_log_likelihood(
            data, ra, dec, psi, incl, phiref, distMpc,
            interp=interp, phase_marginalization=phase_marginalization)
    return jax.jit(f) if jit else f
