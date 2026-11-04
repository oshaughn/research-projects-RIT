Differentiable GP likelihood export and validation
===================================================

RIFT can export its intrinsic likelihood as a portable, **differentiable**
``lnL(theta)`` surrogate (pure JAX, ``jax.grad``-able, no RIFT dependency at load
time), for downstream use such as population inference and gradient-based samplers.
This page summarizes how the surrogate is built and, importantly, how we **test**
that it is accurate enough *for inference* against RIFT's production random-forest
(RF) fit.

The optional subpackage is :mod:`RIFT.interpolators.jax_gp` (requires the JAX stack:
``jax``, ``numpyro``, ``tinygp``; it is never imported on the default CIP path).

Surrogate
---------

``quadgp`` = a quadratic **Fisher core** (posterior-weighted quadratic, exact local
curvature on the sharp eigen-directions) **plus a GP residual** on ``lnL - Q``. A
single stationary GP cannot match a razor-sharp ``lnL`` peak to PE precision; the
quadratic captures the sharp curvature and the GP fits the smooth remainder. Both are
pure JAX, so the export is differentiable. The model is fit in RIFT's decorrelated CIP
fit coordinates.

Building the fit (RF as an on-support oracle)
---------------------------------------------

RF is RIFT's robust, evaluate-anywhere fit; treat it as an **accuracy oracle on its
support** (where the ILE points are). Principles:

- **On-support only.** Design and validate where the oracle is valid — the real
  high-``lnL`` points (peak-outward) — not a Gaussian-ellipsoid guess that wanders
  off-support where RF extrapolates blockily.
- **Average away MC noise.** Use the oracle's smoothed value (or a noise-modelled fit)
  as the target, not the raw per-point ``lnL``.
- **Two-tier dynamic range.** Fit accurately over the posterior region *plus a buffer*
  (strong external mass/spin priors can pull the effective posterior into our tail).
  Beyond it, no accuracy is needed but the surrogate must return smoothly toward zero —
  enforced by far-field residual-zero anchors (no ringing / phantom tail features).
- **Boundaries.** A general per-coordinate boundary + special-locus framework; equal
  mass and zero spin default on for BNS/aligned cases.

Sampling
--------

The posterior is a razor-thin ridge in chirp mass. ``--sampler nuts-mu`` runs NUTS in
the low-level coordinates with its dense mass matrix **preconditioned** by a covariance
built in the well-conditioned decorrelated (mu) frame and pulled back — so NUTS mixes
and explores the weakly-constrained directions by construction (unlike importance
sampling, which is proposal-limited).

Testing it
----------

Two questions, of which the second is decisive:

#. **Surface agreement** (necessary, but an over-strict proxy): relative ``lnL`` error
   over the dynamic range, on a leave-some-out split. RF and GP make correlated
   smoothing errors near the sharp peak, so raw surface RMS overstates the problem.
#. **Posterior agreement** (the metric that matters): the GP posterior vs the
   production RF+AV benchmark, by Jensen-Shannon divergence per marginal. On the BNS
   GW170817 test the marginals agree to JS ~ 0.004-0.05 bits, with the sharp chirp-mass
   width reproduced to ~5 %.

A runnable, reproducible test (fixed committed data, ``make`` targets, and the two
figures) lives at
``MonteCarloMarginalizeCode/Code/demo/rift/export_likelihoods/head_to_head/``; the
create/validate export demo and the full-pipeline GP-vs-standard ladder are in the
parent ``export_likelihoods/`` directory.

.. note::

   These results are tested-so-far on one BNS example and are revisited as code and
   settings change. Surface RMS is correlated and sharp-peak-dominated, so gate
   conclusions on the posterior, not the surface.
