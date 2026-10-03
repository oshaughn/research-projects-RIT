Native exact GPyTorch interpolation
==================================

``--fit-method gp-torch`` uses the optional PyTorch/GPyTorch stack. It fits the
native coordinate array already selected by CIP; it does not change waveform,
likelihood, priors, or integration coordinates. Install ``torch`` and
``gpytorch`` in the intended environment; a CUDA-capable PyTorch installation
is required to use an NVIDIA GPU.

Behavior
--------

* The training coordinates and targets are standardized once. Stored scalers
  also transform every later query; caller arrays are never changed.
* Calculations use float64 on CPU or CUDA. ``--gp-torch-device auto`` selects
  CUDA when available, otherwise CPU. Explicit ``cpu`` or ``cuda:0`` selection
  is available. MPS is not supported because this backend requires float64.
* The kernel is an ARD Matérn-5/2 with learned amplitude and zero prior mean in
  normalized target units. Lengthscale, amplitude and additional-noise bounds
  follow the independent Matérn diagnostic conventions: [0.03, 30], [0.01, 100]
  and [1e-5, 0.1], respectively. Initial values are 1, 1 and 1e-3.
* Per-point ``y_errors`` are standard deviations in the original target units.
  The observation covariance includes ``(y_errors / target_std)**2`` (with a
  numerical floor of 1e-10) plus learned diagonal noise. The returned mean is
  the latent-function posterior mean, with no observational noise added to
  cross-covariances.
* Prediction is deterministic. A cached exact training solve is multiplied by
  bounded query-by-training kernel blocks; no query covariance or random GP
  sample is formed. Direct pairwise distances avoid query-batch centering and
  squared-norm cancellation. Only NumPy arrays of posterior means are returned.
* ``--fit-uncertainty-added`` is explicitly unsupported for this mean-only
  backend, rather than silently changing or ignoring the requested operation.

Resource bounds and fitting
---------------------------

Exact fitting requires O(N**3) work and O(N**2) storage. It is **not** a full-grid
scalable approximation. The default ``--gp-torch-max-train-points 8000`` raises
an error before model construction if more points survive the native cuts.
Choose a subset deliberately with the existing ``--cap-points`` option, or
explicitly raise the maximum after assessing resources. No implicit truncation
is performed by the wrapper. ``--gp-torch-epochs`` defaults to 60 in CIP;
``--gp-torch-batch-size`` defaults to 1024 queries per prediction block.

CIP's existing ``--cap-points`` selection is random. It is not the balanced
training subset used by the independent event diagnostic, and this repair does
not import or reproduce that validated event model. An event posterior claim
requires held-out validation of its chosen training subset, hyperparameters and
integration settings. The corrected backend alone does not establish a
corrected posterior.

Save and load
-------------

``--fit-save-gp fit`` writes ``fit.pt``; an existing ``.pt`` suffix is preserved.
``--fit-load-gp fit.pt`` restores the GP without fitting. CIP still loads and
converts its data as for other GP backends. Saved state includes normalized
training data, known variances, model parameters, input/target scalers,
coordinate names, training hashes, loss history, software versions and the
native likelihood shift. Loading verifies coordinate order and adjusts the
returned likelihood for a different current ``lnL_shift``. The checkpoint
contains tensors and basic metadata and is loaded with ``weights_only=True``;
it does not pickle an executable model object. Loading onto CPU from a CUDA
checkpoint is supported. Prediction caches are recomputed after load.

The wrapper API preserves ``Interpolator(input, target, epochs=...)``,
``train()`` and ``evaluate(input)``. New optional keyword arguments include
``y_errors``, ``device``, ``prediction_batch_size``, ``max_train_points``,
``feature_names`` and ``provenance``. ``save(path)`` writes the exact supplied
path, and ``Interpolator.load(path, ...)`` restores it. The old stochastic,
query-rescaled implementation did not support checkpoint export; its outputs
are not compatible model checkpoints.

Verification
------------

Run the optional-stack regression suite without LAL or physics jobs::

  python -m pytest MonteCarloMarginalizeCode/Code/test/test_gpytorch_wrapper.py -q

The tests compare the predicted mean to an independent dense Matérn posterior
formula, including training/self queries and a zero-standardized query; test
batch/permutation/distractor invariance, repeated determinism, immutable inputs,
normalized heteroscedastic weighting, cache invalidation, bounded kernel blocks,
checkpoint round trips, coordinate checks, and actual native CLI hook save/load
and likelihood-shift behavior. CUDA training/prediction and CPU reload tests run
only when CUDA is available. The full LAL-dependent CIP entrypoint and event
posterior integration require the production environment and are separate
validation steps.

Accelerating an already validated sklearn fit
--------------------------------------------

A separate opt-in path preserves the fitted StandardScaler/Matérn-5/2 model
(including its solved coefficients and target normalization) while evaluating
bounded float64 kernel blocks with CuPy::

    --fit-method gp --fit-load-gp validated-model.pkl \
    --gp-predict-backend cupy --gp-predict-batch-size 4096

This performs no refit and does not change the CIP sampler or prior. It accepts
only a single-target ConstantKernel * Matern(nu=2.5), with optional additive
WhiteKernel, either directly or in a two-step StandardScaler/GP pipeline.
Unsupported kernels, covariance requests and uncertainty-added fits fail
explicitly. CuPy is imported and GPU state allocated only at first prediction.
CUDA allocation/runtime compatibility must be verified for the selected worker;
CPU parity does not certify GPU execution or a posterior.

The frozen S250628am GP2400/4800 model comparison gives maximum CPU mean
prediction differences of 4.18e-11 and 6.46e-11 on 4608 inputs each. Actual CUDA
accuracy and end-to-end speed are being measured in a separate Condor benchmark.
No active scientific workers were changed to this backend during implementation.
