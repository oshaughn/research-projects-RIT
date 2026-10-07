# O4c bandlimited time quadrature

This backport replaces PR 182's August raw-window FFT implementation with the
numerical helper from `rift_O4d` commit
`94840482237b9324553dff8cecc87001cf19560d`. It preserves current O4c's per-row
log offsets, nearest default, and optional cubic detector-time interpolation.
Simpson remains the default. The integration domain remains the closed span
between the first and last gathered sample, at spacing `deltaT`.

The helper reflects `[kappa, reverse(kappa)]` before Fourier refinement.
Unlike treating the original cropped window as a complete period, this avoids
identifying its generally unequal endpoints. The boundary count is diagnostic;
peaks do not silently switch back to an under-resolved Simpson rule because
of their location. Flat, already resolved, and unmeasurable rows use the
caller's Simpson implementation with safe per-row offsets. Dense remeasurement
refines only rows still below the measured-width resolution criterion.

Reflection is a numerical boundary condition, not an exact reconstruction of
the unavailable full correlation or a certified physical-error bound. The
retained-grid FFT is an optimization of the same reflected Fourier polynomial;
its declines retry full padding and report their provenance.

The matched regression suite includes a centered nonperiodic counterexample
where raw periodization errs by over 100 nats and reflected reconstruction
agrees with a direct analytic reference to within 0.001 nat. It also covers
odd windows, nonlinear callbacks, phase marginalization, boundary peaks,
nonfinite rows, per-row refinement, memory chunks, and optimization fallback.
These are deterministic regression tolerances, not universal accuracy claims.

RIFT_roboto_paper's later operational protocol used bandlimited quadrature at
AV batch size 40000, including the Q-pregrid-8 plus cubic configuration. Its
independent time oracle separates detector-time stencil error from quadrature
error. O4c has neither that pregrid nor the modern sinc stencil; those paper
results must not be relabeled as an O4c calibration. Cubic remains available,
with a startup notice that nearest-stencil accuracy comparisons do not
establish its advantage. No production default changes here.

Hand-written CPU commands must pass `--time-marginalization --vectorized
--gpu --force-xpy`. The driver validates after GPU fallback, refuses requests
that cannot reach NoLoop, and refuses `--zero-likelihood`. Production `--resample-time-marginalization --fairdraw-extrinsic-output`
arguments work with either quadrature. Under bandlimited they draw continuously
from the same reflected, width-validated density reconstruction rather than
spline-interpolating coarse log likelihoods. The existing
`--srate-resample-time-marginalization` is accepted as a minimum knot resolution;
width validation may require a finer representation, and output draws have no
lattice. A separate export refinement pass leaves marginal integrals unchanged.
`lnL_raw` is the log of the piecewise-linear density at the drawn time, an
approximation converging with the validated nodal spacing. Exact integer GPS
seconds/nanoseconds serialize the draw at nanosecond precision; `t_ref` remains
a float compatibility view. Simpson export behavior remains unchanged.
Library `return_lnLt` retains its existing coarse-timeseries contract.

Run the three named time-quadrature test files plus
`test_time_marginalization_perrow_offset.py` and `test_noloop_time_interp.py`.
They are registered in GitLab unit tests and the GitHub O4c workflow. GPU
parity and production cost/accuracy still require an appropriate CUDA host;
a CPU pass must not be reported as a real-device validation.
