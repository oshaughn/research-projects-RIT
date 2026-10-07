# O4c bandlimited time quadrature

This backport replaces PR 182's August raw-window FFT implementation with the
numerical helper from `rift_O4d` commit
`94840482237b9324553dff8cecc87001cf19560d`. It preserves current O4c's per-row
log offsets and the historical nearest detector-time default with Simpson.
Simpson remains the default quadrature. For driver calls selecting bandlimited,
an omitted `--interpolate-time` selects cubic fractional detector-time evaluation.
Explicit nearest/cubic and legacy booleans remain authoritative: in particular,
`--interpolate-time nearest` or `False` reproduces the nearest-bin configuration.
The library's own nearest defaults remain unchanged. The integration domain remains the closed span
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
results must not be relabeled as an O4c calibration. A separate full-precut
frequency oracle on 32 S250114ax fixed-intrinsic reference extrinsics at internal
16384 Hz found nearest-origin time-marginal errors up to 3.32 nat and conditional
quantile shifts up to 30.1 microseconds. Existing cubic plus bandlimited reduced
physical-arrival integral error below 0.002 nat on these rows, with conditional
CDF error below 0.00491 and quantile error below 0.536 microseconds. This supports
the new-mode driver default, not a universal calibration or a completed posterior
recovery claim. Selecting bandlimited alone therefore enables cubic; existing
Simpson runs and explicit stencil choices retain their behavior. Startup and
help text distinguish the quadrature-dependent default from explicit choices.
Nearest-matched comparisons cannot establish physical sky/time accuracy.

Hand-written CPU commands must pass `--time-marginalization --vectorized
--gpu --force-xpy`. The driver validates after GPU fallback, refuses requests
that cannot reach NoLoop, and refuses `--zero-likelihood`. Production `--resample-time-marginalization --fairdraw-extrinsic-output`
arguments work with either quadrature. Under bandlimited they draw continuously
from the same reflected, width-validated density reconstruction rather than
spline-interpolating coarse log likelihoods. The existing
`--srate-resample-time-marginalization` is accepted as a minimum knot resolution;
width validation may require a finer representation, and output draws have no
lattice. The draw-only pass uses a separate measured-width requirement
`h <= sigma_t / 16`, remeasured on the actual dense grid and doubled if needed.
The integral retains `h <= sigma_t / 2`; the export pass discards its integral,
so AV weights remain bit-identical. These safety factors serve different errors:
piecewise-linear density interpolation converges more slowly than quadrature.
On 32 accepted S250114ax candidate rows, export safety 16 selected factor 64,
reducing fixed-draw instantaneous error against full physical Fourier evaluation
from 0.1233 to 0.00237 nat; controlled quantile errors fell below 7.7 ns. This is
an event-specific engineering audit, distinct from the paper's 0.02 nat
**time-marginal** row criterion. A warm 1000-row Blackwell GPU export took
0.83 s versus 0.091 s at factor 8, with unchanged integral bytes. The bounded
dense chunk budget and refinement ceiling apply to both passes. A requested
minimum rate remains an additional floor, never an output time lattice.
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

GPU bandlimited calls partition extrinsic likelihood rows before allocating the
coarse kappa/rho/time arrays. This bounds implementation workspace independently
of the sampler proposal batch: production `--n-chunk` still controls the sampler.
The conservative planner caps coarse work at 1 GiB and 4096 rows, with an
additional available-device-memory allowance. Dense reconstruction retains its
own chunk budget. It does not flush the device memory pool or change other jobs.
Simpson, CPU default batching, and coarse `return_lnLt` dispatch are unchanged.
Continuous-draw uniforms are generated once in input order and sliced across
row groups, preserving RNG and per-row conditional results. This budget is not
a total VRAM guarantee for arbitrary callbacks or an extreme single refined row.
The need was exposed by the matched S250114ax IR1 production batch74287 at
internal16384 Hz: unbounded bandlimited coarse arrays exhausted a24GB GPU while
other processes occupied4GB. Those failed attempts are not recovery evidence.
