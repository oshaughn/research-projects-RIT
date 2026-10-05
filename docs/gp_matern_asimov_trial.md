# Opt-in native Matérn GP CIP trial

Use `MonteCarloMarginalizeCode/Code/test/asimov_integration/blueprints/analysis_rift_SEOBNRv5PHM_gp_matern.yaml` as a **dedicated new production's sampler configuration**. Retain the event's existing waveform, data, likelihood settings, calibration treatment, and physical priors. This example does not select an event or submit a run. Use a runtime containing this source revision and compatible NumPy, SciPy, scikit-learn, LAL, and CuPy; an existing container's RIFT installation does not gain the new method automatically.

The opt-in ledger fields are:

| `sampler.cip` field | Native option |
| --- | --- |
| `fitting method: gp-matern` | pseudo-pipe `--cip-fit-method gp-matern` |
| `prediction backend: cupy` | CIP `--gp-predict-backend cupy` |
| `gp matern max train points: 4800` | CIP `--gp-matern-max-train-points 4800` |
| `gp matern optimizer maxiter: 25` | CIP `--gp-matern-optimizer-maxiter 25` |
| `gp matern seed: 25062842` | CIP `--gp-matern-seed 25062842` |
| `av stop metric: kish` | CIP `--av-stop-metric kish` |
| `sampling method: AV` | pseudo-pipe `--cip-sampler-method AV` |
| `explode jobs: 8` / `explode jobs auto: False` | fixed eight CIP sampling workers per stage |
| `request memory: 8192` | pseudo-pipe `--internal-cip-request-memory 8192` (MB) |
| `request gpus: 1` | pseudo-pipe `--internal-cip-request-gpus 1` |
| `require gpus: "Capability >= 6.0 && Capability < 9.0"` | pseudo-pipe `--internal-cip-require-gpus` CUDA11.8 compatibility guard |

GP flags and the `manual extra args` list are rendered into a single `manual-extra-cip-args` value. The example explicitly appends `--internal-use-lnL --n-eff 2500 --n-output-samples 2500 --n-max 100000000`: pseudo-pipe normally divides the global sample count across workers and applies intermediate-stage caps, so the global `n output samples: 2500` alone would **not** produce 2500 per worker. Output count is not effective sample size. The AV default max-weight stopping metric remains unchanged for existing productions.

For a shared-fit DAG, the producer fits once, saves the model, and performs only the existing capped pilot integration (target 5, 10,000 maximum draws, one output sample). Each of eight consumers reloads that stage's model and runs the declared scientific integration. The CuPy producer also needs a real CUDA device because that pilot evaluates the fitted surface. Final extrinsic selection has its own production quota; it should not be confused with the intrinsic worker target.

After an Asimov **build**, before any submission, inspect generated intrinsic stage commands with:

```sh
python MonteCarloMarginalizeCode/Code/test/asimov_integration/audit_gp_matern_schedule.py /absolute/run/path/args_cip_list.txt
```

This rejects a stage that retains RF/quadratic, missing CUDA/AV settings, the inherited random `--cap-points` pre-selection, or a reduced effective output target. It also requires explicit log-likelihood integration and rejects unique-draw output caps, small-neff bounds, and quadratic-puff contingencies in this clean trial. Ordinary native `--fail-unless-n-eff` guards use the selected stopping metric in this source revision; native conservative neff is still recorded and returned separately. Early coordinate schedules remain the event's approved schedules; demanding GP at every stage does not silently change spin coordinates or priors. Check actual generated Condor submits separately: both producer and consumer must request one GPU, have the correct container/NVIDIA binding and compatible CUDA capability, and retain the chosen memory and priority. Do not trust ledger fields alone. A successful template render is not an end-to-end DAG build or GPU execution proof.

The GP trains on a bounded deterministic subset, with native training coordinates and lnL shift recorded. Optimizer convergence and CPU/GPU mean agreement do not establish interpolation accuracy; compare held-out errors and final weighted/posterior distributions. The new trial is an interpolation experiment and does not alter the stopped earlier runs.

## Posterior samples versus the next ILE grid

Systematic posterior resampling can intentionally repeat intrinsic points. Native `util_RandomizeOverlapOrder.py` interleaves equal-size worker blocks and samples row indices without replacement; it does **not** deduplicate identical physical rows. XML serialization preserves those rows, so the normal next ILE prefix can include repeated evaluations. Puffball covariance offsets are drawn independently per row and usually produce distinct points, but this does not deduplicate the ordinary unpuffed grid. Do not cap the scientific posterior to the conservative max-weight unique-draw bound merely to save ILE work: that can sharply reduce the posterior export. A complete end-to-end trial therefore needs explicit next-grid duplicate accounting/reuse or a separate grid-only deduplication step that preserves the posterior sample distribution and declared number of distinct likelihood evaluations. This deployment gate remains unresolved in this example until that route is implemented and audited.

CIP acceptance and optional output-size bounds use the selected Kish statistic in
this opt-in mode. The integrator's historical third return and native annotations
continue to disclose the max-weight statistic. Default productions retain their
existing behavior.

Validation so far: 31 template/schedule/resource/acceptance contract tests pass;
13 local interpolation/stopping tests pass (one CUDA-only test skipped). Actual
reviewed-LAL CIP preprocessing plus fresh/save/reload predictions passed for both
four-coordinate and eight-coordinate stages. Those runtime smoke tests stop
before sampling. CUDA binding, a real event DAG build, grid reuse/count accounting,
and the complete end-to-end posterior remain unverified. The integrator
shape-recovery production merge gate is still required before production merge.
This is an experimental opt-in package, not a production-default recommendation.
