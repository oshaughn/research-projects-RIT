# AV coverage diagnostic (`unseen_fraction`, `unseen_frac_final`)

## What it measures

Since #412, `integrate_log` weights each retained point by the density of the mixture of all
grids it drew from (`log_retained_density`). Each grid holds only bins with a live point. An
emptied bin leaves the next grid, while a retained point keeps its own bin in every later grid.
When the live set leaves bins empty, the mixture density is too high at retained points and the
mass in the empty bins is not counted, so lnZ reads low.

`unseen_fraction` estimates that missing mass on the final grid: the likelihood weight of live
points alone in their bin, as a fraction of the total (Good-Turing). It runs once after the
loop and changes no result. `dict_return['unseen_frac_final']` and the `[AV mc diag]` line report
it; above `COVERAGE_UNSEEN_WARN` (0.05) a `[AV coverage]` warning is printed.

## Evidence

- Synthetic: a Gaussian of width 0.001 on two [-1,1] axes with three flat angular axes reads
  lnZ -0.69 (n_eff 30) and -2.18 (n_eff 300); the final grid holds 4-46% of the mass; unseen
  0.33-0.77. With no flat axes, lnZ is unbiased and unseen is near 0.
- S250114ax, one intrinsic point, six sky configurations, 3 runs each, against an n_eff 325
  reference. Healthy configurations (lnZ within 0.5) read unseen <= 0.036; configurations 0.9-4
  nats low read >= 0.066.

Records: RIFT_roboto_paper `development/OPEN_av_412_narrow_flat_axes.md`.

## Tried and rejected

Coarsening the grid when unseen is high (`claude/av-coverage-guard` before this change) fixed the
synthetic case but on S250114ax never acted where lnZ was low and cut n_eff about 10x in healthy
rotated-phase runs.
