# Cross-model marginalization on a shared grid

A record of what `create_event_parameter_pipeline_BasicMultiApproxIteration`
does and why, as of 2026-08-27. It is expected to be superseded; where this
note and the code disagree, the code and
`test/test_multiapprox_marginalization.py` are live truth.

## What the workflow computes

Running several waveform models is not the same as running several pipelines.
This builder evaluates every model on **one shared intrinsic grid** and
marginalizes over models at each grid point:

    L_marg(lambda) = sum_m  p(m) L_m(lambda)

The iteration loop fits `L_marg`, so the adaptive grid refines on the
model-marginalized posterior — points get placed where the *mixture* has
support, including where one model is good and another is not. This is the
difference from running N single-model pipelines and mixing posteriors
afterwards: there, each run's grid refines on its own model's posterior, and no
amount of post-hoc mixing recovers support that neither run sampled.

The combination is **linear in L**, not in lnL. Averaging lnL would give the
geometric mean — a logarithmic opinion pool — which is not Bayesian model
marginalization and is dominated differently when models disagree.

## Where each piece lives

| stage | model-tagged? | why |
|---|---|---|
| ILE input grid `overlap-grid-N.xml.gz` | no | the shared grid; all models evaluate the same lambda |
| ILE output → `approx_<A>_consolidated_N.composite` | yes | per-model evaluations, kept apart |
| `unify.sh` → `all.net` | no | pools every model, **marginalizes** |
| in-loop CIP → `overlap-grid-N+1` | no | one fit over the marginalized net |
| `unify_model.sh` → `approx_<A>_all.net` | yes | one model only, for the fork |
| terminal CIP → `approx_<A>_overlap-grid-N` + `+annotation.dat` | yes | per-model posterior **and** ln Z |
| extrinsic ILE / convert / resample / cat | yes | per model |
| `util_CombineApproximantPosteriors.py` | — | mixture weighted by p(m) Z_m |

**The loop merges; the terminal stage forks.** The marginalized grid is what
should steer exploration, but the deliverables of a systematics study are the
per-model posteriors and their evidences — so those are produced separately,
left on disk, and combined only at the very end.

## The marginalization is in util_CleanILE.py

It keys on the intrinsic parameters (rounded to 5 decimals), so the same lambda
under several models collapses to one entry. `--model-group-regex` takes the
model label from each input filename and switches on a **two-level** combine:

* **within** a model, replicas estimate one number → ntot-weighted linear mean;
* **across** models, the quantity itself differs → marginalize with p(m).

Flat pooling of everything — the behaviour without the flag — silently uses
*replica counts* as model weights. On a two-model point where one model has two
replicas and the other one, with lnL 100 and 104, flat pooling gives 102.937
where uniform-prior marginalization gives 103.325: **0.39 nats**, growing with
model disagreement and replica imbalance. Correct for replicas of one model,
wrong across models, which is why it is opt-in and why one model reduces to it
exactly (verified byte-identical).

`p(m)` is a *prior over waveform models*, not sampling effort. `ntot` is
sampling effort. Conflating them is the defect above.

## Partial coverage changes the estimator, point by point

A lambda evaluated under only some models is marginalized over that subset, so
the estimator differs across the grid — a model- and lambda-dependent change in
the effective prior. Causes: the `sigmaOverL > 0.9` resolution cut firing for
one model and not another, or one model's ILE failing.

This is reported (never silent) and `--require-all-approx` drops such points
instead. Dropping is not free either — it biases toward regions where the
*worst-resolved* model converged — so it is a choice, not a default.

## Prior work

Done before by hand (Yelikar; Jan), not efficiently in production. The claim
here is the production workflow — one DAG, one shared adaptive grid, evidence-
weighted recombination — not the idea of waveform marginalization.

## Known limits

* The final mixture resamples with replacement, so a model carrying most of the
  weight can be drawn more times than it has samples. Reported as a warning;
  it degrades effective sample size without changing the row count.
* `p(m)` is uniform unless `--approx-prior` is given, and applies at both the
  in-loop marginalization and the final mixture. They are the same p(m) and are
  passed to both from one option.
* A large ln Z gap between waveform models usually means the models disagree by
  far more than the statistical error, not that one is right. The combiner
  warns above 0.99 mixture weight.

## Also fixed here

This builder is not reachable from `util_RIFT_pseudo_pipe.py` (`--pipeline-builder`
offers only BasicIteration, AlternateIteration and Hyperpipe) and asimov drives
pseudo_pipe, so it is a hand-run `bin/` script with no caller — which is how the
following survived. All were read off an emitted two-approximant DAG.

* The terminal extrinsic stage read `overlap-grid-<n_iterations-1>` while the
  final CIP wrote `overlap-grid-<n_iterations>`: a guard,
  `if not ('it' in globals())`, preserved an `it` the iteration loop had already
  left one short. At one iteration it read the raw seed grid.
* `join_grids.sh` interpolated `$(macroapprox)` **inside a bash script** —
  command substitution, not a condor macro, since the `.sub` passes only
  `$1 $2`. It matched nothing. Gone now that the loop grid is not model-tagged.
* `CIP.sub`'s `initialdir` named `iteration_N_cip` while the mkdir loop created
  `approx_<A>_iteration_N_cip`; the consolidate/unify log directories were the
  mirror image. Either holds the job on the execute node, and no DAG-shape
  check sees it — hence `test_every_job_directory_exists`.
* `parent_fit_node` was a single variable spanning `for it: for approx:`, so
  model B's iteration-0 ILE waited on model A's iteration-1 convert. The models
  were serialized into one chain and cross-coupled; they now run in parallel.
* The terminal convert wrote one untagged `posterior_samples-N.dat` from every
  model, while the convergence test read a per-model name nothing wrote.

## Testing status

`test/test_multiapprox_marginalization.py` covers the combination arithmetic
against hand-computed values, and the emitted DAG's shape. **Neither validates
the inference on data.** The prototype for that is a rerun of one of the
Jan/Yelikar cases — cheap, and with large between-model differences — framed as
a reproduction claim first and an efficiency claim second. Nothing from this
workflow belongs in a paper until that runs.
