# Pre-merge review checklist for sampler / estimator changes

Companion to `TESTING.md` (which owns the expensive shape-recovery merge gate). This file is
the **cheap** pass: things to check by reading, before a PR is opened, that have repeatedly
survived CI, a green test suite, and a successful GPU run.

Why it exists: two consecutive integrator PRs each needed **four review rounds**, and in both
cases every finding was the same shape.

* **`oshaughnessy-junior#63`** (AV collapse reporting) — three findings that were all *"a status changes after
  the thing consuming it has already run"*: the caller never read the verdict; the empty-cycle
  test used the cumulative count; replica status was discarded and the reject gate checked the
  pre-replication value.
* **`oshaughnessy-junior#78` / `#84`** (L0 rescue warm seed) — four findings that were all *"an array or a count
  looks like the population but is a filtered or resampled subset of it"*: the seed read the
  fair-draw export resample; the reject gate differenced two differently-resampled lnZ values;
  the reserve's cap kept `-inf`-dominated rows in proportion (2 of 10 finite rows survived)
  and consumed the global RNG; and the capped estimator's *logarithm* carried sampling error
  though its linear total was unbiased.

Every one of those returns a plausible-looking number when it is wrong. None of them crashes.

---

## 1. Always, and cheap

- [ ] **Is the new test file named in `.github/workflows/ci.yml`?** The workflow lists test
      files individually, so a test file existing does not mean CI runs it. **Grep the
      workflow for the filename** — do not rely on any coverage claim, here or elsewhere,
      including this one; the routing drifts.

      *History, as the reason this item exists:* the AV collapse-detection suites
      (`test_av_empty_live_volume.py`, `test_l0_rescue_seed.py`,
      `test_portfolio_fairdraw_backend.py`) were reachable from no CI job at all until
      2026-08-12, one of them having shipped with an already-merged PR. They are routed now,
      in the numpy-matrix sampler job.
- [ ] **Does a test assert the precondition it is about?** For a degenerate-regime fix, a test
      that silently runs in the healthy regime passes vacuously. Open with the assertion —
      e.g. `assert r['n_finite'] < r['n_retained'], 'this target did not underflow'`.
- [ ] **Does the diff contain only your hunks?** `git show --stat` before pushing. Several
      checkouts here are shared between concurrent sessions, and `git add -A` has twice swept
      another branch's uncommitted work into an unrelated commit.
- [ ] **If a default changed, is the previous behaviour still reachable by flag?** and is the
      new default's justification a measurement, not a preference?

## 2. The population-vs-sample invariants

Ask these of every array and every count the change reads:

- [ ] **Is this the population, or a sample of it? What is the denominator?** An importance
      estimate is `mean(w)` over the draws that were *made*; a draw whose likelihood
      underflowed to `-inf` is a real draw contributing a real zero. Filtering those out and
      then averaging divides by the wrong `n`.
- [ ] **`sampler._rvs` may be the fair-draw EXPORT resample rather than the sample set — check
      which.** Every sampler here (`mcsampler`, `mcsamplerGPU`, `mcsamplerEnsemble`,
      `mcsamplerAdaptiveVolume`, `mcsamplerPortfolio`) rebinds `_rvs` **only** when all three
      hold:

          bFairdraw                          # kwargs['igrand_fairdraw_samples']
          and n_extr is not None             # kwargs['igrand_fairdraw_samples_max']
          and n_extr < len(retained)         # n_extr = min(n_extr, 1.5*eff_samp, 1.5*neff)

      Otherwise `_rvs` is still the retained set and reading it is correct. When it *does*
      fire, the rows are drawn **with replacement, proportional to weight**, so on a collapsed
      pass it is *one row* while the live set held 1000 — and duplicates are why a "2-point"
      cloud can have affine rank 0. The ILE sets both kwargs from
      `--fairdraw-extrinsic-output` / `--fairdraw-extrinsic-output-n-max` (see the
      `igrand_fairdraw_samples` entries in `integrate_likelihood_extrinsic_batchmode`), so on
      the extrinsic-export path assume it fired unless you have checked.
      **Do not infer the condition from the sampler class; find the caller's kwargs.**
- [ ] **If you need the draws, is a pre-fair-draw record actually available for this sampler?**
      `_warm_seed_reserve` (snapshotted before the fair draw, carrying `n_retained`,
      `n_finite` and the exact `ln_sum_w_finite`) exists on **`mcsamplerAdaptiveVolume` and
      `mcsamplerPortfolio` only**. It is not a generic sampler facility; `mcsampler`,
      `mcsamplerGPU` and `mcsamplerEnsemble` have no equivalent, so code that must work across
      samplers needs its own fallback and should say which reading it used.
- [ ] **Is the code path reachable at all?** `mcsamplerPortfolio` drives its members through
      `draw_simplified()`, never their `integrate_log()`, so a member never executes anything
      that lives there. Name the caller before assuming a branch runs.
- [ ] **Is the *logarithm* of the estimator unbiased, or only the linear total?** A uniformly
      capped (Horvitz–Thompson) sum is unbiased in `Z` and biased in `ln Z`, and every gate
      here compares logarithms against a nats threshold. Where an exact total is available,
      record it rather than re-estimating it.
- [ ] **Does an error cancel between the two things being compared?** It does not if the two
      passes have different `eff_samp`, different finite fractions, or different subsample
      sizes — which, on the paths that trigger a rescue, they always do.
- [ ] **Is a status written before the thing that consumes it runs?** (the collapse-reporting
      shape above.)

## 3. Side effects when the feature is OFF

- [ ] **Does the change consume the global RNG?** Anything built unconditionally must not
      advance `np.random` — it moves the fair draw, the exported posterior, and every later
      event and replica, so an opt-in feature that is switched off changes a seeded run.
      Use a private `RandomState`.
- [ ] **Does it mutate shared state in place** (`_rvs`, a member's grid, a cached proposal)
      that a later point or replica will read?
- [ ] **Is per-point state cleared on ENTRY**, so "present" means "this pass wrote it" rather
      than "the previous point left it"?

## 4. Blast radius — when to spawn a full code review

Run the cheap list above on everything. **Additionally request a full code review** (the
`code-review` skill, or `/code-review ultra` for a multi-agent pass) when the diff touches:

* anything under `RIFT/integrators/`;
* `bin/integrate_likelihood_extrinsic_batchmode` or the other ILE entry points;
* `RIFT/likelihood/factored_likelihood.py` or the CUDA kernels;
* evidence, weights, normalization, `n_eff`/ESS, or any collapse/rejection gate;
* any changed default that production configurations inherit.

These are the paths where a wrong answer is *plausible* rather than *loud*. The expensive
shape-recovery gate in `TESTING.md` is still required for them — it catches a different class
(posterior shape wrong while the integral is right) and does not subsume this list.

## 5. What does not count as verification

A single run that completed and looks sane is not evidence about the regime a fix targets.
These integrators degrade continuously: a run in the healthy part of the range executes the
same lines and prints the same-looking diagnostics as one in the failing part. Measured
example — a GPU replicate quoted as confirming the warm-seed fix had 239 seed points from a
20001-row reserve, i.e. it never entered the collapsed regime, while the same code kept **2 of
10** finite rows when finite rows were rare.

State which regime the verification run was in, and prefer a replicate campaign over a single
run wherever the failure is a lottery (as it is above rho_net ~ 100).
