# Measuring `--sampler-l0-rescue-reject-dlnZ` after #79

**Result: the default is raised 0.5 -> 3.0, and the gate is not a working truncation
detector.** Across 160 known-lnZ passes on two samplers it caught **0 of 55** genuinely
truncated warm passes at *every* threshold from 0.25 to 4.0 nats, while at 0.5 it binned
**25%** of good portfolio warm passes. 0.5 was strictly dominated: no detection, real cost.

Reproduce:

```
python3 measure_l0_reject_dlnZ.py --sampler AV        --reps 20 --out-json av.json
python3 measure_l0_reject_dlnZ.py --sampler portfolio --reps 20 --out-json pf.json
```

Run logs for the numbers below: `L0_REJECT_DLNZ_run_AV_2026-08-13.log`,
`L0_REJECT_DLNZ_run_portfolio_2026-08-13.log`.

## Why this had to wait for #79

The 0.5 default was chosen when both sides of the comparison were read from the fair-drawn
`_rvs`, which carries a `log(n_retained/eff_samp)` offset that does not cancel between passes
at different `n_eff` (+3.48 nats measured, `verify_skew.py`). Tuning against that would have
been calibrating to the bug. With #79 in (re-landed as #86) both sides come from the retained
reserve, so the number means what its help text says.

## Method

Known-lnZ targets (sums of Gaussians on the unit cube, uniform priors, rho=100, 6-D), driving
the real rescue sequence: cold pass -> reserve -> `build_warm_seed` -> `bootstrap_from_samples`
-> warm pass, with both lnZ readings taken through the ILE's own `_lnZ_of_reserve_or_rvs`.

**Rejection rates are conditioned on what the warm pass ACTUALLY did**, not on the intended
condition -- read out of its own retained set. This matters: on a portfolio the defensive GMM
component means a seeded warm pass usually still reaches every mode, so most "bimodal"
replicates are not truncated at all and a rejection there is a *false* positive. Scoring
against the label rather than the outcome would have reported those as successes.

## The cold reference is unusable on AV, and fine on a portfolio

| | cold lnZ − true | warm lnZ − true |
|---|---|---|
| **AV**, unimodal | **−76.2** (sd 37.9) | −0.75 (sd 0.43) |
| **AV**, bimodal f=0.50 | **−91.6** (sd 29.5) | −0.87 (sd 0.24) |
| **portfolio**, unimodal | −0.23 (sd 16.3) | −0.07 (sd 0.51) |
| **portfolio**, bimodal f=0.50 | −1.02 (sd 0.95) | −0.31 (sd 0.37) |

The gate's premise is *"the cold pass had full support, so a lower warm evidence means the seed
missed mass."* On **AV that premise is false**: the collapsed cold pass reports 70–92 nats below
truth. `cold − warm` is then about −70 to −91 and no positive threshold is ever crossed. This is
not a helper bug -- `lnZ_from_reserve` reproduces `integrate_log`'s own reported lnZ *to the
digit* in all conditions. It is what ESS 1.00 with k-hat ~51 means.

On a **portfolio the premise holds** (cold within ~1 nat), because the GMM member carries a
defensive component -- which is also why its warm pass is rarely truncated in the first place:
it reached every mode in **77 of 80** replicates. The configuration where the gate *could* work
is the configuration that barely needs it.

## Rejection rates, by what actually happened

```
                 AV  (28 good, 52 truncated)      portfolio (77 good, 3 truncated)
    dlnZ      FPR        TPR                    FPR        TPR
    0.25       0%         0%                    34%         0%
    0.50       0%         0%                    25%         0%
    1.00       0%         0%                    12%         0%
    2.00       0%         0%                     5%         0%
    3.00       0%         0%                     0%         0%
```

`TPR` is zero everywhere, on both samplers, across all 55 truncated passes. The AV column is
zero in both directions -- the gate is simply inert there. The portfolio column is *all cost*:
at the old 0.5 default, one good warm pass in four was discarded in favour of a collapsed cold
result.

## What changed, and what did not

**Changed:** the default, 0.5 -> 3.0. That removes the measured false-positive cost (~0% at 3.0)
without giving up any detection, because there is none to give up. It keeps a safety net for a
genuinely large (>3 nat) discrepancy, which no measurement here rules out.

**Not changed:** the gate itself. Replacing it properly means dropping the evidence comparison
for a direct test -- **support containment**: the cold reserve already holds the pass's finite
draws, so ask whether the warm live volume contains them. That needs no second trustworthy lnZ,
which is exactly the resource the collapsed regime cannot supply. Recommended as the follow-up;
deliberately not attempted here, since it is a new detector and wants its own validation.

There is also an irreducible blind spot independent of all this: a threshold `T` can never catch
a missed mode carrying less than `1 - exp(-T)` of the mass -- 95% at T=3.0. That alone means the
evidence comparison cannot be the primary truncation detector at any usable threshold.

## Scope

Synthetic Gaussian / two-Gaussian targets; rho=100; uniform priors on the unit cube; one bimodal
geometry (modes at 0.25 and 0.75 in one coordinate); 20 replicates per condition per sampler
(160 passes total). Not measured: real GW likelihoods, other separations, other rho, GPU
backends. The portfolio truncated-pass count is only 3, so its `TPR` column is weak evidence on
its own -- the strong statement comes from AV's 52.
