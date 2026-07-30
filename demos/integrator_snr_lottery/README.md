# Integrator SNR-lottery demos

Truth-known (`shape_recovery.MixtureTarget.true_lnZ`), CPU-only studies of high-SNR extrinsic
integration failure. SNR ladder = peak width: `sigma_1d = 0.7*(20/SNR)` on the box [-5,5]^d.

## results/restrict_ladder.txt -- restricted-range AV (n=20 per case per SNR)

| case | SNR 40 bias / n_eff / collapse | SNR 80 | SNR 160 |
|------|-------------------------------|--------|---------|
| A_full  standalone AV, full box        | +0.026 / 2029 / 0%  | +0.097 / 1802 / 0% | +0.102 / 1629 / 0% |
| A_sub   standalone AV, CORRECT sub-box | -0.029 / 2495 / 0%  | -0.029 / 2495 / 0% | -0.029 / 2495 / 0% |
| A_sub_wrong  standalone, WRONG sub-box | **-35.9** / 840 / 0% | **-266.8** / 356 / 0% | **-1949.3** / 220 / 0% |
| P_full  portfolio AV+AV, both full     | -0.456 / 21 / 30%   | -0.545 / 16 / 30%  | -0.678 / 30 / 30%  |
| P_mix   portfolio full + CORRECT sub   | -0.427 / 196 / 10%  | -0.428 / 182 / 10% | -0.455 / 186 / 10% |
| P_mix_wrong  portfolio full + WRONG sub| -0.699 / 9 / 55%    | -1.076 / 12 / 45%  | -1.183 / 35 / 25%  |

Readings:
1. **A_sub is SNR-INDEPENDENT** -- identical bias/n_eff at every SNR, because restricting the box to
   the posterior makes the problem self-similar. A_full meanwhile degrades with SNR (n_eff 2029 ->
   1629, bias +0.026 -> +0.102). This is the mechanism working as intended.
2. **A_sub_wrong is the confidently-wrong failure in its purest form**: bias -36 / -267 / -1949 nats
   while `collapse% = 0` and n_eff stays 220-840. n_eff reports a HEALTHY run. This is the strongest
   argument in this whole study that n_eff cannot certify correctness, and it is the ideal generator
   for validating a tail diagnostic (see tools/khat_validation.py).
3. **Fail-safe CONFIRMED**: the same wrong sub-box inside a portfolio (P_mix_wrong) costs ~1 nat and
   some efficiency, not ~1949 nats -- the full-box member keeps q_mix covering. This is the reason to
   prefer the multi-AV variant over mutating a single AV's range.
4. **P_mix improves on P_full**: collapse 30% -> 10%, n_eff ~20 -> ~185, at equal budget.

HONEST CAVEAT: the portfolio rows carry a persistent ~-0.4 nat bias even in the GOOD case (P_mix
-0.43), but P_full shows the same (-0.456), so it is NOT caused by the restriction -- it is the known
downward skew of IS estimates at low n_eff (heavy-tailed weights). The portfolio path is therefore not
demonstrated unbiased at these n_eff; only the restriction's SAFETY and EFFICIENCY are established.

## results/khat_decisive.txt -- does Pareto k-hat catch the confidently-wrong runs? NO (for this failure mode)

Scored against KNOWN truth at SNR 160, n=20 per case. k-hat computed from the true importance
log-weights (log_integrand + log_joint_prior - log_joint_s_prior) via statutils.pareto_khat_from_log.

| case | bias_med | n_eff_med | khat_med | % copies khat>0.7 |
|------|---------:|----------:|---------:|------------------:|
| A_full (accurate)             | +0.102 | 1629 | -0.268 | 0% |
| A_sub (accurate)              | -0.029 | 2495 | -0.302 | 0% |
| **A_sub_wrong (CATASTROPHIC)**| **-1949** | 220 | **0.435** | **10%** |
| P_full                        | -0.678 | 30   | 0.706 | 50% |
| **P_mix (BEST portfolio)**    | -0.455 | 186  | **0.766** | **80%** |
| P_mix_wrong                   | -1.183 | 35   | 0.691 | 45% |

**k-hat misses the catastrophe and fires on the good run.** The run biased by -1949 nats has
khat 0.435 -- BELOW the 0.7 "unresolved tail" threshold -- and only 10% of its copies trip the
threshold, so ~90% of catastrophically-wrong runs pass the check. Meanwhile the most ACCURATE
portfolio configuration (P_mix) has the HIGHEST khat (0.766, 80% firing). Ranking by khat is
anti-correlated with actual error here.

**Mechanism (why this is not a bug in k-hat).** k-hat estimates the tail index of the weights you
ACTUALLY DREW. A sampler confined to a wrong sub-box never draws from the true peak at all, so its
observed weights are narrow and self-consistent -- the tail genuinely IS resolved, for the region it
sampled. The failure is total support non-overlap, not a heavy tail. **k-hat can detect mass whose
tail you have begun to sample; it cannot detect mass you have never touched.** Conversely P_mix has a
legitimately heavy tail (its full-box member occasionally lands a huge-weight point near the peak) --
k-hat correctly flags that, but the run is accurate, so a k-hat gate would reject the best config.

**Implication.** k-hat is a sound tail diagnostic and worth keeping, but it must NOT be used as a
pass/fail correctness gate for proposal/support-mismatch (mode-collapse) failures -- the dominant
high-SNR failure in this study. For that class the working detector remains CROSS-COPY DISAGREEMENT
(replicas / bootstrap quantiles): independent copies that localize differently disagree, and that is
observable, whereas a single run's own weights are not. Note this is one generator, d=4, n=20 --
the mechanism is principled but the numbers are one configuration.
