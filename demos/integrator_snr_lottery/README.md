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
