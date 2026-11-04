# Test data provenance: `data/all.net`

Raw RIFT ILE output (the per-point Monte-Carlo `lnL` evaluations CIP consumes) used
as the fixed test example for the GP↔RF head-to-head.

- **md5:** `9c9f4b0298b4e2bff504835c77937416`
- **size:** 299,792 lines (~32 MB); 186,552 finite intrinsic points after the
  `--sigma-cut 0.6` + de-duplication used by `load_ile_net`.
- **event:** GW170817-like BNS. `lnL` peak ≈ 481.5; median reported `sigma_lnL` ≈ 0.08.
- **columns:** the standard ILE `.net` layout — intrinsic `(m1, m2, s1z, s2z,
  lambda1, lambda2)` plus per-point `lnL` and `sigma_lnL` (col "sigma/L").
- **fit coordinates used in the test:** BNS Morisaki/tidal
  `mu1, mu2, delta_mc, LambdaTilde, DeltaLambdaTilde` (low-level sampling coords
  `mc, delta_mc, s1z, s2z, lambda1, lambda2`).
- **prior box (matches the production benchmark):** `mc ∈ [1.196, 1.199]`,
  `|s1z|,|s2z| ≤ 0.05`, `lambda ∈ [0.01, 4000]`, `delta_mc ∈ [0, 0.9]`.

## TODO (citation / origin) — fill in before paper submission

This file was harvested from a local RIFT run on the dev machine
(`/home/oshaughn/all.net`, 2026-06-02). It is **not** identical to the smaller
`all.net` in the published GW170817 reference run
(`.../rundir_GW17017_knownhost/`, md5 `794f357…`, 19,417 lines) — this is a
larger/independent ILE accumulation. **Record the exact run directory, PSDs,
waveform (IMRPhenomD_NRTidalv2), and event configuration here, and the citation,
so the test example is reproducible and properly attributed.**

## Benchmark posterior (for the corner figure)

The corner overlay compares against the production CIP `--fit-method rf
--sampler-method AV` posterior fleet (10× independent runs, pooled ~50k samples),
built by `../applications/benchmark_condor/` and cached at
`/home/oshaughn/jaxcip_benchmark/out/cip_rf_*.xml.gz` on the dev machine. Point the
`BENCH` make-variable at your own fleet to reproduce.
