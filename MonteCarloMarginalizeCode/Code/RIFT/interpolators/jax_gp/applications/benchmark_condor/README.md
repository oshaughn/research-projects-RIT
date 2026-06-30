# CIP+RF+AV benchmark (Condor) for the jax_cip accuracy comparison

The success metric for `jax_cip` is JS divergence of the 1D marginals (mc first)
against the **production CIP path: RF fit + Adaptive-Volume sampler**. A single CIP
run gives a few thousand samples — not enough for a reliable JS — so we launch it
**10×** and pool. SVGP/`jax_cip` is fast enough to produce its own samples without
this, so only the brute-force benchmark needs the fleet.

## What's here
- `run_cip_rf_av.sh` — one CIP job (`--fit-method rf --sampler-method AV`) on the
  GW170817 `.net`, in the same BNS coordinates/prior as `jax_cip`
  (`--mc-range [1.196,1.199] --chi-max 0.05 --input-tides`). Writes `cip_rf_<idx>.xml.gz`.
- `cip_rf_benchmark.sub` — HTCondor submit, `queue 10`.

## Resource footprint (measured on this box)
| config | peak RSS | wall |
|---|---|---|
| uncapped RF | 3.9 GB | ~70 s |
| `--cap-points 30000` | **1.25 GB** | ~76 s |

So the submit requests `request_memory = 2048` (2 GB, comfortable margin over 1.25 GB)
and `request_cpus = 2` (threads pinned to 2 in the wrapper). The RF dominates memory;
capping points is what keeps the footprint low — keep the cap unless you need a
denser RF.

## Run it
```bash
cd RIFT/interpolators/jax_gp/applications/benchmark_condor
# edit the absolute paths in cip_rf_benchmark.sub (RIFT_CODE / PYTHON / NET) for your site
mkdir -p out logs
condor_submit cip_rf_benchmark.sub
# 10 independent runs (CIP has no --seed, so each randomizes) -> out/cip_rf_{0..9}.xml.gz
```

## Compare (JS, mc first)
```bash
# pool all 10 benchmark runs on the B side; A side is the jax_cip output XML
python -m RIFT.interpolators.jax_gp.applications.compare \
    --a /path/to/jax_cip_out.xml.gz \
    --b out/'cip_rf_*.xml.gz' \
    --param mc
# repeat with --param delta_mc / lambda1 / s1z ... once mc looks good.
```
`compare.py` prints JS in bits with a bootstrap stderr; if the stderr is comparable
to the JS, you are statistics-limited — add more benchmark runs (bump `queue`) and/or
draw more `jax_cip` samples.

## Notes
- No `--seed` in CIP → independent streams per launch; that's why pooling 10 runs
  accumulates valid statistics.
- The benchmark uses the SAME prior box as `jax_cip` (mc-range, chi-max, lambda) so the
  JS reflects method differences, not prior differences.
- `RIFT_CODE` points at a stable checkout (not the ephemeral `.claude/worktrees` copy)
  so the fleet keeps working after the dev worktree is cleaned up.
