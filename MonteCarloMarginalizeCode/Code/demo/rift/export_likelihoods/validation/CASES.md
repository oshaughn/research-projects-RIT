# The validation ladder — what full run to launch per case

Each rung is a **standard RIFT run, to convergence** (you launch it on a cluster /
full machine). The run must produce two things this harness then consumes:

1. **a converged, consolidated likelihood grid** — a named-column `.dat` with an
   `lnL` column plus the physical-parameter columns (`m1 m2 [s1z s2z] [lambda1
   lambda2] [dist] ...`). This is what the GP is fit to.
2. **the standard posterior** — `.dat` samples over the same parameters, from the
   run's own sampler (CIP / `util_ConstructEOSPosterior.py`).

Then: `GRID=... STD=... source config.sh && ./analyze_case.sh <case>`.

> The point of a *full* run is that ILE+CIP **adaptively place grid points** and
> iterate to convergence — that is the only way you know the likelihood was actually
> sampled. Do not substitute a fixed, hand-built grid: the GP (and the comparison)
> would inherit whatever the grid happened to miss.

## Getting distance (dL) as a sampled parameter

Distance is normally marginalized inside ILE, so a vanilla intrinsic run gives no
`dist` marginal. To get the joint `(masses, ..., dist)` posterior, run the
**per-distance likelihood-grid (Plan A) pipeline**: ILE exports
`--export-marginal-distance-grid` at the extrinsic stage, the `.dgrid` files are
consolidated (`util_ConsolidateDistanceGrids.py` → `all_dgrid.dat`, the GRID), and
`util_ConstructEOSPosterior.py` reconstructs the standard `(m1,m2,...,dist)`
posterior (the STD). The local end-to-end mechanics of this path are demonstrated
(on tiny fake data, for wiring only) in
`../../pipeline/zero_spin_phenomD/Makefile`; a real run scales it up with a proper
event/grid and runs to convergence.

Pipeline build is `util_RIFT_pseudo_pipe.py` (see `../../pipeline/` for build
demos). The parameter subspace is set by the run configuration:

| case | restrict the run with | analyze with |
| --- | --- | --- |
| `mcq_dL` | `--assume-nospin`, no tides, `--export-marginal-distance-grid` | `./analyze_case.sh mcq_dL` |
| `mcq_aligned_dL` | aligned-spin (default IMRPhenomD), no tides, `--export-marginal-distance-grid` | `./analyze_case.sh mcq_aligned_dL` |
| precessing + dL | precessing approximant (e.g. IMRPhenomPv2), `--export-marginal-distance-grid` | **harness TODO** (see below) |
| `mcq_aligned_tides` | aligned-spin + `--input-tides` (NRTidal) | `./analyze_case.sh mcq_aligned_tides` |
| full (mass+spin+tides) | the production BNS configuration | the parent `../02_validate.sh` (GW170817) |

(Exact `pseudo_pipe`/`.ini` flags depend on your event and cluster setup; the table
lists the configuration *intent*. Match the run's prior ranges and pass them to the
analysis via `RANGES=...` rather than relying on the grid-extent default.)

## Notes / open ends

- **Precessing** is not yet analyzable here: the NUTS preconditioner and the
  (optional) fit-coordinate path assume aligned spin / BNS. Precessing needs
  precessing fit-coordinate transforms added to
  `RIFT/interpolators/jax_gp/coordinates.py`. The generic `gp_from_grid.py` can fit
  the raw precessing-spin parameters, but expect the GP to need more data and the
  conditioning work that aligned cases avoid.
- **Standard posterior format:** `analyze_case.sh` and `compare_marginals.py` read
  named-column `.dat`. If your run emits the posterior only as XML, convert it (or
  extend `compare_marginals.py` to read it) so both posteriors are in the same
  parameters.
- **Reference is not ground truth:** even a converged RIFT posterior is a reference,
  not assumed-perfect. Read the ladder *relatively* — where the GP first diverges as
  parameters are added — rather than treating a single small JS as exact.
