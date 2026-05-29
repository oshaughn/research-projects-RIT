# Pipeline-build demos (`util_RIFT_pseudo_pipe.py`)

Fast, submission-free smoke tests of the end-to-end pipeline builder. Each
target runs `util_RIFT_pseudo_pipe.py` against a reference `.ini` + `coinc.xml`
with **fake data**, producing a complete RIFT run directory (helper output,
`args_*.txt`, and condor `*.sub` files) **without** submitting anything or
needing real frames, PSDs, or GPUs.

These double as regression tests for argument threading: a flag set on
`util_RIFT_pseudo_pipe.py` must survive through
`create_event_parameter_pipeline_BasicIteration` (CEPP) and land in the correct
condor submit file.

## Running

Inside the RIFT environment:

```bash
# with pixi
pixi run --manifest-path ../../../../../pixi.toml make all
# or, with a pip-installed RIFT on PATH
make all
```

Targets:

| target | what it builds / checks |
| --- | --- |
| `baseline` | a standard pipeline; asserts no per-distance export leaks into `ILE_extr.sub` |
| `grid`     | `--export-marginal-distance-grid` (Plan A); asserts `--export-marginal-distance-grid --internal-use-lnL` land in `ILE_extr.sub`, no `--distance-marginalization` anywhere, and the flag does **not** appear in the intrinsic `ILE.sub` |
| `slices`   | `--export-distance-slices 10 ...` (Plan B); asserts `--export-distance-slices 10`, `--distance-slice-wing-delta-lnL`, `--distance-slice-skip-threshold`, `--internal-use-lnL` land in `ILE_extr.sub`, no `--distance-marginalization`, and nothing leaks into `ILE.sub` |
| `all`      | all three |
| `clean`    | remove generated `rundir_*` and the fake cache |

Inputs default to `.travis/ref_ini/GW150914.ini` and `.travis/ref_ini/coinc.xml`;
override with `make REF_INI=... COINC=... all`.

## Why the extrinsic stage

Per-distance likelihood export (both Plan A density grids and Plan B fixed-`d`
slices) is emitted by the **last-iteration extrinsic** ILE stage (`ILE_extr`),
not by the intrinsic ILE jobs that run every iteration. `util_RIFT_pseudo_pipe.py`
therefore:

1. forces ILE `lnL` mode and **disables distance marginalization** (the export
   integrates the pure likelihood vs distance, which is incompatible with a
   distance-marginalized ILE configuration), and
2. passes the corresponding `--last-iteration-export-*` flag to CEPP, which
   appends the ILE-level export flags to the `ILE_extr` argument string only.

These export flags require `--add-extrinsic` (so the extrinsic stage exists);
the demo passes it explicitly. See
`demo/rift/add_distance_grids/PLAN_B_DESIGN.md` for the Plan-B design and the
`.dslice` re-marginalization API.
