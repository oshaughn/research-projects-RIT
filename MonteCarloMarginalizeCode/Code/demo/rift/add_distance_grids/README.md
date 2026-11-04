# RIFT Distance-Grid Export Demo

This demo builds a small zero-spin RIFT workflow with
`--export-marginal-distance-grid` enabled for ILE jobs.  It reuses the fake
zero-noise frames, PSD, cache, and initial target grid from the CI assets in
`../../../../../.travis/ILE-GPU-Paper/demos`.

`add_distance_grids.ini` records the corresponding zero-spin/fake-data
configuration.  The Makefile builds the runnable DAG through the same
`create_event_parameter_pipeline_BasicIteration` path used by the CI demo,
because the higher-level pseudo-pipe ini path currently trips over an XML
compatibility issue when rereading its temporary target file in this
Python/lalsuite environment.

Run:

```bash
make dag
```

This creates `rundir/`, writes the normal RIFT DAG files, and verifies that
`rundir/args_ile.txt` contains:

- `--export-marginal-distance-grid`
- `--internal-use-lnL`

The demo intentionally does not submit automatically.  To queue the generated
workflow:

```bash
make submit
```

## Environment Note

If the run prints messages like:

```text
swig/python detected a memory leak of type 'struct tagLIGOTimeGPS *', no destructor found.
```

that is an environment compatibility warning from the LALSuite Python bindings,
not a distance-grid failure.  It has been observed with the local `my_rift`
environment (`python 3.12.4`, `lal 7.7.0`, `lalsimulation 6.2.0`,
`lalmetaio 4.0.6`, `lalsuite 7.26`).  Prefer a known-good RIFT/LALSuite
environment whose LAL packages were built with pre-SWIG-4.4 bindings.  Pinning
`swig<4.4` only helps when rebuilding the LAL packages; installing an older SWIG
binary next to already-built LAL Python wheels/conda packages will not change
the generated wrapper code.

After ILE jobs complete, each evaluated intrinsic point should have a companion
`*.dgrid` file.  The grid is a likelihood density in luminosity distance; use
`RIFT.misc.distance_grid.reconstruct_marginal_lnL()` to check that integrating
the distance grid reconstructs the ordinary marginalized likelihood.
