# LISA zero-likelihood CEPP demo

This directory contains the first checked-in LISA/RIFT workflow scaffold.  It is
intended to be small enough for CI contract testing while still looking like the
shape of a real LISA pipeline run.

The demo uses `helper_LISA_Events.py` to write the files consumed by
`create_event_parameter_pipeline_BasicIteration`:

- `proposed-grid.dat`
- `args_ile.txt`
- `args_cip_list.txt`
- `args_test.txt`
- `helper_transfer_files.txt`
- `command-cepp-lisa.sh`

The default path uses `--zero-likelihood`, so it validates file formats,
hyperpipeline sky columns, executable handoffs, and DAG rendering without
requiring real Sangria frames or PSD products.

## Run

From a checkout with the normal RIFT runtime available:

```bash
./MonteCarloMarginalizeCode/Code/demo/rift/lisa/run_lisa_zero_likelihood_cepp.sh
```

Useful environment overrides:

- `RIFT_LISA_PYTHON`: Python executable to use.
- `RIFT_LISA_WORKDIR`: output directory; defaults under `/tmp`.
- `RIFT_LISA_RENDER_CEPP=0`: only write the helper bundle; do not render CEPP.

Additional arguments are passed through to `helper_LISA_Events.py`, for example:

```bash
RIFT_LISA_WORKDIR=/tmp/rift-lisa-demo \
  ./MonteCarloMarginalizeCode/Code/demo/rift/lisa/run_lisa_zero_likelihood_cepp.sh \
  --mass1 120000 --mass2 90000 --ecliptic-longitude 1.4
```

Expected output is a helper bundle and, unless `RIFT_LISA_RENDER_CEPP=0`, CEPP
submit/DAG files in the work directory.  This script does not submit the DAG.

## Synthetic ILE surface

The companion script builds tiny synthetic A/E/T inputs and runs the standalone
LISA ILE against them:

```bash
./MonteCarloMarginalizeCode/Code/demo/rift/lisa/run_lisa_synthetic_ile.sh
```

It writes frequency-domain HDF5 frames, a `lisa.cache`, flat XML PSDs, helper
contract files, and `lisa_ile_0_.dat`.  The run pins most extrinsic parameters
but leaves polarization open, so it exercises a nonzero LISA likelihood integral
without becoming a full PE run.

This surface currently still uses XML PSDs because that is what the ILE path
loads today.  The synthetic input builder keeps PSD generation local and
mechanical, making it a good target for a future ASCII-PSD path once the LISA
workflow no longer needs `lal.series` XML PSD documents.

## Analytic PSD products

For a closer analogue of the toy `generate_iligo_psd` examples, this directory
also provides:

```bash
./MonteCarloMarginalizeCode/Code/demo/rift/lisa/make_lisa_psds.py \
  --output-directory /tmp/rift-lisa-psds --write-ascii
```

It writes analytic LISA A/E/T XML PSDs and, optionally, `LISA_psd.txt`.  The
PP-style LISA surface in `MonteCarloMarginalizeCode/Code/test/pp_lisa` uses
this generator together with the synthetic frame builder and
`util_RIFT_pseudo_pipe.py --lisa-known-sky`.
