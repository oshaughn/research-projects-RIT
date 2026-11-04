# Posterior compression

Fit the joint rows in `extrinsic_posterior_samples.dat`, including their paired
distances and posterior weights. The intrinsic spin prior is the final CIP
stage's prior; early exploratory CIP priors must not be used for de-prioring.
CIP posterior sample files are not read or used to reweight the extrinsic rows.

```bash
util_RIFT_compress_posterior.py --config /path/production.ini \
    --rundir /path/completed_run
```

Enable the pipeline option `build-diagnostics = true` for separate posterior
draw and JS-comparison jobs. The standalone tools are
`util_RIFT_sample_compressed.py` and `util_RIFT_compare_compressed.py`.

See [the RST documentation](../../../../docs/source/posterior_compression.rst)
for the INI block, workflow integration, existing-run inputs, cosmology inheritance,
validation scope and cached evaluation/integration API.

Port lineage: `oshaughnessy-junior/gw-event-likelihood-fits`, commit
`6ecb81245805a478e67443d667dfd7c9cfd533bd`. Catalog coefficients and
campaign-specific importers are excluded.
