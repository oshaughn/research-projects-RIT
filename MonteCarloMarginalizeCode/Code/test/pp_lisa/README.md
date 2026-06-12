# LISA PP Test Surface

This directory is the LISA analogue of `test/pp`: it stages synthetic
injection-like data products and renders a small RIFT analysis surface from a
known truth.  It is intentionally lightweight at first; full PP population
drivers can grow here without putting injection-generation workflows in the
main package path.

The current smoke path builds one known-sky event:

```bash
./run_pp_lisa_known_sky.sh
```

The driver writes, under `RIFT_PP_LISA_WORKDIR` or a temporary directory:

- A/E/T frequency-domain HDF5 frame products
- `lisa.cache`
- analytic LISA A/E/T XML PSDs
- `synthetic-params.env`
- a `pseudo_pipe` known-sky run directory with hyperpipeline CEPP files

Set `RIFT_PP_LISA_RUN_ILE=1` to also run the tiny direct ILE check after the
bundle and DAG are rendered.

Set `RIFT_PP_LISA_VARY_SKY=1` to render the same surface with ecliptic sky
location left as an intrinsic hyperpipeline parameter.  In that mode the helper
omits `--lisa-fixed-sky` from `args_ile.txt` and seeds a small three-point sky
grid.
