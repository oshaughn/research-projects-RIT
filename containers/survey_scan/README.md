# RIFT `survey_scan`

`survey_scan` is a lightweight build/deployment companion for RIFT container
families. It surveys the target Condor GPU pool, emits one warmup job per
container image band, and collects JSON timing/cache reports from common CuPy
and JAX startup probes.

It is deliberately RIFT-specific: the useful probes exercise the NoLoop CuPy
kernels, fused calmarg kernels, and JAX ILE wrapper shapes that dominate startup
cost.

## Commands

```sh
containers/survey_scan.sh survey \
  --out survey/cit-YYYYMMDD \
  --manifest container_family/rift_container_family.generated.yaml
containers/survey_scan.sh emit-jobs \
  --survey survey/cit-YYYYMMDD \
  --manifest container_family/rift_container_family.generated.yaml
containers/survey_scan.sh collect --survey survey/cit-YYYYMMDD
```

The submit-side commands use only the Python standard library. `PyYAML` is used
when available for manifest parsing; otherwise a small parser handles the simple
RIFT container-family YAML schema.

## Profiles

- `rift_cupy_common.py`: warms `Q_inner_product_cupy`,
  `Q_fused_calmarg_cupy`, `Q_fused_calmarg_distmarg_cupy`, and
  `interp_gpu.interp`.
- `rift_jax_ile_common.py`: warms synthetic JAX ILE wrapper modes. Use this only
  for JAX-enabled images.

The generated jobs run the profile inside the chosen container with:

```sh
apptainer exec --nv <image> python3 <profile> --json-out <result>.json
```

If the manifest image is an `osdf://` URL, the generated wrapper fetches only
that image with `stashcp` or `pelican`.
