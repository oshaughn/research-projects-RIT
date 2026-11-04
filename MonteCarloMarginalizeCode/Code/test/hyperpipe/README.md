# RIFT.hyperpipe test suite

Live integration tests for the in-tree `RIFT.hyperpipe` package
(`Code/RIFT/hyperpipe/`) and the `util_RIFT_hyperpipe.py` /
`util_HyperMargGaussian.py` console scripts (`Code/bin/`). Backed by a
self-contained [pixi](https://pixi.sh) environment so the suite works
from any RIFT clone without polluting your global Python.

## Quick run

```sh
# one-time, if you don't have pixi:
curl -fsSL https://pixi.sh/install.sh | bash

cd MonteCarloMarginalizeCode/Code/test/hyperpipe
pixi install        # heavy: pulls lalsuite + igwn-ligolw + gwpy + scientific stack
pixi run test       # full pytest suite (the live test)
```

Auxiliary entry points:

```sh
pixi run test-minimal   # pytest-free smoke check  (also useful in CI without pytest)
pixi run demo-dryrun    # end-to-end dry-run of util_RIFT_hyperpipe.py
pixi run which-rift     # confirm pixi resolved the in-tree RIFT correctly
```

## Layout

```
test/hyperpipe/
├── pixi.toml                       # env spec: python + scientific stack + full lalsuite
├── README.md                       # you are here
└── tests/
    ├── conftest.py                 # auto-detects RIFT_ROOT from this file's location
    ├── test_coords.py              # CIP-mirror coord-spec emission
    ├── test_marg_list.py           # mono + heterogeneous marg-list assembly
    ├── test_config.py              # schema-validation paths
    ├── test_drivers.py             # MargDriverBase + util_HyperMargGaussian end-to-end
    ├── test_hydra_integration.py   # real-Hydra subprocess test of util_RIFT_hyperpipe.py
    ├── standalone_check.py         # pytest-free smoke test
    └── demo_dryrun.py              # full dry-run demo
```

The `[activation.env]` block in `pixi.toml` derives `RIFT_ROOT` /
`RIFT_PY` / `RIFT_BIN` relative to `$PIXI_PROJECT_ROOT`, prepends
`$RIFT_BIN` to `PATH`, and sets `PYTHONPATH` so the pytest subprocesses
can find the in-tree `RIFT` package. No user-specific hardcoded paths.

## What each suite proves

| File | What it proves |
|---|---|
| `test_coords.py` | Coord-spec emission exactly matches the Gaussian-demo `args_*.txt` files (including `-8` vs `-8.0` formatting). NICER-style spec with `--supplementary-coordinate-code` + likelihood-factor trio renders correctly. |
| `test_marg_list.py` | Mono-driver assembly reproduces the Gaussian demo verbatim. Heterogeneous assembly (NICER + GW) preserves per-driver batch sizes and routes the per-driver coord-module override only to the entry that asked for it. |
| `test_config.py` | `validate_config` rejects empty configs / empty `marg-list` / non-positive `n-iterations` / missing `init`. Truthy coercion handles "true"/"True"/"1"/`True`/`1`. |
| `test_drivers.py` | `MargDriverBase` read→mutate→write round-trips without numpy fixed-width truncation. The actual `util_HyperMargGaussian.py` subprocess runs against a 5-point grid and the mode at `(4, 0, 0)` outranks the far point at `(10, 10, 10)`. |
| `test_hydra_integration.py` | The real Hydra entry point of `util_RIFT_hyperpipe.py` runs as a subprocess with CLI overrides, writes every expected artefact (`args_*.txt`, `event-0.net`, `event_nchunk.txt`, `initial_grid.dat`, `transfer_file_list.txt`), and the emitted `create_eos_posterior_pipeline` command line carries every flag we asked for. |

## When something fails

* **`RIFT root not found`** in test skip messages: you're outside a RIFT
  clone, or this suite has been moved away from
  `test/hyperpipe/tests/`. `pixi run which-rift` should print a
  sensible path.
* **lalsuite solver issues on Apple Silicon**: conda-forge's lalsuite is
  reasonably supported on `osx-arm64` now; if you hit solver trouble,
  try `pixi install --platform osx-64` (Rosetta).
* **`PYTHONPATH` collisions** with an existing system RIFT install:
  inside `pixi shell`, `python -c "import RIFT; print(RIFT.__file__)"`
  should resolve to the in-tree copy. If not, check whether some
  outer activation script is overriding `PYTHONPATH`.

## Adding a new test

The `hp_modules` fixture in `conftest.py` exposes the four lightweight
hyperpipe modules (`coords`, `config`, `marg_list`, `drivers_base`) as
attributes on a namespace. New tests should follow the existing pattern:

```python
def test_my_new_thing(hp_modules, tmp_path):
    spec = hp_modules.coords.HyperCoordSpec.from_strings(
        coords_fit="x y", coords_sample="x:[0,1] y:[0,1]",
    )
    ...
```

For tests that need to run `util_RIFT_hyperpipe.py` or `util_HyperMargGaussian.py`
as a subprocess (i.e. exercise the real Hydra entry point), use the
`rift_py` / `rift_bin` fixtures to find the scripts and propagate
`PYTHONPATH` to the child via `os.environ.copy()` — see
`test_hydra_integration.py` for the canonical example.
