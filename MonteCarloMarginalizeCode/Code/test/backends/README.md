# `dag_utils_generic` test suite

This directory exercises `RIFT.misc.dag_utils_generic` -- the backend-neutral
workflow / DAG layer that replaces the old `glue.pipeline`-only
`dag_utils.py`.  Three backends are tested:

| name       | runtime artefact                         | dependency                    |
| ---------- | ---------------------------------------- | ----------------------------- |
| `htcondor` | `*.sub` + DAGMan `*.dag`                 | `pip install htcondor`        |
| `glue`     | `*.sub` + DAGMan `*.dag` (via glue)      | `lscsoft-glue` / `lalsuite`   |
| `slurm`    | `*.sbatch` + shell `*_dag.sh` driver     | none (just a shell)           |

Each test/demo runs **the same workflow code** under all three backends and
compares the artefacts.

## Layout

```
test/backends/
  test_backends_lowlevel.py                # unittest suite for the backend abstraction
  synthesize_pipeline_outputs.py           # self-contained pipeline emitter (no science deps)
  demo_create_event_parameter_pipeline_BasicIteration.sh
  demo_create_eos_posterior_pipeline.sh
  demo_util_RIFT_pseudo_pipe.sh
  run_all_backends.sh                      # top-level runner
  inputs/                                  # minimal fixtures used by the shell demos
  outputs/                                 # populated by the runner; .gitignored in production use
    htcondor/<pipeline>/...
    glue/<pipeline>/...
    slurm/<pipeline>/...
```

## How to run

### Quick sanity check (no RIFT science stack needed)

```sh
cd MonteCarloMarginalizeCode/Code/test/backends
python3 test_backends_lowlevel.py
python3 synthesize_pipeline_outputs.py
```

The first command runs the unittest suite (15 tests) for the abstraction
itself: data-model state preservation, registry behaviour, custom backend
registration, and per-backend artefact emission for each of htcondor, glue
and slurm.

The second command runs a hand-written *synthesizer* that uses
`dag_utils_generic` directly to build workflows whose structure mirrors
what the real production pipelines emit -- enough to inspect what each
backend produces, without needing `lalsimulation` / `scipy` / a working
RIFT install.  Outputs land under `outputs/<backend>/<pipeline>/`.

### Full run (requires a working RIFT install)

```sh
cd MonteCarloMarginalizeCode/Code/test/backends
bash run_all_backends.sh
```

This additionally drives the real pipeline scripts -- the intermediate
`create_event_parameter_pipeline_BasicIteration` and
`create_eos_posterior_pipeline`, plus the high-level
`util_RIFT_pseudo_pipe.py` -- once for each backend.  Each invocation sets
`RIFT_DAG_BACKEND` to select the active backend, and writes its artefacts
under `outputs/<backend>/<pipeline>/`.

The driver scripts use minimal placeholder fixtures from `inputs/`; the
goal is to exercise the DAG-emission stage, not to actually submit a job.
For real production runs you would point them at a valid initial grid /
event coinc / etc.

### Selecting a single backend interactively

```sh
RIFT_DAG_BACKEND=slurm python3 ../../bin/util_RIFT_pseudo_pipe.py …
```

This is the same mechanism the demos use.  Recognised values:

| `RIFT_DAG_BACKEND` | meaning                                                |
| ------------------ | ------------------------------------------------------ |
| `auto` (default)   | prefer `htcondor`, fall back to `glue`                 |
| `htcondor`         | force the htcondor python bindings backend            |
| `glue`             | force the legacy `glue.pipeline` backend               |
| `slurm`            | force the Slurm backend                                |
| any other name     | a backend the user registered via `register_backend()` |

## Inspecting the outputs

After running the synthesizer, compare side-by-side how each backend
encodes the same workflow:

```sh
diff <(cat outputs/htcondor/cepp_basic_iteration/ILE-0.sub) \
     <(cat outputs/slurm/cepp_basic_iteration/ILE-0.sub)
```

The HTCondor version has `request_memory = 8192M / request_gpus = 1 /
queue 8` whereas the Slurm version has `#SBATCH --mem=8192M / --gres=gpu:1
/ --array=0-7`, plus the original Condor commands preserved as `#`
comments for traceability.

The DAG layer does the same trick: HTCondor emits a DAGMan `.dag` with
`JOB / VARS / RETRY / PARENT … CHILD …` lines; Slurm emits a shell driver
that submits `sbatch --dependency=afterok:JOBID …` chains.

## Adding a new backend

```python
from RIFT.misc.dag_utils_generic import (
    WorkflowBackend, register_backend, set_backend,
)

class PBSBackend(WorkflowBackend):
    name = "pbs"
    def emit_job(self, job, path): ...
    def emit_dag(self, dag, path): ...

register_backend(PBSBackend())
set_backend("pbs")
```

To add a test for it: copy `SlurmBackendTests` in
`test_backends_lowlevel.py` and adapt the directive assertions.
