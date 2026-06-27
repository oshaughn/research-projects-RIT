# Multi-GPU ILE fan-out

**What it does.** A single RIFT ILE "batchmode" job evaluates a contiguous block
of intrinsic grid points — `[--event, --event + --n-events-to-analyze)` — one at
a time on **one** GPU. (In production that block is large: the distexport runs use
`jobs per worker: 100`, i.e. 100 points per ILE job.) On a node where several GPUs
are reserved, the other GPUs sit idle.

The fan-out splits that block into **N disjoint shards run concurrently, one per
GPU**. Each shard is pinned with `CUDA_VISIBLE_DEVICES`, gets a disjoint
sub-range, and writes to a distinct `--output-file` prefix (`<orig>.gpu<dev>`) so
the per-point output files never collide. Downstream collection is unchanged:
`util_ILEdagPostprocess.sh` globs `CME*.dat` and `util_CleanILE.py` de-duplicates
by *parameter value*, not filename. Coverage is identical to the serial run (the
shards partition the range exactly), and the launcher returns the first non-zero
shard exit code, so condor retry/hold behaviour is preserved.

### Default policy: grab every GPU on the node

**GPU ILE jobs now fan out across all of the node's GPUs by default** (`RIFT_ILE_GPU_FANOUT`
unset ⇒ `all`). The job still **requests one GPU** (`request_GPUs=1`), so condor matching
is unchanged — it lands on any GPU node exactly as before — but at runtime the launcher
enumerates **every physical GPU** (`nvidia-smi`, ignoring `CUDA_VISIBLE_DEVICES`) and splits
the point block across all of them. On a 1-GPU node this is a byte-for-byte no-op; only
multi-GPU nodes change.

> **This assumes you reserve whole nodes.** Requesting 1 GPU but using all of them means
> condor still thinks the other GPUs are free — on a *shared* multi-GPU node it would step
> on co-scheduled jobs. If you are not reserving whole nodes, set the single-GPU fallback.

**Fallback to the old single-GPU run:** `RIFT_ILE_GPU_FANOUT=1` (or `single` / `off`, or
`--ile-gpu-fanout 1`). Then `ile_pre.sh` just `exec`s the ILE binary exactly as before.

Why this default and not the partitionable-slot `auto-max-N`? HTCondor *does* support
partitionable GPU slots today (verified on the CIT pool: a 2-GPU partitionable slot carves
per-GPU dynamic slots), but that path is not considered sustainable long-term, so the
default is the reservation-based `all` instead.

---

## How it is wired (and why it works under asimov)

The fan-out lives entirely in **generated submit files**, decided at **DAG-build
time**:

- `ile_pre.sh` wraps the ILE binary in a tiny launcher (`RIFT.misc.dag_utils*.
  ile_invocation_shell`). The chosen fan-out value is **baked into `ile_pre.sh`**
  as `export RIFT_ILE_GPU_FANOUT="${RIFT_ILE_GPU_FANOUT:-N}"`, so the running job
  needs **no environment variable** — critical because asimov submits in a clean
  shell. (A runtime `RIFT_ILE_GPU_FANOUT` still overrides the baked default.)
- The ILE `.sub` gets `request_GPUs=N` and `request_CPUs=N`, so HTCondor only
  matches a slot/node that actually has N GPUs, and gives you N CPUs to drive them.

> **Requires the `ile_pre.sh` path.** The launcher is the `ile_pre.sh` the pipeline
> generates for frames-based / singularity / OSG-file-transfer runs (the standard
> production configuration). A bare `--fake-data-cache` run with no frames does not
> emit `ile_pre.sh`, so there is nothing to fan out there.

---

## Setting the policy

The default (`all`) needs no action — GPU ILE jobs fan out across the node automatically.
To **override** it (a fixed count, or the single-GPU fallback), set `RIFT_ILE_GPU_FANOUT`
any of these ways (all resolve to the same baked value):

| Context | How |
|---|---|
| **Env var** (interactive) | `export RIFT_ILE_GPU_FANOUT=1` before `util_RIFT_pseudo_pipe.py` (e.g. fallback) |
| **CLI flag** (direct) | `util_RIFT_pseudo_pipe.py ... --ile-force-gpu --ile-gpu-fanout 1` (also on `create_event_parameter_pipeline_BasicIteration`) |
| **asimov blueprint** | `scheduler.environment variables: {RIFT_ILE_GPU_FANOUT: 1}` **or** `scheduler.pipeline: {ile-gpu-fanout: 1}` |

### Values

| Value | `request_GPUs` (condor) | What the launcher splits across | Use when |
|---|---|---|---|
| `all` **(unset ⇒ default)** | 1 | **every physical GPU** (ignores `CUDA_VISIBLE_DEVICES`) | **dedicated / whole node you reserved** |
| `1` / `single` / `off` | 1 | — (no fan-out) | **fallback: shared node / single-GPU run** |
| `N` (int) | `N` | the N granted GPUs (adapts down if fewer) | every node has exactly N GPUs |
| `auto-max-N` | **expression** ≤ N | exactly the GPUs condor granted | shared pool, partitionable GPU slots (not the recommended path) |
| `auto` | 1 | the GPUs condor granted | you sized a multi-GPU slot some other way |

**The runtime split is already fully adaptive** — the launcher splits the point block
across however many GPUs it is handed (1, 2, 3, 4 …), always covering the whole block.
So "adapt to the number found" is solved regardless of value. The only real question is
how to make condor *hand you* a variable number; that is what the two adaptive values do:

- **`auto-max-N` (shared pool):** HTCondor's plain `request_GPUs` is a single fixed count,
  so it cannot natively say "give me 1 to 4". This value instead emits a ClassAd
  **expression** for `request_GPUs`/`request_CPUs` that asks for *up to N of the
  capability-matching GPUs available on the matched (partitionable) slot* — so ONE job
  flavour lands on a 1/2/3/4-GPU slot and grabs them all, and the launcher (`auto`) fans
  out across exactly that many. The default expression is
  `ifThenElse(countMatches(RequireGPUs,AvailableGPUs) >= N, N, ifThenElse(... >= 1, ..., 1))`
  (same `countMatches` idiom RIFT already uses for cross-platform GPU matching). It
  **requires partitionable slots and a GPU-aware negotiator** — verify on your pool with
  `condor_status -long <gpu-node> | grep -i gpu`, and if the attribute differs, override the
  whole expression with `RIFT_ILE_GPU_REQUEST_EXPR='<your expr>'` (no code change).

- **`all` (dedicated node):** if you already reserve whole nodes, keep `request_GPUs=1`
  (so it matches a node with *any* number of GPUs) and let the launcher enumerate **all
  physical GPUs** via `nvidia-smi`, ignoring `CUDA_VISIBLE_DEVICES`. Simplest, needs no
  partitionable-slot support. **Caveats:** only safe when the node is exclusively yours
  (otherwise you would step on co-scheduled jobs), and it assumes condor is *not* cgroup-
  isolating the GPU devices (with strict device isolation a shard pinned to a non-granted
  GPU cannot use it). It also still requests 1 CPU — bump `request_CPUs` if your scheduler
  confines CPUs too.

Is a variable request "even possible"? Not as a plain `request_GPUs` number — that is one
value. It *is* possible either as the `auto-max-N` expression (condor sizes the dynamic
slot to what's available) or by reserving the node and using `all`. Pick by pool type.

---

## Demonstrations in this directory

### 1. `make smoke-local` — proof the split + pin logic works (runs anywhere)

Builds a **real** `ile_pre.sh` from the shipped helper, wrapping a stub ILE
(`fake_ile.py`), and runs it across this node's GPUs. No cupy, no condor, no data —
it isolates the only new logic. Asserts every grid point is covered exactly once,
spread across the GPUs, with distinct per-shard output prefixes:

```
make smoke-local                 # uses nvidia-smi to find GPUs
make smoke-local FANOUT=2        # split across 2
make smoke-local DEVICES=0,1,2,3 # force a device list (shared node)
```

### 2. `make build` + `make verify` — a real pipeline run dir

Builds a pipeline on the CI synthetic data (same data as `demo/rift/calmarg`),
in singularity/OSG mode so `ile_pre.sh` is generated, with the fan-out baked in.
Needs the container-family manifest:

```
export SINGULARITY_RIFT_IMAGE=$(pwd)/blueprints/rift_container_family.cit.yaml
export SINGULARITY_BASE_EXE_DIR=/usr/local/bin/
make build FANOUT=4
make verify        # asserts ILE.sub has request_GPUs=4/request_CPUs=4
                   #   and ile_pre.sh bakes RIFT_ILE_GPU_FANOUT=4
make inspect       # show the generated launcher + sub resource lines
```

`make build` only builds the DAG (it does not submit). To actually run it you need
a pool with ≥ N-GPU nodes; submit with `condor_submit_dag` from the run dir.

### 3. `blueprints/` — the asimov path

- `rift-multigpu.yaml` — analysis blueprint showing **both** blueprint encodings
  (environment-variable and pipeline-CLI) plus host/GPU matching.
- `rift_container_family.cit.yaml` — frozen container-family pin (per-machine image
  selection + GPU capability floor).

Apply with `asimov apply -f blueprints/rift-multigpu.yaml` (after the matching
event blueprint). The RIFT asimov pipeline (`RIFT/asimov/rift.py`) copies
`environment variables` into the build environment / turns `pipeline` keys into
`util_RIFT_pseudo_pipe.py` flags, so the value reaches the DAG build with **no
assumption about the submit shell**.

---

## Host / GPU matching

`request_GPUs=N` is the primary matcher: HTCondor will only place the job where N
GPUs are available. Layer on a capability floor and host pins as needed:

- `scheduler.gpu architectures:` → `RIFT_REQUIRE_GPUS` device exclusions (drop slow
  cards), and `RIFT_REQUIRE_GPUS='(Capability >= 8.0)'`-style floors via the env.
- `scheduler.avoid hosts:` → `RIFT_AVOID_HOSTS` (blacklist single-GPU or bad nodes).

**Fan-out targets local / dedicated multi-GPU pools.** OSG glide-in slots typically
expose one GPU each, so `request_GPUs=4` will not match there — use your local pool
(`osg: False`, or a site requirement) for fan-out runs.

---

## Sizing

- `jobs per worker` (= `--ile-n-events-to-analyze`, the per-job block) should be a
  comfortable multiple of the fan-out, e.g. 100 points / 4 GPUs = 25 points/GPU.
- GPU memory is per shard on its own device, so N shards on N distinct GPUs do not
  contend; an A100 (80 GB) runs one ILE shard with room to spare.
