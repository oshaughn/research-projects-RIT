# Simulation archive design

This document describes the on-disk format and core abstractions for a
RIFT simulation archive. The goals are:

1. **Self-contained.** An archive is a directory you can copy, ship, and
   read months later without access to the originating source tree.
2. **Cheap to query.** Listing what's in the archive, what's complete,
   and basic per-sim summaries should not require opening every output
   file. There is a single `index.jsonl` for that.
3. **Two-tier queueing.** The queue used to *request* simulations is
   independent of the queue used to *run* them. A typical OSG setup is
   "Condor DAG on the submit host requests sims; a Slurm cluster runs
   them." A test setup might be "local request, local run." The archive
   does not care.
4. **Frozen code.** The user-supplied generator is captured at archive
   creation time and stored under `code/`. Workers `exec` the frozen
   copy. The archive is portable without the original repository.

The existing classes in `BaseManager.py` / `CondorManager.py` /
`SlurmManager.py` evolve toward this design; the new schemas live
alongside them in `database.py` (stub, this commit) and can be adopted
incrementally.


## On-disk layout

```
<base_location>/
  manifest.json                 # archive-level metadata (schema, name, code refs, queue config)
  code/                         # frozen interfaces — required to *run* workers, not to read summaries
    generator.py                # entrypoint: generator.run(params, sim_dir) -> writes output
    summarizer.py               # OPTIONAL: summarize(sim_dir) -> dict (used post-completion)
    requirements.txt            # OPTIONAL: package versions known at archive creation
    README                      # human note: what this archive is, how it was built
  index.jsonl                   # one JSON object per line, one line per sim — canonical and cheap
  sims/
    <sim_name>/                 # one subdirectory per sim (sim_name is the archive's internal id)
      params.json               # canonical input parameters
      status.json               # FSM state + per-queue tracking + history
      summary.json              # summarizer output (written when status -> complete)
      output.<ext>              # raw output written by the worker; format up to the generator
      logs/
        worker.stdout
        worker.stderr
        run.log                 # generator-internal logging if any
  request_queue/                # workspace for the queue that *requests* sims
    dags/                       # condor DAGs (when request_queue.kind == 'condor')
    *.sub
  run_queue/                    # workspace for the queue that *runs* sims
    workers/                    # generated per-sim worker scripts
    submit_files/
  state.json                    # cached cluster ids, last-poll timestamps, etc.
```

Anything that's "configuration we may want to overwrite atomically"
lives at the archive root; anything per-sim lives under
`sims/<sim_name>/`. Anything queue-system-specific lives under
`request_queue/` or `run_queue/` — the archive itself stays neutral.


## Manifest

`manifest.json` at the archive root, written once at archive creation
and updated only when archive-level configuration changes (e.g. a queue
swap). Schema:

```jsonc
{
  "schema_version": 1,
  "name": "GW190521_grid_v3",
  "created_at": "2026-04-26T15:00:00Z",
  "rift_version": "0.0.18.0",
  "code": {
    "generator":            "code/generator.py",
    "generator_entrypoint": "generator:run",       // module:callable
    "summarizer":           "code/summarizer.py",
    "summarizer_entrypoint":"summarizer:summarize",
    "requirements":         "code/requirements.txt"
  },
  "request_queue": { "kind": "condor", "extra": { } },
  "run_queue":     { "kind": "slurm",  "extra": { "partition": "compute" } },
  "params_schema": { "type": "scalar", "dtype": "float" },   // hint only
  "summary_schema": { "fields": { "value": "float", "wall_time_s": "float" } }
}
```

The split between `code.generator_entrypoint` ("module:callable")
matches the `setuptools` convention and works equally well for a
single-file generator and a multi-file package frozen under `code/`.


## Index

`index.jsonl` is one record per sim. It is the **fast, canonical view**:
listing the archive, filtering by status, computing aggregate summaries
all use this file and never touch `sims/<sim_name>/` directly. The full
per-sim story lives in `status.json` + `summary.json`; the index is a
projection.

```jsonl
{"name":"1","status":"complete","params":0.5,"summary":{"value":0.7071}}
{"name":"2","status":"running", "params":1.5,"summary":null}
{"name":"3","status":"ready",   "params":2.5,"summary":null}
```

Append-only with periodic compaction. A run that registers many sims
appends one line per sim under a single index lock; status updates
compact (rewrite) the file under the same lock. Compaction is cheap
because the file is JSON-lines and rarely larger than a few MB even
for 100k sims.


## Per-sim status

`sims/<sim_name>/status.json`. Contains the FSM state plus per-queue
tracking and a transition log. The FSM is the same as today's
`QUEUE_STATES`:
`ready -> submit_ready -> running -> complete | stuck`.

```jsonc
{
  "name": "1",
  "params": 0.5,
  "status": "running",
  "history": [
    { "status": "ready",        "ts": "2026-04-26T15:00:00Z" },
    { "status": "submit_ready", "ts": "2026-04-26T15:00:05Z" },
    { "status": "running",      "ts": "2026-04-26T15:00:30Z" }
  ],
  "request_queue": { "kind": "condor", "cluster_id": 12345, "proc_id": 0, "dag_node": "sim_1" },
  "run_queue":     { "kind": "slurm",  "job_id": 98765, "host": "compute-7" },
  "output": { "path": "sims/1/output.npy", "size": null, "sha256": null },
  "started_at":   "2026-04-26T15:00:30Z",
  "completed_at": null
}
```

The `request_queue` and `run_queue` blocks are independent — the same
sim can be tracked in two systems at once. `kind` matches the value in
the manifest's queue spec.


## Frozen code

At archive creation, the user supplies one of:

1. **A Python callable.** The archive captures `inspect.getsource(fn)`
   and writes `code/generator.py` with a `run(params, sim_dir)`
   entrypoint. Closure captures and unusual import scopes are *not*
   supported; the function must be self-contained.
2. **A path to a `.py` file.** Copied verbatim. The user is responsible
   for naming the entrypoint per the manifest.
3. **A spec dict.** `{ "module_path": "...", "entrypoint": "mod:fn",
   "extra_files": ["aux.py", "data.csv"] }`. All listed files are copied
   into `code/`; `module_path` becomes `code/generator.py` (or whatever
   the spec names).

A worker invocation is then approximately:

```bash
python -c "import sys; sys.path.insert(0, 'code'); \
  from generator import run; run(params=..., sim_dir='sims/1')"
```

— so the archive is runnable without RIFT being installed, given a
Python interpreter and whatever the generator's own deps are. The
optional `code/requirements.txt` documents those.

Reading **summaries** does not need `code/` at all — `summary.json` is
plain JSON. Reading **outputs** may need `code/` if the format is
opaque (e.g. pickled objects), so a generator that writes `.npy`/`.h5`/
JSON is preferred.


## Similarity / dedup

Users come back. They request "the same" sim again. Sometimes "the
same" means bit-identical params; sometimes it means "the same up to
floating-point noise"; sometimes it means "the same up to a domain-
specific tolerance" (mass within 1e-6 solar masses, spin within 1e-3,
etc.). The archive should not redo work in any of these cases.

Two user-supplied callables, both optional, both frozen alongside the
generator under `code/`:

```python
def same_q(params_a, params_b) -> bool:
    """Reflexive, symmetric, transitive equality on parameters.
    Defaults to exact equality (params_a == params_b)."""

def lookup_key(params) -> "JSON-serializable":
    """Maps params to a coarse bucket for fast dedup.
    Must be consistent with same_q: same_q(a, b) == True implies
    lookup_key(a) == lookup_key(b). Defaults to str(params)."""
```

**`lookup_key` must be JSON-serializable, not merely hashable.** The
bucket key is a *persisted* value — written to `index.jsonl`, with the
dedup buckets rebuilt from that file on every `Archive` construction. So
the real requirement is that it survive the archive's JSON normalization
unchanged. That is both stricter and weaker than hashability: a
`frozenset` is hashable but cannot be persisted, while a plain `list` can
be persisted but is not hashable.

The engine normalizes on the way in and canonicalizes the same way on the
way out, so these are handled rather than silently breaking dedup:

* **tuples** — JSON has no tuple type, so a tuple key returns as a list;
  both canonicalize to the same form.
* **dict keys** — JSON coerces them to strings, and not via `str()`:
  `True`/`False`/`None` become `"true"`/`"false"`/`"null"`, float
  infinities `"Infinity"`. Keys colliding once coerced
  (`{True: 'a', "true": 'b'}`) collapse last-wins, consistently on both
  sides.

A key JSON cannot represent at all raises at `register()` with a message
naming this contract, rather than surfacing as a `sorted()` TypeError
from inside the index write.

The safest choice is a **string**: it round-trips to itself, sorts, and
cannot collide by coercion. RIFT's own `gw_pe_synthetic` returns a tuple,
which the normalization handles.

These together give O(1) average lookup: bucket by `lookup_key`, then
run `same_q` against only the (typically zero or one) entries in that
bucket. The archive keeps an in-memory `{lookup_key: [sim_name, ...]}`
map, rebuilt from `index.jsonl` on rehydration.

Each row in `index.jsonl` carries its `lookup_key` so the dedup index
can be rebuilt without ever calling the user's code (useful when
inspecting an archive on a machine that doesn't have the generator's
runtime deps).

`Archive.register(params)` becomes idempotent under same_q: it returns
the existing sim's name when a match is found, registers a fresh sim
otherwise. `Archive.find_existing(params) -> Optional[str]` is exposed
for callers that want to look without registering.

The manifest grows two optional `code.*` entries:

```jsonc
"code": {
  "generator": "code/generator.py",         "generator_entrypoint": "generator:run",
  "same_q":    "code/same_q.py",            "same_q_entrypoint":    "same_q:same_q",
  "lookup_key":"code/lookup_key.py",        "lookup_key_entrypoint":"lookup_key:lookup_key"
}
```

Typical usage, gravitational-wave style:

```python
def lookup_key(p):
    # Round into a coarse grid; tolerance below same_q's threshold.
    return (round(p["mtotal"], 4),
            round(p["q"],      4),
            round(p["chi1z"],  3),
            round(p["chi2z"],  3))

def same_q(a, b):
    return (abs(a["mtotal"] - b["mtotal"]) < 1e-3 and
            abs(a["q"]      - b["q"])      < 1e-3 and
            abs(a["chi1z"]  - b["chi1z"])  < 1e-2 and
            abs(a["chi2z"]  - b["chi2z"])  < 1e-2)
```


## Refinement (iterative improvement)

A simulation is rarely "done" in a binary sense. For GW parameter
inference the lnL at a single (mc, eta) is computed via a Monte Carlo
integral; the answer can always be made tighter by running more
samples. A user might come back and say "this point doesn't have the
precision I need — improve it."

The archive models this as a `level` counter per sim:

* Each sim has a `target_level` (what's been requested) and a
  `current_level` (what's been computed). They're both integers,
  starting at 0.
* A sim's output is a sequence of per-level files:
  `sims/<name>/level_1.json`, `level_2.json`, ... Each level is
  self-contained (or builds on its predecessors — that's the
  generator's choice).
* The summarizer can fold all completed levels into a single rolling
  summary (e.g. weighted average of lnL across levels with errors that
  shrink with sample count).

The FSM extends:

```
ready          -> never run, target_level >= 1, current_level == 0
submit_ready   -> queued for the run system (next level to compute is current_level + 1)
running        -> a level is being computed
complete       -> current_level == target_level
refine_ready   -> output exists (current_level >= 1) but target_level > current_level
stuck          -> failure
```

`Archive.register(params, target_level=N)` is *idempotent on bump*:
- doesn't exist: register fresh with `target_level=N`
- exists at `current_level >= N`: return name, no work
- exists at `current_level < N`: bump `target_level` to `max(existing, N)`,
  transition to `refine_ready`, return name

`Archive.refine(name, target_level=N)` is the explicit form for
callers that already know the sim_name.

The generator signature is `(params, sim_dir, level, prev_levels)`:
- `level`: integer, the level being computed right now
- `prev_levels`: list of paths to outputs from levels 1..level-1, so
  the generator can read prior work and extend it (e.g. a Monte Carlo
  generator can accumulate samples from earlier levels)

Each `register_simulation` invocation in the run queue may compute one
or many levels. The simplest run queue computes one level per submit
and re-submits the sim for the next level. A smarter run queue might
batch level-1-through-N for a sim into a single condor job to amortize
startup cost.

Status records grow a `levels[]` list:

```jsonc
{
  "name": "1",
  "params": {"mc": 25.0, "eta": 0.22},
  "status": "complete",
  "target_level": 4,
  "current_level": 4,
  "levels": [
    { "level": 1, "output_path": "sims/1/level_1.json", "completed_at": "..." },
    { "level": 2, "output_path": "sims/1/level_2.json", "completed_at": "..." },
    { "level": 3, "output_path": "sims/1/level_3.json", "completed_at": "..." },
    { "level": 4, "output_path": "sims/1/level_4.json", "completed_at": "..." }
  ],
  "summary": { "lnL_mean": -123.4, "lnL_stderr": 0.05, "n_samples": 4000 }
}
```


## Two-tier queueing

Two interfaces, composed by the archive. The same archive can mix and
match: `(LocalRequestQueue, LocalRunQueue)` for tests,
`(CondorRequestQueue, SlurmRunQueue)` for an OSG-fed cluster,
`(CondorRequestQueue, CondorRunQueue)` if the request and run queues
are the same condor pool, etc.

```python
class RequestQueue:
    """Decides which sims should next be sent to the run queue, and
    tracks their state in the request system. The request queue does
    NOT execute sims; it forwards them to the run queue."""
    kind: str
    def submit_pending(self, archive) -> list[str]: ...   # 'ready'   -> 'submit_ready'/'running'
    def poll(self, archive)            -> dict[str, str]: ... # name -> request-side status

class RunQueue:
    """Actually runs sims (writes the output file)."""
    kind: str
    def build_worker(self, archive, sim_name) -> str: ...      # path to per-sim worker script
    def submit(self, archive, sim_names)      -> list[tuple[str, str]]: ...   # (name, run_queue_id)
    def poll(self, archive, sim_names)        -> dict[str, str]: ... # name -> run-side status
```

In a production OSG setup the implementations look like:
- `CondorRequestQueue.submit_pending`: builds a DAG over all `ready`
  sims, calls `condor_submit_dag`, stamps `submit_ready`.
- `CondorRequestQueue.poll`: queries the schedd via `_htcondor_module`
  (htcondor or htcondor2), updates `request_queue.cluster_id` etc.
- `SlurmRunQueue.build_worker`: writes `run_queue/workers/<name>.sh`
  that loads `code/generator.py` and calls `run(params, sim_dir)`.
- `SlurmRunQueue.submit`: `sbatch` per worker, capture job ids.
- `SlurmRunQueue.poll`: `squeue --json` parse → status.

The archive composes them and owns the FSM transitions: queue
implementations *report* state, the archive *applies* it.


## Topology: dual condor

A common production setup at OSG is *two condor pools*: a "request"
pool on a dedicated submit host that runs the long-lived planner DAG,
and a "run" pool (typically OSG itself, or a remote site) that runs
the actual workers. The archive doesn't care that both happen to be
condor — they're configured as two separate `(RequestQueue, RunQueue)`
instances. Concretely:

```
+-------------------------------+        +---------------------------------+
| request pool (e.g. CIT submit)|        | run pool (e.g. OSG / remote site)|
|                               |        |                                  |
|   meta-DAG                    |        |   sub-DAG (one per scout pass)  |
|   ┌─────────────┐             |        |   ┌─────────────┐               |
|   │ scout       │             |        |   │ sim_a lvl 1 │               |
|   │ (read       │             |        |   │ sim_b lvl 1 │               |
|   │  archive,   │             |        |   │ sim_c lvl 2 │  (refine)     |
|   │  pick work) │             |        |   │ sim_a lvl 2 │  (refine)     |
|   └─────┬───────┘             |        |   └─────┬───────┘               |
|         │                     |        |         │                       |
|   ┌─────▼───────┐ -- submit ──┼────────┼──> writes outputs to shared FS  |
|   │ submit_run  │  via        |        |         │                       |
|   │ (condor_    │ condor_     |        |   ┌─────▼───────┐               |
|   │  submit_dag │ submit_dag  |        |   │ workers     │               |
|   │  -name run) │ -name run   |        |   │ (run gen)   │               |
|   └─────┬───────┘             |        |   └─────────────┘               |
|         │                     |        |                                  |
|   ┌─────▼───────┐             |        +---------------------------------+
|   │ wait/poll   │ <-- htcondor.Schedd(run_pool).query(...)
|   │ collect     │
|   └─────┬───────┘
|         │
|   ┌─────▼───────┐
|   │ iterate     │ -- back to scout
|   └─────────────┘
+-------------------------------+

                shared filesystem (the archive)
            <base>/manifest.json, index.jsonl, sims/...
```

The shared filesystem typically only spans the *submit* nodes of each
pool (and is most often the same machine for both pools, or two
machines NFS-cross-mounted). Execute hosts on the run pool — OSG
machines, leased EC2 nodes, etc. — generally have no view of the
archive directory. The framework's job is to ferry exactly the files
each (sim, level) worker needs onto the execute host, run there, and
ferry the level output back.

### File-transfer model (no shared FS at execute hosts)

Each (sim, level) worker is a self-contained condor job that:

1. **Receives** these files via `transfer_input_files`:
   * `code/` — the entire frozen code directory (generator,
     summarizer, etc.). Condor preserves the directory name in the
     sandbox.
   * `sims/<name>/params.json` — flattened by condor to `params.json`
     in the sandbox cwd.
   * `sims/<name>/level_1.json`, ..., `sims/<name>/level_<N-1>.json` —
     prior levels needed by an accumulating generator. Condor flattens
     these to `level_<i>.json` in cwd.
2. **Runs** a small bootstrap script (also transferred in `code/` or
   composed inline) that does:
       sys.path.insert(0, 'code'); from generator import run
       params = json.load(open('params.json'))
       run(params, sim_dir=os.getcwd(), level=N, prev_levels=['level_1.json', ...])
3. **Writes** `level_<N>.json` in cwd.
4. **Ships** that file back via `transfer_output_files = level_<N>.json`
   plus `transfer_output_remaps = "level_<N>.json = <abs_path>/sims/<name>/level_<N>.json"`
   so the file lands at the correct location in the archive on the
   submit node.

The submit-side helper `Archive.transfer_input_files_for(sim_name,
level)` returns the input list; `Archive.expected_output(sim_name,
level)` returns the output basename + remap target.

For very large inputs (NR catalogs, conditioned data) condor's
`osdf://` URLs are the right escape hatch — same pattern as the
existing CondorManager's `singularity_image` handling. The framework
should treat anything beyond a small `params.json + level files` budget
as a hint to push it through OSDF rather than `transfer_input_files`.

Neither pool talks to the other directly. The submit-side queue
implementations are responsible for assembling the input set, naming
the remap, and parsing condor events to know when transfer-back is
complete. Once the level file exists at its remapped location, the
archive's existing `refresh_status_from_disk` style sweep promotes the
sim to `complete` (or to `refine_ready` if more levels remain).

Implementation notes for the dual-condor queue classes:

* `DualCondorRequestQueue` runs on the request pool. Its
  `submit_pending(archive)` writes a sub-DAG file under
  `<base>/request_queue/dags/`, the sub-DAG's nodes target the *run*
  pool via condor's `-name <run_schedd>` argument, and returns the
  cluster id of the spawned `condor_submit_dag` process.
* `DualCondorRunQueue` runs on the run pool. Its `build_worker(sim,
  level)` writes a per-(sim,level) submit description under
  `<base>/run_queue/submit_files/<sim>_lvl<N>.sub`, with the executable
  set to a small bootstrap that does:
    `python -c "from generator import run; run(<params>, <sim_dir>, <level>, <prev>)"`
  (`generator` is loaded from the frozen `code/` dir).
* Polling on the request side queries the run pool's schedd:
  `htcondor.Schedd(<run_pool_collector>).query(...)`. The
  `_htcondor_module` caching machinery from `CondorManager.py` is
  reusable as-is.

The `kind` discriminator in the manifest stays as `"condor"` for both
queue records; the `extra` dict carries the `pool` / `schedd` / DAG
template differences.

## Topology: SLURM

The SLURM equivalent of `DualCondorRunQueue` is `SlurmRunQueue` (paired
with `SlurmRequestQueue`). Two important differences from condor:

* **Shared filesystem at compute nodes is assumed.** SLURM has no
  built-in `transfer_input_files`/`transfer_output_files` analog. The
  framework treats the archive directory as visible from the worker
  (the standard HPC model with NFS-mounted home/scratch). Sites
  without that property should arrange a Singularity container with
  bind-mounts (`singularity exec -B <archive>:<archive> ... python3 ...`)
  in the queue's `prelude` snippet, or use `sbcast` to broadcast a
  manifest to `/tmp` on the workers.
* **Inter-job dependencies use `--dependency=afterok:<jobid>`** rather
  than DAGMan's PARENT/CHILD edges. `SlurmRunQueue.submit` chains each
  sim's level submissions; level N's sbatch waits for level N-1 to
  exit cleanly.

The per-(sim, level) submit script is a normal `#!/bin/bash` script
with `#SBATCH` directives at the top, a `prelude` shell snippet (e.g.
`module load python/3.11`, virtualenv activate), `cd <sim_dir>`, and a
final `exec python3 <archive>/run_queue/workers/bootstrap.py …`. The
bootstrap is the same one condor uses; the only adaptation is the
`--code-dir` argument (passed as the absolute path to `<archive>/code`
because the worker's cwd is `<sim_dir>`, not a flattened sandbox).

OOM auto-recovery is not built into the SLURM submit description.
SLURM's standard idiom is `--requeue` plus operator-driven `scontrol
update jobid=… ReqMem=…`; the archive's `Archive.unstick(name,
bump_memory=True)` covers this from the framework side. For sites that
want fully automatic retry, wrap the sbatch in a re-submit harness in
the queue's `prelude`.

Configuration in the manifest's `run_queue.extra`:

```yaml
partition:           compute        # required
time_limit:          "02:00:00"     # default 01:00:00
nodes:               1
ntasks:              1
cpus_per_task:       1
request_memory:      4096           # MB
account:             ligo-research  # or env SLURM_ACCOUNT
qos:                 normal         # or env SLURM_QOS
prelude: |
  module load python/3.11
  source /home/me/rift-env/bin/activate
partition_extra:                    # extra `--key=value` directives
  gres:              gpu:1
  exclusive:         ""
```

Per-sim overrides supported via `Archive.set_resources(name, ...)`:
`request_memory`, `time_limit`, `partition`, plus
`extra_sbatch_directives` (a dict; merged into the queue's
`partition_extra`).


### Configuration defaults

`DualCondorRunQueue` reads the following from the constructor first,
then falls back to environment variables, then to hard-coded defaults:

| Setting                 | Constructor kwarg        | Env-var fallback | Default                                                |
|-------------------------|--------------------------|------------------|--------------------------------------------------------|
| Accounting group        | `accounting_group`       | `LIGO_ACCOUNTING`| (omit)                                                 |
| Accounting group user   | `accounting_group_user`  | `LIGO_USER_NAME` | (omit)                                                 |
| Memory request          | `request_memory`         | —                | 4096 MB                                                |
| Disk request            | `request_disk`           | —                | `4G`                                                   |
| `getenv` condor command | `getenv`                 | `RIFT_GETENV`    | `LD_LIBRARY_PATH,PATH,PYTHONPATH,*RIFT*,LIBRARY_PATH`  |
| Run-pool schedd `-name` | `run_pool`               | —                | local                                                  |
| Cross-pool collector    | `run_collector`          | —                | local                                                  |

The `getenv` default is an *allowlist* rather than `True` because
`getenv = True` is blocked by many sites (CIT among them). The
allowlist is the OSG-blessed convention from `docs/source/osg.rst`.
Sites that permit blanket env forwarding can set `RIFT_GETENV=True`
in the submit-side environment, pass `getenv='True'` to the
`DualCondorRunQueue` constructor, or set `run_queue.extra.getenv` in
the manifest.

The env-var fallback matches the legacy `CondorManager` behavior, so
existing LIGO/IGWN submit hosts that already export `LIGO_ACCOUNTING`
and `LIGO_USER_NAME` work without manifest changes. Manifest values in
`run_queue.extra` override constructor defaults, which override env
vars — manifest > constructor > env > hard-coded.

OSG site-selection knobs (`+DESIRED_SITES`, `+UNDESIRED_SITES`,
`Requirements`, etc.) are passed via the constructor's
`extra_condor_cmds` dict (or `run_queue.extra.extra_condor_cmds` in
the manifest). The bindings appear verbatim as additional `key =
value` lines in every per-(sim, level) submit description.

Because those lines are emitted **last**, a key that the queue already
writes would replace its line rather than extend it — and
`condor_submit` reports success either way. `transfer_input_files`,
`transfer_output_files`, `transfer_output_remaps` and
`periodic_release` are therefore refused in `extra_condor_cmds`
(case-insensitively; HTCondor command names are). Each has an
append-only alternative:

| instead of `extra_condor_cmds[...]` | use |
|---|---|
| `transfer_input_files` | `extra_transfer_input_files` (appended) |
| `transfer_output_files` | `extra_transfer_output_files` (appended, `{level}`/`{sim_name}` substituted) |
| `periodic_release` | `extra_periodic_release` (OR'd in) |

`extra_periodic_release` takes a single-line ClassAd expression for
sites whose pool holds jobs for reasons the queue does not model — an
opportunistic pool produces transient holds a dedicated cluster never
sees. While `auto_release_on_oom` is on, hold codes 26 and 34 belong to
the OOM policy and the term is scoped away from them, so
`oom_max_retries` remains a real cap and `request_memory` cannot be
multiplied without bound; the term governs every other hold code. With
the OOM policy off it governs all of them.

```python
DualCondorRunQueue(
    auto_release_on_oom=True,
    extra_periodic_release="(HoldReasonCode =!= 1) && (NumJobStarts < 50)",
)
```


## Hyperpipeline / glue.pipeline integration

Many existing RIFT workflows are built with `glue.pipeline.CondorDAG`
(see `RIFT/misc/dag_utils.py`, the various `bin/create_*_pipeline*`
scripts, and the hyperpipeline). The simulation archive must drop into
that ecosystem without forcing a top-level rewrite: the user adds *one
node* to their DAG that says "ensure simulation X is present at level
N", and downstream nodes simply add it as a parent.

The integration is two pieces:

1. **`request_sim` CLI** (`RIFT.simulation_manager.cli.request_sim`).
   A small executable that opens an archive, calls `Archive.register`,
   dispatches run-pool work via the configured request_queue if
   needed, and (in `--ensure` mode) blocks until the requested level
   is satisfied. Three modes: `--ensure` (the DAGMan-friendly default),
   `--check` (poll-once and exit), `--submit-async` (dispatch and
   return). Exit code drives DAGMan node success.

2. **`RequestSimulationJob`** (`RIFT.simulation_manager.glue_compat`).
   A `glue.pipeline.CondorDAGJob` subclass whose executable is the CLI
   above. Construct once per archive/profile; call `make_node(params,
   target_level)` to get a `CondorDAGNode` that the user attaches to
   their existing DAG. Per-node values arrive as standard glue macros
   (`$(macro_params)`, `$(macro_target_level)`).

A consumer's hyperpipeline-style flow looks like:

```python
from glue import pipeline
from RIFT.simulation_manager.glue_compat import RequestSimulationJob

dag = pipeline.CondorDAG(log='hyperpipe.dag.log')

sim_job = RequestSimulationJob(
    archive_path='/data/archives/lnL_grid',
    accounting_group='ligo.dev.o4.cbc.pe.lalinferencerapid',
    accounting_group_user='albert.einstein',
    timeout=3600,
)
sim_job.set_sub_file('request_sim.sub')
sim_job.write_sub_file()

for params in points_of_interest:
    sim_node = sim_job.make_node(params=params, target_level=4)
    dag.add_node(sim_node)
    downstream_node.add_parent(sim_node)   # blocks until ready

dag.set_dag_file('hyperpipe')
dag.write_concrete_dag()
```

The CLI runs on the *submit node* (where the archive lives). The
parent DAG can therefore live on a different pool from the run pool;
the request-sim node only needs the archive's submit-node filesystem.
When the CLI dispatches further work via the configured request_queue,
the file-transfer machinery from the previous section ferries the
inputs out to execute hosts and the outputs back.

`glue.pipeline` is treated as an optional dependency of the
simulation_manager package: importing `glue_compat` is safe without it
(only the `RequestSimulationJob` constructor raises a clear error if
glue is actually missing). The CLI itself has no glue dependency.


## Lightweight example

`examples/sim_database_hello.py` registers three sims (params 0.5,
1.5, 2.5), uses `LocalRequestQueue + LocalRunQueue` (which run the
generator inline, no schedd), inspects `manifest.json` and
`index.jsonl`, and reloads the archive in a fresh `Archive(...)` call
to confirm everything round-trips. This is the smallest end-to-end
exercise of the design and should always pass without any cluster.
