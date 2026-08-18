"""Simulation archive database — schema and queue interfaces.

This module defines the v2 archive layout described in DESIGN.md:
manifest.json + index.jsonl + per-sim status.json/summary.json under
sims/<name>/, with frozen code under code/, and an explicit split
between RequestQueue (decides what to submit) and RunQueue (actually
runs the generator).

The existing classes in BaseManager.py / CondorManager.py /
SlurmManager.py keep working unchanged; this module is additive. Real
condor/slurm queue implementations should land here as small
subclasses of RequestQueue / RunQueue.

Status:
    - Manifest, Index, StatusRecord, code-freeze: implemented.
    - LocalRequestQueue + LocalRunQueue: implemented (no schedd; runs
      the frozen generator inline). Used by examples and tests.
    - DualCondorRequestQueue + DualCondorRunQueue: implemented. The
      run queue writes per-(sim, level) submit descriptions, assembles
      a chained DAG, and dispatches via condor_submit_dag (with -name
      <run_pool> for cross-pool). poll() queries the run pool's schedd
      via the cached _htcondor_module from CondorManager (htcondor /
      htcondor2). Output-on-disk drives final 'complete' transitions
      via Archive.refresh_status_from_disk.
    - SlurmRunQueue: not yet provided (stubbed out at the design
      level only). Build on simple_slurm or pyslurmutils; mirror the
      condor implementation.
    - make_queues_from_manifest(archive): instantiates queues from
      the manifest's request_queue / run_queue config. Used by
      cli/request_sim.py --ensure to attach queues automatically.
"""

from __future__ import annotations

import contextlib
import datetime
import errno
import inspect
import json
import logging
import os
import warnings
from types import MappingProxyType
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, Mapping

try:
    import fcntl   # POSIX-only; archive multi-writer safety relies on flock(2)
    _HAS_FCNTL = True
except ImportError:                                              # pragma: no cover
    fcntl = None
    _HAS_FCNTL = False

logger = logging.getLogger(__name__)

# Re-export the FSM constants so callers don't have to import BaseManager.
from .BaseManager import QUEUE_STATES  # noqa: F401

SCHEMA_VERSION = 1
DEFAULT_GENERATOR_FILE = "generator.py"
DEFAULT_SUMMARIZER_FILE = "summarizer.py"
DEFAULT_SAME_Q_FILE = "same_q.py"
DEFAULT_LOOKUP_KEY_FILE = "lookup_key.py"

# Safe default for the condor 'getenv' command. Many sites (CIT among
# them) refuse `getenv = True` outright. The allowlist below mirrors
# the OSG convention documented in docs/source/osg.rst:
#   RIFT_GETENV=LD_LIBRARY_PATH,PATH,PYTHONPATH,*RIFT*,LIBRARY_PATH
# A user who needs the legacy `True` behavior can either set
# RIFT_GETENV=True in the environment or pass getenv='True' to the
# DualCondorRunQueue constructor (or via run_queue.extra.getenv in the
# manifest).
DEFAULT_GETENV_ALLOWLIST = "LD_LIBRARY_PATH,PATH,PYTHONPATH,*RIFT*,LIBRARY_PATH"


def _freeze(x: Any) -> Any:
    """Recursively map a JSON-shaped value onto a hashable one.

    Lists and tuples collapse onto the same tuple form. Dicts become a
    tuple of (key, frozen-value) pairs sorted by key. Anything still
    unhashable falls back to the repr sentinel.
    """
    if isinstance(x, (list, tuple)):
        return tuple(_freeze(v) for v in x)
    if isinstance(x, dict):
        # Sort by key alone: after _safe_hashable's JSON pass the keys
        # are strings and unique, and sorting on the pair could otherwise
        # try to order two frozen values of unrelated types.
        return tuple(sorted(
            ((str(k), _freeze(v)) for k, v in x.items()),
            key=lambda kv: kv[0],
        ))
    try:
        hash(x)
        return x
    except TypeError:
        return ("__unhashable__", repr(x))


# Canonicalize a lookup_key into something hashable AND identical to what
# comes back out of index.jsonl, because dedup buckets are rebuilt from
# that file on every Archive construction — which makes the bucket key a
# persisted value.
#
# Getting this wrong is silent: the rehydrated bucket key stops matching
# the freshly-computed one, find_existing misses, and register() mints a
# duplicate sim for physics the archive already holds. The caller just
# pays twice, from the second session onward, with nothing in the logs.
#
# Rather than model JSON's coercion rules by hand, we run the value
# through an actual JSON round-trip first, so the canonical form matches
# the persisted form *by construction*. That covers, in one step, every
# way the two could otherwise diverge:
#
#   * tuples, which JSON has no type for, coming back as lists;
#   * dict keys, which JSON coerces to strings — and not via str():
#     True/False/None serialize as "true"/"false"/"null", and float
#     infinities as "Infinity", none of which str() reproduces;
#   * dict keys that collide once coerced ({True: 'a', "true": 'b'}),
#     which JSON collapses last-wins — applying the same round-trip
#     means fresh and rehydrated agree on the survivor instead of
#     disagreeing about how many entries there are.
#
# Values that JSON cannot represent at all (a tuple used as a dict key,
# say) fall through to _freeze on the original. Such a lookup_key could
# not have been persisted in the first place, so there is no rehydrated
# form for it to disagree with.
#
# Collisions this introduces between distinct inputs — a list and the
# equal tuple, say — are harmless: buckets only nominate same_q
# candidates, and same_q still makes the decision.
def _json_normalized(x: Any) -> Any:
    """The value as it will exist after a round-trip through index.jsonl.

    This is the form that must be *stored*, not merely the form used for
    bucketing. Normalizing only at bucket time is not enough: the index
    row keeps whatever `lookup_key` returned, and `Index._write_all`
    serializes rows with ``sort_keys=True``. A dict key set that JSON
    would coerce to strings is still raw at that point, so a key like
    ``{True: 'a', 'true': 'b'}`` reaches `sorted()` as a bool beside a
    str and raises

        TypeError: '<' not supported between instances of 'str' and 'bool'

    from inside `register`. Normalizing on the way in makes the stored
    value sortable and makes persisted and canonical forms identical by
    construction.
    """
    try:
        return json.loads(json.dumps(x))
    except (TypeError, ValueError, RecursionError):
        return x


def _safe_hashable(x: Any) -> Any:
    return _freeze(_json_normalized(x))


def _require_persistable_lookup_key(key: Any) -> Any:
    """Normalize a lookup_key for storage, or say clearly why it cannot be.

    Backends control `lookup_key`, and a value JSON cannot represent —
    a set, a frozenset, a tuple used as a dict key — cannot live in
    index.jsonl at all. Catching it here names the contract instead of
    surfacing a json/sorted TypeError from deep in the write path.
    """
    try:
        json.dumps(key)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "lookup_key must be JSON-serializable so it can be persisted in "
            "index.jsonl and compared after reopen; got {!r} ({}). Return a "
            "string, number, or a list/dict of them.".format(key, exc)
        ) from exc
    return _json_normalized(key)
def _reject_reserved_basename(entry: str, what: str) -> None:
    """Refuse an entry whose basename shadows a file the archive stages.

    Condor flattens basenames into the sandbox cwd, so on the input side
    this would overwrite the archive's own copy on the worker. On the
    OUTPUT side it is worse: the remap points back at sims/<name>/, so a
    returned `params.json` overwrites the sim's recorded inputs in the
    archive itself, corrupting state every later level reads.
    """
    base = entry.rstrip("/").rsplit("/", 1)[-1]
    if base in _RESERVED_SANDBOX_BASENAMES or (
            base.startswith("level_") and base.endswith(".json")):
        raise ValueError(
            "{}: {!r} has basename {!r}, which collides with a file the "
            "archive itself stages or writes.".format(what, entry, base))


def _reject_duplicate_basenames(entries: Sequence[str], what: str) -> None:
    """Refuse two entries that flatten to the same sandbox filename.

    Condor flattens basenames into the job's cwd, so
    `osdf:///siteA/data.h5` and `osdf:///siteB/data.h5` are two different
    objects that land on top of each other. The reserved-name check does
    not see this: neither entry collides with anything the archive
    stages, only with the other one.
    """
    seen = {}
    for entry in entries:
        base = str(entry).rstrip("/").rsplit("/", 1)[-1]
        if base in seen:
            # Identical entries count too: naming the same file twice is
            # at best a wasted transfer of a multi-GB object, and on the
            # output side it emits a duplicate remap pair. Two templates
            # that expand to the same name land here as equal strings.
            raise ValueError(
                "{}: {!r} and {!r} both resolve to {!r} in the job sandbox, "
                "so one would overwrite the other on the worker.".format(
                    what, seen[base], entry, base))
        seen[base] = str(entry)


def _validate_hold_codes(value: Any, *, what: str) -> Tuple[int, ...]:
    """Hold codes naming the condition a policy acts on.

    Deliberately data rather than an expression: these round-trip
    through the manifest as JSON, and they are what a site operator
    reads off their own infrastructure record. Order is preserved so
    the emitted expression is stable across runs.
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(
            "{0} must be a sequence of integer hold codes, got {1!r}".format(
                what, type(value).__name__))
    codes = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, int):
            # bool is an int subclass and `True` would silently become 1.
            raise TypeError(
                "{0} entries must be integer hold codes, got {1!r}".format(
                    what, entry))
        if entry not in codes:
            codes.append(entry)
    return tuple(codes)


def _validate_subcode_exclusions(value: Any, *, what: str
                                 ) -> Dict[int, Tuple[int, ...]]:
    """Sub-codes to carve out of a hold code, as {code: (subcode, ...)}.

    A hold code says which subsystem held the job; the sub-code says
    why. `SYSTEM_PERIODIC_HOLD` is the case that forces this to exist --
    every site expression it evaluates produces the same hold code, and
    only the sub-code distinguishes "over memory" from "restarted too
    many times".

    Keys are coerced from str, because JSON has no integer keys and
    these arrive back from the manifest as strings. Skipping that turns
    a configured exclusion into a silently ignored one after a round
    trip, which is the same class of bug as a lookup_key that is not
    JSON-stable.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "{0} must be a mapping of {{hold_code: [subcode, ...]}}, got "
            "{1!r}".format(what, type(value).__name__))
    out: Dict[int, Tuple[int, ...]] = {}
    for key, subs in value.items():
        if isinstance(key, bool):
            raise TypeError("{0} keys must be hold codes".format(what))
        if isinstance(key, str):
            try:
                key = int(key)
            except ValueError:
                raise TypeError(
                    "{0} key {1!r} is not a hold code".format(what, key))
        if not isinstance(key, int):
            raise TypeError(
                "{0} keys must be hold codes, got {1!r}".format(what, key))
        out[key] = _validate_hold_codes(
            subs, what="{0}[{1}]".format(what, key))
    return out


def _validate_release_expression(value: Any, *, what: str) -> str:
    """Check a ClassAd expression destined for a submit command.

    The expression is not parsed. The HTCondor python bindings are
    optional here, and a check that runs only where they happen to be
    installed is worse than no check at all: it moves the failure off
    the author's machine and onto someone else's. condor_submit rejects
    a malformed expression, loudly, at submit time.

    What is checked is the part that is not the author's own mistake to
    make. A newline ends a submit command, so a value carrying one --
    from a manifest, a config file, a `run_queue.extra` dict written by
    another tool -- would have its remainder read as further submit
    commands, free to set `getenv = True` or replace
    transfer_output_files. That is refused.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(
            "{0} must be a string ClassAd expression, got {1!r}".format(
                what, type(value).__name__))
    text = value.strip()
    if not text:
        return ""
    if "\n" in text or "\r" in text:
        raise ValueError(
            "{0} must be a single line: a newline would end the submit "
            "command and let the rest of the value be read as further "
            "commands".format(what))
    return text


def _validate_transfer_entries(entries: Any, *, what: str,
                               remap_syntax: bool = False) -> List[str]:
    """Check a backend-supplied transfer list, or say why it is unusable.

    Every rejection here is something HTCondor accepts without complaint
    and then gets wrong on a remote worker, which is the worst place to
    find out. `condor_submit` exits 0 for all of them.

      * a bare string is a Sequence[str], so it iterates as CHARACTERS
        and becomes one transfer request per letter. This is the likeliest
        operator mistake and the type annotation invites it.
      * transfer_input_files is comma-separated, so an entry containing a
        comma silently splits into two bogus entries. URLs with query
        strings hit this routinely.
      * a newline ends the submit command, so the remainder becomes its
        own submit line. Later duplicates win in Condor, so a stray
        newline can silently override request_memory, the executable, or
        the output remaps.
    """
    if entries is None:
        return []
    if isinstance(entries, (str, bytes)):
        raise TypeError(
            "{} must be a list of entries, not a bare string: a string is a "
            "Sequence[str] and would iterate as one transfer request per "
            "character. Wrap it: [{!r}].".format(what, entries))
    out: List[str] = []
    for entry in entries:
        text = str(entry)
        if not text.strip():
            raise ValueError("{}: empty entry".format(what))
        bad_chars = [(",", "separates entries in the transfer list"),
                     ("\n", "ends the submit command"),
                     ("\r", "ends the submit command")]
        if remap_syntax:
            # transfer_output_remaps is a ';'-separated list of name=path
            # pairs, so either character makes the remap unparseable.
            bad_chars += [(";", "separates pairs in transfer_output_remaps"),
                          ("=", "separates name from path in "
                                "transfer_output_remaps")]
        for bad, why in bad_chars:
            if bad in text:
                raise ValueError(
                    "{}: entry {!r} contains {!r}, which {}. HTCondor accepts "
                    "the submit file and the job fails later on the execute "
                    "host.".format(what, text, bad, why))
        out.append(text)
    return out


#: Hold codes this class treats as "the job ran out of memory", and the
#: attribute that rations retries. Both are DEFAULTS, not facts: what a
#: hold code means is a property of the site, not of HTCondor. 34 is the
#: unambiguous memory code; 26 is SystemPolicy, which means whatever the
#: site's SYSTEM_PERIODIC_HOLD expressions say it means -- on the LIGO
#: clusters this policy was written for that is usually memory, and on
#: an OSG access point it is as likely to be an anti-thrash limiter
#: whose precondition is a high NumJobStarts. Sites that differ pass
#: oom_hold_codes / oom_hold_subcode_exclusions / oom_retry_counter
#: rather than editing this.
#:
#: Deliberately NOT recorded here: which sites differ, and how. That
#: belongs in whatever inventory the operator already keeps about their
#: own infrastructure. A table of site facts in shared code is stale the
#: day after it is written and wrong for everyone it does not name.
DEFAULT_OOM_HOLD_CODES = (34, 26)
DEFAULT_OOM_RETRY_COUNTER = "NumJobStarts"


#: Submit commands the archive composes itself. A backend that sets any
#: of these through extra_condor_cmds replaces the archive's line rather
#: than extending it, because extra_condor_cmds is emitted last. Stored
#: casefolded: HTCondor command names are case-insensitive, so the guard
#: has to be too.
_PROTECTED_SUBMIT_COMMANDS = frozenset({
    "transfer_input_files", "transfer_output_files", "transfer_output_remaps",
    # periodic_release joined this set when extra_periodic_release gave it
    # a supported additive alternative. Setting it here replaced the
    # queue's line and silently discarded the auto_release_on_oom memory
    # policy -- the exact bug the additive hook exists to remove, which
    # would otherwise stay reachable, unguarded, right beside the fix.
    "periodic_release",
    # request_memory is the other half of the same policy. Replacing it
    # leaves periodic_release intact, so the job is released the full
    # oom_max_retries times at a fixed size and OOMs every time -- it
    # spends the whole budget achieving nothing, which is a worse end
    # than losing the release arm. Per-sim sizes go through
    # Archive.set_resources, which composes rather than substitutes.
    "request_memory",
})

#: What to use instead of each refused key. Kept beside the set so a new
#: entry cannot be added without answering "and what should they do?" --
#: a guard that refuses without a remedy just moves the dead end.
_PROTECTED_ALTERNATIVES = {
    "transfer_input_files": "extra_transfer_input_files, which appends",
    "transfer_output_files": "extra_transfer_output_files, which appends",
    "transfer_output_remaps": "extra_transfer_output_files, whose entries "
                              "accept remap syntax",
    "periodic_release": "extra_periodic_release, which is OR'd into the "
                        "expression instead of replacing it",
    "request_memory": "the request_memory argument, or "
                      "Archive.set_resources for a per-sim override",
}

#: Basenames the archive itself stages into the worker sandbox. Condor
#: flattens transferred basenames into cwd, so a backend input sharing one
#: of these silently clobbers it on the worker.
_RESERVED_SANDBOX_BASENAMES = ("code", "params.json")


def _default_same_q(a: Any, b: Any) -> bool:
    return a == b


def _default_lookup_key(p: Any) -> Any:
    return str(p)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# Archive locking is implemented as an instance-method context manager on
# Archive itself (see Archive._with_lock); reads stay unlocked.


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class Manifest:
    """archive-level metadata stored at <base>/manifest.json."""

    FILENAME = "manifest.json"

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    @classmethod
    def new(cls, name: str,
            request_queue_kind: str,
            run_queue_kind: str,
            generator_entrypoint: str = "generator:run",
            summarizer_entrypoint: Optional[str] = None,
            same_q_entrypoint: Optional[str] = None,
            lookup_key_entrypoint: Optional[str] = None,
            params_schema: Optional[Dict[str, Any]] = None,
            summary_schema: Optional[Dict[str, Any]] = None,
            rift_version: Optional[str] = None,
            request_queue_extra: Optional[Dict[str, Any]] = None,
            run_queue_extra: Optional[Dict[str, Any]] = None,
            ) -> "Manifest":
        data: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "created_at": _now(),
            "rift_version": rift_version or _detect_rift_version(),
            "code": {
                "generator": "code/" + DEFAULT_GENERATOR_FILE,
                "generator_entrypoint": generator_entrypoint,
            },
            "request_queue": {"kind": request_queue_kind,
                              "extra": request_queue_extra or {}},
            "run_queue":     {"kind": run_queue_kind,
                              "extra": run_queue_extra or {}},
        }
        if summarizer_entrypoint is not None:
            data["code"]["summarizer"] = "code/" + DEFAULT_SUMMARIZER_FILE
            data["code"]["summarizer_entrypoint"] = summarizer_entrypoint
        if same_q_entrypoint is not None:
            data["code"]["same_q"] = "code/" + DEFAULT_SAME_Q_FILE
            data["code"]["same_q_entrypoint"] = same_q_entrypoint
        if lookup_key_entrypoint is not None:
            data["code"]["lookup_key"] = "code/" + DEFAULT_LOOKUP_KEY_FILE
            data["code"]["lookup_key_entrypoint"] = lookup_key_entrypoint
        if params_schema is not None:
            data["params_schema"] = params_schema
        if summary_schema is not None:
            data["summary_schema"] = summary_schema
        return cls(data)

    def write(self, base: Union[str, Path]) -> None:
        path = Path(base) / self.FILENAME
        path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, base: Union[str, Path]) -> "Manifest":
        path = Path(base) / cls.FILENAME
        return cls(json.loads(path.read_text()))


def _detect_rift_version() -> Optional[str]:
    try:
        from importlib.metadata import version as _v
        return _v("RIFT")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class Index:
    """One JSON object per line, one line per sim. The canonical, cheap
    view of the archive. Status updates and registrations rewrite the
    file under a simple file lock (not implemented yet — single-writer
    assumption is documented for now)."""

    FILENAME = "index.jsonl"

    def __init__(self, base: Union[str, Path]):
        self.base = Path(base)

    @property
    def path(self) -> Path:
        return self.base / self.FILENAME

    def all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for row in self.all():
            if row.get("name") == name:
                return row
        return None

    def with_status(self, status: str) -> List[Dict[str, Any]]:
        return [r for r in self.all() if r.get("status") == status]

    def upsert(self, row: Dict[str, Any]) -> None:
        rows = self.all()
        for i, existing in enumerate(rows):
            if existing.get("name") == row["name"]:
                rows[i] = row
                break
        else:
            rows.append(row)
        self._write_all(rows)

    def remove(self, name: str) -> None:
        self._write_all([r for r in self.all() if r.get("name") != name])

    def _write_all(self, rows: Iterable[Dict[str, Any]]) -> None:
        # Use a per-PID/thread-id temp suffix so two writers (in pathological
        # cases where the archive lock isn't held — shouldn't happen in
        # normal use) don't stomp each other's temp file.
        tmp = self.path.with_name(
            "{}.{}.{}.tmp".format(self.path.name, os.getpid(),
                                   threading.get_ident()))
        with open(tmp, "w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(tmp, self.path)   # atomic on POSIX


# ---------------------------------------------------------------------------
# Per-sim status
# ---------------------------------------------------------------------------

class StatusRecord:
    """sims/<name>/status.json. The full per-sim story; the index is a
    projection of this. Mutations go through transition() so the history
    log and the FSM stay consistent."""

    FILENAME = "status.json"

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    @classmethod
    def new(cls, name: str, params: Any, target_level: int = 1) -> "StatusRecord":
        ts = _now()
        return cls({
            "name": name,
            "params": params,
            "status": "ready",
            "target_level": int(target_level),
            "current_level": 0,
            "levels": [],   # list of {"level": int, "output_path": str, "completed_at": str}
            "history": [{"status": "ready", "ts": ts}],
            "request_queue": None,
            "run_queue": None,
            "resources": {},  # per-sim overrides: request_memory, request_disk, extra_condor_cmds
            "started_at": None,
            "completed_at": None,
        })

    def transition(self, new_status: str, **fields: Any) -> None:
        if new_status not in QUEUE_STATES:
            raise ValueError("Unknown status %r; expected one of %s" %
                             (new_status, QUEUE_STATES))
        self.data["status"] = new_status
        self.data["history"].append({"status": new_status, "ts": _now()})
        if new_status == "running" and self.data["started_at"] is None:
            self.data["started_at"] = _now()
        if new_status == "complete" and self.data["completed_at"] is None:
            self.data["completed_at"] = _now()
        for k, v in fields.items():
            self.data[k] = v

    def append_level(self, level: int, output_path: str) -> None:
        """Record a successful level computation. Updates current_level
        and the levels[] list. Caller should subsequently transition
        to 'complete' or 'refine_ready' as appropriate."""
        self.data["levels"].append({
            "level": int(level),
            "output_path": output_path,
            "completed_at": _now(),
        })
        self.data["current_level"] = max(self.data.get("current_level", 0), int(level))

    def bump_target(self, target_level: int) -> bool:
        """Raise target_level if `target_level` is higher than the current
        target. Returns True iff a bump occurred."""
        cur = self.data.get("target_level", 0)
        if int(target_level) > cur:
            self.data["target_level"] = int(target_level)
            return True
        return False

    def needs_more_work(self) -> bool:
        return self.data.get("current_level", 0) < self.data.get("target_level", 0)

    def write(self, sim_dir: Union[str, Path]) -> None:
        # Atomic write: a plain write_text() truncates the file to 0 bytes
        # before refilling it, so a concurrent reader (parallel marg jobs all
        # share one archive) can observe an EMPTY status.json -> JSONDecodeError.
        # Mirror IndexAppend._write_all: write a per-PID/thread temp then
        # os.replace() it in (atomic on POSIX). Reader never sees a partial file.
        path = Path(sim_dir, self.FILENAME)
        tmp = path.with_name(
            "{}.{}.{}.tmp".format(path.name, os.getpid(), threading.get_ident()))
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)   # atomic on POSIX

    @classmethod
    def read(cls, sim_dir: Union[str, Path]) -> "StatusRecord":
        # Defensive: even with atomic writes, a networked fs can briefly expose
        # an empty/partial read between create and rename. Retry a few times on
        # a decode error before giving up.
        path = Path(sim_dir, cls.FILENAME)
        for attempt in range(5):
            text = path.read_text()
            try:
                return cls(json.loads(text))
            except json.JSONDecodeError:
                if attempt == 4:
                    raise
                time.sleep(0.05)


# ---------------------------------------------------------------------------
# Code freezing
# ---------------------------------------------------------------------------

CodeSpec = Union[Callable[..., Any], str, os.PathLike, Dict[str, Any]]


def freeze_code(spec: CodeSpec, code_dir: Union[str, Path],
                target_filename: str = DEFAULT_GENERATOR_FILE,
                ) -> str:
    """Snapshot the generator (or summarizer) source into <code_dir>/.

    Returns the entrypoint string ("module:callable") that should be
    stored in the manifest. Three input shapes are accepted:

      * callable: inspect.getsource() is captured into a single-file
        module named after `target_filename`. The function must be
        self-contained (no closure captures, no module-relative imports
        beyond the standard library and explicitly-listed deps).
      * path: a .py file is copied verbatim.
      * dict: {"module_path": ..., "entrypoint": "mod:fn",
               "extra_files": [...]}. All listed files are copied; the
        named module becomes the canonical generator.
    """
    code_dir = Path(code_dir)
    code_dir.mkdir(parents=True, exist_ok=True)

    if callable(spec):
        src = inspect.getsource(spec)
        # Dedent so `def` starts at column 0 (matters when capturing a
        # local function defined inside a test).
        src = textwrap.dedent(src)
        out = code_dir / target_filename
        out.write_text(src)
        module_name = Path(target_filename).stem
        return "{}:{}".format(module_name, spec.__name__)

    if isinstance(spec, (str, os.PathLike)):
        src_path = Path(spec)
        if not src_path.is_file():
            raise FileNotFoundError(src_path)
        shutil.copy(src_path, code_dir / target_filename)
        # Caller is responsible for naming the entrypoint.
        return "{}:run".format(Path(target_filename).stem)

    if isinstance(spec, dict):
        module_path = Path(spec["module_path"])
        shutil.copy(module_path, code_dir / target_filename)
        for extra in spec.get("extra_files", []):
            shutil.copy(extra, code_dir / Path(extra).name)
        return spec.get("entrypoint", "{}:run".format(Path(target_filename).stem))

    raise TypeError("spec must be a callable, a path, or a dict; got %r" %
                    type(spec))


def load_entrypoint(code_dir: Union[str, Path], entrypoint: str) -> Callable[..., Any]:
    """Resolve a 'module:callable' entrypoint against <code_dir>/. Used
    by workers (and by LocalRunQueue for the no-schedd path)."""
    module_name, _, attr = entrypoint.partition(":")
    if not attr:
        raise ValueError("entrypoint must be 'module:callable'; got %r" % entrypoint)
    code_dir = str(Path(code_dir).resolve())
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import importlib
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

class Archive:
    """Thin facade over the on-disk layout. Owns the FSM; queues report
    state, this class applies it. Construction either creates a new
    archive (when `manifest` is given) or rehydrates an existing one
    (when only `base_location` is given)."""

    def __init__(self, base_location: Union[str, Path],
                 manifest: Optional[Manifest] = None,
                 request_queue: Optional["RequestQueue"] = None,
                 run_queue: Optional["RunQueue"] = None,
                 generator_spec: Optional[CodeSpec] = None,
                 summarizer_spec: Optional[CodeSpec] = None,
                 same_q_spec: Optional[CodeSpec] = None,
                 lookup_key_spec: Optional[CodeSpec] = None):
        self.base = Path(base_location)
        self._lock_path = self.base / ".archive.lock"
        # Cross-process serialization: fcntl.flock on _lock_fd.
        # Intra-process serialization across threads: threading.RLock
        # (reentrant so nested locked calls in one thread don't deadlock).
        # The flock fd is acquired on the OUTERMOST entry inside one
        # process; the RLock guards entry/exit so flock state is always
        # held by exactly one thread at a time per process.
        self._intra_process_lock = threading.RLock()
        self._lock_fd: Optional[int] = None
        self.request_queue = request_queue
        self.run_queue = run_queue
        if manifest is not None:
            self._initialize_new(manifest, generator_spec, summarizer_spec,
                                 same_q_spec, lookup_key_spec)
        else:
            self.manifest = Manifest.read(self.base)
        self.index = Index(self.base)
        # Resolve dedup callables (frozen versions if available, else defaults).
        self._same_q: Callable[[Any, Any], bool] = self._resolve_same_q()
        self._lookup_key: Callable[[Any], Any] = self._resolve_lookup_key()
        # Build the in-memory dedup index from index.jsonl. Buckets are
        # {hashable_lookup_key: [sim_name, ...]}; values are kept in
        # registration order so the first match wins under same_q.
        # For legacy rows missing a stored lookup_key, compute it now.
        self._dedup_buckets: Dict[Any, List[str]] = {}
        for row in self.index.all():
            if "lookup_key" in row:
                key = row["lookup_key"]
            else:
                try:
                    key = self._lookup_key(row.get("params"))
                except Exception:
                    key = row.get("name")
            self._dedup_buckets.setdefault(_safe_hashable(key), []).append(row["name"])

    # ---- bootstrap / rehydrate -------------------------------------------
    def _initialize_new(self, manifest: Manifest,
                        generator_spec: Optional[CodeSpec],
                        summarizer_spec: Optional[CodeSpec],
                        same_q_spec: Optional[CodeSpec],
                        lookup_key_spec: Optional[CodeSpec]) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        for sub in ("code", "sims", "request_queue", "run_queue"):
            (self.base / sub).mkdir(exist_ok=True)
        if generator_spec is None:
            raise ValueError("generator_spec is required when creating a new archive")
        gen_entry = freeze_code(generator_spec, self.base / "code",
                                target_filename=DEFAULT_GENERATOR_FILE)
        manifest.data["code"]["generator_entrypoint"] = gen_entry
        if summarizer_spec is not None:
            sum_entry = freeze_code(summarizer_spec, self.base / "code",
                                    target_filename=DEFAULT_SUMMARIZER_FILE)
            manifest.data["code"]["summarizer"] = "code/" + DEFAULT_SUMMARIZER_FILE
            manifest.data["code"]["summarizer_entrypoint"] = sum_entry
        if same_q_spec is not None:
            sq_entry = freeze_code(same_q_spec, self.base / "code",
                                   target_filename=DEFAULT_SAME_Q_FILE)
            manifest.data["code"]["same_q"] = "code/" + DEFAULT_SAME_Q_FILE
            manifest.data["code"]["same_q_entrypoint"] = sq_entry
        if lookup_key_spec is not None:
            lk_entry = freeze_code(lookup_key_spec, self.base / "code",
                                   target_filename=DEFAULT_LOOKUP_KEY_FILE)
            manifest.data["code"]["lookup_key"] = "code/" + DEFAULT_LOOKUP_KEY_FILE
            manifest.data["code"]["lookup_key_entrypoint"] = lk_entry
        manifest.write(self.base)
        self.manifest = manifest

    # ---- locking ----------------------------------------------------------
    @contextlib.contextmanager
    def _with_lock(self) -> Iterator[None]:
        """Hold the archive lock for the duration of the block.

        Two layers:
          * threading.RLock: serializes threads within one process and
            allows re-entry from the same thread (so transition() called
            from inside register() doesn't deadlock).
          * fcntl.flock on <base>/.archive.lock: serializes processes.
            Acquired on the outermost re-entry per process so concurrent
            workers (e.g. multiple request_sim CLIs against the same
            archive) coordinate cleanly.

        Reads (Index.all, get_status, StatusRecord.read) intentionally
        don't take this lock — they're a snapshot view and operate on
        immutable JSON files."""
        with self._intra_process_lock:
            outermost = self._lock_fd is None
            if outermost and _HAS_FCNTL:
                self._lock_path.parent.mkdir(parents=True, exist_ok=True)
                self._lock_fd = os.open(str(self._lock_path),
                                        os.O_RDWR | os.O_CREAT, 0o644)
                t0 = time.time()
                warned = False
                while True:
                    try:
                        fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except (BlockingIOError, OSError) as exc:
                        if exc.errno not in (errno.EAGAIN, errno.EACCES):
                            raise
                        if not warned and time.time() - t0 > 5.0:
                            logger.info("archive lock %s held; waiting...",
                                        self._lock_path)
                            warned = True
                        time.sleep(0.1)
            elif outermost and not getattr(Archive, "_warned_no_fcntl", False):
                logger.warning("fcntl unavailable; archive operations are NOT "
                               "safe across processes (intra-process locking "
                               "via threading.RLock still active).")
                Archive._warned_no_fcntl = True
            try:
                yield
            finally:
                if outermost and self._lock_fd is not None:
                    try:
                        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                    finally:
                        os.close(self._lock_fd)
                        self._lock_fd = None

    # ---- callable resolution ---------------------------------------------
    def _resolve_same_q(self) -> Callable[[Any, Any], bool]:
        ep = self.manifest.data.get("code", {}).get("same_q_entrypoint")
        if not ep:
            return _default_same_q
        return load_entrypoint(self.base / "code", ep)

    def _resolve_lookup_key(self) -> Callable[[Any], Any]:
        ep = self.manifest.data.get("code", {}).get("lookup_key_entrypoint")
        if not ep:
            return _default_lookup_key
        return load_entrypoint(self.base / "code", ep)

    # ---- registration -----------------------------------------------------
    def sim_dir(self, name: str) -> Path:
        return self.base / "sims" / name

    def find_existing(self, params: Any) -> Optional[str]:
        """Return the name of an existing sim whose params satisfy
        same_q(params, existing.params), else None. O(1) average via
        the lookup_key bucket; O(K) worst case where K is the bucket
        size (typically 1)."""
        key = _safe_hashable(self._lookup_key(params))
        for cand_name in self._dedup_buckets.get(key, []):
            cand = self.index.by_name(cand_name)
            if cand is None:
                continue
            try:
                if self._same_q(params, cand["params"]):
                    return cand_name
            except Exception:
                continue
        return None

    def register(self, params: Any, target_level: int = 1,
                 name: Optional[str] = None) -> str:
        """Idempotent under same_q + level. Behavior:

          * sim does not exist:    register fresh with `target_level=N`
          * sim exists, current_level >= N:  return name, no work
          * sim exists, current_level < N:   bump target_level to
            max(existing, N), transition to 'refine_ready', return name

        Returns the (existing or newly allocated) sim_name in all cases.
        Multi-writer-safe: holds the archive lock for the duration.
        """
        target_level = int(target_level)
        with self._with_lock():
            existing = self.find_existing(params)
            if existing is not None:
                self._maybe_bump_target(existing, target_level)
                return existing
            # Compute and validate the key BEFORE allocating a name or
            # writing anything. Validating after the mkdir left sims/<name>/
            # with params.json and status.json behind when the key turned
            # out to be unpersistable — a half-registered simulation that
            # the index has never heard of, and that the next register()
            # will silently allocate around.
            lk = _require_persistable_lookup_key(self._lookup_key(params))
            if name is None:
                name = str(len(list((self.base / "sims").iterdir())) + 1)
            sd = self.sim_dir(name)
            sd.mkdir(parents=True, exist_ok=True)
            (sd / "logs").mkdir(exist_ok=True)
            (sd / "params.json").write_text(json.dumps(params) + "\n")
            rec = StatusRecord.new(name, params, target_level=target_level)
            rec.write(sd)
            self.index.upsert({"name": name, "params": params,
                               "status": "ready", "summary": None,
                               "lookup_key": lk,
                               "target_level": target_level,
                               "current_level": 0})
            self._dedup_buckets.setdefault(_safe_hashable(lk), []).append(name)
            return name

    def refine(self, name: str, target_level: int) -> bool:
        """Explicit refinement request. Bumps the target_level if needed
        and (when the sim already had output) transitions it to
        'refine_ready'. Returns True iff a bump occurred."""
        with self._with_lock():
            return self._maybe_bump_target(name, int(target_level))

    def _maybe_bump_target(self, name: str, target_level: int) -> bool:
        # Caller must hold the archive lock.
        rec = StatusRecord.read(self.sim_dir(name))
        if not rec.bump_target(target_level):
            return False
        new_status = rec.data["status"]
        if rec.data["current_level"] >= 1 and rec.data["status"] == "complete":
            new_status = "refine_ready"
        rec.transition(new_status)  # records the history line even if status unchanged
        rec.data["status"] = new_status
        rec.write(self.sim_dir(name))
        row = self.index.by_name(name) or {"name": name}
        row["status"] = new_status
        row["target_level"] = rec.data["target_level"]
        row["current_level"] = rec.data["current_level"]
        self.index.upsert(row)
        return True

    # ---- transitions ------------------------------------------------------
    def transition(self, name: str, new_status: str, **fields: Any) -> None:
        with self._with_lock():
            rec = StatusRecord.read(self.sim_dir(name))
            rec.transition(new_status, **fields)
            rec.write(self.sim_dir(name))
            row = self.index.by_name(name) or {"name": name}
            row["status"] = new_status
            self.index.upsert(row)

    def refresh_status_from_disk(self) -> Dict[str, str]:
        """Sweep every sim and reconcile its `levels[]` history with the
        files actually present under sims/<name>/. New level_<N>.json
        files (e.g. dropped in by a condor worker via transfer_output_remaps)
        are absorbed; sims promote to 'complete' when current_level
        reaches target_level, or to 'refine_ready' if they're not done.

        Returns a dict mapping sim_name -> new_status for every sim
        whose status changed. Safe to call repeatedly (idempotent
        when no new outputs have arrived)."""
        changed: Dict[str, str] = {}
        with self._with_lock():
            for sim_name in list(self.simulations_iter_names()):
                sd = self.sim_dir(sim_name)
                if not sd.exists():
                    continue
                try:
                    rec = StatusRecord.read(sd)
                except Exception:
                    continue
                known_levels = {l["level"] for l in rec.data.get("levels", [])}
                target = rec.data.get("target_level", 0)
                mutated = False
                for lvl in range(1, target + 1):
                    if lvl in known_levels:
                        continue
                    level_file = sd / "level_{}.json".format(lvl)
                    if level_file.exists() and level_file.stat().st_size > 0:
                        rec.append_level(lvl, str(level_file.relative_to(self.base)))
                        mutated = True
                if not mutated and rec.data.get("status") in ("complete", "stuck"):
                    continue
                cur = rec.data.get("current_level", 0)
                old_status = rec.data.get("status")
                if cur >= target and target > 0:
                    new_status = "complete"
                elif cur >= 1:
                    new_status = "refine_ready"
                else:
                    new_status = old_status   # nothing computed yet; leave alone
                if new_status != old_status or mutated:
                    rec.transition(new_status)
                    rec.write(sd)
                    row = self.index.by_name(sim_name) or {"name": sim_name}
                    row["status"] = new_status
                    row["current_level"] = cur
                    self.index.upsert(row)
                    if new_status != old_status:
                        changed[sim_name] = new_status
        return changed

    def simulations_iter_names(self) -> Iterable[str]:
        for row in self.index.all():
            yield row["name"]

    def update_summary(self, name: str, summary: Dict[str, Any]) -> None:
        with self._with_lock():
            sd = self.sim_dir(name)
            (sd / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
            row = self.index.by_name(name) or {"name": name}
            row["summary"] = summary
            self.index.upsert(row)

    # ---- per-sim resource overrides --------------------------------------
    def set_resources(self, name: str,
                      request_memory: Optional[int] = None,
                      request_disk: Optional[str] = None,
                      extra_condor_cmds: Optional[Dict[str, str]] = None,
                      ) -> None:
        """Per-sim overrides on top of the queue's defaults. Only the
        keys you pass are updated; pass `None` to leave a field
        unchanged. The queue's `build_worker` reads these and merges
        them on top of its own defaults at submit time."""
        with self._with_lock():
            sd = self.sim_dir(name)
            rec = StatusRecord.read(sd)
            res = dict(rec.data.get("resources") or {})
            if request_memory is not None:
                res["request_memory"] = int(request_memory)
            if request_disk is not None:
                res["request_disk"] = request_disk
            if extra_condor_cmds is not None:
                merged = dict(res.get("extra_condor_cmds") or {})
                merged.update(extra_condor_cmds)
                res["extra_condor_cmds"] = merged
            rec.data["resources"] = res
            rec.write(sd)

    def get_resources(self, name: str) -> Dict[str, Any]:
        sd = self.sim_dir(name)
        rec = StatusRecord.read(sd)
        return dict(rec.data.get("resources") or {})

    # ---- stuck-state recovery --------------------------------------------
    def unstick(self, name: str, bump_memory: bool = False,
                bump_factor: float = 1.5) -> None:
        """Clear `stuck`, transition to `refine_ready` (if has prior
        levels) or `ready` (if not). With `bump_memory=True`, also
        multiplies the per-sim request_memory by `bump_factor` so the
        next attempt asks for more headroom."""
        with self._with_lock():
            sd = self.sim_dir(name)
            rec = StatusRecord.read(sd)
            if rec.data.get("status") != "stuck":
                logger.info("unstick: %s is not stuck (status=%s); no-op",
                            name, rec.data.get("status"))
                return
            new_status = "refine_ready" if rec.data.get("current_level", 0) >= 1 else "ready"
            if bump_memory:
                res = dict(rec.data.get("resources") or {})
                # Start from per-sim override if present; else use a
                # reasonable baseline (the run-queue default would be
                # the right thing but we don't have it here — caller
                # can pass it through set_resources first if desired).
                base = int(res.get("request_memory", 4096))
                res["request_memory"] = int(base * float(bump_factor))
                rec.data["resources"] = res
                logger.info("unstick: bumped %s request_memory %d -> %d",
                            name, base, res["request_memory"])
            rec.transition(new_status)
            rec.data["status"] = new_status
            rec.write(sd)
            row = self.index.by_name(name) or {"name": name}
            row["status"] = new_status
            self.index.upsert(row)

    def unstick_all(self, bump_memory: bool = False,
                    bump_factor: float = 1.5) -> List[str]:
        names = self.with_status("stuck")
        for n in names:
            self.unstick(n, bump_memory=bump_memory, bump_factor=bump_factor)
        return names

    # ---- admin: resummarize / verify / rebuild_index ---------------------
    def resummarize_all(self, *, only_complete: bool = False
                        ) -> Dict[str, str]:
        """Re-load the manifest's summarizer and re-run it for every sim
        with at least one completed level. Used after the summarizer's
        source has been updated (a new freeze + manifest rewrite, or
        an in-place edit of code/summarizer.py for archives whose
        summarizer is allowed to evolve).

        Returns {sim_name: 'ok' | 'no-summarizer' | 'no-levels' | 'error: <msg>'}.
        With `only_complete=True` skips sims whose status isn't 'complete'.
        """
        report: Dict[str, str] = {}
        summarizer = self.load_summarizer()
        if summarizer is None:
            for n in self.simulations_iter_names():
                report[n] = "no-summarizer"
            return report
        with self._with_lock():
            for sim_name in list(self.simulations_iter_names()):
                sd = self.sim_dir(sim_name)
                try:
                    rec = StatusRecord.read(sd)
                except Exception as exc:
                    report[sim_name] = "error: cannot read status ({})".format(exc)
                    continue
                if only_complete and rec.data.get("status") != "complete":
                    report[sim_name] = "skipped: status={}".format(
                        rec.data.get("status"))
                    continue
                level_paths = [str(self.base / l["output_path"])
                               for l in rec.data.get("levels", [])
                               if l.get("output_path")]
                if not level_paths:
                    report[sim_name] = "no-levels"
                    continue
                params = rec.data.get("params")
                try:
                    summary = summarizer(sim_dir=str(sd), params=params,
                                         levels=level_paths)
                except TypeError:
                    summary = summarizer(sim_dir=str(sd), params=params)
                except Exception as exc:
                    report[sim_name] = "error: {}".format(exc)
                    continue
                if summary is not None:
                    sd_summary = sd / "summary.json"
                    sd_summary.write_text(
                        json.dumps(summary, indent=2, sort_keys=True) + "\n")
                    row = self.index.by_name(sim_name) or {"name": sim_name}
                    row["summary"] = summary
                    self.index.upsert(row)
                report[sim_name] = "ok"
        return report

    def verify(self) -> Dict[str, Any]:
        """Cross-check index.jsonl, per-sim status.json, and on-disk
        files. Returns a structured report — meant to be inspected by
        an operator before acting on it. Does NOT mutate; pair with
        rebuild_index if the report shows drift you want to repair."""
        report: Dict[str, Any] = {
            "manifest_ok": True,
            "manifest_issues": [],
            "missing_in_index": [],     # sim_dirs on disk not in index
            "orphan_in_index": [],      # index rows whose sim_dir is gone
            "status_drift": [],         # status.json vs index disagreement
            "missing_levels": [],       # status claims level N but file absent
            "extra_levels": [],         # level files present but not in status
            "stuck_sims": [],
            "complete_sims": 0,
            "incomplete_sims": 0,
        }

        # --- manifest sanity checks ------------------------------------
        try:
            man = self.manifest.data
        except Exception as exc:
            report["manifest_ok"] = False
            report["manifest_issues"].append("read failed: {}".format(exc))
            return report
        for ep_key in ("generator", "summarizer", "same_q", "lookup_key"):
            ep = man.get("code", {}).get(ep_key)
            if ep is None:
                continue
            if not (self.base / ep).exists():
                report["manifest_ok"] = False
                report["manifest_issues"].append(
                    "manifest.code.{} -> {!r} but file is missing".format(ep_key, ep))

        # --- sims directory vs index -----------------------------------
        sims_dir = self.base / "sims"
        on_disk = {p.name for p in sims_dir.iterdir() if p.is_dir()} \
            if sims_dir.exists() else set()
        in_index = {row["name"] for row in self.index.all()}
        report["missing_in_index"] = sorted(on_disk - in_index)
        report["orphan_in_index"] = sorted(in_index - on_disk)

        # --- per-sim cross-check ---------------------------------------
        for sim_name in sorted(on_disk | in_index):
            row = self.index.by_name(sim_name)
            sd = self.sim_dir(sim_name)
            if not sd.exists():
                continue   # already accounted for in orphan_in_index
            try:
                rec = StatusRecord.read(sd)
            except Exception as exc:
                report["status_drift"].append(
                    {"name": sim_name, "issue": "status.json unreadable: {}".format(exc)})
                continue
            if row is not None:
                if row.get("status") != rec.data.get("status"):
                    report["status_drift"].append(
                        {"name": sim_name,
                         "index_status": row.get("status"),
                         "record_status": rec.data.get("status")})
                if row.get("current_level") != rec.data.get("current_level"):
                    report["status_drift"].append(
                        {"name": sim_name,
                         "index_current_level": row.get("current_level"),
                         "record_current_level": rec.data.get("current_level")})
            recorded_levels = {l["level"] for l in rec.data.get("levels", [])}
            target = rec.data.get("target_level", 0)
            for lvl in range(1, target + 1):
                level_file = sd / "level_{}.json".format(lvl)
                exists = level_file.exists() and level_file.stat().st_size > 0
                if lvl in recorded_levels and not exists:
                    report["missing_levels"].append({"name": sim_name, "level": lvl})
                if exists and lvl not in recorded_levels:
                    report["extra_levels"].append({"name": sim_name, "level": lvl})
            status = rec.data.get("status")
            if status == "complete":
                report["complete_sims"] += 1
            elif status == "stuck":
                report["stuck_sims"].append(sim_name)
            else:
                report["incomplete_sims"] += 1

        report["healthy"] = (
            report["manifest_ok"]
            and not report["missing_in_index"]
            and not report["orphan_in_index"]
            and not report["status_drift"]
            and not report["missing_levels"]
            and not report["extra_levels"]
        )
        return report

    def rebuild_index(self) -> int:
        """Reconstruct index.jsonl from per-sim status.json files. Useful
        if the index is corrupted, lost, or out of sync (e.g. after
        manual file ops). Returns the count of rows written. Re-runs the
        summarizer would be a separate pass (resummarize_all)."""
        with self._with_lock():
            sims_dir = self.base / "sims"
            if not sims_dir.exists():
                logger.info("rebuild_index: no sims/ dir; clearing index")
                self.index._write_all([])
                return 0
            rows: List[Dict[str, Any]] = []
            for sd in sorted(sims_dir.iterdir()):
                if not sd.is_dir():
                    continue
                try:
                    rec = StatusRecord.read(sd)
                except Exception:
                    logger.warning("rebuild_index: skipping %s (no readable "
                                   "status.json)", sd.name)
                    continue
                params = rec.data.get("params")
                summary = None
                summary_path = sd / "summary.json"
                if summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text())
                    except Exception:
                        summary = None
                row = {
                    "name": sd.name,
                    "params": params,
                    "status": rec.data.get("status"),
                    "summary": summary,
                    # Normalized exactly as register() does. Storing the
                    # raw key here meant an archive that registered and
                    # reopened cleanly still blew up in rebuild_index with
                    # the original sorted() TypeError, because _write_all
                    # serializes rows with sort_keys=True.
                    "lookup_key": (
                        _require_persistable_lookup_key(
                            self._lookup_key(params))
                        if params is not None else None),
                    "target_level": rec.data.get("target_level", 0),
                    "current_level": rec.data.get("current_level", 0),
                }
                rows.append(row)
            self.index._write_all(rows)
            # Rebuild dedup buckets from scratch.
            self._dedup_buckets = {}
            for row in rows:
                if row.get("lookup_key") is None:
                    continue
                key = _safe_hashable(row["lookup_key"])
                self._dedup_buckets.setdefault(key, []).append(row["name"])
            return len(rows)

    # ---- introspection ----------------------------------------------------
    def with_status(self, status: str) -> List[str]:
        return [r["name"] for r in self.index.with_status(status)]

    def get_status(self, name: str) -> Optional[str]:
        r = self.index.by_name(name)
        return r["status"] if r else None

    def load_generator(self) -> Callable[..., Any]:
        return load_entrypoint(self.base / "code",
                               self.manifest.data["code"]["generator_entrypoint"])

    def load_summarizer(self) -> Optional[Callable[..., Any]]:
        ep = self.manifest.data.get("code", {}).get("summarizer_entrypoint")
        if not ep:
            return None
        return load_entrypoint(self.base / "code", ep)

    # ---- file-transfer helpers (for run pools without shared FS) ---------
    def transfer_input_files_for(self, sim_name: str, level: int) -> List[str]:
        """Files a worker needs on the execute host to compute (sim, level).

        Suitable as the value of condor's `transfer_input_files`. Paths
        are returned as filesystem paths on the submit side; condor
        flattens basenames into the worker sandbox cwd.

        Includes:
          * <archive>/code   — the whole frozen code directory
          * <archive>/sims/<name>/params.json
          * <archive>/sims/<name>/level_1.json ... level_<level-1>.json
            (whichever are present on disk)
        """
        sd = self.sim_dir(sim_name)
        files: List[str] = [str(self.base / "code")]
        params_path = sd / "params.json"
        if params_path.exists():
            files.append(str(params_path))
        for i in range(1, int(level)):
            p = sd / "level_{}.json".format(i)
            if p.exists():
                files.append(str(p))
        return files

    def expected_output(self, sim_name: str, level: int) -> Tuple[str, str]:
        """Return (output_basename, absolute_remap_target) for a (sim, level)
        worker. Use the basename in `transfer_output_files` and the pair
        in `transfer_output_remaps` so condor places the worker's output
        at the canonical archive location on the submit node."""
        basename = "level_{}.json".format(int(level))
        target = str(self.sim_dir(sim_name) / basename)
        return basename, target

    def worker_bootstrap_script(self) -> str:
        """Return a self-contained Python bootstrap script (as a string)
        that the run pool's executable should be set to. The script
        expects argv:
            --sim-name <n> --level <N>
            [--code-dir <path>]              (default: 'code' relative to cwd;
                                              slurm passes the archive's
                                              absolute code/ path here)
            [--prev-levels FILE [...]]
        Reads ./params.json from cwd, writes ./level_<N>.json to cwd.
        Used by both DualCondorRunQueue (sandbox cwd, files flattened)
        and SlurmRunQueue (cwd = sim_dir on shared FS)."""
        ep = self.manifest.data["code"]["generator_entrypoint"]
        module_name, _, fn_name = ep.partition(":")
        return textwrap.dedent('''\
            #!/usr/bin/env python3
            """Auto-generated worker bootstrap for the simulation_manager
            v2 archive. Loads the frozen generator and runs one level."""
            import argparse, json, os, sys

            ap = argparse.ArgumentParser()
            ap.add_argument("--sim-name", required=True)
            ap.add_argument("--level", type=int, required=True)
            ap.add_argument("--code-dir", default="code",
                            help="path to the frozen code/ directory; "
                                 "default 'code' is relative to cwd, "
                                 "matching condor's flattened sandbox.")
            ap.add_argument("--prev-levels", nargs="*", default=[])
            args = ap.parse_args()

            sys.path.insert(0, args.code_dir)
            from {module_name} import {fn_name} as _gen

            with open("params.json") as f:
                params = json.load(f)

            prev = [os.path.abspath(p) for p in args.prev_levels]
            _gen(params, sim_dir=os.getcwd(), level=args.level, prev_levels=prev)
        ''').format(module_name=module_name, fn_name=fn_name)


# ---------------------------------------------------------------------------
# Queue interfaces
# ---------------------------------------------------------------------------

class RequestQueue:
    """Decides which sims should next be sent to the run queue, and
    tracks their state in the request system. Subclasses set `kind`."""
    kind: str = "abstract"

    def submit_pending(self, archive: Archive) -> List[str]:
        raise NotImplementedError

    def poll(self, archive: Archive) -> Dict[str, str]:
        raise NotImplementedError


class RunQueue:
    """Actually runs sims (writes their output)."""
    kind: str = "abstract"

    def build_worker(self, archive: Archive, sim_name: str) -> str:
        raise NotImplementedError

    def submit(self, archive: Archive, sim_names: Iterable[str]
               ) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def poll(self, archive: Archive, sim_names: Iterable[str]
             ) -> Dict[str, str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Local queues (no schedd) — used for tests and worked examples
# ---------------------------------------------------------------------------

class LocalRequestQueue(RequestQueue):
    """Trivial pass-through: every sim that needs work ('ready' or
    'refine_ready') is immediately handed to the run queue."""
    kind = "local"

    def __init__(self, run_queue: "LocalRunQueue"):
        self.run_queue = run_queue

    def submit_pending(self, archive: Archive) -> List[str]:
        pending = (archive.with_status("ready")
                   + archive.with_status("refine_ready"))
        for n in pending:
            archive.transition(n, "submit_ready")
        if pending:
            self.run_queue.submit(archive, pending)
        return pending

    def poll(self, archive: Archive) -> Dict[str, str]:
        return {n: archive.get_status(n) for n in archive.with_status("running")}


class LocalRunQueue(RunQueue):
    """Runs the frozen generator inline in the current process. Computes
    every missing level (current_level + 1 ... target_level) for each
    submitted sim. Useful for end-to-end tests without a cluster."""
    kind = "local"

    def build_worker(self, archive: Archive, sim_name: str) -> str:
        # No external script needed: we call the generator in-process.
        return ""

    def submit(self, archive: Archive, sim_names: Iterable[str]
               ) -> List[Tuple[str, str]]:
        gen = archive.load_generator()
        summarizer = archive.load_summarizer()
        results: List[Tuple[str, str]] = []
        for name in sim_names:
            sd = archive.sim_dir(name)
            rec = StatusRecord.read(sd)
            archive.transition(name, "running",
                               run_queue={"kind": self.kind, "job_id": name})
            params = json.loads((sd / "params.json").read_text())
            target = rec.data["target_level"]
            current = rec.data["current_level"]
            stuck = False
            for lvl in range(current + 1, target + 1):
                prev_levels = [str(sd / "level_{}.json".format(i))
                               for i in range(1, lvl)]
                try:
                    gen(params, sim_dir=str(sd), level=lvl,
                        prev_levels=prev_levels)
                except Exception as exc:
                    rec = StatusRecord.read(sd)
                    rec.transition("stuck",
                                   run_queue={"kind": self.kind,
                                              "job_id": name,
                                              "error": str(exc)})
                    rec.write(sd)
                    archive.index.upsert({**(archive.index.by_name(name) or {"name": name}),
                                          "status": "stuck"})
                    stuck = True
                    break
                level_output = sd / "level_{}.json".format(lvl)
                # Generator may write to a different filename; record
                # whatever it produced for this level.
                if not level_output.exists():
                    candidates = sorted(sd.glob("level_{}*".format(lvl)))
                    if candidates:
                        level_output = candidates[0]
                rec = StatusRecord.read(sd)
                rec.append_level(lvl, str(level_output.relative_to(archive.base))
                                 if level_output.exists() else "")
                rec.write(sd)
            if stuck:
                results.append((name, "stuck"))
                continue
            archive.transition(name, "complete")
            # Re-read to attach final level info on the index row.
            rec = StatusRecord.read(sd)
            row = archive.index.by_name(name) or {"name": name}
            row["current_level"] = rec.data["current_level"]
            row["target_level"] = rec.data["target_level"]
            row["status"] = "complete"
            archive.index.upsert(row)
            if summarizer is not None:
                level_paths = [str(archive.base / l["output_path"])
                               for l in rec.data["levels"] if l["output_path"]]
                try:
                    summary = summarizer(sim_dir=str(sd), params=params,
                                         levels=level_paths)
                except TypeError:
                    # Summarizer may have the simpler (sim_dir, params) signature.
                    summary = summarizer(sim_dir=str(sd), params=params)
                except Exception:
                    summary = None
                if summary is not None:
                    archive.update_summary(name, summary)
            results.append((name, "complete"))
        return results

    def poll(self, archive: Archive, sim_names: Iterable[str]
             ) -> Dict[str, str]:
        # Local runs are synchronous; a poll after submit always shows complete.
        return {n: archive.get_status(n) for n in sim_names}


# ---------------------------------------------------------------------------
# Dual-condor queue stubs
# ---------------------------------------------------------------------------
#
# Topology:
#
#   request pool (e.g. CIT submit host)         run pool (e.g. OSG / remote)
#   ┌──────────────────────────────┐            ┌──────────────────────────────┐
#   │ planner DAG                  │            │ per-sim, per-level workers   │
#   │  - scout: pick what to do    │            │  - read frozen code/         │
#   │  - DualCondorRequestQueue    │  submit   │  - run gen(params, sd, lvl)  │
#   │    builds & submits sub-DAG  │ ────────> │  - write level_<N>.json      │
#   │    via condor_submit_dag     │            │                              │
#   │    -name <run_pool_schedd>   │            │ shared FS: <base>/sims/...   │
#   │  - polls run-pool schedd via │ <──────── │                              │
#   │    htcondor.Schedd(run_pool) │  output    │                              │
#   └──────────────────────────────┘            └──────────────────────────────┘
#
# Both pools mount the same archive; communication is the filesystem.
# The classes below sketch the interfaces that need fleshing out;
# detailed pseudocode lives in the docstrings.

class DualCondorRequestQueue(RequestQueue):
    """Runs on the *request* condor pool. Its job is orchestration:
    pick the sims that need work, ask the run queue to actually
    dispatch them, and poll for completion.

    In the simplest deployment the request queue is essentially a
    thin wrapper over the run queue — the planner that decides
    *what* to ask for lives upstream (e.g. the user's hyperpipeline
    DAG calling `Archive.register(params, target_level=N)` per node).

    Configuration in the manifest's request_queue.extra:
        request_pool   : str  # informational; the schedd this DAG runs on
        run_pool       : str  # passthrough to the run queue
        run_collector  : str  # collector host for htcondor.Schedd lookups
        accounting_group / accounting_group_user
    """
    kind = "condor"

    def __init__(self,
                 run_queue: Optional["DualCondorRunQueue"] = None,
                 request_pool: Optional[str] = None,
                 run_pool: Optional[str] = None,
                 run_collector: Optional[str] = None,
                 **submit_kwargs: Any):
        self.run_queue = run_queue
        self.request_pool = request_pool
        self.run_pool = run_pool
        self.run_collector = run_collector
        self.submit_kwargs = submit_kwargs

    def submit_pending(self, archive: Archive) -> List[str]:
        if self.run_queue is None:
            raise RuntimeError("DualCondorRequestQueue: run_queue not attached")
        pending = (archive.with_status("ready")
                   + archive.with_status("refine_ready"))
        if not pending:
            return []
        for n in pending:
            archive.transition(n, "submit_ready")
        try:
            self.run_queue.submit(archive, pending)
        except Exception:
            # Roll back the submit_ready transitions on dispatch failure
            # so a retry sees the sims as 'ready' / 'refine_ready' again.
            for n in pending:
                archive.transition(n, "ready")
            raise
        return pending

    def poll(self, archive: Archive) -> Dict[str, str]:
        if self.run_queue is None:
            return {}
        observed = self.run_queue.poll(archive, archive.simulations_iter_names())
        # Output-on-disk is the authoritative completion signal.
        archive.refresh_status_from_disk()
        return observed


class DualCondorRunQueue(RunQueue):
    """Run-pool queue: ferries per-(sim, level) work to condor execute
    hosts and tracks completion via the schedd.

    Each (sim, level) is one condor job. Levels for the same sim are
    chained as DAG parents/children so an accumulating generator sees
    its prior levels' outputs already on disk before its job starts.

    Configuration:
        run_pool         : str  -- target schedd (-name <run_pool> on
                                   condor_submit_dag for cross-pool).
                                   None = local schedd.
        run_collector    : str  -- collector host for cross-pool
                                   htcondor.Schedd(<collector>) queries
                                   in poll(). None = local.
        accounting_group        -- defaults to env LIGO_ACCOUNTING
        accounting_group_user   -- defaults to env LIGO_USER_NAME
                                   Both follow the standard LIGO/IGWN
                                   convention; the matching env-var
                                   fallback matches the legacy
                                   CondorManager behaviour so existing
                                   submit hosts work unchanged.
        request_memory   : int (MB), default 4096
        request_disk     : str (e.g. '4G'), default '4G'
        getenv           : str  -- value of condor 'getenv' command.
                                   Default precedence: constructor kwarg >
                                   $RIFT_GETENV env > safe allowlist
                                   ('LD_LIBRARY_PATH,PATH,PYTHONPATH,*RIFT*,
                                   LIBRARY_PATH'). NOTE: `getenv = True`
                                   is blocked by many sites (CIT among
                                   them); the allowlist is the OSG-blessed
                                   alternative. Pass getenv='True'
                                   explicitly only on sites that allow it.
        use_singularity  : bool
        singularity_image: str   -- required if use_singularity=True
        oom_hold_codes   : seq  -- hold codes this site reports when a
                                   job runs out of memory. Default
                                   DEFAULT_OOM_HOLD_CODES = (34, 26). 34
                                   is unambiguous; 26 is SystemPolicy and
                                   means whatever the site's
                                   SYSTEM_PERIODIC_HOLD expressions say,
                                   which elsewhere may be an anti-thrash
                                   limiter rather than memory.
        oom_hold_subcode_exclusions: {code: [subcode, ...]} -- sub-codes
                                   to carve out of a code above. Needed
                                   because every SYSTEM_PERIODIC_HOLD at
                                   a site reports one hold code and only
                                   the sub-code separates "over memory"
                                   from "restarted too many times". A
                                   sub-code keyed on a code not listed in
                                   oom_hold_codes is refused rather than
                                   ignored.
        oom_retry_counter: str  -- ClassAd expression rationing the
                                   retries and scaling the bump. Default
                                   DEFAULT_OOM_RETRY_COUNTER =
                                   "NumJobStarts". NumHolds is the other
                                   obvious choice and is not better
                                   everywhere: it counts holds of every
                                   kind, including transfer failures that
                                   increment it without the job ever
                                   running.
        extra_periodic_release: str -- a ClassAd expression OR'd into
                                   periodic_release alongside the OOM
                                   policy, for sites that hold jobs for
                                   reasons this class does not model.
                                   While auto_release_on_oom is on, the
                                   term is scoped away from whatever
                                   codes oom_hold_codes names, so
                                   oom_max_retries stays a real cap and
                                   the term governs every other code.
                                   With the OOM policy off it governs all
                                   of them. Setting periodic_release
                                   through extra_condor_cmds is refused
                                   -- it replaced the whole expression and
                                   dropped the memory handling with it.
        extra_condor_cmds: dict  -- additional `key = value` lines
                                    appended verbatim to the submit
                                    description (e.g. +DESIRED_SITES,
                                    +UNDESIRED_SITES for OSG site
                                    selection, requirements clauses).
        extra_transfer_input_files: list -- extra entries APPENDED to
                                    transfer_input_files for every job.
                                    Intended for bulk inputs addressed
                                    by URL (osdf://, http://) so they
                                    are fetched from a cache instead of
                                    staged through the submit host's
                                    spool. Setting `transfer_input_files`
                                    via extra_condor_cmds would instead
                                    *replace* the archive's own entries
                                    and strip the frozen code/ directory,
                                    leaving the worker nothing to run.
        extra_transfer_output_files: list -- products to bring BACK
                                    beyond the level_<N>.json marker,
                                    named relative to the job sandbox.
                                    `{level}` and `{sim_name}` are
                                    substituted, so e.g. "level_{level}"
                                    returns a per-level output directory.
                                    Each is remapped to the same relative
                                    path under sims/<name>/.
                                    transfer_output_files is explicit, so
                                    without this HTCondor returns only the
                                    marker and everything else the worker
                                    produced dies with the sandbox — the
                                    job completes having discarded its
                                    own results.

    The defaults above also apply when DualCondorRunQueue is
    instantiated via make_queues_from_manifest() — keys absent from
    `run_queue.extra` in the manifest fall back to the constructor
    defaults, which in turn fall back to the env vars listed above.
    """
    kind = "condor"

    def __init__(self,
                 run_pool: Optional[str] = None,
                 run_collector: Optional[str] = None,
                 request_memory: int = 4096,
                 request_disk: str = "4G",
                 accounting_group: Optional[str] = None,
                 accounting_group_user: Optional[str] = None,
                 getenv: Optional[str] = None,
                 use_singularity: bool = False,
                 singularity_image: Optional[str] = None,
                 extra_condor_cmds: Optional[Dict[str, str]] = None,
                 extra_transfer_input_files: Optional[Sequence[str]] = None,
                 extra_transfer_output_files: Optional[Sequence[str]] = None,
                 auto_release_on_oom: bool = True,
                 extra_periodic_release: Optional[str] = None,
                 oom_hold_codes: Optional[Sequence[int]] = None,
                 oom_hold_subcode_exclusions: Optional[Mapping[int, Sequence[int]]] = None,
                 oom_retry_counter: Optional[str] = None,
                 oom_max_retries: int = 5,
                 oom_memory_factor: float = 1.5,
                 subdag_factory: Optional[Callable[[Any, str, int], str]] = None,
                 submit_mode: str = "submit",
                 **submit_kwargs: Any):
        self.run_pool = run_pool
        self.run_collector = run_collector
        self.extra_transfer_input_files = extra_transfer_input_files
        self.extra_transfer_output_files = extra_transfer_output_files
        if (self.extra_transfer_input_files or self.extra_transfer_output_files) \
                and subdag_factory is not None:
            # Fail early for the common case. submit() re-checks, because
            # both of these are plain attributes and assigning either after
            # construction reaches the same silently-ignoring path.
            raise ValueError(
                "extra_transfer_{input,output}_files are applied by "
                "build_worker, which is bypassed when subdag_factory is set: "
                "the sub-DAG owns its own submit descriptions. Put the extra "
                "entries in the sub-DAG the factory generates instead.")
        self.request_memory = int(request_memory)
        self.request_disk = request_disk
        self.accounting_group = accounting_group or os.environ.get("LIGO_ACCOUNTING")
        self.accounting_group_user = (accounting_group_user
                                      or os.environ.get("LIGO_USER_NAME"))
        if getenv is not None:
            self.getenv = getenv
        else:
            self.getenv = os.environ.get("RIFT_GETENV", DEFAULT_GETENV_ALLOWLIST)
        self.use_singularity = use_singularity
        self.singularity_image = singularity_image
        self.extra_condor_cmds = extra_condor_cmds or {}
        self.auto_release_on_oom = bool(auto_release_on_oom)
        self.extra_periodic_release = extra_periodic_release
        self.oom_hold_codes = (DEFAULT_OOM_HOLD_CODES if oom_hold_codes is None
                               else oom_hold_codes)
        self.oom_hold_subcode_exclusions = oom_hold_subcode_exclusions
        self.oom_retry_counter = (DEFAULT_OOM_RETRY_COUNTER
                                  if oom_retry_counter is None
                                  else oom_retry_counter)
        self.oom_max_retries = int(oom_max_retries)
        self.oom_memory_factor = float(oom_memory_factor)
        # Per-(sim, level) work-unit factory. When set, each level emits
        # a `SUBDAG EXTERNAL <node_id> <factory(archive, sim, level)>`
        # node in the wrapper DAG instead of a vanilla-universe `JOB`
        # backed by build_worker(). Used by backends whose work unit is
        # itself a condor DAG (e.g. GW PE via util_RIFT_pseudo_pipe).
        self.subdag_factory = subdag_factory
        # submit_mode controls dispatch:
        #   "submit"  -> call condor_submit_dag (default; archive owns
        #                its own dispatch)
        #   "embed"   -> write the wrapper DAG and return without
        #                dispatching, so a parent workflow can include
        #                it as `SUBDAG EXTERNAL`. Path is recorded on
        #                self.last_wrapper_dag_path.
        if submit_mode not in ("submit", "embed"):
            raise ValueError("submit_mode must be 'submit' or 'embed'; got {!r}"
                             .format(submit_mode))
        self.submit_mode = submit_mode
        self.submit_kwargs = submit_kwargs
        if submit_kwargs:
            # submit_kwargs is stored and never read. Silence here makes
            # the manifest a one-way hatch across versions: a RIFT that
            # predates a key lands it in here and submits under different
            # policy than the archive was built with, with nothing in the
            # log. That is the same silent-substitution failure the
            # transfer and periodic_release guards exist to stop, on the
            # version axis instead of the config one.
            warnings.warn(
                "DualCondorRunQueue ignoring unrecognised option(s) {0}. "
                "If these came from a manifest's run_queue.extra, this "
                "RIFT is older than the archive and the jobs will submit "
                "under different policy than intended.".format(
                    ", ".join(sorted(map(repr, submit_kwargs)))),
                RuntimeWarning, stacklevel=2)
        # Per-archive state.
        self.dag_cluster_id: Optional[int] = None
        self.last_wrapper_dag_path: Optional[str] = None

    # -------- per-(sim, level) submit description --------------------------

    # These are validated on ASSIGNMENT, not only in __init__. Checking
    # once at construction is not protection: they are ordinary public
    # attributes, and configuring a queue by assigning to them after the
    # fact is the natural thing to do — which walked straight past every
    # guard.
    @property
    def extra_transfer_input_files(self) -> Tuple[str, ...]:
        # A tuple, not the live list: returning the list let a caller do
        # `q.extra_transfer_input_files.append("/bad,entry")`, which never
        # goes through the setter and so skipped every check. Handing back
        # something immutable makes that attempt fail at the append.
        return tuple(self._extra_transfer_input_files)

    @extra_transfer_input_files.setter
    def extra_transfer_input_files(self, value: Any) -> None:
        entries = _validate_transfer_entries(
            value, what="extra_transfer_input_files")
        for entry in entries:
            _reject_reserved_basename(entry, "extra_transfer_input_files")
        self._extra_transfer_input_files = entries

    @property
    def extra_transfer_output_files(self) -> Tuple[str, ...]:
        return tuple(self._extra_transfer_output_files)

    @extra_transfer_output_files.setter
    def extra_transfer_output_files(self, value: Any) -> None:
        entries = _validate_transfer_entries(
            value, what="extra_transfer_output_files", remap_syntax=True)
        for entry in entries:
            _reject_reserved_basename(entry, "extra_transfer_output_files")
        self._extra_transfer_output_files = entries

    @property
    def extra_periodic_release(self) -> str:
        return self._extra_periodic_release

    @extra_periodic_release.setter
    def extra_periodic_release(self, value: Any) -> None:
        self._extra_periodic_release = _validate_release_expression(
            value, what="extra_periodic_release")

    @property
    def oom_hold_codes(self) -> Tuple[int, ...]:
        return self._oom_hold_codes

    @oom_hold_codes.setter
    def oom_hold_codes(self, value: Any) -> None:
        # None means "the default", as it does in the constructor and for
        # oom_retry_counter. Reading it as "own no codes" would let
        # `q.oom_hold_codes = None` disable the memory policy outright,
        # which is a thing to have to ask for -- pass () for that.
        self._oom_hold_codes = (
            DEFAULT_OOM_HOLD_CODES if value is None
            else _validate_hold_codes(value, what="oom_hold_codes"))
        self._reject_orphan_subcode_exclusions()

    @property
    def oom_hold_subcode_exclusions(self) -> Mapping[int, Tuple[int, ...]]:
        # A read-only view, not a copy: a copy makes
        # `q.oom_hold_subcode_exclusions[26] = (100,)` a silent no-op,
        # where this makes it raise. Same reasoning as the transfer
        # properties handing back tuples rather than live lists.
        return MappingProxyType(self._oom_hold_subcode_exclusions)

    @oom_hold_subcode_exclusions.setter
    def oom_hold_subcode_exclusions(self, value: Any) -> None:
        self._oom_hold_subcode_exclusions = _validate_subcode_exclusions(
            value, what="oom_hold_subcode_exclusions")
        self._reject_orphan_subcode_exclusions()

    def _reject_orphan_subcode_exclusions(self) -> None:
        """An exclusion on a code the policy does not own does nothing.

        Silently ignoring it means a typo'd key reads as configured and
        has no effect -- the site believes it has carved out its
        anti-thrash sub-code and has not. Only checked once both
        attributes exist, because the constructor sets them in sequence.
        """
        codes = getattr(self, "_oom_hold_codes", None)
        orphans = getattr(self, "_oom_hold_subcode_exclusions", None)
        if codes is None or not orphans:
            return
        unknown = sorted(k for k in orphans if k not in codes)
        if unknown:
            raise ValueError(
                "oom_hold_subcode_exclusions names hold code(s) {0} that "
                "oom_hold_codes does not include ({1}), so the exclusion "
                "would have no effect".format(
                    ", ".join(map(str, unknown)),
                    ", ".join(map(str, codes)) or "none"))

    @property
    def oom_retry_counter(self) -> str:
        return self._oom_retry_counter

    @oom_retry_counter.setter
    def oom_retry_counter(self, value: Any) -> None:
        self._oom_retry_counter = _validate_release_expression(
            value, what="oom_retry_counter") or DEFAULT_OOM_RETRY_COUNTER

    def _oom_hold_predicate(self, code_attr: str, subcode_attr: str) -> str:
        """"This hold is one the OOM policy owns", as a ClassAd expression.

        Built twice per submit description against different attributes:
        periodic_release asks about the CURRENT hold, request_memory about
        the LAST one. Same policy, two vantage points -- which is why this
        is a builder and not a string the caller supplies ready-made.
        """
        terms = []
        for code in self._oom_hold_codes:
            term = "({0} =?= {1})".format(code_attr, code)
            excluded = self._oom_hold_subcode_exclusions.get(code) or ()
            if excluded:
                term = "({0}{1})".format(term, "".join(
                    " && ({0} =!= {1})".format(subcode_attr, sub)
                    for sub in excluded))
            terms.append(term)
        if not terms:
            # No codes configured means the policy owns nothing. Emit a
            # constant rather than an empty string, so the surrounding
            # expression stays well-formed instead of becoming a parse
            # error at submit time.
            return "false"
        return " || ".join(terms)

    def _bootstrap_path(self, archive: Archive) -> Path:
        path = archive.base / "run_queue" / "workers" / "bootstrap.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        # (Re)write each call so a manifest change in the entrypoint
        # picks up immediately.
        path.write_text(archive.worker_bootstrap_script())
        path.chmod(0o755)
        return path

    def build_worker(self, archive: Archive, sim_name: str,
                     level: int = 1) -> str:
        """Write the per-(sim, level) condor submit description and
        return its absolute path. Idempotent: re-running it overwrites.

        Per-sim resource overrides (set via Archive.set_resources) merge
        on top of the queue's defaults: per-sim values for
        request_memory / request_disk / extra_condor_cmds win where set.
        """
        sd = archive.sim_dir(sim_name)
        if not sd.exists():
            raise FileNotFoundError("sim_dir does not exist: {}".format(sd))

        # Per-sim overrides on top of the queue's defaults.
        try:
            res = archive.get_resources(sim_name)
        except Exception:
            res = {}
        request_memory = int(res.get("request_memory", self.request_memory))
        request_disk = res.get("request_disk", self.request_disk)
        extra_cmds = dict(self.extra_condor_cmds)
        extra_cmds.update(res.get("extra_condor_cmds") or {})
        # extra_condor_cmds is emitted last, so these would REPLACE the
        # lines built above rather than extend them — dropping the frozen
        # code/ directory, the sim's params, or the output remaps, with
        # condor_submit reporting success either way.
        # Compared case-insensitively: HTCondor submit command names are
        # case-insensitive, so `Transfer_Input_Files` is the same directive
        # as `transfer_input_files` and an exact lowercase match let it
        # straight through — reinstating the very substitution this guard
        # exists to prevent, with the frozen code/ and params.json silently
        # dropped. `extra_cmds` is the merged dict, so per-sim overrides
        # from Archive.set_resources are covered by the same pass.
        for _key in extra_cmds:
            if str(_key).strip().casefold() in _PROTECTED_SUBMIT_COMMANDS:
                raise ValueError(
                    "extra_condor_cmds must not set {0!r}: it is emitted "
                    "after the archive's own line, so it replaces that line "
                    "rather than extending it (compared case-insensitively, "
                    "because HTCondor command names are). Use {1} "
                    "instead.".format(_key, _PROTECTED_ALTERNATIVES.get(
                        str(_key).strip().casefold(),
                        "the corresponding append-only option")))

        bootstrap = self._bootstrap_path(archive)
        log_dir = archive.base / "run_queue" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        sub_dir = archive.base / "run_queue" / "submit_files"
        sub_dir.mkdir(parents=True, exist_ok=True)

        # Declare ALL chained prior levels regardless of disk presence;
        # PARENT/CHILD edges guarantee they exist by the time level N runs.
        prev_basenames = ["level_{}.json".format(i) for i in range(1, level)]
        prev_paths = [str(sd / b) for b in prev_basenames]
        out_base, out_target = archive.expected_output(sim_name, level)
        transfer_in = [str(archive.base / "code"),
                       str(sd / "params.json")] + prev_paths
        # Backend-supplied inputs every job also needs — typically bulk
        # objects addressed by URL (osdf://, http://) so they come from a
        # cache rather than the submit host's spool. Appended, never
        # substituted: dropping the entries above would leave the worker
        # with no frozen code to run.
        # Re-validated here, not merely at assignment. The output side
        # already did this; the input side trusted whatever the attribute
        # happened to hold, so writing to the private backing attribute
        # reached a submit file with a comma-split entry or an injected
        # submit command. Validate what we are about to emit.
        extra_in = _validate_transfer_entries(
            self._extra_transfer_input_files,
            what="extra_transfer_input_files (at submit)")
        for entry in extra_in:
            _reject_reserved_basename(
                entry, "extra_transfer_input_files (at submit)")
        transfer_in += extra_in
        _reject_duplicate_basenames(transfer_in, "transfer_input_files")

        lines: List[str] = [
            "# Auto-generated by RIFT.simulation_manager.database."
            "DualCondorRunQueue",
            "universe                = vanilla",
            "executable              = {}".format(bootstrap),
        ]
        args_tail = ["--sim-name", sim_name, "--level", str(int(level))]
        if prev_basenames:
            args_tail.append("--prev-levels")
            args_tail.extend(prev_basenames)
        lines.append('arguments              = "{}"'.format(
            " ".join(_condor_arg_quote(a) for a in args_tail)))
        if transfer_in:
            lines.append("transfer_input_files    = {}".format(",".join(transfer_in)))
        lines.append("should_transfer_files   = YES")
        lines.append("when_to_transfer_output = ON_EXIT")
        # Backend-supplied products, beyond the level_<N>.json marker.
        # transfer_output_files is explicit, so HTCondor returns ONLY what
        # is named here: anything else the worker wrote is destroyed with
        # the sandbox. A backend whose science *is* output files (rather
        # than a single JSON marker) has to be able to name them, or its
        # jobs complete having thrown their results away.
        out_names = [out_base]
        out_remaps = ["{}={}".format(out_base, out_target)]
        # Validate the raw COLLECTION first, exactly as the input side
        # does. Iterating the property alone was not symmetric: a bare
        # str reaching the backing attribute tuple()s into one entry per
        # character, and each single character then passes the per-entry
        # checks cleanly — so build_worker emitted
        # `transfer_output_files = level_1.json,w,x,y,z` and returned
        # successfully, instead of raising the way the input side does.
        for entry in _validate_transfer_entries(
                self._extra_transfer_output_files,
                what="extra_transfer_output_files (at submit)",
                remap_syntax=True):
            try:
                name = str(entry).format(level=int(level), sim_name=sim_name)
            except (KeyError, IndexError, AttributeError) as exc:
                raise ValueError(
                    "extra_transfer_output_files: {!r} uses an unknown "
                    "placeholder {}; only {{level}} and {{sim_name}} are "
                    "substituted.".format(entry, exc)) from None
            # Re-validate AFTER substitution: the checks at assignment saw
            # the template, and expansion can introduce a space or a path
            # separator that HTCondor's transfer list cannot express.
            _validate_transfer_entries([name],
                                       what="extra_transfer_output_files "
                                            "(after substitution)",
                                       remap_syntax=True)
            _reject_reserved_basename(
                name, "extra_transfer_output_files (after substitution)")
            if " " in name or "/" in name:
                raise ValueError(
                    "extra_transfer_output_files: {!r} expands to {!r}; "
                    "HTCondor transfer lists cannot express a space or a "
                    "path separator in an entry.".format(entry, name))
            out_names.append(name)
            out_remaps.append("{}={}".format(name, sd / name))
        _reject_duplicate_basenames(out_names, "transfer_output_files")
        lines.append("transfer_output_files   = {}".format(",".join(out_names)))
        lines.append('transfer_output_remaps  = "{}"'.format(";".join(out_remaps)))
        lines.append("getenv                  = {}".format(self.getenv))

        release_terms = []
        if self.auto_release_on_oom:
            # Stuart's catch-and-release pattern: on a hold this site
            # calls "out of memory", bump request_memory by
            # oom_memory_factor and release. After oom_max_retries the job
            # stays held and the archive's stuck-detection takes over.
            #
            # Which holds those are, and what counts the retries, come from
            # oom_hold_codes / oom_hold_subcode_exclusions /
            # oom_retry_counter. See DEFAULT_OOM_HOLD_CODES for why they
            # cannot be constants.
            was_oom = self._oom_hold_predicate(
                "LastHoldReasonCode", "LastHoldReasonSubCode")
            is_oom = self._oom_hold_predicate(
                "HoldReasonCode", "HoldReasonSubCode")
            lines.append("MY.InitialRequestMemory = {}".format(request_memory))
            # MemoryUsage is the attribute here that can actually be
            # undefined: in the job ad it is itself an expression over
            # ResidentSetSize, which a job held before it ever executed
            # does not have. int(factor * n * undefined) is undefined, an
            # undefined request_memory matches no slot, and the job then
            # sits Idle with nothing in its log to say why. Fall back to
            # the original request: released unchanged it may hold again,
            # but the retry cap bounds that, whereas never matching is
            # bounded by nothing.
            lines.append(
                "request_memory          = ifthenelse(({was_oom}) && "
                "(MemoryUsage =!= undefined), "
                "int({factor} * ({counter}) * MemoryUsage), "
                "MY.InitialRequestMemory)".format(
                    was_oom=was_oom, factor=self.oom_memory_factor,
                    counter=self.oom_retry_counter))
            release_terms.append(
                "({is_oom}) && ({counter} < {n})".format(
                    is_oom=is_oom, counter=self.oom_retry_counter,
                    n=self.oom_max_retries))
        else:
            lines.append("request_memory          = {}M".format(request_memory))

        # A backend with its own release condition contributes a term
        # rather than a replacement. Before this, the only way to add one
        # was extra_condor_cmds, which is emitted last and so overwrites
        # periodic_release outright -- taking the OOM policy above with
        # it, silently, and leaving that copy of the expression to drift
        # away from this one. Site policy varies enough that the hook is
        # necessary (an opportunistic pool holds jobs for reasons a
        # dedicated cluster never sees); losing the memory handling to
        # get it is not.
        if self.extra_periodic_release:
            site_term = self.extra_periodic_release
            if self.auto_release_on_oom:
                # Scope the site term away from the codes the OOM policy
                # owns. Without this, OR-ing does not partition anything:
                # a term like `(HoldReasonCode =!= 1) && (NumJobStarts <
                # 50)` matches 26 and 34 as well, so it re-releases a job
                # whose memory budget is deliberately spent. oom_max_retries
                # then caps nothing, request_memory keeps being multiplied
                # by a NumHolds nothing bounds, and the job climbs past
                # every slot in the pool and sits Idle forever -- a worse
                # end than the Held state the cap exists to produce.
                # ...away from whatever codes the policy is CONFIGURED to
                # own, not a second hardcoded copy of the default set.
                site_term = "({site}) && !({is_oom})".format(
                    site=site_term,
                    is_oom=self._oom_hold_predicate(
                        "HoldReasonCode", "HoldReasonSubCode"))
            release_terms.append(site_term)
        if release_terms:
            # One term is emitted bare so that configuring no site term
            # leaves the expression byte-identical to what this class
            # emitted before the hook existed.
            #
            # Term order is load-bearing when there are two. The OOM term
            # comes first and `||` short-circuits on True, so a site term
            # that evaluates to Error cannot suppress a memory release.
            # Reversing them would let a malformed site expression take
            # the memory policy down with it.
            body = (release_terms[0] if len(release_terms) == 1
                    else " || ".join("({})".format(t) for t in release_terms))
            lines.append("periodic_release        = " + body)

        lines.append("request_disk            = {}".format(request_disk))
        if self.accounting_group:
            lines.append("accounting_group        = {}".format(self.accounting_group))
        if self.accounting_group_user:
            lines.append("accounting_group_user   = {}".format(self.accounting_group_user))
        if self.use_singularity:
            if not self.singularity_image:
                raise ValueError("use_singularity=True but no singularity_image set")
            lines.append("MY.SingularityImage     = \"{}\"".format(self.singularity_image))
            lines.append("MY.SingularityBindCVMFS = True")
            lines.append('Requirements            = HAS_SINGULARITY=?=TRUE')
            lines.append("transfer_executable     = False")
        for k, v in extra_cmds.items():
            lines.append("{:24s}= {}".format(k, v))
        tag = "{}_lvl{}".format(sim_name, level)
        lines.append("log                     = {}/{}.log".format(log_dir, tag))
        lines.append("output                  = {}/{}.out".format(log_dir, tag))
        lines.append("error                   = {}/{}.err".format(log_dir, tag))
        lines.append("queue 1")

        sub_path = sub_dir / "{}.sub".format(tag)
        sub_path.write_text("\n".join(lines) + "\n")
        return str(sub_path)

    # -------- DAG assembly + submit ---------------------------------------
    def submit(self, archive: Archive, sim_names: Iterable[str]
               ) -> List[Tuple[str, str]]:
        """For each sim, build its per-level work units (vanilla-universe
        .sub via build_worker, OR a sub-DAG via subdag_factory) and
        assemble a wrapper DAG with PARENT/CHILD edges between the
        levels of each sim. Behavior of the dispatch step depends on
        self.submit_mode:

          * "submit" (default): invoke condor_submit_dag and return
            [(sim, cluster_id), ...].
          * "embed": write the wrapper DAG and return [(sim, ""), ...]
            without dispatching. The wrapper DAG path is recorded on
            self.last_wrapper_dag_path so a parent workflow can pick
            it up via `SUBDAG EXTERNAL <id> <wrapper_path>`.

        When self.subdag_factory is set, each per-(sim, level) node in
        the wrapper DAG is `SUBDAG EXTERNAL` rather than `JOB`. This
        lets backends whose work unit is itself a DAG (e.g. GW PE via
        util_RIFT_pseudo_pipe) compose cleanly.
        """
        sim_names = list(sim_names)
        # nodes: list of (sim, level, work_path, is_subdag)
        nodes: List[Tuple[str, int, str, bool]] = []
        edges: List[Tuple[str, str]] = []

        for sim in sim_names:
            rec = StatusRecord.read(archive.sim_dir(sim))
            cur = rec.data.get("current_level", 0)
            tgt = rec.data.get("target_level", 0)
            prev_id: Optional[str] = None
            for lvl in range(cur + 1, tgt + 1):
                node_id = "{}_lvl{}".format(sim, lvl)
                if self.subdag_factory is not None:
                    # Checked here, not just in __init__: subdag_factory and
                    # the extras are plain attributes, and assigning either
                    # after construction reached this path with the extras
                    # silently ignored.
                    if (self.extra_transfer_input_files
                            or self.extra_transfer_output_files):
                        raise ValueError(
                            "extra_transfer_{input,output}_files are applied by "
                            "build_worker, which this sub-DAG path bypasses: the "
                            "sub-DAG owns its own submit descriptions. Put the "
                            "entries in the DAG the factory generates, or clear "
                            "subdag_factory.")
                    work_path = self.subdag_factory(archive, sim, lvl)
                    nodes.append((sim, lvl, work_path, True))
                else:
                    work_path = self.build_worker(archive, sim, level=lvl)
                    nodes.append((sim, lvl, work_path, False))
                if prev_id is not None:
                    edges.append((prev_id, node_id))
                prev_id = node_id

        if not nodes:
            return []

        dag_dir = archive.base / "run_queue" / "dags"
        dag_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(dag_dir.glob("run_*.dag"))
        idx = len(existing) + 1
        dag_path = dag_dir / "run_{:04d}.dag".format(idx)

        dag_lines: List[str] = []
        for sim, lvl, work_path, is_subdag in nodes:
            node_id = "{}_lvl{}".format(sim, lvl)
            if is_subdag:
                dag_lines.append("SUBDAG EXTERNAL {} {}".format(node_id, work_path))
            else:
                dag_lines.append("JOB {} {}".format(node_id, work_path))
        for parent, child in edges:
            dag_lines.append("PARENT {} CHILD {}".format(parent, child))
        dag_path.write_text("\n".join(dag_lines) + "\n")
        self.last_wrapper_dag_path = str(dag_path)

        if self.submit_mode == "embed":
            # Mark sims submit_ready; the parent workflow that includes
            # this DAG via SUBDAG EXTERNAL is responsible for actual
            # dispatch + completion semantics.
            for sim in sim_names:
                archive.transition(sim, "submit_ready",
                                   request_queue={"kind": "condor",
                                                  "pool": self.run_pool,
                                                  "wrapper_dag_path": str(dag_path),
                                                  "submit_mode": "embed"})
            return [(sim, "") for sim in sim_names]

        # "submit" mode: dispatch now via condor_submit_dag.
        cluster_id = self._submit_dag(dag_path)
        self.dag_cluster_id = cluster_id
        for sim in sim_names:
            archive.transition(sim, "submit_ready",
                               request_queue={"kind": "condor",
                                              "pool": self.run_pool,
                                              "dag_cluster_id": cluster_id,
                                              "dag_path": str(dag_path),
                                              "submit_mode": "submit"})
        return [(sim, str(cluster_id) if cluster_id is not None else "")
                for sim in sim_names]

    def _submit_dag(self, dag_path: Path) -> Optional[int]:
        """Invoke condor_submit_dag. Returns the cluster id of the
        submitted DAGMan, or None if condor_submit_dag isn't on PATH
        (in which case the call is logged + skipped — useful for dry
        runs and CI without a real condor)."""
        cmd = ["condor_submit_dag", "-f"]
        if self.run_pool:
            cmd[1:1] = ["-name", self.run_pool]
        cmd.append(str(dag_path))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    check=True)
        except FileNotFoundError:
            logger.warning("condor_submit_dag not on PATH; skipping submit "
                           "of %s. Set up a real condor environment to "
                           "actually dispatch work.", dag_path)
            return None
        except subprocess.CalledProcessError as exc:
            logger.error("condor_submit_dag failed: %s", exc.stderr)
            raise
        # Parse "submitted to cluster <N>" out of stdout.
        import re
        m = re.search(r"submitted to cluster (\d+)", result.stdout)
        return int(m.group(1)) if m else None

    # -------- polling ------------------------------------------------------
    def poll(self, archive: Archive, sim_names: Iterable[str]
             ) -> Dict[str, str]:
        """Query the run pool's schedd; sims with at least one job still
        in the queue go to 'running'; sims that previously had jobs but
        now don't, with their target levels' output files present, are
        reconciled by Archive.refresh_status_from_disk (called by the
        request queue right after this returns).

        Works with htcondor or htcondor2 via the cached _htcondor_module
        from CondorManager. If neither binding is available, returns the
        currently-recorded statuses without modification."""
        sim_names = list(sim_names)
        try:
            from .CondorManager import _htcondor_module, has_htcondor
        except ImportError:
            _htcondor_module, has_htcondor = None, False
        if not has_htcondor or _htcondor_module is None:
            return {n: archive.get_status(n) for n in sim_names}
        try:
            if self.run_collector:
                collector = _htcondor_module.Collector(self.run_collector)
                schedd_ad = collector.locate(
                    _htcondor_module.DaemonTypes.Schedd, self.run_pool)
                schedd = _htcondor_module.Schedd(schedd_ad)
            else:
                schedd = _htcondor_module.Schedd()
            constraint = None
            if self.dag_cluster_id is not None:
                constraint = "DAGManJobId =?= {}".format(self.dag_cluster_id)
            ads = schedd.query(constraint=constraint,
                               projection=["ClusterId", "ProcId", "JobStatus",
                                           "Args", "Cmd"])
        except Exception as exc:
            logger.warning("DualCondorRunQueue.poll: schedd query failed: %s", exc)
            return {n: archive.get_status(n) for n in sim_names}

        in_queue: Dict[str, set] = {}    # sim -> set of levels still in queue
        for ad in ads:
            args = ad.get("Args") or ""
            sim, lvl = _parse_args_for_sim_level(args)
            if sim is None:
                continue
            in_queue.setdefault(sim, set()).add(lvl)

        results: Dict[str, str] = {}
        for sim in sim_names:
            current = archive.get_status(sim)
            if sim in in_queue:
                if current not in ("running", "complete", "stuck"):
                    archive.transition(sim, "running")
                results[sim] = "running"
            else:
                results[sim] = current
        return results


def _condor_arg_quote(s: str) -> str:
    """condor_submit's `arguments = "..."` uses a particular quoting
    convention: single quotes wrap any token containing whitespace,
    embedded double-quotes are doubled. For our usage (sim names,
    integer levels, level_<N>.json basenames) we don't need quoting,
    but keep the helper for future safety."""
    if not s:
        return "''"
    if any(c in s for c in (" ", "\t", '"', "'")):
        return "'" + s.replace("'", "''") + "'"
    return s


def _parse_args_for_sim_level(args: str) -> Tuple[Optional[str], Optional[int]]:
    """Recover (sim_name, level) from a condor job ad's Args string.
    The submit description sets:
        arguments = "--sim-name <name> --level <N> [--prev-levels ...]"
    """
    parts = args.split()
    sim, lvl = None, None
    i = 0
    while i < len(parts):
        if parts[i] == "--sim-name" and i + 1 < len(parts):
            sim = parts[i + 1]
            i += 2
        elif parts[i] == "--level" and i + 1 < len(parts):
            try:
                lvl = int(parts[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1
    return sim, lvl


# ---------------------------------------------------------------------------
# Slurm queue
# ---------------------------------------------------------------------------
#
# SLURM differs from condor in two important ways:
#
#   1. There is no transfer_input_files / transfer_output_files. Workers run
#      on compute nodes that are assumed to share a filesystem with the
#      submit host (the standard HPC model: NFS-mounted home/scratch). The
#      worker reads sims/<n>/params.json directly and writes sims/<n>/
#      level_<N>.json directly. No remap, no flattening.
#   2. Inter-job dependencies use --dependency=afterok:<jobid> on sbatch
#      rather than DAGMan's PARENT/CHILD edges. We chain each sim's level
#      submissions with that flag so level N's job waits for level N-1.
#
# For sites where compute nodes do NOT share the archive's FS, the
# operator should arrange a container with a bind-mount (singularity exec
# -B <archive>:<archive> ...) or use sbcast to broadcast a manifest into
# /tmp on the workers. That's deployment configuration; the framework
# treats the archive directory as visible from the worker.

class SlurmRunQueue(RunQueue):
    """Per-(sim, level) worker is one sbatch script. Levels for the same
    sim are chained via `--dependency=afterok:<previous-jobid>`.

    Configuration:
        partition          : str  -- required: SLURM partition / queue
        time_limit         : str  -- e.g. '02:00:00' (default '01:00:00')
        nodes              : int  -- default 1
        ntasks             : int  -- default 1
        cpus_per_task      : int  -- default 1
        request_memory     : int (MB), default 4096
        request_disk       : str  -- ignored by slurm directly; informational
        account            : str  -- defaults to env SLURM_ACCOUNT
        qos                : str  -- defaults to env SLURM_QOS
        partition_extra    : dict -- additional `#SBATCH --key=value` lines
                                     emitted verbatim
        sbatch_path        : str  -- default 'sbatch'
        squeue_path        : str  -- default 'squeue'
        sacct_path         : str  -- default 'sacct'
        python_executable  : str  -- default 'python3'; what to invoke the
                                     bootstrap with
        prelude            : str  -- shell snippet inserted before the
                                     bootstrap (`module load python`,
                                     `source /path/to/env/bin/activate`,
                                     `singularity exec ... python3 ...`)

    Compared to DualCondorRunQueue, this queue does NOT bake in an
    auto-OOM-release mechanism. Slurm's standard tool for that is the
    job's `--requeue` policy combined with operator-driven `scontrol
    update jobid=... ReqMem=...`; users who want automated retry should
    use Archive.unstick(name, bump_memory=True) and re-submit, or wrap
    the sbatch in a re-submit shell harness.
    """
    kind = "slurm"

    def __init__(self,
                 partition: Optional[str] = None,
                 time_limit: str = "01:00:00",
                 nodes: int = 1,
                 ntasks: int = 1,
                 cpus_per_task: int = 1,
                 request_memory: int = 4096,
                 request_disk: Optional[str] = None,
                 account: Optional[str] = None,
                 qos: Optional[str] = None,
                 partition_extra: Optional[Dict[str, str]] = None,
                 sbatch_path: str = "sbatch",
                 squeue_path: str = "squeue",
                 sacct_path: str = "sacct",
                 python_executable: str = "python3",
                 prelude: str = "",
                 **submit_kwargs: Any):
        self.partition = partition
        self.time_limit = time_limit
        self.nodes = int(nodes)
        self.ntasks = int(ntasks)
        self.cpus_per_task = int(cpus_per_task)
        self.request_memory = int(request_memory)
        self.request_disk = request_disk
        self.account = account or os.environ.get("SLURM_ACCOUNT")
        self.qos = qos or os.environ.get("SLURM_QOS")
        self.partition_extra = partition_extra or {}
        self.sbatch_path = sbatch_path
        self.squeue_path = squeue_path
        self.sacct_path = sacct_path
        self.python_executable = python_executable
        self.prelude = prelude
        self.submit_kwargs = submit_kwargs
        # Per-archive bookkeeping: sim_name -> [(level, jobid), ...]
        self.submitted_jobs: Dict[str, List[Tuple[int, str]]] = {}



    # ---- bootstrap helpers ------------------------------------------------
    def _bootstrap_path(self, archive: Archive) -> Path:
        path = archive.base / "run_queue" / "workers" / "bootstrap.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(archive.worker_bootstrap_script())
        path.chmod(0o755)
        return path

    # ---- per-(sim, level) sbatch script ---------------------------------
    def build_worker(self, archive: Archive, sim_name: str,
                     level: int = 1) -> str:
        """Write the per-(sim, level) sbatch shell script and return its
        absolute path. Idempotent."""
        sd = archive.sim_dir(sim_name)
        if not sd.exists():
            raise FileNotFoundError("sim_dir does not exist: {}".format(sd))

        # Per-sim resource overrides on top of queue defaults.
        try:
            res = archive.get_resources(sim_name)
        except Exception:
            res = {}
        request_memory = int(res.get("request_memory", self.request_memory))
        time_limit = res.get("time_limit", self.time_limit)
        partition = res.get("partition", self.partition)
        partition_extra = dict(self.partition_extra)
        partition_extra.update(res.get("extra_sbatch_directives") or {})

        bootstrap = self._bootstrap_path(archive)
        log_dir = archive.base / "run_queue" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        sub_dir = archive.base / "run_queue" / "submit_files"
        sub_dir.mkdir(parents=True, exist_ok=True)

        prev_basenames = ["level_{}.json".format(i) for i in range(1, level)]
        tag = "{}_lvl{}".format(sim_name, level)

        sbatch_lines: List[str] = [
            "#!/bin/bash",
            "# Auto-generated by RIFT.simulation_manager.database.SlurmRunQueue",
            "#SBATCH --job-name={}".format(tag),
            "#SBATCH --output={}/{}.out".format(log_dir, tag),
            "#SBATCH --error={}/{}.err".format(log_dir, tag),
            "#SBATCH --time={}".format(time_limit),
            "#SBATCH --nodes={}".format(self.nodes),
            "#SBATCH --ntasks={}".format(self.ntasks),
            "#SBATCH --cpus-per-task={}".format(self.cpus_per_task),
            "#SBATCH --mem={}M".format(request_memory),
        ]
        if partition:
            sbatch_lines.append("#SBATCH --partition={}".format(partition))
        if self.account:
            sbatch_lines.append("#SBATCH --account={}".format(self.account))
        if self.qos:
            sbatch_lines.append("#SBATCH --qos={}".format(self.qos))
        for k, v in partition_extra.items():
            sbatch_lines.append("#SBATCH --{}={}".format(k.lstrip("-"), v))
        sbatch_lines.append("")
        sbatch_lines.append("set -euo pipefail")
        if self.prelude:
            sbatch_lines.append(self.prelude)
        # Worker logic.
        sbatch_lines.append("cd {}".format(sd))
        cmd = [self.python_executable, str(bootstrap),
               "--sim-name", sim_name, "--level", str(int(level)),
               "--code-dir", str(archive.base / "code")]
        if prev_basenames:
            cmd.append("--prev-levels")
            cmd.extend(prev_basenames)
        sbatch_lines.append("exec " + " ".join(cmd))
        sbatch_lines.append("")

        sub_path = sub_dir / "{}.sh".format(tag)
        sub_path.write_text("\n".join(sbatch_lines))
        sub_path.chmod(0o755)
        return str(sub_path)

    # ---- submission ------------------------------------------------------
    def submit(self, archive: Archive, sim_names: Iterable[str]
               ) -> List[Tuple[str, str]]:
        """Build per-(sim, level) sbatch scripts and submit them with
        --dependency=afterok chains within each sim. Returns
        [(sim_name, last_level_jobid_or_empty), ...]."""
        sim_names = list(sim_names)
        results: List[Tuple[str, str]] = []
        for sim in sim_names:
            rec = StatusRecord.read(archive.sim_dir(sim))
            cur = rec.data.get("current_level", 0)
            tgt = rec.data.get("target_level", 0)
            level_jobids: List[Tuple[int, str]] = []
            prev_jobid: Optional[str] = None
            failed = False
            for lvl in range(cur + 1, tgt + 1):
                sub_path = self.build_worker(archive, sim, level=lvl)
                jobid = self._sbatch(sub_path, depends_on=prev_jobid)
                if jobid is None:
                    # sbatch missing or failed — skip remaining levels for
                    # this sim; the chain would be invalid without the
                    # parent jobid anyway.
                    failed = True
                    break
                level_jobids.append((lvl, jobid))
                prev_jobid = jobid
            self.submitted_jobs.setdefault(sim, []).extend(level_jobids)
            archive.transition(sim, "submit_ready",
                               request_queue={"kind": "slurm",
                                              "partition": self.partition,
                                              "jobs": [
                                                  {"level": lvl, "jobid": jid}
                                                  for lvl, jid in level_jobids]})
            last_jobid = level_jobids[-1][1] if level_jobids else ""
            if failed and not level_jobids:
                last_jobid = ""
            results.append((sim, last_jobid))
        return results

    def _sbatch(self, sub_path: str,
                depends_on: Optional[str] = None) -> Optional[str]:
        """Invoke `sbatch [--dependency=afterok:<id>] <sub_path>` and
        return the jobid. Returns None (and logs a warning) if sbatch
        is not on PATH — useful for dry runs and for running the v2
        unit tests on a non-slurm host."""
        cmd = [self.sbatch_path, "--parsable"]
        if depends_on:
            cmd.append("--dependency=afterok:{}".format(depends_on))
        cmd.append(sub_path)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    check=True)
        except FileNotFoundError:
            logger.warning("sbatch not on PATH; skipping submit of %s. "
                           "Set up a slurm environment to actually dispatch.",
                           sub_path)
            return None
        except subprocess.CalledProcessError as exc:
            logger.error("sbatch failed: %s", exc.stderr)
            raise
        # `--parsable` prints just the jobid (or jobid;cluster).
        first_line = (result.stdout or "").strip().splitlines()[0:1]
        if not first_line:
            return None
        return first_line[0].split(";", 1)[0]

    # ---- polling ---------------------------------------------------------
    def poll(self, archive: Archive, sim_names: Iterable[str]
             ) -> Dict[str, str]:
        """Use squeue to find jobs still in the queue, then sacct to
        confirm completion/failure for ones that have left. Updates
        archive statuses; output-on-disk is the authoritative
        completion signal (applied by the request queue's poll via
        Archive.refresh_status_from_disk)."""
        sim_names = list(sim_names)
        # Build the union of jobids we care about.
        all_jobids: Dict[str, str] = {}   # jobid -> sim
        for sim in sim_names:
            for lvl, jid in self.submitted_jobs.get(sim, []):
                if jid:
                    all_jobids[jid] = sim
        if not all_jobids:
            return {n: archive.get_status(n) for n in sim_names}

        in_queue = self._squeue(list(all_jobids.keys()))
        results: Dict[str, str] = {}
        for sim in sim_names:
            current = archive.get_status(sim)
            sim_jobs = self.submitted_jobs.get(sim, [])
            still_running = [jid for _, jid in sim_jobs
                             if jid and jid in in_queue]
            if still_running:
                if current not in ("running", "complete", "stuck"):
                    archive.transition(sim, "running")
                results[sim] = "running"
            else:
                results[sim] = current
        return results

    def _squeue(self, jobids: List[str]) -> set:
        """Return the subset of `jobids` still showing in squeue.
        Returns the input set unchanged on squeue failure (conservative:
        we'd rather wait too long than declare a sim done prematurely)."""
        if not jobids:
            return set()
        try:
            result = subprocess.run(
                [self.squeue_path, "-j", ",".join(jobids),
                 "--noheader", "-o", "%i"],
                capture_output=True, text=True, check=False)
        except FileNotFoundError:
            logger.info("squeue not on PATH; assuming all jobs still queued")
            return set(jobids)
        # squeue exits non-zero if NONE of the jobids match (slurm versions
        # vary). Treat empty stdout as "all done".
        out = (result.stdout or "").strip()
        if not out:
            return set()
        return {line.strip() for line in out.splitlines() if line.strip()}


class SlurmRequestQueue(RequestQueue):
    """Slurm-side orchestrator. Identifies pending sims and delegates
    submission to SlurmRunQueue."""
    kind = "slurm"

    def __init__(self,
                 run_queue: Optional["SlurmRunQueue"] = None,
                 **submit_kwargs: Any):
        self.run_queue = run_queue
        self.submit_kwargs = submit_kwargs

    def submit_pending(self, archive: Archive) -> List[str]:
        if self.run_queue is None:
            raise RuntimeError("SlurmRequestQueue: run_queue not attached")
        pending = (archive.with_status("ready")
                   + archive.with_status("refine_ready"))
        if not pending:
            return []
        try:
            self.run_queue.submit(archive, pending)
        except Exception:
            for n in pending:
                archive.transition(n, "ready")
            raise
        return pending

    def poll(self, archive: Archive) -> Dict[str, str]:
        if self.run_queue is None:
            return {}
        observed = self.run_queue.poll(archive, archive.simulations_iter_names())
        archive.refresh_status_from_disk()
        return observed


# ---------------------------------------------------------------------------
# Queue auto-resolution from manifest config
# ---------------------------------------------------------------------------

QUEUE_REGISTRY: Dict[Tuple[str, str], type] = {
    ("request", "local"):  LocalRequestQueue,
    ("run",     "local"):  LocalRunQueue,
    ("request", "condor"): DualCondorRequestQueue,
    ("run",     "condor"): DualCondorRunQueue,
    ("request", "slurm"):  SlurmRequestQueue,
    ("run",     "slurm"):  SlurmRunQueue,
}


def make_queues_from_manifest(archive: Archive
                              ) -> Tuple[RequestQueue, RunQueue]:
    """Instantiate (request_queue, run_queue) from the archive's manifest.
    The run queue's `extra` dict is passed as keyword arguments; the
    request queue gets the run_queue plumbed in.

    Used by the request_sim CLI's --ensure mode so the user doesn't
    have to wire queues by hand."""
    rq_cfg = archive.manifest.data.get("request_queue", {})
    runq_cfg = archive.manifest.data.get("run_queue", {})
    rq_kind = rq_cfg.get("kind", "local")
    runq_kind = runq_cfg.get("kind", "local")

    runq_cls = QUEUE_REGISTRY.get(("run", runq_kind))
    if runq_cls is None:
        raise ValueError("No registered run queue for kind={!r}".format(runq_kind))
    run_queue = runq_cls(**(runq_cfg.get("extra") or {}))

    rq_cls = QUEUE_REGISTRY.get(("request", rq_kind))
    if rq_cls is None:
        raise ValueError("No registered request queue for kind={!r}".format(rq_kind))
    request_queue = rq_cls(run_queue=run_queue, **(rq_cfg.get("extra") or {}))
    return request_queue, run_queue
