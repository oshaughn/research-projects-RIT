"""
RIFT.hyperpipe.marg_list
========================

Multi-event / multi-driver "marg-list" assembly.

A *marg* in the hyperpipeline vocabulary is a per-event (or per-likelihood)
job that evaluates the underlying likelihood on the current grid of
hyperparameters. ``create_eos_posterior_pipeline`` consumes parallel
lists describing each marg:

    * ``--marg-event-exe-list-file``    -- one executable path per line
    * ``--marg-event-args-list-file``   -- one args string per line
    * ``--marg-event-nchunk-list-file`` -- one int per line (heterogeneous OK)
    * ``--event-file <path>``           -- repeated; one per marg entry

This module assembles those four files from a Hydra ``marg-list:`` section
and is responsible for:

  * resolving each ``exe`` (via :func:`RIFT.misc.dag_utils_generic.which`
    or a literal path)
  * copying non-RIFT-core executables into the working directory so they
    can be transferred (mirroring the current
    ``util_RIFT_hyperpipe.py`` behaviour)
  * materializing per-entry event files (``event-<idx>.net``) and a
    sentinel ``empty_event_file`` when no event data are supplied
  * stitching in per-driver coord-module overrides for heterogeneous
    analyses (e.g. GW + NICER), where each driver carries its own coord
    convention.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class MargAssembly:
    """Output of :func:`assemble_marg_list`.

    All paths are absolute. Lengths of the per-entry sequences are equal
    to ``len(cfg['marg-list'])``.
    """

    args_lines: List[str] = field(default_factory=list)
    exe_paths: List[str] = field(default_factory=list)
    event_files: List[str] = field(default_factory=list)
    n_chunks: List[int] = field(default_factory=list)
    transfer_files: List[str] = field(default_factory=list)
    names: List[str] = field(default_factory=list)

    # ----- emission helpers ------------------------------------------------
    def write_args_file(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("\n".join(self.args_lines))
            if self.args_lines:
                f.write("\n")

    def write_exe_file(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("\n".join(self.exe_paths))
            if self.exe_paths:
                f.write("\n")

    def write_nchunk_file(self, path: str) -> None:
        with open(path, "w") as f:
            for n in self.n_chunks:
                f.write(f"{int(n)}\n")

    def write_transfer_file_list(self, path: str, extra: Optional[List[str]] = None) -> None:
        """Write the transfer-file-list HTCondor consumes.

        Appends ``extra`` (e.g. user-supplied general.transfer-files) after
        the auto-detected exe transfers.
        """
        all_files = list(self.transfer_files)
        if extra:
            all_files.extend(extra)
        with open(path, "w") as f:
            for line in all_files:
                f.write(line + "\n")


# --------------------------------------------------------------------------
# Core assembly
# --------------------------------------------------------------------------


_RIFT_CORE_EXE_HINTS = (
    # Substrings that, if present in an exe's resolved path, mark it as
    # a core RIFT executable we should *not* duplicate into the working
    # directory or transfer-file-list. Mirrors what the original
    # util_RIFT_hyperpipe.py did for "MonteCarloMarginalizeCode/Code/bin"
    # but is more forgiving of editable installs.
    "MonteCarloMarginalizeCode/Code/bin",
    "/RIFT/",
    "/site-packages/RIFT",
)


def _looks_like_core_rift(exe_path: str) -> bool:
    if not exe_path:
        return False
    return any(h in exe_path for h in _RIFT_CORE_EXE_HINTS)


def _resolve_exe(exe: str, base_dir: str) -> str:
    """Resolve an executable name to an absolute path.

    Tries, in order:
      1. literal absolute path
      2. :func:`shutil.which`
      3. :func:`RIFT.misc.dag_utils_generic.which`
      4. ``<base_dir>/<exe>``
    """
    if not exe:
        raise ValueError("marg-list entry is missing 'exe'.")
    exe = os.path.expanduser(exe.strip())
    if os.path.isabs(exe) and os.path.exists(exe):
        return exe
    bare = os.path.basename(exe)
    found = shutil.which(bare)
    if found:
        return found
    try:
        from RIFT.misc.dag_utils_generic import which as rift_which
    except Exception:  # pragma: no cover -- only at runtime
        rift_which = None
    if rift_which is not None:
        found = rift_which(bare)
        if found:
            return found
    fallback = os.path.join(base_dir, bare)
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"Could not locate marg exe {exe!r}. Searched PATH, RIFT, and {base_dir!r}."
    )


def _stage_event_file(
    entry: Any,
    indx: int,
    base_dir: str,
    run_dir: str,
) -> Tuple[str, bool]:
    """Materialize this entry's event file at run_dir/event-<indx>.net.

    Returns ``(abs_path, is_empty_sentinel)``. If the entry has no
    ``event-file`` set, we write a sentinel file with the single token
    ``empty_event_file`` so the downstream pipeline still sees a
    well-formed input.
    """
    src = None
    if hasattr(entry, "get"):
        src = entry.get("event-file") or entry.get("event_file")
    elif "event-file" in entry:
        src = entry["event-file"]
    dest = os.path.join(run_dir, f"event-{indx}.net")
    if src:
        src = os.path.expanduser(src)
        if not os.path.isabs(src):
            src = os.path.join(base_dir, src)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"marg-list entry {indx}: event-file {src!r} does not exist."
            )
        shutil.copyfile(src, dest)
        return dest, False
    # sentinel
    with open(dest, "w") as f:
        f.write("empty_event_file\n")
    return dest, True


def _entry_get(entry: Any, key: str, default=None):
    """Get a key from a marg-list entry, tolerating dict or DictConfig."""
    try:
        return entry.get(key, default)  # type: ignore[union-attr]
    except AttributeError:
        return entry[key] if key in entry else default


def assemble_marg_list(
    cfg,
    *,
    base_dir: str,
    run_dir: str,
) -> MargAssembly:
    """Walk ``cfg['marg-list']`` and prepare all per-entry artefacts.

    Parameters
    ----------
    cfg
        The full hyperpipe config (DictConfig or dict). Only the
        ``marg-list`` key is consulted here.
    base_dir
        Directory the user invoked ``util_RIFT_hyperpipe.py`` from;
        relative paths in the config are resolved against this.
    run_dir
        Working directory for the pipeline (where ``event-<i>.net``
        files and copies of non-core exes are written).

    Returns
    -------
    MargAssembly
        Holds parallel lists of args / exe / event-file / n-chunk per
        marg entry, plus the auto-detected transfer-file list.
    """
    marg_entries = cfg["marg-list"]
    out = MargAssembly()

    for indx, entry in enumerate(marg_entries):
        name = _entry_get(entry, "name") or f"marg_{indx}"
        out.names.append(name)

        # ----- exe ------------------------------------------------------
        raw_exe = _entry_get(entry, "exe")
        if not raw_exe:
            raise ValueError(f"marg-list[{indx}] ({name!r}) missing required key 'exe'.")
        exe_path = _resolve_exe(raw_exe, base_dir=base_dir)
        out.exe_paths.append(exe_path)
        if not _looks_like_core_rift(exe_path):
            # Stage a local copy so file-transfer (e.g. OSG) can ship it
            local_copy = os.path.join(run_dir, os.path.basename(exe_path))
            if os.path.abspath(local_copy) != os.path.abspath(exe_path):
                shutil.copyfile(exe_path, local_copy)
                try:
                    os.chmod(local_copy, 0o755)
                except OSError:
                    pass
                out.transfer_files.append(local_copy)
            else:
                out.transfer_files.append(exe_path)

        # ----- args -----------------------------------------------------
        args = (_entry_get(entry, "args") or "").strip()
        extra = (_entry_get(entry, "extra-args") or "").strip()
        coord_mod = _entry_get(entry, "coord-module")
        bits = [args]
        if extra:
            bits.append(extra)
        if coord_mod:
            # Per-driver coord-module override. We always emit
            # --supplementary-coordinate-code; drivers that don't consume
            # it should be wrapped in a shell that ignores it.
            bits.append(f"--supplementary-coordinate-code {coord_mod}")
        out.args_lines.append(" ".join(b for b in bits if b))

        # ----- event file ----------------------------------------------
        event_path, _ = _stage_event_file(entry, indx, base_dir=base_dir, run_dir=run_dir)
        out.event_files.append(event_path)

        # ----- n-chunk -------------------------------------------------
        nchunk = _entry_get(entry, "n-chunk", 1)
        try:
            nchunk = int(nchunk)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"marg-list[{indx}] ({name!r}) has non-integer n-chunk={nchunk!r}."
            ) from exc
        if nchunk <= 0:
            raise ValueError(
                f"marg-list[{indx}] ({name!r}) n-chunk must be positive; got {nchunk}."
            )
        out.n_chunks.append(nchunk)

    return out
