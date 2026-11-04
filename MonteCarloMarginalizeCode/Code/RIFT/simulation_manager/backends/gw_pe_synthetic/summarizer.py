"""Summarizer for the GW PE synthetic-targeted backend.

Reads the latest level's output (the PE sub-DAG's posterior file or
the stub's level_<N>.json) and rolls a small summary into
`<sim_dir>/summary.json` / the index.

The real summarizer (when pseudo_pipe is producing posteriors) reads
the canonical RIFT output files. For the stub the summary just
echoes the marker. Generator/factory branching is by file presence —
if a posterior file is found at the expected location we use it,
otherwise we fall back to the level_<N>.json marker.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def summarize(sim_dir, params, levels=None):
    latest = _pick_latest(sim_dir, levels)
    if latest is None:
        return None

    # Look for a RIFT posterior file in the same level rundir as the
    # level_<N>.json (when the real factory wrote one); summarize from
    # it. Otherwise return a stub summary.
    level_num = _level_from_path(latest)
    rundir = os.path.join(sim_dir, "level_{}".format(level_num))
    posterior_path = _find_posterior(rundir)
    if posterior_path:
        return _summarize_posterior(posterior_path, params, level_num)

    # Stub fallback: just record what the marker file says.
    with open(latest) as f:
        marker = json.load(f)
    return {
        "level": int(marker.get("level", level_num)),
        "stub": True,
        "_note": "no posterior found; summarizer fell back to marker"
    }


def _pick_latest(sim_dir, levels) -> Optional[str]:
    if levels:
        return levels[-1]
    cands = sorted(p for p in os.listdir(sim_dir)
                   if p.startswith("level_") and p.endswith(".json"))
    if not cands:
        return None
    return os.path.join(sim_dir, cands[-1])


def _level_from_path(path: str) -> int:
    name = os.path.basename(path)
    # name is "level_<N>.json"
    try:
        return int(name[len("level_"):].split(".")[0])
    except Exception:
        return 0


def _find_posterior(rundir: str) -> Optional[str]:
    """Return the path of a RIFT posterior file in `rundir`, if any.
    RIFT writes posteriors under the run dir as
    `posterior_samples-final.dat` or similar; we look for a few
    canonical names to keep this robust to small naming changes."""
    if not os.path.isdir(rundir):
        return None
    candidates = [
        "posterior_samples-final.dat",
        "extrinsic_posterior_samples.dat",
        "posterior_samples.dat",
    ]
    for c in candidates:
        p = os.path.join(rundir, c)
        if os.path.isfile(p):
            return p
    return None


def _summarize_posterior(posterior_path: str, params: Any,
                         level: int) -> Dict[str, Any]:
    """Load a small set of summary stats from the posterior. Stays
    stdlib-friendly so the summarizer is invokable on any host the
    archive might be opened from; numpy is used only if available."""
    summary: Dict[str, Any] = {
        "level": int(level),
        "posterior_path": os.path.relpath(posterior_path),
    }
    try:
        import numpy as np
        with open(posterior_path) as f:
            header = f.readline().strip()
        cols = header.lstrip("#").split()
        data = np.loadtxt(posterior_path, comments="#")
        if data.ndim == 1:
            data = data[None, :]
        summary["n_samples"] = int(len(data))
        # Pull a few canonical columns when available.
        for col in ("mc", "eta", "lnL", "distance"):
            if col in cols:
                idx = cols.index(col)
                vals = data[:, idx]
                summary["{}_median".format(col)] = float(np.median(vals))
                summary["{}_std".format(col)] = float(np.std(vals))
    except ImportError:
        summary["_note"] = "numpy unavailable; only path recorded"
    except Exception as exc:
        summary["_warning"] = "summary parsing failed: {}".format(exc)
    return summary
