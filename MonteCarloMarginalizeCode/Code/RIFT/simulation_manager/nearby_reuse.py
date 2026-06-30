"""Engine-agnostic simulation-reuse layer over the v2 ``Archive``.

Design
------
The "simulation library" is the :class:`~RIFT.simulation_manager.database.Archive`
itself: a directory of completed simulations, each tagged with the formation
parameters Lambda it was run at, the seed(s) it used, and the catalog(s) of
observable samples it produced. This module is the *read / query / pool* layer
on top of that library. It is purely additive and read-only: it never mutates
the archive, never takes the archive lock, and never imports an engine. It uses
numpy + the standard library only.

The point is reuse. A population-inference loop that wants samples at some
parameter point Lambda should ask the library what it already has *before*
generating anything new:

  * **Exact-match reuse (unbiased).** Sims whose params satisfy the archive's
    ``same_q`` predicate are the SAME Lambda run with (typically) different
    seeds. They are independent realizations of one population, so pooling their
    catalogs is statistically clean seed-averaging -- pure variance reduction,
    no bias. Use :func:`find_matching` / :func:`gather_samples`.

  * **Nearby reuse (approximate).** Sims at a *different but close* Lambda can
    stand in for a not-yet-computed point, at the cost of a controlled shape
    bias that grows with the parameter distance. Use :func:`find_nearby`; it
    annotates every record with its distance so a caller can distance-weight,
    interpolate, or restrict to ``distance == 0``.

Pooling caveat (see :func:`pool_catalogs`): concatenating catalog *columns*
across sims is valid for SHAPE (the distribution of observables). It is NOT
valid to sum a rate proxy (``mu_detect`` / ``n_mergers``) across pooled sims --
each pooled sim is an independent realization of the SAME expected rate, so the
rate must be AVERAGED per sim, never summed. Summing multiplies the predicted
rate by the number of pooled sims and corrupts any Poisson rate term.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "iter_completed",
    "param_distance",
    "find_matching",
    "find_nearby",
    "pool_catalogs",
    "gather_samples",
]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def _level_catalog_paths(archive, name: str, status_levels: Sequence[Dict[str, Any]],
                         summary: Optional[Dict[str, Any]]) -> List[Path]:
    """Absolute paths to every per-level catalog file for one sim.

    Prefer the ``levels`` list from status.json (one entry per independent
    larger draw). Fall back to ``summary["output_path"]`` when ``levels`` is
    absent (legacy / hand-built rows). Only paths that exist on disk are
    returned, in level order.
    """
    paths: List[Path] = []
    seen: set = set()
    for entry in status_levels or []:
        op = entry.get("output_path")
        if not op:
            continue
        p = (archive.base / op).resolve()
        if p not in seen and p.exists():
            paths.append(p)
            seen.add(p)
    if not paths and summary and summary.get("output_path"):
        p = (archive.base / summary["output_path"]).resolve()
        if p.exists():
            paths.append(p)
    return paths


def iter_completed(archive):
    """Yield a small record for every ``status == "complete"`` sim.

    Each record is a dict::

        {"name", "params", "summary",
         "levels":        [ {level entries from status.json}, ... ],
         "catalog_paths": [ absolute Path per completed level ]}

    ``levels`` is read from the sim's status.json; if that list is empty we fall
    back to ``summary["output_path"]`` for the catalog path. The records are
    cheap (no catalog data is loaded here -- only paths); call
    :func:`pool_catalogs` to actually read sample columns.
    """
    from .database import StatusRecord

    for row in archive.index.all():
        if row.get("status") != "complete":
            continue
        name = row["name"]
        summary = row.get("summary")
        status_levels: List[Dict[str, Any]] = []
        try:
            rec = StatusRecord.read(archive.sim_dir(name))
            status_levels = list(rec.data.get("levels", []) or [])
        except Exception:
            status_levels = []
        catalog_paths = _level_catalog_paths(archive, name, status_levels, summary)
        yield {
            "name": name,
            "params": row.get("params"),
            "summary": summary,
            "levels": status_levels,
            "catalog_paths": catalog_paths,
        }


# ---------------------------------------------------------------------------
# Parameter distance
# ---------------------------------------------------------------------------

def _as_param_dict(p: Any, ignore: Sequence[str]) -> Optional[Dict[str, float]]:
    """Coerce a params object to a {key: float} dict, dropping ignored and
    underscore-prefixed keys. Returns None for scalar params (handled
    separately)."""
    if isinstance(p, dict):
        out: Dict[str, float] = {}
        for k, v in p.items():
            if str(k).startswith("_") or k in ignore:
                continue
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                # Non-numeric param value: skip (distance is over numeric dims).
                continue
        return out
    return None


def param_distance(p1: Any, p2: Any,
                   ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                   ignore: Sequence[str] = ("seed",)) -> float:
    """Normalized Euclidean distance between two parameter objects.

    Handles both dict params (``{"s": .., "r": ..}``) and scalar params
    (a bare number). Mirrors the ``rapster_same_q`` convention: keys in
    ``ignore`` and any underscore-prefixed keys are dropped before comparing.

    Only keys present in BOTH params are compared -- a key on one side only is
    ignored rather than treated as a large/zero offset, because we cannot know
    the "missing" value's scale. (Document this so callers who care about
    differing schemas can pre-fill defaults.)

    If ``ranges`` (``{key: (lo, hi)}``) is given, each dimension is normalized by
    ``(hi - lo)`` so unlike-scale params are comparable (e.g. ``s in [0, 0.3]``
    vs ``r in [0.5, 2]``). Without ``ranges`` raw differences are used. The
    distance is symmetric and zero on identical inputs.
    """
    d1 = _as_param_dict(p1, ignore)
    d2 = _as_param_dict(p2, ignore)

    # Scalar params on both sides.
    if d1 is None and d2 is None:
        try:
            diff = float(p1) - float(p2)
        except (TypeError, ValueError):
            return 0.0 if p1 == p2 else math.inf
        if ranges:
            # A scalar param has no key; accept a single-entry ranges dict.
            span = next(iter(ranges.values()))
            lo, hi = float(span[0]), float(span[1])
            if hi != lo:
                diff /= (hi - lo)
        return abs(diff)

    # Mixed scalar/dict: not comparable.
    if d1 is None or d2 is None:
        return math.inf

    common = set(d1) & set(d2)
    acc = 0.0
    for k in common:
        diff = d1[k] - d2[k]
        if ranges and k in ranges:
            lo, hi = float(ranges[k][0]), float(ranges[k][1])
            if hi != lo:
                diff /= (hi - lo)
        acc += diff * diff
    return math.sqrt(acc)


# ---------------------------------------------------------------------------
# Matching / nearby queries
# ---------------------------------------------------------------------------

def _resolve_same_q(archive, same_q: Optional[Callable[[Any, Any], bool]]
                    ) -> Callable[[Any, Any], bool]:
    if same_q is not None:
        return same_q
    cand = getattr(archive, "_same_q", None)
    if callable(cand):
        return cand
    # Fall back to exact param equality via zero distance.
    return lambda a, b: param_distance(a, b) == 0.0


def find_matching(archive, params: Any,
                  same_q: Optional[Callable[[Any, Any], bool]] = None
                  ) -> List[Dict[str, Any]]:
    """Completed-sim records whose params ``same_q``-match ``params``.

    These are the SAME Lambda (any seed) as ``params`` -- independent
    realizations whose catalogs are UNBIASED to pool together (seed-averaging).

    If ``same_q`` is None the archive's own predicate (``archive._same_q``,
    e.g. the frozen ``rapster_same_q``) is used; if that is unavailable we fall
    back to ``param_distance == 0``.
    """
    sq = _resolve_same_q(archive, same_q)
    out: List[Dict[str, Any]] = []
    for rec in iter_completed(archive):
        try:
            if sq(params, rec["params"]):
                out.append(rec)
        except Exception:
            # A same_q that chokes on a record's params just doesn't match.
            continue
    return out


def find_nearby(archive, params: Any,
                k: Optional[int] = None,
                tol: Optional[float] = None,
                ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                ignore: Sequence[str] = ("seed",)) -> List[Dict[str, Any]]:
    """Completed-sim records sorted by :func:`param_distance` to ``params``.

    Each returned record is annotated with ``"distance"``. Selection:

      * ``k``   -- keep the k nearest records.
      * ``tol`` -- keep records within ``tol`` (inclusive).

    Both may be given (then both constraints apply: the k nearest *that are
    also* within tol). At least one of ``k`` / ``tol`` is required. Records at
    ``distance > 0`` are an APPROXIMATION -- the caller should distance-weight
    or restrict to ``distance == 0`` for unbiased use.
    """
    if k is None and tol is None:
        raise ValueError("find_nearby: provide at least one of k or tol")
    scored: List[Dict[str, Any]] = []
    for rec in iter_completed(archive):
        rec = dict(rec)
        rec["distance"] = param_distance(params, rec["params"],
                                         ranges=ranges, ignore=ignore)
        scored.append(rec)
    scored.sort(key=lambda r: r["distance"])
    if tol is not None:
        scored = [r for r in scored if r["distance"] <= tol]
    if k is not None:
        scored = scored[:k]
    return scored


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def _load_catalog(path: Path) -> Tuple[Dict[str, List[Any]], int, Optional[float]]:
    """Load one level file. Returns (catalog_dict, n_samples, mu_proxy).

    ``catalog`` is the level's ``catalog`` dict (column-per-key). ``n_samples``
    is the per-event count (len of any column, or ``n_mergers``). ``mu_proxy``
    is the level's ``mu_detect`` if present (for the caller's reference -- this
    function never sums it)."""
    with open(path) as f:
        d = json.load(f)
    catalog = d.get("catalog") or {}
    n = 0
    for col in catalog.values():
        if isinstance(col, list):
            n = len(col)
            break
    if n == 0:
        n = int(d.get("n_mergers", 0))
    mu = d.get("mu_detect")
    return catalog, n, (float(mu) if mu is not None else None)


def pool_catalogs(records: Sequence[Dict[str, Any]],
                  keys: Optional[Sequence[str]] = None
                  ) -> Tuple[Dict[str, List[Any]], List[Dict[str, Any]]]:
    """Concatenate per-level catalogs across records into one pooled catalog.

    Returns ``(pooled_catalog, provenance)``:

      * ``pooled_catalog`` -- ``{column_key: [values...]}``, the column-wise
        concatenation of every level of every record. ALL levels of each sim are
        pooled, since successive levels are independent larger draws (extra valid
        samples). ``keys`` selects which catalog columns to keep (default: the
        union of keys seen across all loaded levels).

      * ``provenance`` -- one entry per contributing sim::

            {"name", "distance" (if the record carried one),
             "n_samples", "n_levels", "mu_detect_per_level": [...]}

    CORRECTNESS -- this routine is SHAPE-focused and that is deliberate:

      * Concatenating observable columns is valid: pooled samples are a larger
        draw from the (possibly mixed-Lambda) population.
      * A rate proxy (``mu_detect`` / ``n_mergers``) is reported per-level in
        provenance and is NEVER summed here. Each pooled sim/level is an
        independent realization of the SAME expected rate, so a caller that
        needs a rate must AVERAGE it across the independent realizations, not
        sum it. Summing would multiply the predicted rate by the number of
        pooled draws and corrupt any Poisson rate term.
      * Records at ``distance > 0`` bias the pooled SHAPE; ``provenance`` carries
        each sim's distance so the caller can distance-weight or restrict to
        ``distance == 0``.
    """
    # First pass: load every level, collect column keys.
    loaded: List[Tuple[Dict[str, Any], List[Dict[str, List[Any]]]]] = []
    all_keys: List[str] = []
    seen_keys: set = set()
    for rec in records:
        level_cats: List[Dict[str, List[Any]]] = []
        mus: List[Optional[float]] = []
        n_total = 0
        for path in rec.get("catalog_paths", []):
            catalog, n, mu = _load_catalog(path)
            level_cats.append(catalog)
            mus.append(mu)
            n_total += n
            for kk in catalog:
                if kk not in seen_keys:
                    seen_keys.add(kk)
                    all_keys.append(kk)
        prov = {
            "name": rec.get("name"),
            "n_samples": n_total,
            "n_levels": len(level_cats),
            "mu_detect_per_level": mus,
        }
        if "distance" in rec:
            prov["distance"] = rec["distance"]
        loaded.append((prov, level_cats))

    out_keys = list(keys) if keys is not None else all_keys
    pooled: Dict[str, List[Any]] = {kk: [] for kk in out_keys}
    provenance: List[Dict[str, Any]] = []

    for prov, level_cats in loaded:
        for catalog in level_cats:
            # Use the column length of this level to keep columns aligned even
            # when a particular key is absent from one level (pad with None).
            level_n = 0
            for col in catalog.values():
                if isinstance(col, list):
                    level_n = len(col)
                    break
            for kk in out_keys:
                col = catalog.get(kk)
                if isinstance(col, list):
                    pooled[kk].extend(col)
                else:
                    pooled[kk].extend([None] * level_n)
        provenance.append(prov)

    return pooled, provenance


def gather_samples(archive, params: Any,
                   same_q: Optional[Callable[[Any, Any], bool]] = None,
                   include_nearby: bool = False,
                   keys: Optional[Sequence[str]] = None,
                   **nearby_kw: Any) -> Dict[str, Any]:
    """Reuse entry point: pooled catalog the library already has for ``params``.

    By default returns the pooled catalog of all SAME-Lambda completed sims
    (exact ``same_q`` matches, any seed) -- the unbiased, seed-averaged reuse an
    inference driver wants before deciding whether to generate more.

    With ``include_nearby=True`` the same-Lambda matches are UNIONED with the
    nearby records from :func:`find_nearby` (pass ``k=`` / ``tol=`` / ``ranges=``
    via ``nearby_kw``); the nearby additions are approximate (distance > 0) and
    carry their distance in the returned provenance.

    Returns::

        {"catalog":     {column: [values...]},
         "provenance":  [per-sim provenance...],
         "n_samples":   total pooled sample count,
         "n_sims":      number of contributing sims,
         "params":      the queried params}
    """
    matches = find_matching(archive, params, same_q=same_q)
    records = list(matches)
    if include_nearby:
        if not nearby_kw.get("k") and not nearby_kw.get("tol"):
            raise ValueError("gather_samples: include_nearby requires k= or tol=")
        seen = {r["name"] for r in records}
        for rec in find_nearby(archive, params, **nearby_kw):
            if rec["name"] not in seen:
                records.append(rec)
                seen.add(rec["name"])

    pooled, provenance = pool_catalogs(records, keys=keys)
    n_samples = sum(p["n_samples"] for p in provenance)
    return {
        "catalog": pooled,
        "provenance": provenance,
        "n_samples": n_samples,
        "n_sims": len(provenance),
        "params": params,
    }
