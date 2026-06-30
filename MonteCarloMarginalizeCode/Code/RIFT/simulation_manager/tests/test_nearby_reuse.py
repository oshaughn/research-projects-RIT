"""Tests for the engine-agnostic nearby-sim-reuse layer (nearby_reuse.py).

We build a SMALL real archive with the v2 Archive + LocalRunQueue, driven by a
self-contained toy generator that writes a multi-column ``catalog`` dict (the
same shape rapster emits). Params are dicts ``{"s":.., "r":..}`` plus a ``seed``.

The archive's own same_q/lookup_key are SEED-SENSITIVE, so different seeds are
stored as distinct sims (independent realizations the library keeps separately).
The reuse layer then pools across seed by passing an explicit seed-ignoring
``same_q`` to :func:`find_matching` (and ``param_distance`` ignores seed by
default). One sim is refined to level 2 so we can test that pooling spans
multiple independent levels of one sim.

Run with the RIFT-importable interpreter, e.g.:

    PYTHONPATH=/home/oshaughn/20260613-Me-RebootAuto/RIFT_ralph_copy/MonteCarloMarginalizeCode/Code \
      /home/oshaughn/20260613-Me-RebootAuto/popsynth_hyperpipe/.pixi/envs/default/bin/python \
      -m pytest -q .../simulation_manager/tests/test_nearby_reuse.py
"""

from __future__ import annotations

import math

import pytest

from RIFT.simulation_manager.database import (
    Archive, Manifest, LocalRequestQueue, LocalRunQueue,
)
from RIFT.simulation_manager import nearby_reuse as nr


# --- self-contained engine hooks (frozen by inspect.getsource) -------------
def toy_generator(params, sim_dir, level, prev_levels):
    """Write level_<N>.json with a multi-column catalog. Sample count scales
    with level; values depend on params + level + seed so levels differ."""
    import json, os, random
    os.makedirs(sim_dir, exist_ok=True)
    s = float(params.get("s", 0.0))
    r = float(params.get("r", 1.0))
    seed = int(params.get("seed", 0))
    n = 3 * int(level)
    rng = random.Random((round(s, 9), round(r, 9), seed, int(level)).__hash__()
                        & 0xFFFFFFFF)
    m1 = [10.0 + 5.0 * r + rng.random() for _ in range(n)]
    m2 = [5.0 + 2.0 * r + rng.random() for _ in range(n)]
    chiEff = [s + 0.01 * rng.random() for _ in range(n)]
    out = os.path.join(sim_dir, "level_{}.json".format(int(level)))
    with open(out, "w") as f:
        json.dump({"level": int(level), "params": dict(params),
                   "samples": m1,
                   "catalog": {"m1": m1, "m2": m2, "chiEff": chiEff},
                   "n_mergers": n, "mu_detect": float(n)}, f)


def toy_summarizer(sim_dir, params, levels=None):
    import json, os
    if levels:
        latest = levels[-1]
    else:
        cands = sorted(p for p in os.listdir(sim_dir)
                       if p.startswith("level_") and p.endswith(".json"))
        latest = os.path.join(sim_dir, cands[-1])
    with open(latest) as f:
        d = json.load(f)
    return {"n_mergers": d["n_mergers"], "mu_detect": d["mu_detect"],
            "level": d["level"], "output_path": latest}


def toy_same_q(a, b):
    # SEED-SENSITIVE: distinct seeds are distinct stored sims.
    def tup(p):
        return tuple(sorted((k, round(float(v), 9)) for k, v in p.items()
                            if not str(k).startswith("_")))
    return tup(a) == tup(b)


def toy_lookup_key(p):
    return repr(sorted((k, round(float(v), 9)) for k, v in p.items()
                       if not str(k).startswith("_")))


def seedless_same_q(a, b):
    # Reuse-layer predicate: pool across seed (same Lambda).
    def tup(p):
        return tuple(sorted((k, round(float(v), 9)) for k, v in p.items()
                            if not str(k).startswith("_") and k != "seed"))
    return tup(a) == tup(b)


# --- fixture: a small populated archive ------------------------------------
@pytest.fixture()
def archive(tmp_path):
    base = tmp_path / "lib"
    manifest = Manifest.new(
        name="toy", request_queue_kind="local", run_queue_kind="local",
        summarizer_entrypoint="summarizer:toy_summarizer",
        same_q_entrypoint="same_q:toy_same_q",
        lookup_key_entrypoint="lookup_key:toy_lookup_key",
    )
    arc = Archive(base_location=base, manifest=manifest,
                  generator_spec=toy_generator,
                  summarizer_spec=toy_summarizer,
                  same_q_spec=toy_same_q,
                  lookup_key_spec=toy_lookup_key)
    run_q = LocalRunQueue()
    arc.request_queue = LocalRequestQueue(run_queue=run_q)
    arc.run_queue = run_q

    # Two same-Lambda sims (s=0.0,r=1.0) with different seeds; one refined to L2.
    arc.register({"s": 0.0, "r": 1.0, "seed": 1}, target_level=2)  # 2 levels
    arc.register({"s": 0.0, "r": 1.0, "seed": 2}, target_level=1)  # 1 level
    # A nearby sim (r shifted) and a far sim (s shifted a lot).
    arc.register({"s": 0.0, "r": 1.2, "seed": 1}, target_level=1)
    arc.register({"s": 0.3, "r": 1.0, "seed": 1}, target_level=1)
    # Run them all to completion.
    arc.request_queue.submit_pending(arc)

    # An extra registered-but-not-run sim, to prove iter_completed filters it.
    arc.register({"s": 0.0, "r": 5.0, "seed": 9}, target_level=1)
    return arc


# --- tests ------------------------------------------------------------------
def test_iter_completed_only_complete_with_paths(archive):
    recs = list(nr.iter_completed(archive))
    names = {r["name"] for r in recs}
    # 4 completed; the 5th (s=5.0) is 'ready', excluded.
    assert len(recs) == 4
    statuses = {n: archive.get_status(n) for n in archive.simulations_iter_names()}
    for n in names:
        assert statuses[n] == "complete"
    for r in recs:
        assert r["catalog_paths"], "completed sim must expose catalog paths"
        for p in r["catalog_paths"]:
            assert p.is_absolute() and p.exists()
    # The L2 sim has two catalog paths; the L1 sims have one.
    by_params = {(round(r["params"]["s"], 3), round(r["params"]["r"], 3),
                  r["params"]["seed"]): r for r in recs}
    assert len(by_params[(0.0, 1.0, 1)]["catalog_paths"]) == 2
    assert len(by_params[(0.0, 1.0, 2)]["catalog_paths"]) == 1


def test_param_distance_dict_and_scalar():
    # identical -> 0, symmetric
    p = {"s": 0.1, "r": 1.0, "seed": 7}
    q = {"s": 0.1, "r": 1.0, "seed": 99}      # seed ignored
    assert nr.param_distance(p, q) == 0.0
    assert nr.param_distance(p, q) == nr.param_distance(q, p)

    # underscore keys ignored
    assert nr.param_distance({"s": 0.0, "_x": 100.0}, {"s": 0.0, "_x": -100.0}) == 0.0

    # raw vs range-normalized
    a = {"s": 0.0, "r": 0.5}
    b = {"s": 0.3, "r": 2.0}
    raw = nr.param_distance(a, b)
    assert raw == pytest.approx(math.hypot(0.3, 1.5))
    ranges = {"s": (0.0, 0.3), "r": (0.5, 2.0)}
    normd = nr.param_distance(a, b, ranges=ranges)
    # each dim becomes 1.0 after normalization
    assert normd == pytest.approx(math.sqrt(2.0))

    # scalar params
    assert nr.param_distance(0.001, 0.001) == 0.0
    assert nr.param_distance(0.001, 0.002) == pytest.approx(0.001)
    assert nr.param_distance(0.0, 1.0, ranges={"x": (0.0, 2.0)}) == pytest.approx(0.5)


def test_find_matching_pools_same_lambda_only(archive):
    matches = nr.find_matching(archive, {"s": 0.0, "r": 1.0, "seed": 123},
                               same_q=seedless_same_q)
    got = {(round(r["params"]["s"], 3), round(r["params"]["r"], 3)) for r in matches}
    # both seed=1 and seed=2 at (0.0, 1.0); neither the r=1.2 nor s=0.3 sims
    assert got == {(0.0, 1.0)}
    assert len(matches) == 2
    seeds = {r["params"]["seed"] for r in matches}
    assert seeds == {1, 2}


def test_find_nearby_orders_and_honors_k_tol(archive):
    target = {"s": 0.0, "r": 1.0, "seed": 0}
    ranges = {"s": (0.0, 0.3), "r": (0.5, 2.0)}
    nearby = nr.find_nearby(archive, target, k=10, ranges=ranges)
    dists = [r["distance"] for r in nearby]
    assert dists == sorted(dists)               # ordered by distance
    assert dists[0] == 0.0 and dists[1] == 0.0  # the two exact matches first

    # k limit
    assert len(nr.find_nearby(archive, target, k=2, ranges=ranges)) == 2

    # tol limit: r=1.2 is (1.2-1.0)/(2.0-0.5)=0.1333 away; s=0.3 is 1.0 away.
    within = nr.find_nearby(archive, target, tol=0.2, ranges=ranges)
    assert all(r["distance"] <= 0.2 for r in within)
    # the two exact + the r=1.2 nearby = 3; the s=0.3 far one excluded
    assert len(within) == 3

    with pytest.raises(ValueError):
        nr.find_nearby(archive, target)         # neither k nor tol


def test_pool_catalogs_spans_sims_and_levels(archive):
    matches = nr.find_matching(archive, {"s": 0.0, "r": 1.0, "seed": 0},
                               same_q=seedless_same_q)
    pooled, prov = nr.pool_catalogs(matches)

    # provenance: one entry per sim; counts sum to pooled length
    total = sum(p["n_samples"] for p in prov)
    assert len(pooled["m1"]) == total
    assert len(pooled["m2"]) == total
    assert len(pooled["chiEff"]) == total

    # the seed=1 sim has 2 levels (n=3 and n=6 -> 9); seed=2 has 1 level (n=3)
    by_name = {p["name"]: p for p in prov}
    counts = sorted(p["n_samples"] for p in prov)
    assert counts == [3, 9]
    assert sorted(p["n_levels"] for p in prov) == [1, 2]
    assert total == 12

    # mu_detect is reported per-level, never summed
    multi = [p for p in prov if p["n_levels"] == 2][0]
    assert multi["mu_detect_per_level"] == [3.0, 6.0]

    # keys= selects columns
    only_m1, _ = nr.pool_catalogs(matches, keys=["m1"])
    assert set(only_m1) == {"m1"}
    assert len(only_m1["m1"]) == total


def test_gather_samples_same_lambda_and_nearby(archive):
    target = {"s": 0.0, "r": 1.0, "seed": 0}
    res = nr.gather_samples(archive, target, same_q=seedless_same_q)
    assert res["n_sims"] == 2
    assert res["n_samples"] == 12
    assert len(res["catalog"]["m1"]) == 12

    ranges = {"s": (0.0, 0.3), "r": (0.5, 2.0)}
    res2 = nr.gather_samples(archive, target, same_q=seedless_same_q,
                             include_nearby=True, tol=0.2, ranges=ranges)
    # adds the r=1.2 nearby sim (1 level, n=3) -> 3 sims, 15 samples
    assert res2["n_sims"] == 3
    assert res2["n_samples"] == 15
    # the nearby addition carries a positive distance in provenance
    assert any(p.get("distance", 0.0) > 0.0 for p in res2["provenance"])
