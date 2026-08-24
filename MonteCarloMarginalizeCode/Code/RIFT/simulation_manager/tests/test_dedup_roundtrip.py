"""Dedup must survive reopening an archive.

`Archive` keeps its dedup buckets in memory, keyed by
``_safe_hashable(lookup_key(params))``, and rebuilds them from
``index.jsonl`` every time the archive is constructed. That makes the
bucket key a *persisted* value, so it has to be stable across a JSON
round-trip as well as hashable.

A tuple is not. JSON has no tuple type, so a backend whose
``lookup_key`` returns a tuple — including this tree's own
``backends/gw_pe_synthetic/lookup_key.py`` — gets a list back on
reopen. Hashing the list fails, `_safe_hashable` falls back to the
repr sentinel, and that never equals the freshly-computed tuple. The
bucket misses, `find_existing` returns None, and `register` mints a
duplicate sim for physics the archive already has. Nothing errors; the
campaign just quietly pays twice.

Run with the RIFT-importable interpreter, e.g.:

    PYTHONPATH=<...>/MonteCarloMarginalizeCode/Code \
      python -m pytest -q .../simulation_manager/tests/test_dedup_roundtrip.py
"""

from __future__ import annotations

import json

import pytest

from RIFT.simulation_manager.database import (
    Archive, Manifest, _safe_hashable,
)


# ---------------------------------------------------------------------------
# _safe_hashable, directly
# ---------------------------------------------------------------------------

def test_list_and_tuple_canonicalize_together():
    """The core invariant: a JSON-round-tripped tuple must land in the
    same bucket as the tuple it came from."""
    key = (0.05, 0.2, "1d_spherical")
    restored = json.loads(json.dumps(list(key)))
    assert _safe_hashable(restored) == _safe_hashable(key)


def test_canonical_form_is_hashable():
    assert hash(_safe_hashable([1, 2, [3, 4]])) is not None


def test_nested_lists_canonicalize():
    assert _safe_hashable([1, [2, 3]]) == _safe_hashable((1, (2, 3)))


def test_dicts_canonicalize_regardless_of_insertion_order():
    a = {"x": 1, "y": [2, 3]}
    b = {"y": [2, 3], "x": 1}
    assert _safe_hashable(a) == _safe_hashable(b)
    assert hash(_safe_hashable(a)) is not None


@pytest.mark.parametrize("key", [
    1, 1.0, 1.5, -0.0, "a",
    True, False, None,                    # str() gives True/False/None,
    float("inf"), float("-inf"),          # JSON gives true/false/null/Infinity
])
def test_dict_keys_survive_jsons_own_coercion(key):
    """JSON coerces dict keys to strings, but *not* via str():
    True -> "true", None -> "null", inf -> "Infinity". Canonicalizing
    with str() would put the fresh and rehydrated forms in different
    buckets for exactly those keys."""
    d = {key: "v"}
    restored = json.loads(json.dumps(d))
    assert _safe_hashable(restored) == _safe_hashable(d)


def test_colliding_coerced_keys_agree_with_json():
    """{True: 'a', "true": 'b'} both coerce to "true"; JSON collapses
    them last-wins. The canonical form has to collapse the same way, or
    fresh and rehydrated disagree on how many entries there are."""
    d = {True: "a", "true": "b"}
    restored = json.loads(json.dumps(d))
    assert _safe_hashable(restored) == _safe_hashable(d)
    assert len(_safe_hashable(d)) == 1


def test_dict_ordering_is_total_across_mixed_key_types():
    """Mixed key types must not raise on sort."""
    key = _safe_hashable({1: "a", "b": 2, 3.5: "c", None: "d", True: "e"})
    assert hash(key) is not None


def test_unserializable_key_falls_back_without_raising():
    """A tuple dict-key is not JSON-representable, so such a lookup_key
    could never have been persisted; canonicalization must degrade
    rather than explode."""
    key = _safe_hashable({(1, 2): "x"})
    assert hash(key) is not None


def test_scalars_pass_through():
    for v in ("a", 1, 1.5, None, True):
        assert _safe_hashable(v) == v


def test_genuinely_unhashable_still_falls_back():
    class Weird:
        __hash__ = None

    got = _safe_hashable(Weird())
    assert isinstance(got, tuple) and got[0] == "__unhashable__"


# ---------------------------------------------------------------------------
# Through a real Archive
# ---------------------------------------------------------------------------

def _generator_src():
    return (
        "import json, os\n"
        "def run(params, sim_dir, level, prev_levels):\n"
        "    p = os.path.join(sim_dir, 'level_%d.json' % level)\n"
        "    with open(p, 'w') as f:\n"
        "        json.dump({'level': level}, f)\n"
        "    return p\n"
    )


def _tuple_lookup_key_src():
    """A tuple-returning lookup_key, exactly the shape gw_pe_synthetic
    (and any natural backend) uses."""
    return (
        "def lookup_key(params):\n"
        "    return (round(float(params.get('mc', 0.0)), 3),\n"
        "            round(float(params.get('eta', 0.0)), 4))\n"
    )


def _dict_lookup_key_src():
    """A dict-returning lookup_key keyed on bools, which JSON coerces to
    "true"/"false" — not the "True"/"False" that str() produces.

    Mixed key types are fine now: `register` normalizes the key through
    JSON before storing it, so `Index._write_all`'s sort_keys=True sees
    strings. That was not always true — see the mixed-key tests at the
    bottom of this file for the regression.
    """
    return (
        "def lookup_key(params):\n"
        "    return {True: round(float(params.get('mc', 0.0)), 3),\n"
        "            False: round(float(params.get('eta', 0.0)), 4)}\n"
    )


def _same_q_src():
    return (
        "def same_q(a, b):\n"
        "    return (abs(float(a.get('mc', 0)) - float(b.get('mc', 0))) < 1e-6\n"
        "            and abs(float(a.get('eta', 0)) - float(b.get('eta', 0))) < 1e-8)\n"
    )


@pytest.fixture
def archive_factory(tmp_path):
    def _make(subdir, lookup_key_src=None):
        code = tmp_path / (subdir + "_src")
        code.mkdir(parents=True, exist_ok=True)
        (code / "generator.py").write_text(_generator_src())
        (code / "lookup_key.py").write_text(
            lookup_key_src or _tuple_lookup_key_src())
        (code / "same_q.py").write_text(_same_q_src())

        manifest = Manifest.new(
            name="dedup_roundtrip",
            request_queue_kind="local",
            run_queue_kind="local",
            same_q_entrypoint="same_q:same_q",
            lookup_key_entrypoint="lookup_key:lookup_key",
        )
        return Archive(
            base_location=tmp_path / subdir,
            manifest=manifest,
            generator_spec={"module_path": str(code / "generator.py"),
                            "entrypoint": "generator:run"},
            same_q_spec={"module_path": str(code / "same_q.py"),
                         "entrypoint": "same_q:same_q"},
            lookup_key_spec={"module_path": str(code / "lookup_key.py"),
                             "entrypoint": "lookup_key:lookup_key"},
        )
    return _make


PARAMS = {"mc": 1.2, "eta": 0.24}


def test_dedup_within_one_session(archive_factory):
    a = archive_factory("arch")
    first = a.register(dict(PARAMS), target_level=1)
    assert a.register(dict(PARAMS), target_level=1) == first
    assert len(list(a.index.all())) == 1


def test_dedup_survives_reopen(archive_factory, tmp_path):
    """The regression. Before the _safe_hashable fix this registered a
    second sim for identical physics."""
    a = archive_factory("arch")
    first = a.register(dict(PARAMS), target_level=1)

    reopened = Archive(base_location=tmp_path / "arch")
    assert reopened.register(dict(PARAMS), target_level=1) == first
    assert len(list(reopened.index.all())) == 1


def test_find_existing_matches_after_reopen(archive_factory, tmp_path):
    a = archive_factory("arch")
    name = a.register(dict(PARAMS), target_level=1)

    reopened = Archive(base_location=tmp_path / "arch")
    assert reopened.find_existing(dict(PARAMS)) == name


def test_distinct_physics_still_separates_after_reopen(archive_factory, tmp_path):
    a = archive_factory("arch")
    first = a.register(dict(PARAMS), target_level=1)

    reopened = Archive(base_location=tmp_path / "arch")
    other = reopened.register({"mc": 9.9, "eta": 0.1}, target_level=1)
    assert other != first
    assert len(list(reopened.index.all())) == 2


def test_dict_lookup_key_dedups_across_reopen(archive_factory, tmp_path):
    """The reopen regression for dict-valued lookup keys whose keys JSON
    coerces differently from str() — True -> "true", None -> "null".
    Under str()-based canonicalization this registered a duplicate."""
    a = archive_factory("dictarch", lookup_key_src=_dict_lookup_key_src())
    first = a.register(dict(PARAMS), target_level=1)

    reopened = Archive(base_location=tmp_path / "dictarch")
    assert reopened.find_existing(dict(PARAMS)) == first
    assert reopened.register(dict(PARAMS), target_level=1) == first
    assert len(list(reopened.index.all())) == 1


def test_dict_lookup_key_still_separates_distinct_physics(archive_factory,
                                                          tmp_path):
    a = archive_factory("dictarch", lookup_key_src=_dict_lookup_key_src())
    first = a.register(dict(PARAMS), target_level=1)

    reopened = Archive(base_location=tmp_path / "dictarch")
    other = reopened.register({"mc": 9.9, "eta": 0.1}, target_level=1)
    assert other != first
    assert len(list(reopened.index.all())) == 2


def test_stored_key_is_what_we_think_it_is(archive_factory, tmp_path):
    """Guard the premise: the key really is persisted as a JSON list."""
    a = archive_factory("arch")
    a.register(dict(PARAMS), target_level=1)
    row = list(a.index.all())[0]
    assert isinstance(row["lookup_key"], list)


# ---------------------------------------------------------------------------
# Dict-valued lookup_key, through a real archive
#
# _safe_hashable alone is not enough evidence. `register` stores the key in
# the index row and `Index._write_all` serializes rows with sort_keys=True,
# so a key set JSON would coerce to strings still reaches sorted() raw. A
# key like {True: 'a', 'true': 'b'} passed the unit test above while
# register() raised
#     TypeError: '<' not supported between instances of 'str' and 'bool'
# These drive register -> reopen instead.
# ---------------------------------------------------------------------------

def _dict_key_archive(tmp_path, subdir, lookup_body):
    code = tmp_path / (subdir + "_src")
    code.mkdir(parents=True, exist_ok=True)
    (code / "generator.py").write_text(_generator_src())
    (code / "lookup_key.py").write_text(lookup_body)
    (code / "same_q.py").write_text(
        "def same_q(a, b):\n"
        "    return a.get('tag') == b.get('tag')\n")
    manifest = Manifest.new(
        name="dict_key", request_queue_kind="local", run_queue_kind="local",
        same_q_entrypoint="same_q:same_q",
        lookup_key_entrypoint="lookup_key:lookup_key",
    )
    return Archive(
        base_location=tmp_path / subdir, manifest=manifest,
        generator_spec={"module_path": str(code / "generator.py"),
                        "entrypoint": "generator:run"},
        same_q_spec={"module_path": str(code / "same_q.py"),
                     "entrypoint": "same_q:same_q"},
        lookup_key_spec={"module_path": str(code / "lookup_key.py"),
                         "entrypoint": "lookup_key:lookup_key"},
    )


_MIXED_KEY_LOOKUP = (
    "def lookup_key(params):\n"
    "    return {True: 'a', 'true': 'b', 'tag': params.get('tag')}\n"
)

_NESTED_COMPOSITION_LOOKUP = (
    # SuperNu-shaped: a composition dict mixing atomic numbers and symbols.
    "def lookup_key(params):\n"
    "    return {'tag': params.get('tag'),\n"
    "            'comp': {26: 0.5, 'Fe': 0.5, None: 0.0}}\n"
)


def test_mixed_type_dict_keys_can_be_registered(tmp_path):
    """The regression: register() raised on sorted() before this fix."""
    a = _dict_key_archive(tmp_path, "arch", _MIXED_KEY_LOOKUP)
    name = a.register({"tag": "x"}, target_level=1)
    assert name


def test_mixed_type_dict_keys_dedup_across_reopen(tmp_path):
    a = _dict_key_archive(tmp_path, "arch", _MIXED_KEY_LOOKUP)
    first = a.register({"tag": "x"}, target_level=1)

    reopened = Archive(base_location=tmp_path / "arch")
    assert reopened.register({"tag": "x"}, target_level=1) == first
    assert len(list(reopened.index.all())) == 1


def test_nested_composition_keys_dedup_across_reopen(tmp_path):
    """SuperNu-shaped: atomic numbers, element symbols and None together."""
    a = _dict_key_archive(tmp_path, "arch", _NESTED_COMPOSITION_LOOKUP)
    first = a.register({"tag": "x"}, target_level=1)

    reopened = Archive(base_location=tmp_path / "arch")
    assert reopened.register({"tag": "x"}, target_level=1) == first
    assert len(list(reopened.index.all())) == 1


def test_stored_dict_key_is_json_normalized(tmp_path):
    """What lands in index.jsonl must already be the coerced form, or
    the next write hits the same sorted() failure."""
    a = _dict_key_archive(tmp_path, "arch", _MIXED_KEY_LOOKUP)
    a.register({"tag": "x"}, target_level=1)
    stored = list(a.index.all())[0]["lookup_key"]
    assert all(isinstance(k, str) for k in stored)
    assert stored["true"] == "b"          # collision collapsed last-wins


def test_distinct_physics_still_separates_with_dict_keys(tmp_path):
    a = _dict_key_archive(tmp_path, "arch", _MIXED_KEY_LOOKUP)
    first = a.register({"tag": "x"}, target_level=1)

    reopened = Archive(base_location=tmp_path / "arch")
    other = reopened.register({"tag": "y"}, target_level=1)
    assert other != first
    assert len(list(reopened.index.all())) == 2


def test_index_survives_a_second_write_after_reopen(tmp_path):
    """_write_all runs again on the next upsert; the stored key must
    still be sortable then."""
    a = _dict_key_archive(tmp_path, "arch", _MIXED_KEY_LOOKUP)
    a.register({"tag": "x"}, target_level=1)

    reopened = Archive(base_location=tmp_path / "arch")
    reopened.register({"tag": "y"}, target_level=1)      # triggers a rewrite
    again = Archive(base_location=tmp_path / "arch")
    assert len(list(again.index.all())) == 2


def test_unpersistable_lookup_key_names_the_contract(tmp_path):
    """A set cannot live in index.jsonl. Say so, rather than surfacing a
    json TypeError from the write path."""
    a = _dict_key_archive(
        tmp_path, "arch",
        "def lookup_key(params):\n    return {frozenset(['a']): 1}\n")
    with pytest.raises(TypeError, match="JSON-serializable"):
        a.register({"tag": "x"}, target_level=1)


# ---------------------------------------------------------------------------
# rebuild_index and failed registration
# ---------------------------------------------------------------------------

def test_rebuild_index_normalizes_the_key(tmp_path):
    """register -> rebuild -> reopen. rebuild_index stored the raw key,
    so a mixed-key archive that registered and reopened cleanly still
    failed here with the original sorted() TypeError."""
    a = _dict_key_archive(tmp_path, "arch", _MIXED_KEY_LOOKUP)
    first = a.register({"tag": "x"}, target_level=1)

    assert a.rebuild_index() == 1

    reopened = Archive(base_location=tmp_path / "arch")
    assert reopened.register({"tag": "x"}, target_level=1) == first
    assert len(list(reopened.index.all())) == 1


def test_rebuild_index_keeps_dedup_working_for_nested_keys(tmp_path):
    a = _dict_key_archive(tmp_path, "arch", _NESTED_COMPOSITION_LOOKUP)
    first = a.register({"tag": "x"}, target_level=1)
    a.rebuild_index()
    reopened = Archive(base_location=tmp_path / "arch")
    assert reopened.find_existing({"tag": "x"}) == first


def test_failed_registration_leaves_no_partial_sim(tmp_path):
    """An unpersistable key raised the intended error but left sims/1
    behind, with params.json and status.json, unknown to the index."""
    a = _dict_key_archive(
        tmp_path, "arch",
        "def lookup_key(params):\n    return {frozenset(['a']): 1}\n")

    with pytest.raises(TypeError, match="JSON-serializable"):
        a.register({"tag": "x"}, target_level=1)

    sims = tmp_path / "arch" / "sims"
    assert list(sims.iterdir()) == [], "left a half-registered simulation"
    assert list(a.index.all()) == []


#: Fails only for tag == "bad", so one archive can see both outcomes.
_SOMETIMES_BAD_LOOKUP = (
    "def lookup_key(params):\n"
    "    if params.get('tag') == 'bad':\n"
    "        return {frozenset(['a']): 1}\n"
    "    return 'ok|' + str(params.get('tag'))\n"
)


def test_a_failed_registration_does_not_consume_a_name(tmp_path):
    """Names are allocated by counting entries in sims/, so an orphan
    directory shifts every later name. Using two archives here would not
    test that — the orphan has to be in the SAME archive."""
    a = _dict_key_archive(tmp_path, "arch", _SOMETIMES_BAD_LOOKUP)

    with pytest.raises(TypeError, match="JSON-serializable"):
        a.register({"tag": "bad"}, target_level=1)

    assert a.register({"tag": "good"}, target_level=1) == "1"
    assert len(list(a.index.all())) == 1
