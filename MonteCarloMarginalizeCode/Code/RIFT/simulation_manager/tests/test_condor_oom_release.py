"""DualCondorRunQueue's catch-and-release policy for memory holds.

`auto_release_on_oom` bumps `request_memory` and releases a job held
because it ran out of memory, up to `oom_max_retries` times.

Three things in that sentence are site facts, not HTCondor facts, and
this module is mostly about not pretending otherwise:

  * **which hold codes mean "out of memory".** 34 is unambiguous. 26 is
    SystemPolicy -- it means whatever the site's SYSTEM_PERIODIC_HOLD
    expressions say, which on the LIGO clusters this policy came from is
    usually memory, and on an OSG access point may be an anti-thrash
    limiter that fires on a high NumJobStarts. Same code, opposite
    meaning.
  * **which sub-codes to carve out**, because every SYSTEM_PERIODIC_HOLD
    at a site produces the same hold code and only the sub-code
    separates them.
  * **what rations the retries.** NumJobStarts counts execution
    attempts; NumHolds counts holds of every kind, including transfer
    failures that increment it while NumJobStarts stays at 0. Neither is
    "the memory retry count" at every site.

So they are arguments with defaults, and the defaults are exactly what
this class emitted before they existed. Sites that differ pass their own
and keep the knowledge of which-site-is-which in whatever inventory they
already maintain, not here.

Expressions are EVALUATED against synthetic job ads rather than
string-matched, so the tests describe scheduler behaviour rather than the
text encoding it. Evaluation needs the HTCondor python bindings; the
shape tests do not and stay live without them.

Run with the RIFT-importable interpreter, e.g.:

    PYTHONPATH=<...>/MonteCarloMarginalizeCode/Code \
      python -m pytest -q .../tests/test_condor_oom_release.py
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from RIFT.simulation_manager.database import (
    Archive, DualCondorRunQueue, Manifest,
)

try:                                                 # pragma: no cover
    import classad2 as classad
except ImportError:                                  # pragma: no cover
    try:
        import classad
    except ImportError:
        classad = None

needs_classad = pytest.mark.skipif(
    classad is None, reason="HTCondor python bindings not importable")

#: What this class emitted before any of these knobs existed. The
#: default configuration must still produce it, or every existing
#: deployment silently changes policy on upgrade.
PRE_EXISTING_RELEASE = (
    "((HoldReasonCode =?= 34) || (HoldReasonCode =?= 26)) "
    "&& (NumJobStarts < 5)")


def _generator_src():
    return (
        "import json, os\n"
        "def run(params, sim_dir, level, prev_levels):\n"
        "    p = os.path.join(sim_dir, 'level_%d.json' % level)\n"
        "    with open(p, 'w') as f:\n"
        "        json.dump({'level': level}, f)\n"
        "    return p\n"
    )


@pytest.fixture
def archive(tmp_path):
    code = tmp_path / "src"
    code.mkdir()
    (code / "generator.py").write_text(_generator_src())
    manifest = Manifest.new(name="oom_release",
                            request_queue_kind="condor",
                            run_queue_kind="condor")
    return Archive(
        base_location=tmp_path / "arch", manifest=manifest,
        generator_spec={"module_path": str(code / "generator.py"),
                        "entrypoint": "generator:run"},
    )


def _build(archive, queue, level=1):
    name = archive.register({"x": 1}, target_level=level)
    return open(queue.build_worker(archive, name, level)).read()


def _command(sub_text, key):
    hits = [l for l in sub_text.splitlines()
            if l.split("=")[0].strip().lower() == key]
    assert len(hits) == 1, hits
    return hits[0].split("=", 1)[1].strip()


def _eval(expr, **job_ad):
    """Evaluate a submit expression against a synthetic job ad.

    `MY.` is stripped first: it is submit-language scope syntax that
    condor resolves against the job ad at evaluation time, but the
    python bindings evaluate a lone ad and return Undefined for it.
    Stripping keeps the rest of the real emitted text under test. Note
    this proves the expression PARSES and evaluates -- the dry-run tests
    are what show condor accepts the `MY.` form itself.
    """
    got = classad.ExprTree(expr.replace("MY.", "")).eval(
        classad.ClassAd(dict(job_ad)))
    # classad.Value is an IntEnum, so Undefined and Error are truthy
    # ints: `assert _eval(...)` would pass on either and every
    # behavioural test here would be vacuous. Refuse them at the door.
    if isinstance(got, classad.Value):
        raise AssertionError(
            "expression evaluated to {!r}, not a value: {}".format(got, expr))
    return got


# --------------------------------------------------------------------
# the defaults are the old behaviour, exactly
# --------------------------------------------------------------------

def test_the_default_release_expression_is_unchanged(archive):
    """Byte-for-byte what the class emitted before these knobs existed.

    Anything less and every deployment that never heard of this change
    gets a different policy on upgrade."""
    sub = _build(archive, DualCondorRunQueue(auto_release_on_oom=True,
                                             oom_max_retries=5))
    assert _command(sub, "periodic_release") == PRE_EXISTING_RELEASE


@needs_classad
def test_the_default_memory_bump_matches_the_old_one(archive):
    """request_memory is restructured (see the MemoryUsage guard below),
    so it is not text-identical. It must still agree with the old
    expression everywhere the old one produced a value."""
    old = ("ifthenelse((LastHoldReasonCode =!= 34 && LastHoldReasonCode =!= 26)"
           ", InitialRequestMemory, int(1.5 * NumJobStarts * MemoryUsage))")
    new = _command(_build(archive, DualCondorRunQueue(request_memory=4096)),
                   "request_memory")
    for code in (34, 26, 13, 1, 47):
        for starts in (1, 3, 9):
            ad = dict(LastHoldReasonCode=code, NumJobStarts=starts,
                      MemoryUsage=1000, InitialRequestMemory=4096)
            assert _eval(new, **ad) == _eval(old, **ad), (code, starts)


@needs_classad
def test_the_default_counts_starts_not_holds(archive):
    """Not an accident and not a leftover: on an OSG access point
    NumHolds is incremented by transfer failures that never ran the job
    at all, so it is not a better default -- only a different one."""
    release = _command(_build(archive, DualCondorRunQueue(oom_max_retries=5)),
                       "periodic_release")
    assert _eval(release, HoldReasonCode=34, NumJobStarts=1, NumHolds=99)
    assert _eval(release, HoldReasonCode=34, NumJobStarts=9,
                 NumHolds=1) is False


# --------------------------------------------------------------------
# the site supplies the policy
# --------------------------------------------------------------------

@needs_classad
def test_a_site_can_narrow_which_codes_mean_memory(archive):
    """The OSG case: code 26 there is SystemPolicy, and the site policy
    it reports is an anti-thrash limiter, not memory. Releasing it with
    a bigger memory request fights the pool's own protection."""
    q = DualCondorRunQueue(auto_release_on_oom=True, oom_hold_codes=(34,))
    release = _command(_build(archive, q), "periodic_release")
    assert _eval(release, HoldReasonCode=34, NumJobStarts=1)
    assert _eval(release, HoldReasonCode=26, NumJobStarts=1) is False


@needs_classad
def test_a_site_can_carve_out_one_subcode_of_a_shared_code(archive):
    """The finer case, and why codes alone are not enough: a site whose
    memory holds DO arrive as 26 still needs to exclude the limiter,
    which arrives as 26 too and is told apart only by its sub-code."""
    q = DualCondorRunQueue(auto_release_on_oom=True,
                           oom_hold_subcode_exclusions={26: (100, 101)})
    release = _command(_build(archive, q), "periodic_release")
    # the anti-thrash limiter: same code, excluded sub-code
    assert _eval(release, HoldReasonCode=26, HoldReasonSubCode=100,
                 NumJobStarts=1) is False
    # a real memory hold reported by the same site policy
    assert _eval(release, HoldReasonCode=26, HoldReasonSubCode=7,
                 NumJobStarts=1)
    # 34 is untouched by an exclusion keyed on 26
    assert _eval(release, HoldReasonCode=34, HoldReasonSubCode=100,
                 NumJobStarts=1)


@needs_classad
def test_a_site_can_choose_what_rations_the_retries(archive):
    q = DualCondorRunQueue(auto_release_on_oom=True, oom_max_retries=5,
                           oom_retry_counter="NumHolds")
    release = _command(_build(archive, q), "periodic_release")
    assert "NumJobStarts" not in release
    assert _eval(release, HoldReasonCode=34, NumHolds=1, NumJobStarts=99)
    assert _eval(release, HoldReasonCode=34, NumHolds=9,
                 NumJobStarts=1) is False


@needs_classad
def test_the_counter_also_scales_the_memory_bump(archive):
    q = DualCondorRunQueue(auto_release_on_oom=True, oom_memory_factor=1.5,
                           oom_retry_counter="NumHolds")
    mem = _command(_build(archive, q), "request_memory")
    assert _eval(mem, LastHoldReasonCode=34, NumHolds=2, NumJobStarts=11,
                 MemoryUsage=1000, InitialRequestMemory=4096) == 3000


@needs_classad
def test_owning_no_codes_disables_the_policy_without_breaking_the_file(
        archive):
    """An empty set must emit a well-formed expression, not an empty one
    that condor rejects at submit time."""
    q = DualCondorRunQueue(auto_release_on_oom=True, oom_hold_codes=())
    release = _command(_build(archive, q), "periodic_release")
    assert _eval(release, HoldReasonCode=34, NumJobStarts=1) is False
    mem = _command(_build(archive, q), "request_memory")
    assert _eval(mem, LastHoldReasonCode=34, NumJobStarts=3,
                 MemoryUsage=1000, InitialRequestMemory=4096) == 4096


# A test asserting "no site names appear in database.py" used to live
# here. It was theatre: none of its needles occurred in the module even
# before this change, so it passed unconditionally and on the parent
# commit too, while the module does say "LIGO clusters" and "OSG access
# point" in prose the needle list happened not to cover. A grep cannot
# express "no site-to-policy table" -- the constraint is a review one,
# and it is stated in DEFAULT_OOM_HOLD_CODES and DESIGN.md instead.


# --------------------------------------------------------------------
# the guard belongs on the attribute that can actually be undefined
# --------------------------------------------------------------------

@needs_classad
def test_an_undefined_memory_usage_does_not_wedge_the_job(archive):
    """MemoryUsage is itself an expression over ResidentSetSize, which a
    job held before it ever executed does not have. int(1.5 * n *
    undefined) is undefined, an undefined request_memory matches no
    slot, and the job sits Idle with nothing in its log -- worse than
    releasing it unchanged, which the retry cap at least bounds."""
    mem = _command(_build(archive, DualCondorRunQueue(request_memory=4096)),
                   "request_memory")
    got = _eval(mem, LastHoldReasonCode=34, NumJobStarts=3,
                InitialRequestMemory=4096)          # no MemoryUsage
    assert got == 4096


@needs_classad
def test_a_non_memory_hold_leaves_the_request_alone(archive):
    mem = _command(_build(archive, DualCondorRunQueue(request_memory=4096)),
                   "request_memory")
    assert _eval(mem, LastHoldReasonCode=13, NumJobStarts=3,
                 MemoryUsage=1000, InitialRequestMemory=4096) == 4096


# --------------------------------------------------------------------
# extra_periodic_release: additive, and scoped to the configured codes
# --------------------------------------------------------------------

SITE_TERM = "(HoldReasonCode =!= 1) && (NumJobStarts < 50)"


@needs_classad
def test_a_site_term_does_not_cost_the_memory_policy(archive):
    """The point of the hook. Routing this through extra_condor_cmds
    instead replaced periodic_release outright and the OOM arm was gone
    -- silently, and only on the sites that needed the site term."""
    q = DualCondorRunQueue(auto_release_on_oom=True, oom_max_retries=5,
                           extra_periodic_release=SITE_TERM)
    release = _command(_build(archive, q), "periodic_release")
    assert _eval(release, HoldReasonCode=34, NumJobStarts=1)
    assert _eval(release, HoldReasonCode=7, NumJobStarts=1)


@needs_classad
def test_each_term_keeps_its_own_budget(archive):
    q = DualCondorRunQueue(auto_release_on_oom=True, oom_max_retries=5,
                           extra_periodic_release=SITE_TERM)
    release = _command(_build(archive, q), "periodic_release")
    # THE case, and the one an earlier version of this test dodged by
    # asserting it with HoldReasonCode=1 -- the single code the site
    # term excludes by construction, so it could not fail however the
    # terms composed. Unscoped, the site term matches 34 happily and
    # oom_max_retries caps nothing while request_memory climbs past
    # every slot in the pool.
    assert _eval(release, HoldReasonCode=34, NumJobStarts=9) is False
    assert _eval(release, HoldReasonCode=26, NumJobStarts=9) is False
    # site budget spent, memory arm still live
    assert _eval(release, HoldReasonCode=34, NumJobStarts=1)
    # a user hold is nobody's business
    assert _eval(release, HoldReasonCode=1, NumJobStarts=1) is False


@needs_classad
def test_the_scoping_follows_the_configured_codes(archive):
    """Not a second hardcoded copy of the default set: a site that has
    told the policy it does not own code 26 gets to release 26 from its
    own term."""
    q = DualCondorRunQueue(auto_release_on_oom=True, oom_hold_codes=(34,),
                           oom_max_retries=5,
                           extra_periodic_release=SITE_TERM)
    release = _command(_build(archive, q), "periodic_release")
    assert _eval(release, HoldReasonCode=26, NumJobStarts=1)
    assert _eval(release, HoldReasonCode=34, NumJobStarts=9) is False


@needs_classad
def test_a_site_term_can_stand_alone(archive):
    q = DualCondorRunQueue(auto_release_on_oom=False,
                           extra_periodic_release=SITE_TERM)
    release = _command(_build(archive, q), "periodic_release")
    assert _eval(release, HoldReasonCode=7, NumJobStarts=1)
    assert _eval(release, HoldReasonCode=1, NumJobStarts=1) is False


def test_no_site_term_changes_nothing(archive):
    for empty in (None, "", "   "):
        got = _build(archive, DualCondorRunQueue(auto_release_on_oom=True,
                                                 oom_max_retries=5,
                                                 extra_periodic_release=empty))
        assert _command(got, "periodic_release") == PRE_EXISTING_RELEASE


# --------------------------------------------------------------------
# rejections
# --------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "(HoldReasonCode =!= 1)\ngetenv = True",
    "(HoldReasonCode =!= 1)\r\ntransfer_output_files = nothing",
])
def test_a_newline_cannot_smuggle_in_another_submit_command(archive, bad):
    """A newline ends the submit command; the remainder would be read as
    a fresh one. This value can arrive from a manifest written by
    another tool, so it is not only the author's own typing."""
    with pytest.raises(ValueError):
        DualCondorRunQueue(extra_periodic_release=bad)
    with pytest.raises(ValueError):
        DualCondorRunQueue(oom_retry_counter=bad)


@pytest.mark.parametrize("bad", [17, ["a", "b"], {"x": 1}, object()])
def test_a_non_string_expression_is_refused(bad):
    with pytest.raises(TypeError):
        DualCondorRunQueue(extra_periodic_release=bad)


@pytest.mark.parametrize("bad", ["34", 34, {"a": 1}, [34, "35"], [34, True]])
def test_hold_codes_must_be_integers(bad):
    """A bare string is iterable and would become one code per
    character; True is an int subclass and would silently become 1."""
    with pytest.raises(TypeError):
        DualCondorRunQueue(oom_hold_codes=bad)


@pytest.mark.parametrize("bad", [[26], "26", {26: 100}, {"x": [100]}])
def test_subcode_exclusions_must_be_a_code_to_subcodes_mapping(bad):
    with pytest.raises(TypeError):
        DualCondorRunQueue(oom_hold_subcode_exclusions=bad)


def test_assignment_after_construction_is_validated(archive):
    """Constructor-only checks are bypassed by plain assignment -- the
    failure mode the transfer-file guards had to be fixed for."""
    q = DualCondorRunQueue(auto_release_on_oom=True)
    with pytest.raises(ValueError):
        q.extra_periodic_release = "(HoldReasonCode =!= 1)\ngetenv = True"
    with pytest.raises(TypeError):
        q.oom_hold_codes = "34"
    with pytest.raises(TypeError):
        q.oom_hold_subcode_exclusions = [26]
    assert "getenv = True" not in _build(archive, q)


def test_periodic_release_cannot_be_replaced_through_extra_condor_cmds(archive):
    """The bug the additive hook exists to remove, closed rather than
    routed around. extra_condor_cmds is emitted last, so a
    periodic_release key there replaced the queue's line and took the
    whole OOM policy with it -- silently, condor_submit reporting
    success. Leaving that path open beside the additive one means the
    next backend author still finds it first."""
    q = DualCondorRunQueue(
        auto_release_on_oom=True,
        extra_condor_cmds={"periodic_release": "(HoldReasonCode =?= 13)"})
    with pytest.raises(ValueError, match="periodic_release"):
        _build(archive, q)


def test_the_refusal_is_case_insensitive(archive):
    """HTCondor command names are case-insensitive, so an exact-lowercase
    check would let Periodic_Release straight through."""
    q = DualCondorRunQueue(
        auto_release_on_oom=True,
        extra_condor_cmds={"Periodic_Release": "(HoldReasonCode =?= 13)"})
    with pytest.raises(ValueError):
        _build(archive, q)


def test_disabling_the_policy_leaves_a_plain_memory_request(archive):
    sub = _build(archive, DualCondorRunQueue(auto_release_on_oom=False,
                                             request_memory=4096))
    assert _command(sub, "request_memory") == "4096M"
    assert "periodic_release" not in sub


# --------------------------------------------------------------------
# the manifest carries the policy
# --------------------------------------------------------------------

def test_the_policy_survives_the_manifest(tmp_path):
    """A relocated archive must submit under the policy it was built
    with. Note the sub-code map: JSON has no integer keys, so it comes
    back as {"26": [100]} and an implementation that does not coerce
    turns a configured exclusion into a silently ignored one."""
    from RIFT.simulation_manager.database import make_queues_from_manifest

    code = tmp_path / "src"
    code.mkdir()
    (code / "generator.py").write_text(_generator_src())
    manifest = Manifest.new(
        name="oom_manifest", request_queue_kind="condor",
        run_queue_kind="condor",
        run_queue_extra={"oom_hold_codes": [34, 26],
                         "oom_hold_subcode_exclusions": {"26": [100]},
                         "oom_retry_counter": "NumHolds",
                         "extra_periodic_release": SITE_TERM})
    Archive(base_location=tmp_path / "arch", manifest=manifest,
            generator_spec={"module_path": str(code / "generator.py"),
                            "entrypoint": "generator:run"})
    reopened = Archive(base_location=tmp_path / "arch")
    _, run_queue = make_queues_from_manifest(reopened)
    assert run_queue.oom_hold_codes == (34, 26)
    assert dict(run_queue.oom_hold_subcode_exclusions) == {26: (100,)}
    assert run_queue.oom_retry_counter == "NumHolds"
    assert run_queue.extra_periodic_release == SITE_TERM


# --------------------------------------------------------------------
# the scheduler's own opinion
# --------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {},
    {"oom_hold_codes": (34,)},
    {"oom_hold_subcode_exclusions": {26: (100, 101)}},
    {"oom_retry_counter": "NumHolds"},
    {"oom_hold_codes": ()},
    {"extra_periodic_release": SITE_TERM},
    {"oom_hold_codes": (34,), "extra_periodic_release": SITE_TERM,
     "oom_retry_counter": "NumHolds"},
])
def test_condor_accepts_every_shape_of_policy(archive, tmp_path, kwargs):
    """No expression evaluator substitutes for condor parsing it, and
    each knob changes the emitted text in a different place. -dry-run
    contacts no schedd and queues nothing."""
    condor_submit = shutil.which("condor_submit")
    if condor_submit is None:
        pytest.skip("condor_submit not on PATH")
    sub = _build(archive, DualCondorRunQueue(auto_release_on_oom=True,
                                             **kwargs))
    path = tmp_path / "oom.sub"
    path.write_text(sub)
    out = tmp_path / "oom.dry"
    proc = subprocess.run([condor_submit, "-dry-run", str(out), str(path)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    materialised = [l for l in out.read_text().splitlines()
                    if l.split("=")[0].strip().lower()
                    in ("requestmemory", "periodicrelease")]
    assert len(materialised) == 2, materialised


# --------------------------------------------------------------------
# the counter is spliced into arithmetic, not only into a comparison
# --------------------------------------------------------------------

@needs_classad
def test_a_compound_counter_is_not_mangled_by_precedence(archive):
    """`oom_retry_counter` is validated as an EXPRESSION, so a compound
    one is advertised input. Unparenthesised in the bump it reassociates:
    `int(1.5 * NumHolds - NumJobStarts * MemoryUsage)` is
    (1.5*NumHolds) - (NumJobStarts*MemoryUsage), which for a job at
    NumHolds=6, NumJobStarts=2, MemoryUsage=1000 asks for -1991 MB.
    condor_submit accepts that, and a negative request matches no slot --
    the wedged-Idle failure this policy's MemoryUsage guard exists to
    prevent, reintroduced through a different door."""
    q = DualCondorRunQueue(auto_release_on_oom=True, oom_memory_factor=1.5,
                           oom_retry_counter="NumHolds - NumJobStarts")
    mem = _command(_build(archive, q), "request_memory")
    assert _eval(mem, LastHoldReasonCode=34, NumHolds=6, NumJobStarts=2,
                 MemoryUsage=1000, InitialRequestMemory=4096) == 6000


@needs_classad
def test_the_comparison_form_is_unaffected(archive):
    """`<` has lower precedence than any arithmetic, so the release arm
    was already safe -- which is why the fix is confined to the bump and
    the default release text stays byte-identical."""
    q = DualCondorRunQueue(auto_release_on_oom=True, oom_max_retries=5,
                           oom_retry_counter="NumHolds - NumJobStarts")
    release = _command(_build(archive, q), "periodic_release")
    assert _eval(release, HoldReasonCode=34, NumHolds=6, NumJobStarts=2)
    assert _eval(release, HoldReasonCode=34, NumHolds=9,
                 NumJobStarts=2) is False


# --------------------------------------------------------------------
# configuration that would quietly do nothing
# --------------------------------------------------------------------

def test_an_exclusion_on_an_unowned_code_is_refused(archive):
    """Silently ignoring it means a typo reads as configured: the site
    believes it has carved out its anti-thrash sub-code and has not."""
    with pytest.raises(ValueError, match="99"):
        DualCondorRunQueue(oom_hold_codes=(34,),
                           oom_hold_subcode_exclusions={99: (1,)})
    q = DualCondorRunQueue(oom_hold_codes=(34, 26),
                           oom_hold_subcode_exclusions={26: (100,)})
    with pytest.raises(ValueError):
        q.oom_hold_codes = (34,)        # orphans the exclusion after the fact


def test_a_refused_assignment_leaves_the_previous_policy_in_place(archive):
    """A setter that stores first and validates after leaves the queue
    configured with the value it just rejected: the caller sees the
    ValueError, reads it as "nothing changed", and submits under (34,)
    anyway. Both directions of the pair have to hold."""
    q = DualCondorRunQueue(auto_release_on_oom=True, oom_hold_codes=(34, 26),
                           oom_hold_subcode_exclusions={26: (100,)})
    with pytest.raises(ValueError):
        q.oom_hold_codes = (34,)
    assert q.oom_hold_codes == (34, 26)
    with pytest.raises(ValueError):
        q.oom_hold_subcode_exclusions = {99: (1,)}
    assert dict(q.oom_hold_subcode_exclusions) == {26: (100,)}
    # ...and what it submits is the surviving policy, not the rejected
    # one: the attribute reading right is no use if the emitted text
    # disagrees with it.
    intact = DualCondorRunQueue(auto_release_on_oom=True,
                                oom_hold_codes=(34, 26),
                                oom_hold_subcode_exclusions={26: (100,)})
    assert _command(_build(archive, q), "periodic_release") == \
        _command(_build(archive, intact), "periodic_release")


def test_none_means_the_default_not_the_empty_set(archive):
    """As it does in the constructor and for oom_retry_counter. Reading
    it as "own no codes" would let an assignment disable the memory
    policy outright; pass () to ask for that."""
    q = DualCondorRunQueue(auto_release_on_oom=True, oom_max_retries=5)
    q.oom_hold_codes = None
    assert _command(_build(archive, q), "periodic_release") == \
        PRE_EXISTING_RELEASE


def test_the_exclusion_view_cannot_be_mutated_in_place(archive):
    """A copy would make this a silent no-op, the same trap the transfer
    properties avoid by handing back tuples."""
    q = DualCondorRunQueue(oom_hold_codes=(34, 26))
    with pytest.raises(TypeError):
        q.oom_hold_subcode_exclusions[26] = (100,)


# --------------------------------------------------------------------
# the other half of the memory policy
# --------------------------------------------------------------------

def test_request_memory_cannot_be_replaced_through_extra_condor_cmds(archive):
    """Protecting periodic_release alone did not close the path.
    Replacing request_memory leaves the release arm intact, so the job is
    released the full oom_max_retries times at a fixed size and OOMs
    every time -- it spends the whole budget achieving nothing."""
    q = DualCondorRunQueue(auto_release_on_oom=True,
                           extra_condor_cmds={"request_memory": "8G"})
    with pytest.raises(ValueError, match="request_memory"):
        _build(archive, q)


@pytest.mark.parametrize("key,expected", [
    ("periodic_release", "extra_periodic_release"),
    ("request_memory", "set_resources"),
    ("transfer_input_files", "extra_transfer_input_files"),
])
def test_the_refusal_names_the_thing_to_use_instead(archive, key, expected):
    """A guard that refuses without a remedy just moves the dead end.
    The periodic_release message used to point at the transfer options."""
    q = DualCondorRunQueue(auto_release_on_oom=True,
                           extra_condor_cmds={key: "whatever"})
    with pytest.raises(ValueError, match=expected):
        _build(archive, q)
