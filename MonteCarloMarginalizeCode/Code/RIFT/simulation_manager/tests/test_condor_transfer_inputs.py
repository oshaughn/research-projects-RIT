"""DualCondorRunQueue.extra_transfer_input_files.

A backend often needs every job to stage a bulk input the archive knows
nothing about — an opacity table, a reference catalogue — and on OSG
those belong in `transfer_input_files` as `osdf://` URLs so Condor
fetches them through a cache instead of the submit host's spool.

Before this hook the only way in was `extra_condor_cmds`, which is
appended verbatim and so *replaces* the `transfer_input_files` line the
queue already wrote. That silently strips the frozen `code/` directory
and the sim's params, leaving the worker with nothing to run. Hence an
append-only knob.

Run with the RIFT-importable interpreter, e.g.:

    PYTHONPATH=<...>/MonteCarloMarginalizeCode/Code \
      python -m pytest -q .../tests/test_condor_transfer_inputs.py
"""

from __future__ import annotations

import pytest

from RIFT.simulation_manager.database import (
    Archive, DualCondorRunQueue, Manifest,
)

BULK = [
    "osdf:///ospool/ap41/data/u/r3/opacities-v2.h5",
    "osdf:///ospool/ap41/data/u/r3/compositions-v1.tar.gz",
]


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
    manifest = Manifest.new(name="transfer_inputs",
                            request_queue_kind="condor",
                            run_queue_kind="condor")
    return Archive(
        base_location=tmp_path / "arch", manifest=manifest,
        generator_spec={"module_path": str(code / "generator.py"),
                        "entrypoint": "generator:run"},
    )


def _transfer_line(sub_text):
    lines = [l for l in sub_text.splitlines()
             if l.strip().startswith("transfer_input_files")]
    assert len(lines) == 1, lines
    return lines[0]


def _build(archive, queue, level=1):
    name = archive.register({"x": 1}, target_level=level)
    return name, open(queue.build_worker(archive, name, level)).read()


def test_extras_are_appended(archive):
    q = DualCondorRunQueue(extra_transfer_input_files=BULK)
    _, sub = _build(archive, q)
    line = _transfer_line(sub)
    for url in BULK:
        assert url in line


def test_archive_entries_are_preserved(archive, tmp_path):
    """The whole point: extras must be present AND must not displace the
    frozen code.

    Asserting only the archive entries made this pass unmodified against
    the base revision — the old constructor swallowed the unknown kwarg
    into **submit_kwargs rather than raising, so the test could not
    detect the failure mode it names."""
    q = DualCondorRunQueue(extra_transfer_input_files=BULK)
    name, sub = _build(archive, q)
    line = _transfer_line(sub)
    assert str(tmp_path / "arch" / "code") in line
    assert "params.json" in line
    for url in BULK:
        assert url in line


def test_default_is_unchanged(archive, tmp_path):
    """No extras configured means the submit description is exactly what
    it was before this knob existed."""
    q = DualCondorRunQueue()
    _, sub = _build(archive, q)
    line = _transfer_line(sub)
    assert "osdf://" not in line
    assert str(tmp_path / "arch" / "code") in line


def test_extras_appear_once_per_job(archive):
    q = DualCondorRunQueue(extra_transfer_input_files=BULK)
    _, sub = _build(archive, q)
    line = _transfer_line(sub)
    for url in BULK:
        assert line.count(url) == 1


def test_extras_survive_repeated_builds(archive):
    """build_worker is documented idempotent; the extras list must not
    accumulate across calls."""
    q = DualCondorRunQueue(extra_transfer_input_files=BULK)
    name = archive.register({"x": 1}, target_level=1)
    q.build_worker(archive, name, 1)
    sub = open(q.build_worker(archive, name, 1)).read()
    assert _transfer_line(sub).count(BULK[0]) == 1


def test_chained_levels_still_declare_prior_outputs(archive):
    """Extras must not disturb the prior-level entries, which are
    declared regardless of disk presence because the DAG guarantees they
    exist by the time level N runs."""
    q = DualCondorRunQueue(extra_transfer_input_files=BULK)
    name = archive.register({"x": 1}, target_level=3)
    sub = open(q.build_worker(archive, name, 3)).read()
    line = _transfer_line(sub)
    assert "level_1.json" in line
    assert "level_2.json" in line
    assert BULK[0] in line


def test_accepts_path_like_entries(archive, tmp_path):
    local = tmp_path / "aux.dat"
    local.write_text("x")
    q = DualCondorRunQueue(extra_transfer_input_files=[local])
    _, sub = _build(archive, q)
    assert str(local) in _transfer_line(sub)


def test_reaches_the_queue_through_the_manifest(tmp_path):
    """make_queues_from_manifest passes run_queue.extra as kwargs, so a
    reopened archive must keep its bulk inputs."""
    from RIFT.simulation_manager.database import make_queues_from_manifest

    code = tmp_path / "src"
    code.mkdir()
    (code / "generator.py").write_text(_generator_src())
    manifest = Manifest.new(
        name="transfer_inputs", request_queue_kind="condor",
        run_queue_kind="condor",
        run_queue_extra={"extra_transfer_input_files": BULK},
    )
    a = Archive(base_location=tmp_path / "arch", manifest=manifest,
                generator_spec={"module_path": str(code / "generator.py"),
                                "entrypoint": "generator:run"})
    reopened = Archive(base_location=tmp_path / "arch")
    _, run_queue = make_queues_from_manifest(reopened)
    assert run_queue.extra_transfer_input_files == BULK


# ---------------------------------------------------------------------------
# Rejections: each of these is something condor_submit accepts with exit 0
# and then gets wrong on a remote worker.
# ---------------------------------------------------------------------------

def test_bare_string_is_rejected():
    """A str is a Sequence[str], so it would iterate as one transfer
    request per character."""
    with pytest.raises(TypeError, match="not a bare string"):
        DualCondorRunQueue(extra_transfer_input_files="osdf:///a/b.h5")


@pytest.mark.parametrize("bad", [
    "/data/tab,v2.h5",                      # comma separates entries
    "/data/a.h5\nrequest_memory = 999999",  # newline injects a submit command
    "   ",                                  # empty
])
def test_corrupting_entries_are_rejected(bad):
    with pytest.raises(ValueError):
        DualCondorRunQueue(extra_transfer_input_files=[bad])


@pytest.mark.parametrize("colliding", [
    "osdf:///bulk/params.json", "osdf:///bulk/code", "osdf:///bulk/level_1.json",
])
def test_basename_collisions_are_rejected(colliding):
    """Condor flattens basenames into the sandbox, so these would
    overwrite the archive's own staged files on the worker."""
    with pytest.raises(ValueError, match="collides"):
        DualCondorRunQueue(extra_transfer_input_files=[colliding])


def test_extras_with_subdag_factory_are_rejected():
    """submit() dispatches to the sub-DAG and never calls build_worker,
    so the extras would be stored, persisted to the manifest, and reach
    nothing at all."""
    with pytest.raises(ValueError, match="subdag_factory"):
        DualCondorRunQueue(extra_transfer_input_files=BULK,
                           subdag_factory=lambda a, s, l: "x.dag")


@pytest.mark.parametrize("key", [
    "transfer_input_files", "transfer_output_files", "transfer_output_remaps",
])
def test_extra_condor_cmds_cannot_replace_the_transfer_lines(archive, key):
    """extra_condor_cmds is emitted last, so setting these would replace
    the archive's own line and strip what the worker needs."""
    q = DualCondorRunQueue(extra_condor_cmds={key: "/other/thing"})
    name = archive.register({"x": 1}, target_level=1)
    with pytest.raises(ValueError, match=key):
        q.build_worker(archive, name, 1)


# ---------------------------------------------------------------------------
# Output side
# ---------------------------------------------------------------------------

def test_extra_outputs_are_returned_and_remapped(archive, tmp_path):
    """transfer_output_files is explicit, so anything not named here is
    destroyed with the sandbox — a backend whose science IS output files
    completes having discarded its own results."""
    q = DualCondorRunQueue(extra_transfer_output_files=["level_{level}"])
    name = archive.register({"x": 1}, target_level=1)
    sub = open(q.build_worker(archive, name, 1)).read()
    out = next(l for l in sub.splitlines()
               if l.strip().startswith("transfer_output_files"))
    remap = next(l for l in sub.splitlines()
                 if l.strip().startswith("transfer_output_remaps"))
    assert "level_1.json" in out and "level_1" in out
    assert str(tmp_path / "arch" / "sims" / name / "level_1") in remap
    assert remap.count(";") == 1          # marker remap plus ours


def test_output_placeholders_track_the_level(archive):
    """Asserting `"level_2" in out` was vacuous — the marker is already
    named level_2.json, so it passed with the feature unimplemented.
    Check the actual entry list instead."""
    q = DualCondorRunQueue(extra_transfer_output_files=["work_{level}"])
    name = archive.register({"x": 1}, target_level=2)
    sub = open(q.build_worker(archive, name, 2)).read()
    out = next(l for l in sub.splitlines()
               if l.strip().startswith("transfer_output_files"))
    entries = [e.strip() for e in out.split("=", 1)[1].split(",")]
    assert entries == ["level_2.json", "work_2"]


def test_output_default_is_unchanged(archive):
    q = DualCondorRunQueue()
    name = archive.register({"x": 1}, target_level=1)
    sub = open(q.build_worker(archive, name, 1)).read()
    out = next(l for l in sub.splitlines()
               if l.strip().startswith("transfer_output_files"))
    remap = next(l for l in sub.splitlines()
                 if l.strip().startswith("transfer_output_remaps"))
    assert out.split("=", 1)[1].strip() == "level_1.json"
    assert ";" not in remap


# ---------------------------------------------------------------------------
# Guards must survive attribute assignment, not just __init__
#
# All of these were reachable after the first round of "fixes": the
# attributes are public, and configuring a queue by assigning to them is
# the natural thing to do, which walked past every constructor check.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    ["/data/tab,v2.h5"],
    ["/data/a.h5\nrequest_memory = 999999"],
    "osdf:///a/b.h5",
    ["osdf:///bulk/params.json"],
])
def test_input_assignment_after_construction_is_validated(bad):
    q = DualCondorRunQueue()
    with pytest.raises((ValueError, TypeError)):
        q.extra_transfer_input_files = bad


@pytest.mark.parametrize("bad", [
    ["evil;name=/etc/hosts"],
    ["a=b"],
    ["params.json"],
    ["out,put"],
])
def test_output_assignment_after_construction_is_validated(bad):
    q = DualCondorRunQueue()
    with pytest.raises((ValueError, TypeError)):
        q.extra_transfer_output_files = bad


def test_subdag_factory_assigned_late_still_refuses_extras(archive):
    """The P0: setting subdag_factory after construction reached the
    sub-DAG path with the extras stored and silently ignored."""
    q = DualCondorRunQueue(extra_transfer_input_files=BULK,
                           submit_mode="embed")
    q.subdag_factory = lambda a, s, l: "/some/external.dag"
    name = archive.register({"x": 1}, target_level=1)
    with pytest.raises(ValueError, match="sub-DAG"):
        q.submit(archive, [name])


def test_extras_assigned_late_still_refuse_a_subdag(archive):
    """...and the same in the other order."""
    q = DualCondorRunQueue(submit_mode="embed",
                           subdag_factory=lambda a, s, l: "/some/external.dag")
    q.extra_transfer_input_files = BULK
    name = archive.register({"x": 1}, target_level=1)
    with pytest.raises(ValueError, match="sub-DAG"):
        q.submit(archive, [name])


# ---------------------------------------------------------------------------
# Output-side hazards the shared validator did not originally cover
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["evil;x", "a=b"])
def test_remap_delimiters_are_rejected(bad):
    """transfer_output_remaps is a ';'-separated list of name=path pairs,
    so either character makes the remap unparseable on the execute side."""
    with pytest.raises(ValueError):
        DualCondorRunQueue(extra_transfer_output_files=[bad])


@pytest.mark.parametrize("bad", ["params.json", "code", "level_{level}.json"])
def test_output_basename_collisions_are_rejected(bad):
    """An output entry is remapped back under sims/<name>/, so a returned
    params.json overwrites the sim's recorded inputs in the archive —
    corrupting state every later level reads."""
    with pytest.raises(ValueError, match="collides"):
        q = DualCondorRunQueue(extra_transfer_output_files=[bad])
        q.build_worker.__self__            # constructed: force the check


def test_expanded_names_are_revalidated(archive):
    """Validation at assignment sees the template; expansion can still
    introduce a space or a path separator."""
    name = archive.register({"x": 1}, target_level=1)
    for template in ("my file_{level}", "sub/dir_{level}"):
        q = DualCondorRunQueue(extra_transfer_output_files=[template])
        with pytest.raises(ValueError):
            q.build_worker(archive, name, 1)


def test_unknown_placeholder_names_the_contract(archive):
    q = DualCondorRunQueue(extra_transfer_output_files=["stuff_{foo}"])
    name = archive.register({"x": 1}, target_level=1)
    with pytest.raises(ValueError, match="placeholder"):
        q.build_worker(archive, name, 1)
