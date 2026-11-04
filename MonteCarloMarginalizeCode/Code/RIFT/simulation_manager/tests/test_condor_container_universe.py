"""DualCondorRunQueue's container universe and output-transfer timing.

Two things a backend previously had to reach through `extra_condor_cmds`,
which is emitted last and therefore *replaces* the queue's own lines
rather than extending them.

`universe = container` + `container_image` is how OSG documents running
in a container today. The queue only knew the legacy `+SingularityImage`
form, so every backend targeting OSG hand-rolled the modern one — and in
doing so silently replaced the queue's `universe` line, with the winner
decided by which line condor read last.

`when_to_transfer_output` was hardcoded `ON_EXIT`. On a preemptable pool
that discards the sandbox when a job is evicted, throwing away whatever
the job had already written. `ON_EXIT_OR_EVICT` keeps it, and a backend
whose science *is* output files needs that.

Run with the RIFT-importable interpreter, e.g.:

    PYTHONPATH=<...>/MonteCarloMarginalizeCode/Code \
      python -m pytest -q .../tests/test_condor_container_universe.py
"""

from __future__ import annotations

import inspect
import shutil
import subprocess

import pytest

from RIFT.simulation_manager.database import (
    Archive, DualCondorRunQueue, Manifest, WHEN_TO_TRANSFER_OUTPUT,
)

IMAGE = "osdf:///ospool/ap41/data/example/supernu-v2.sif"


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
    manifest = Manifest.new(name="container", request_queue_kind="condor",
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


# --------------------------------------------------------------------
# container universe
# --------------------------------------------------------------------

def test_no_image_is_a_vanilla_job(archive):
    """The default must not move: every existing deployment submits
    vanilla and none of them asked for this."""
    sub = _build(archive, DualCondorRunQueue())
    assert _command(sub, "universe") == "vanilla"
    assert "container_image" not in sub


def test_an_image_selects_the_container_universe(archive):
    """One argument, not two. A backend that has to remember to set the
    universe as well will eventually forget, and a container_image under
    a vanilla universe is silently ignored by condor."""
    sub = _build(archive, DualCondorRunQueue(container_image=IMAGE))
    assert _command(sub, "universe") == "container"
    assert _command(sub, "container_image") == IMAGE


def test_the_image_is_not_resolved_or_fetched(archive):
    """An osdf:// or docker:// reference is not readable from the submit
    host, and demanding that it be would refuse the ordinary OSG case."""
    for ref in ("docker://library/python:3.11",
                "osdf:///ospool/ap41/data/nobody/nothing.sif",
                "/no/such/path/local.sif"):
        sub = _build(archive, DualCondorRunQueue(container_image=ref))
        assert _command(sub, "container_image") == ref


def test_the_two_container_mechanisms_are_refused_together(archive):
    """Emitting both leaves the outcome to whichever the site honours."""
    q = DualCondorRunQueue(container_image=IMAGE, use_singularity=True,
                           singularity_image="/cvmfs/x.sif")
    with pytest.raises(ValueError, match="container_image"):
        _build(archive, q)


def test_the_legacy_form_still_works_alone(archive):
    """Not a deprecation. Sites that only honour +SingularityImage exist."""
    sub = _build(archive, DualCondorRunQueue(
        use_singularity=True, singularity_image="/cvmfs/x.sif"))
    assert _command(sub, "universe") == "vanilla"
    assert "SingularityImage" in sub


@pytest.mark.parametrize("bad", [
    "osdf:///x.sif\ngetenv = True",
    "osdf:///x.sif\r\ntransfer_output_files = nothing",
])
def test_a_newline_in_the_image_cannot_add_a_submit_command(bad):
    with pytest.raises(ValueError):
        DualCondorRunQueue(container_image=bad)


@pytest.mark.parametrize("bad", [17, ["a"], {"x": 1}, object()])
def test_a_non_string_image_is_refused(bad):
    with pytest.raises(TypeError):
        DualCondorRunQueue(container_image=bad)


# --------------------------------------------------------------------
# when_to_transfer_output
# --------------------------------------------------------------------

def test_the_default_transfer_timing_is_unchanged(archive):
    assert _command(_build(archive, DualCondorRunQueue()),
                    "when_to_transfer_output") == "ON_EXIT"


@pytest.mark.parametrize("value", [v for v in WHEN_TO_TRANSFER_OUTPUT
                                   if v != "ON_EXIT_OR_EVICT"])
def test_every_usable_value_is_emitted(archive, value):
    sub = _build(archive, DualCondorRunQueue(when_to_transfer_output=value))
    assert _command(sub, "when_to_transfer_output") == value


def test_the_vocabulary_is_condors(archive):
    """Guards the set itself. It drifted once already -- NEVER was in it,
    and no test noticed because every test read the emitted text rather
    than asking condor."""
    assert set(WHEN_TO_TRANSFER_OUTPUT) == {
        "ON_EXIT", "ON_EXIT_OR_EVICT", "ON_SUCCESS"}


def test_it_is_normalised_not_passed_through(archive):
    sub = _build(archive,
                 DualCondorRunQueue(when_to_transfer_output=" on_success "))
    assert _command(sub, "when_to_transfer_output") == "ON_SUCCESS"


def test_never_is_refused_because_condor_discards_it(archive):
    """NEVER is not in the JDL. Measured against condor 25.13.1 it
    submits rc=0 and materialises as ON_EXIT -- so a caller setting it to
    suppress transfer gets transfer, silently. An illegal value like
    BANANA condor rejects loudly on its own, so accepting NEVER was the
    only thing this validator actually changed, in the wrong direction."""
    with pytest.raises(ValueError, match="when_to_transfer_output"):
        DualCondorRunQueue(when_to_transfer_output="NEVER")


def test_on_exit_or_evict_is_refused_by_this_queue(archive):
    """HTCondor holds a job whose listed output is missing at eviction,
    and this queue always lists level_<N>.json, which exists only after
    the job succeeds. Every mid-run eviction would hold rather than
    reschedule -- worse than the ON_EXIT it is reached for. Constructing
    is allowed; building the submit description is where it raises, so
    the message lands with the job that would have been broken."""
    q = DualCondorRunQueue(when_to_transfer_output="ON_EXIT_OR_EVICT")
    with pytest.raises(ValueError, match="checkpoint_exit_code"):
        _build(archive, q)


@pytest.mark.parametrize("bad", ["ON_EVICT", "always", "", "ON_EXIT_OR_EVIC"])
def test_a_value_condor_does_not_know_is_refused(bad):
    """Caught here rather than by the schedd: by the time condor_submit
    refuses it the archive has already recorded the sim as dispatched, so
    it presents as a stuck simulation, not a configuration error."""
    with pytest.raises(ValueError, match="when_to_transfer_output"):
        DualCondorRunQueue(when_to_transfer_output=bad)


@pytest.mark.parametrize("bad", [17, ["ON_EXIT"], object()])
def test_a_non_string_timing_is_refused(bad):
    with pytest.raises(TypeError):
        DualCondorRunQueue(when_to_transfer_output=bad)


# --------------------------------------------------------------------
# the old way in is closed
# --------------------------------------------------------------------

@pytest.mark.parametrize("key,expected", [
    ("container_image", "container_image"),
    ("when_to_transfer_output", "when_to_transfer_output"),
])
def test_the_old_route_is_refused_and_names_the_argument(archive, key,
                                                         expected):
    """These were reachable only through extra_condor_cmds before there
    were arguments for them. Leaving that open beside the argument means
    the next backend author finds it first, and a universe set twice is
    decided by whichever line condor reads last."""
    q = DualCondorRunQueue(extra_condor_cmds={key: "whatever"})
    with pytest.raises(ValueError, match=expected):
        _build(archive, q)


def test_assignment_after_construction_is_validated(archive):
    q = DualCondorRunQueue()
    with pytest.raises(ValueError):
        q.container_image = "osdf:///x.sif\ngetenv = True"
    with pytest.raises(ValueError):
        q.when_to_transfer_output = "ON_EVICT"
    assert "getenv = True" not in _build(archive, q)


def test_the_policy_survives_the_manifest(tmp_path):
    from RIFT.simulation_manager.database import make_queues_from_manifest

    code = tmp_path / "src"
    code.mkdir()
    (code / "generator.py").write_text(_generator_src())
    manifest = Manifest.new(
        name="container_manifest", request_queue_kind="condor",
        run_queue_kind="condor",
        run_queue_extra={"container_image": IMAGE,
                         "when_to_transfer_output": "ON_SUCCESS"})
    Archive(base_location=tmp_path / "arch", manifest=manifest,
            generator_spec={"module_path": str(code / "generator.py"),
                            "entrypoint": "generator:run"})
    reopened = Archive(base_location=tmp_path / "arch")
    _, run_queue = make_queues_from_manifest(reopened)
    assert run_queue.container_image == IMAGE
    sub = _build(reopened, run_queue)
    assert _command(sub, "universe") == "container"
    assert _command(sub, "when_to_transfer_output") == "ON_SUCCESS"


@pytest.mark.parametrize("kwargs", [
    {},
    {"container_image": IMAGE},
    {"when_to_transfer_output": "ON_SUCCESS"},
    {"container_image": IMAGE, "when_to_transfer_output": "ON_SUCCESS"},
    {"use_singularity": True, "singularity_image": "/cvmfs/x.sif"},
])
def test_condor_accepts_every_shape(archive, tmp_path, kwargs):
    """No hand-reading of the submit file substitutes for condor parsing
    it. -dry-run contacts no schedd and queues nothing."""
    condor_submit = shutil.which("condor_submit")
    if condor_submit is None:
        pytest.skip("condor_submit not on PATH")
    path = tmp_path / "c.sub"
    path.write_text(_build(archive, DualCondorRunQueue(**kwargs)))
    out = tmp_path / "c.dry"
    proc = subprocess.run([condor_submit, "-dry-run:oauth=1", str(out), str(path)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    # rc=0 only proves the file parses -- condor accepts a garbage image
    # string with rc=0, and silently rewrites values it dislikes. Read
    # the materialised ad back and compare it to what was asked for.
    ad = {}
    for line in out.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            ad[k.strip().lower()] = v.strip().strip('"')
    assert ad.get("whentotransferoutput") == kwargs.get(
        "when_to_transfer_output", "ON_EXIT")
    if kwargs.get("container_image"):
        assert ad.get("wantcontainer") == "true", ad.get("wantcontainer")
        assert IMAGE.rsplit("/", 1)[-1] in out.read_text()


# --------------------------------------------------------------------
# the sub-DAG path bypasses build_worker entirely
# --------------------------------------------------------------------

def test_a_container_and_a_subdag_are_refused_together():
    """submit() bypasses build_worker when subdag_factory is set, so the
    image would never be emitted and the sub-DAG's nodes would run the
    science OUTSIDE the container -- no error, nothing in the submit
    files to show it. The transfer extras are guarded on this path for
    the same reason; the image was not."""
    with pytest.raises(ValueError, match="container_image"):
        DualCondorRunQueue(container_image=IMAGE,
                           subdag_factory=lambda a, s, l: "/tmp/x")


def test_assigning_either_one_late_is_still_refused(archive):
    """Both are plain attributes, so a constructor-only check is
    bypassed by assignment -- which is why submit() re-checks."""
    q = DualCondorRunQueue(container_image=IMAGE)
    q.subdag_factory = lambda a, s, l: "/tmp/x"
    name = archive.register({"x": 1}, target_level=1)
    with pytest.raises(ValueError, match="container_image"):
        q.submit(archive, [name])


@pytest.mark.parametrize("value", [v for v in WHEN_TO_TRANSFER_OUTPUT
                                   if v != "ON_EXIT"])
def test_a_transfer_timing_and_a_subdag_are_refused_together(value):
    """The timing is emitted by build_worker too, so the sub-DAG's nodes
    would transfer at the default ON_EXIT and the request would vanish.
    ON_EXIT_OR_EVICT is worse than dropped: build_worker is also where it
    is REFUSED, so this path routed the one value the queue rejects
    around its own rejection."""
    with pytest.raises(ValueError, match="when_to_transfer_output"):
        DualCondorRunQueue(when_to_transfer_output=value,
                           subdag_factory=lambda a, s, l: "/tmp/x")


@pytest.mark.parametrize("value", [v for v in WHEN_TO_TRANSFER_OUTPUT
                                   if v != "ON_EXIT"])
def test_assigning_the_timing_late_is_still_refused(archive, value):
    """Same plain-attribute hole as the image, in both orders."""
    name = archive.register({"x": 1}, target_level=1)

    q = DualCondorRunQueue(when_to_transfer_output=value, submit_mode="embed")
    q.subdag_factory = lambda a, s, l: "/tmp/x"
    with pytest.raises(ValueError, match="when_to_transfer_output"):
        q.submit(archive, [name])

    q = DualCondorRunQueue(submit_mode="embed",
                           subdag_factory=lambda a, s, l: "/tmp/x")
    q.when_to_transfer_output = value
    with pytest.raises(ValueError, match="when_to_transfer_output"):
        q.submit(archive, [name])


def test_the_default_timing_still_composes_with_a_subdag(archive, tmp_path):
    """The refusal is of a NON-default policy that would be dropped, not
    of sub-DAGs: a backend whose work unit is itself a DAG (GW PE via
    util_RIFT_pseudo_pipe) never asked for a timing and must still
    submit."""
    made = tmp_path / "child.dag"
    made.write_text("# noop\n")
    q = DualCondorRunQueue(submit_mode="embed",
                           subdag_factory=lambda a, s, l: str(made))
    name = archive.register({"x": 1}, target_level=1)
    q.submit(archive, [name])
    wrapper = open(q.last_wrapper_dag_path).read()
    assert "SUBDAG EXTERNAL {}_lvl1 {}".format(name, made) in wrapper


@pytest.mark.parametrize("bad", ["osdf:///x.sif \\", "osdf:///x.sif\x00"])
def test_an_image_that_would_corrupt_the_submit_file_is_refused(bad):
    """A trailing backslash is a submit-file line continuation: it
    swallows the next command, which is `arguments`. Measured before the
    fix -- condor_submit returned 0 and the workers ran the bootstrap
    with no --sim-name/--level, exiting 2 forever with nothing in the
    submit file to explain it."""
    with pytest.raises(ValueError):
        DualCondorRunQueue(container_image=bad)


def test_the_new_arguments_did_not_move_the_old_ones():
    """Both arrived after the constructor's positional sequence was in
    use. An earlier draft spliced them in next to use_singularity, where
    they belong by topic -- which rebound every positional argument from
    that point on, so a caller's positional True for use_singularity
    reached the container_image validator and raised TypeError. They are
    keyword-only and last instead."""
    q = DualCondorRunQueue(None, None, 4096, "4G", None, None, None,
                           True, "/cvmfs/x.sif")
    assert q.use_singularity is True
    assert q.singularity_image == "/cvmfs/x.sif"
    assert q.container_image == ""
    assert q.when_to_transfer_output == "ON_EXIT"
    params = inspect.signature(DualCondorRunQueue.__init__).parameters
    for name in ("container_image", "when_to_transfer_output"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
    positional = [n for n, p in params.items()
                  if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional[:10] == [
        "self", "run_pool", "run_collector", "request_memory",
        "request_disk", "accounting_group", "accounting_group_user",
        "getenv", "use_singularity", "singularity_image"]


def test_universe_is_not_protected(archive):
    """Deliberate. An earlier draft protected it, on the claim that a
    container_image under a vanilla universe is ignored by condor.
    Measured: vanilla+image, container+image and no universe at all give
    byte-identical ads. Protecting it would be a breaking change with no
    defect behind it, and would leave no route to local/scheduler/grid."""
    sub = _build(archive, DualCondorRunQueue(
        extra_condor_cmds={"universe": "scheduler"}))
    assert "scheduler" in sub
