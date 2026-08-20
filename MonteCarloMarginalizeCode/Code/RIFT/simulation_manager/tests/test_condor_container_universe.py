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


@pytest.mark.parametrize("value", WHEN_TO_TRANSFER_OUTPUT)
def test_every_legal_value_is_emitted(archive, value):
    sub = _build(archive, DualCondorRunQueue(when_to_transfer_output=value))
    assert _command(sub, "when_to_transfer_output") == value


def test_it_is_normalised_not_passed_through(archive):
    sub = _build(archive,
                 DualCondorRunQueue(when_to_transfer_output=" on_exit_or_evict "))
    assert _command(sub, "when_to_transfer_output") == "ON_EXIT_OR_EVICT"


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
    ("universe", "container_image"),
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
                         "when_to_transfer_output": "ON_EXIT_OR_EVICT"})
    Archive(base_location=tmp_path / "arch", manifest=manifest,
            generator_spec={"module_path": str(code / "generator.py"),
                            "entrypoint": "generator:run"})
    reopened = Archive(base_location=tmp_path / "arch")
    _, run_queue = make_queues_from_manifest(reopened)
    assert run_queue.container_image == IMAGE
    sub = _build(reopened, run_queue)
    assert _command(sub, "universe") == "container"
    assert _command(sub, "when_to_transfer_output") == "ON_EXIT_OR_EVICT"


@pytest.mark.parametrize("kwargs", [
    {},
    {"container_image": IMAGE},
    {"when_to_transfer_output": "ON_EXIT_OR_EVICT"},
    {"container_image": IMAGE, "when_to_transfer_output": "ON_EXIT_OR_EVICT"},
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
    proc = subprocess.run([condor_submit, "-dry-run", str(out), str(path)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    materialised = out.read_text()
    if kwargs.get("container_image"):
        assert "ContainerImage" in materialised
