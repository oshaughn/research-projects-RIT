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
    """The whole point: extras must not displace the frozen code."""
    q = DualCondorRunQueue(extra_transfer_input_files=BULK)
    name, sub = _build(archive, q)
    line = _transfer_line(sub)
    assert str(tmp_path / "arch" / "code") in line
    assert "params.json" in line


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
