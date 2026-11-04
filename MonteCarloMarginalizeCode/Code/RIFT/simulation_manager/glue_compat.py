"""glue.pipeline integration: drop a 'request a simulation' node into an
existing DAG.

The use case: a user has a hyperpipeline DAG (or any glue.pipeline DAG)
and wants to add ONE node that says "ensure simulation X is present in
the archive at level >= N" before downstream nodes run. The user
shouldn't have to reimplement the top-level workflow.

Example:

    from glue import pipeline
    from RIFT.simulation_manager.glue_compat import RequestSimulationJob

    dag = pipeline.CondorDAG(log='/tmp/dag.log')

    sim_job = RequestSimulationJob(
        archive_path='/path/to/archive',
        log_dir='/tmp/sim_logs',
        request_memory=2048,
        accounting_group='ligo.dev.o4.cbc.pe.lalinferencerapid',
        accounting_group_user='albert.einstein',
    )
    sim_job.set_sub_file('request_sim.sub')
    sim_job.write_sub_file()

    for params in points_of_interest:
        node = sim_job.make_node(params=params, target_level=4)
        dag.add_node(node)
        downstream_node.add_parent(node)   # everything downstream
                                           # blocks until the sim is ready

The node's executable is the request_sim CLI (a thin Python entrypoint
that registers + dispatches + blocks). When DAGMan runs the node it
calls the CLI, which (a) returns immediately if the sim is already at
the requested level or (b) dispatches run-pool work and blocks until
ready. Either way DAGMan sees a normal exit-code-driven node.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# We don't fail-import on missing glue; we just defer the error until use.
try:
    from glue import pipeline as _glue_pipeline  # type: ignore
    _has_glue = True
except Exception:                                # pragma: no cover
    _glue_pipeline = None
    _has_glue = False


def _require_glue() -> None:
    if not _has_glue:
        raise ImportError(
            "glue.pipeline is not available; install lscsoft-glue to use "
            "RequestSimulationJob. The same archive can still be driven "
            "directly via RIFT.simulation_manager.cli.request_sim.")


def _default_request_sim_argv() -> List[str]:
    """Argv prefix that invokes the request_sim CLI as a Python module.
    Using `python -m` rather than a path lets the node work whether or
    not the package was installed via `pip install -e .`."""
    return [sys.executable or "python3", "-m",
            "RIFT.simulation_manager.cli.request_sim"]


if _has_glue:

    class RequestSimulationJob(_glue_pipeline.CondorDAGJob):
        """A glue.pipeline CondorDAGJob that runs the request_sim CLI.

        Construct once per (archive, request profile); call `make_node`
        per (params, target_level) you want to ensure. The job's
        executable invokes
            python -m RIFT.simulation_manager.cli.request_sim
        with --archive / --params / --target-level supplied as macros
        on each node.
        """

        def __init__(self,
                     archive_path: Union[str, Path],
                     log_dir: Union[str, Path] = ".",
                     universe: str = "vanilla",
                     mode: str = "ensure",      # passed to the CLI
                     request_memory: int = 1024,
                     request_disk: Optional[str] = None,
                     accounting_group: Optional[str] = None,
                     accounting_group_user: Optional[str] = None,
                     getenv: str = "True",
                     extra_condor_cmds: Optional[Dict[str, str]] = None,
                     poll_interval: Optional[float] = None,
                     timeout: Optional[float] = None,
                     argv_prefix: Optional[List[str]] = None,
                     ):
            argv = list(argv_prefix or _default_request_sim_argv())
            executable = argv[0]
            super().__init__(universe=universe, executable=executable)

            self.archive_path = str(Path(archive_path).resolve())
            self.log_dir = str(Path(log_dir).resolve())
            os.makedirs(self.log_dir, exist_ok=True)

            # Build the argument string. Everything after the executable
            # becomes condor `arguments`. Per-node values (params,
            # target_level) come in as macros.
            tail = argv[1:] + [
                "--archive", self.archive_path,
                "--params", "$(macro_params)",
                "--target-level", "$(macro_target_level)",
                "--{}".format(mode.replace("_", "-")),
            ]
            if poll_interval is not None:
                tail += ["--poll-interval", str(poll_interval)]
            if timeout is not None:
                tail += ["--timeout", str(timeout)]
            self.add_arg(" ".join(shlex.quote(s) for s in tail))

            # Standard condor knobs.
            self.add_condor_cmd("getenv", getenv)
            self.add_condor_cmd("request_memory", "{}M".format(int(request_memory)))
            if request_disk is not None:
                self.add_condor_cmd("request_disk", str(request_disk))
            if accounting_group:
                self.add_condor_cmd("accounting_group", accounting_group)
            if accounting_group_user:
                self.add_condor_cmd("accounting_group_user", accounting_group_user)
            for k, v in (extra_condor_cmds or {}).items():
                self.add_condor_cmd(k, str(v))

            tag = "request_sim"
            self.set_log_file(
                "{ld}/{tag}-$(cluster)-$(process).log".format(ld=self.log_dir, tag=tag))
            self.set_stdout_file(
                "{ld}/{tag}-$(cluster)-$(process).out".format(ld=self.log_dir, tag=tag))
            self.set_stderr_file(
                "{ld}/{tag}-$(cluster)-$(process).err".format(ld=self.log_dir, tag=tag))

        # -------- node factory --------------------------------------------
        def make_node(self,
                      params: Any,
                      target_level: int = 1,
                      category: Optional[str] = None,
                      retry: Optional[int] = None,
                      ) -> "_glue_pipeline.CondorDAGNode":
            """Build a CondorDAGNode for one request. Caller adds it to
            their DAG and wires up parents/children as usual."""
            node = _glue_pipeline.CondorDAGNode(self)
            node.add_macro("macro_params", json.dumps(params))
            node.add_macro("macro_target_level", str(int(target_level)))
            if category is not None:
                node.set_category(category)
            if retry is not None:
                node.set_retry(int(retry))
            return node

else:

    class RequestSimulationJob:  # type: ignore[no-redef]
        """Stand-in raised at construction when glue.pipeline is missing."""

        def __init__(self, *a: Any, **kw: Any):
            _require_glue()


def make_request_simulation_node(dag: Any,
                                 archive_path: Union[str, Path],
                                 params: Any,
                                 target_level: int = 1,
                                 *,
                                 sub_file: Optional[Union[str, Path]] = None,
                                 reuse_job: Optional[Any] = None,
                                 **job_kwargs: Any,
                                 ) -> Any:
    """One-shot helper: build (or reuse) a RequestSimulationJob, attach
    a node for `params` to `dag`, and return the node.

    For multiple nodes per DAG, prefer constructing the job once and
    calling `job.make_node(...)` repeatedly — that writes a single
    .sub file shared across all nodes.
    """
    _require_glue()
    if reuse_job is None:
        job = RequestSimulationJob(archive_path=archive_path, **job_kwargs)
        if sub_file is not None:
            job.set_sub_file(str(sub_file))
            job.write_sub_file()
    else:
        job = reuse_job
    node = job.make_node(params=params, target_level=target_level)
    dag.add_node(node)
    return node
