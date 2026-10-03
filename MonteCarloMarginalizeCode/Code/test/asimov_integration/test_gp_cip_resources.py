"""Execute the real submit-writer body without LAL/glue imports or submission.

Command-plumbing checks execute source AST fragments, preserving the actual
formatting and quoting of recursively generated commands. These are not full
DAG-build tests: a site/runtime CUDA smoke remains a separate deployment gate.
"""
import ast
import os
import shlex
import sys
import types
from pathlib import Path
import pytest

CODE = Path(__file__).resolve().parents[2]
GUARD = "Capability >= 6.0 && Capability < 9.0"


class FakeJob:
    def __init__(self, **kwargs):
        self.init = kwargs
        self.commands = {}
        self.options = []
        self.paths = {}

    def add_condor_cmd(self, key, value):
        self.commands[key.lower()] = value

    def add_opt(self, key, value):
        self.options.append((key, value))

    def set_sub_file(self, value):
        self.paths["sub"] = value

    def set_log_file(self, value):
        self.paths["log"] = value

    def set_stderr_file(self, value):
        self.paths["err"] = value

    def set_stdout_file(self, value):
        self.paths["out"] = value


def real_writer(backend="dag_utils.py"):
    path = CODE / "RIFT/misc" / backend
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "write_CIP_sub")
    scope = {"os": os, "sys": sys, "CondorDAGJob": FakeJob, "pipeline": types.SimpleNamespace(CondorDAGJob=FakeJob),
             "which": lambda value: "/test/bin/" + value,
             "default_resolved_env": None, "default_getenv_value": "True",
             "is_container_manifest": lambda value: False}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
    return scope["write_CIP_sub"]


@pytest.mark.parametrize("backend", ["dag_utils.py", "dag_utils_generic.py"])
def test_real_gpu_submit_writer_resources_and_container(monkeypatch, backend):
    monkeypatch.setenv("RIFT_NOSTREAM_LOG", "1")
    job, name = real_writer(backend)(tag="CIP_worker", arg_str="--fit-method gp-matern", out_dir="/run",
        ncopies=8, request_memory=8192, request_gpus=1, require_gpus=GUARD,
        use_singularity=True, singularity_image="/scratch/reviewed-cuda118.sif", transfer_files=["all.net"])
    assert name == "CIP_worker.sub"
    assert job._CondorJob__queue == 8
    assert job.commands["request_gpus"] == "1"
    assert job.commands["require_gpus"] == GUARD
    assert job.commands["request_cpus"] == "1"
    assert job.commands["request_memory"] == "8192M"
    assert job.commands["my.singularityimage"] == '"/scratch/reviewed-cuda118.sif"'
    assert "stream_output" not in job.commands and "stream_error" not in job.commands
    assert job.commands["transfer_input_files"] == "all.net"


@pytest.mark.parametrize("backend", ["dag_utils.py", "dag_utils_generic.py"])
def test_real_cpu_writer_default_does_not_request_gpu(backend):
    job, _ = real_writer(backend)(arg_str="--fit-method rf", out_dir="/run")
    assert "request_gpus" not in job.commands
    assert "require_gpus" not in job.commands
    assert job.commands["request_memory"] == "8192M"


@pytest.mark.parametrize("kwargs", [{"request_gpus": -1}, {"request_gpus": "1"}, {"require_gpus": GUARD}])
@pytest.mark.parametrize("backend", ["dag_utils.py", "dag_utils_generic.py"])
def test_real_writer_rejects_invalid_gpu_resource_request(kwargs, backend):
    with pytest.raises(ValueError):
        real_writer(backend)(arg_str="--fit-method gp-matern", out_dir="/run", **kwargs)


def test_actual_four_main_producer_consumer_calls_receive_resources():
    tree = ast.parse((CODE / "bin/create_event_parameter_pipeline_BasicIteration").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "write_CIP_sub"
             and any(k.arg == "tag" and isinstance(k.value, ast.Constant) and k.value.value in ("CIP", "CIP_worker")
                     or k.arg == "tag" and isinstance(k.value, ast.BinOp) and isinstance(k.value.left, ast.Constant)
                     and k.value.left.value in ("CIP_", "CIP_worker") for k in n.keywords)]
    assert any(isinstance(n,ast.Import) and any(a.name=='RIFT.misc.dag_utils_generic' and a.asname=='dag_utils' for a in n.names) for n in tree.body)
    assert len(calls) == 4
    for call in calls:
        keywords = {k.arg: k.value for k in call.keywords}
        assert ast.unparse(keywords["request_gpus"]) == "opts.request_gpus_CIP"
        assert ast.unparse(keywords["require_gpus"]) == "opts.require_gpus_CIP"


def execute_resource_fragments(relative, opts):
    path = CODE / relative
    tree = ast.parse(path.read_text())
    fragments = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.AugAssign)):
            continue
        # Keep outer resource block intact so its conditional quoting runs.
        if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute) and node.test.attr.startswith("internal_cip_"):
            if node.test.attr in ("internal_cip_request_gpus", "internal_cip_require_gpus"):
                fragments.append(node)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "cmd":
            value = ast.unparse(node.value)
            if "--request-gpus-CIP" in value and "opts.request_gpus_CIP" in value:
                fragments.append(node)
        elif isinstance(node, ast.If) and isinstance(node.test, ast.Attribute) and node.test.attr == "require_gpus_CIP":
            fragments.append(node)
    assert len(fragments) == 2
    scope = {"cmd": "builder", "opts": opts, "shlex": shlex}
    exec(compile(ast.Module(body=fragments, type_ignores=[]), str(path), "exec"), scope)
    return shlex.split(scope["cmd"])


def test_actual_pseudo_and_recursive_commands_preserve_gpu_guard_single_argument():
    pseudo = execute_resource_fragments("bin/util_RIFT_pseudo_pipe.py", types.SimpleNamespace(
        internal_cip_request_gpus=1, internal_cip_require_gpus=GUARD))
    recursive = execute_resource_fragments("bin/create_event_parameter_pipeline_BasicIteration", types.SimpleNamespace(
        request_gpus_CIP=1, require_gpus_CIP=GUARD))
    for command in (pseudo, recursive):
        assert command[command.index("--request-gpus-CIP") + 1] == "1"
        assert command[command.index("--require-gpus-CIP") + 1] == GUARD


def test_actual_cip_and_ile_calls_use_independent_runtime_images():
    tree = ast.parse((CODE / "bin/create_event_parameter_pipeline_BasicIteration").read_text())
    cip = []
    ile = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        keywords = {k.arg: k.value for k in call.keywords}
        if "singularity_image" not in keywords:
            continue
        if call.func.attr == "write_CIP_sub":
            cip.append(ast.unparse(keywords["singularity_image"]))
        elif call.func.attr == "write_ILE_sub_simple":
            ile.append(ast.unparse(keywords["singularity_image"]))
    assert len(cip) == 4 and set(cip) == {"cip_singularity_image"}
    assert ile and set(ile) == {"singularity_image"}
    source = (CODE / "bin/create_event_parameter_pipeline_BasicIteration").read_text()
    assert "--cip-singularity-image" in source
    assert "shlex.quote(opts.cip_singularity_image)" in source
