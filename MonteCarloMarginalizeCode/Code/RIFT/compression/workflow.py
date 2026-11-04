"""Terminal DAG job consuming the selected final paired posterior."""
from pathlib import Path
import shlex
import shutil
from .config import enabled, file_hash, read_ini, resolve
from .cli import write_json
import json


def preflight(config, rundir, *, extrinsic, distance_reweighting=False):
    if not config or not enabled(config):
        return None
    if not extrinsic:
        raise ValueError("auto-fit requires final paired extrinsic posterior export")
    if distance_reweighting:
        raise ValueError("auto-fit with comoving-distance post-reweighting requires a resolved final-prior adapter; use the original posterior standalone")
    return read_ini(config)


def attach(dag, pipeline, config, rundir, parent, posterior, runtime_prefix="", final_cip_args=None):
    """The local-universe job uses products already returned to the submit host."""
    run = Path(rundir).resolve()
    source = Path(config).resolve()
    copy = run/"auto_fit.production.ini"
    if copy.exists() and copy.read_bytes() != source.read_bytes():
        raise ValueError("Existing auto-fit INI snapshot differs; preserve the previous workflow")
    if source != copy:
        shutil.copyfile(source, copy)
    if final_cip_args is not None:
        priorpath = run/"auto_fit.final_cip_args.json"
        selected = dict(arguments=final_cip_args, cip_list_sha256=file_hash(run/"args_cip_list.txt"))
        if priorpath.exists() and json.loads(priorpath.read_text()) != selected:
            raise ValueError("Existing final CIP snapshot differs; preserve the previous workflow")
        write_json(priorpath, selected)
    cfg = read_ini(copy)
    settings = cfg["rift-auto-fit"]
    _, _, resolved = resolve(copy, run)
    diagnostic = settings.getboolean("build-diagnostics", fallback=False)
    samples = settings.getint("diagnostic_samples", fallback=20000)
    bins = settings.getint("diagnostic_bins", fallback=64)
    if diagnostic and (not 100 <= samples <= 1000000 or not 8 <= bins <= 512):
        raise ValueError("Diagnostics require 100..1000000 samples and 8..512 bins")
    def command(name, arguments):
        executable = shutil.which(name)
        if executable is None:
            raise ValueError(name+" is not installed/on PATH")
        return shlex.join([executable]+[str(a) for a in arguments])+"\n"
    fit_command = command("util_RIFT_compress_posterior.py", [
        "--config", copy, "--rundir", run, "--posterior", posterior])
    sample_commands, compare_commands = "", ""
    if diagnostic:
        output = Path(resolved["output"])
        if not output.is_absolute():
            output = run/output
        for target in resolved["targets"]:
            directory = output/target/"diagnostics"
            draws = directory/"posterior_samples.dat"
            sample_commands += command("util_RIFT_sample_compressed.py", [
                "--model", output/target/"model.json", "--output", draws,
                "--n-samples", samples, "--seed", resolved["seed"]+2000])
            compare_commands += command("util_RIFT_compare_compressed.py", [
                "--fit-directory", output, "--target", target, "--samples", draws,
                "--output", directory/"js.json", "--bins", bins])
    node = _attach_job(dag, pipeline, run, parent, settings, runtime_prefix,
                       "AUTO_FIT", "auto_fit", fit_command)
    if diagnostic:
        node = _attach_job(dag, pipeline, run, node, settings, runtime_prefix,
                           "AUTO_FIT_SAMPLE", "auto_fit_sample", sample_commands)
        node = _attach_job(dag, pipeline, run, node, settings, runtime_prefix,
                           "AUTO_FIT_COMPARE", "auto_fit_compare", compare_commands)
    return node


def _attach_job(dag, pipeline, run, parent, settings, runtime_prefix, category, stem, commands):
    wrapper = run/(stem+".sh")
    wrapper.write_text("#!/bin/bash\nset -eu\n"+runtime_prefix+"\n"+commands)
    wrapper.chmod(0o755)
    job = pipeline.CondorDAGJob(universe="local", executable=str(wrapper))
    job.set_sub_file(str(run/(category+".sub")))
    job.set_log_file(str(run/(stem+".log")))
    job.set_stdout_file(str(run/(stem+".out")))
    job.set_stderr_file(str(run/(stem+".err")))
    job.add_condor_cmd("initialdir", str(run))
    job.add_condor_cmd("getenv", "True")
    for option, default in (("request_cpus", 1), ("request_memory", 4096), ("request_disk", 2048)):
        key = option+"_mb" if option != "request_cpus" else option
        value = settings.getint(key, fallback=default)
        if value <= 0 or (option == "request_cpus" and value != 1):
            raise ValueError("auto-fit currently requires one CPU and positive memory/disk budgets")
        job.add_condor_cmd(option, str(value*1024 if option == "request_disk" else value))
    seconds = settings.getint("max_runtime_seconds", fallback=1800)
    if seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive")
    job.add_condor_cmd("environment", '"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"')
    job.add_condor_cmd("periodic_remove", "JobStatus == 2 && (CurrentTime-JobStartDate) > "+str(seconds))
    job.write_sub_file()
    node = pipeline.CondorDAGNode(job)
    node.set_category(category)
    node.set_retry(0)
    node.add_parent(parent)
    dag.add_node(node)
    return node
