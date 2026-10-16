"""Container-family checks for BasicIteration's production dag_utils backend."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from RIFT.misc import dag_utils
from RIFT.misc.container_manifest import ContainerManifestError
from RIFT import lalsimutils as lsu
import lal


CODE = Path(__file__).resolve().parents[1]


MANIFEST = """\
version: 1
fallback: old
containers:
  - label: old
    image: osdf:///igwn/rift-old.sif
    cuda_capability_min: 3.0
    cuda_capability_max: 7.0
  - label: new
    image: osdf:///igwn/rift-new.sif
    cuda_capability_min: 7.0
"""


def _manifest(tmp_path):
    path = tmp_path / "family.yaml"
    path.write_text(MANIFEST)
    return str(path)


def _ile(tmp_path, monkeypatch, image, *, gpu=True, frames=False):
    monkeypatch.chdir(tmp_path)
    kwargs = dict(
        tag="ILE", log_dir=str(tmp_path) + "/", exe="/usr/bin/true",
        arg_str="--foo bar", transfer_files=["../all.net"],
        use_singularity=True, singularity_image=image, request_gpu=gpu,
        cache_file="local.cache", use_oauth_files="scitokens",
    )
    if frames:
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        kwargs["frames_dir"] = str(frame_dir)
    job, _ = dag_utils.write_ILE_sub_simple(**kwargs)
    return job, dict(job.get_condor_cmds())


def test_gpu_container_universe_keeps_fanout_and_selective_transfer(tmp_path, monkeypatch):
    monkeypatch.setenv("RIFT_CONTAINER_UNIVERSE", "1")
    monkeypatch.setenv("RIFT_ILE_GPU_FANOUT", "2")
    monkeypatch.setenv("RIFT_REQUIRE_GPUS", 'DeviceName =!= "old"')
    job, cmds = _ile(tmp_path, monkeypatch, _manifest(tmp_path), frames=True)
    job.write_sub_file()
    assert "universe = container" in (tmp_path / "ILE.sub").read_text().lower()
    assert cmds["request_GPUs"] == "2"
    assert cmds["request_CPUs"] == "2"
    assert cmds["container_image"].startswith("$$([")
    assert "/" not in cmds["container_image"]
    assert "MY.SingularityImage" not in cmds
    assert cmds["transfer_input_files"].count("$$([") == 1
    assert "osdf:///igwn/rift-new.sif" in cmds["transfer_input_files"]
    assert cmds["MY.TransferInput"] == '"' + cmds["transfer_input_files"].replace('"', '\\"') + '"'
    assert cmds["use_oauth_services"] == "scitokens"
    assert "TARGET.GPUs_Capability =!= undefined" in cmds["requirements"]
    assert 'DeviceName =!= "old"' in cmds["require_gpus"]
    assert "Capability >= 3.0" in cmds["require_gpus"]
    pre = (tmp_path / "ile_pre.sh").read_text()
    assert "RIFT_ILE_GPU_FANOUT" in pre
    assert "/usr/bin/true" in pre


def test_cpu_container_universe_cip_calibration_and_ile_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("RIFT_CONTAINER_UNIVERSE", "1")
    monkeypatch.delenv("RIFT_REQUIRE_GPUS", raising=False)
    image = _manifest(tmp_path)
    _, ile = _ile(tmp_path, monkeypatch, image, gpu=False)
    assert ile["container_image"] == "osdf:///igwn/rift-old.sif"
    assert "$$([" not in ile.get("transfer_input_files", "")
    assert "require_gpus" not in ile

    job, _ = dag_utils.write_CIP_sub(
        tag="CIP", log_dir=str(tmp_path) + "/", out_dir=str(tmp_path),
        exe="/usr/bin/true", arg_str="--foo bar", transfer_files=["../all.net"],
        use_singularity=True, singularity_image=image, use_oauth_files="scitokens",
    )
    job.write_sub_file()
    cip = dict(job.get_condor_cmds())
    assert cip["container_image"] == "osdf:///igwn/rift-old.sif"
    assert "MY.SingularityImage" not in cip
    assert cip["use_oauth_services"] == "scitokens"

    job, _ = dag_utils.write_calibration_uncertainty_reweighting_sub(
        tag="CAL", log_dir=str(tmp_path) + "/", exe="/usr/bin/true",
        transfer_files=["../samples.dat"], use_singularity=True, use_oauth_files="scitokens",
        singularity_image=image, pickle_file="event.pkl", posterior_file="posterior.dat",
    )
    job.write_sub_file()
    cal = dict(job.get_condor_cmds())
    assert cal["container_image"] == "osdf:///igwn/rift-old.sif"
    assert "MY.SingularityImage" not in cal
    assert cal["use_oauth_services"] == "scitokens"


def test_legacy_family_and_single_image_routes(tmp_path, monkeypatch):
    monkeypatch.delenv("RIFT_CONTAINER_UNIVERSE", raising=False)
    image = _manifest(tmp_path)
    _, family = _ile(tmp_path, monkeypatch, image)
    assert family["MY.SingularityImage"].startswith("ifThenElse(")
    assert "$$([" in family["transfer_input_files"]
    _, cpu_family = _ile(tmp_path, monkeypatch, image, gpu=False)
    assert cpu_family["MY.SingularityImage"] == '"./rift-old.sif"'
    assert "$$([" not in cpu_family["transfer_input_files"]
    assert "osdf:///igwn/rift-old.sif" in cpu_family["transfer_input_files"]
    assert "require_gpus" not in cpu_family
    _, plain = _ile(tmp_path, monkeypatch, "./single.sif")
    assert plain["MY.SingularityImage"] == '"./single.sif"'
    assert "container_image" not in plain

    job, _ = dag_utils.write_CIP_sub(
        tag="CIP", log_dir=str(tmp_path) + "/", out_dir=str(tmp_path),
        exe="/usr/bin/true", arg_str="--foo bar", transfer_files=["../all.net"],
        use_singularity=True, singularity_image=image,
    )
    cip = dict(job.get_condor_cmds())
    assert cip["MY.SingularityImage"] == '"./rift-old.sif"'
    assert "osdf:///igwn/rift-old.sif" in cip["transfer_input_files"]


def test_runtime_wrapper_opt_in_fails_instead_of_silent_legacy_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("RIFT_CONTAINER_UNIVERSE", raising=False)
    monkeypatch.setenv("RIFT_CONTAINER_RUNTIME_SELECT", "1")
    with pytest.raises(ContainerManifestError, match="not supported by O4c"):
        _ile(tmp_path, monkeypatch, _manifest(tmp_path))


def test_real_basic_iteration_build_emits_family_ile_and_dag(tmp_path):
    image = _manifest(tmp_path)
    p = lsu.ChooseWaveformParams()
    p.m1, p.m2 = 60 * lal.MSUN_SI, 40 * lal.MSUN_SI
    p.fmin = p.fref = 30
    lsu.ChooseWaveformParams_array_to_xml([p], fname=str(tmp_path / "input-grid"))
    (tmp_path / "args_ile.txt").write_text(
        "integrate_likelihood_extrinsic_batchmode --approx IMRPhenomXPHM"
    )
    (tmp_path / "args_cip.txt").write_text(
        "X --parameter m1 --parameter m2 --n-output-samples 10"
    )
    (tmp_path / "frames").mkdir()
    env = dict(os.environ)
    env.update(
        SINGULARITY_RIFT_IMAGE=image, SINGULARITY_BASE_EXE_DIR="/usr/bin",
        RIFT_CONTAINER_UNIVERSE="1", RIFT_ILE_GPU_FANOUT="2",
        PYTHONPATH=str(CODE), PATH=str(CODE / "bin") + os.pathsep + env["PATH"],
        GW_SURROGATE="",
    )
    command = [
        sys.executable, str(CODE / "bin/create_event_parameter_pipeline_BasicIteration"),
        "--working-directory", str(tmp_path),
        "--input-grid", str(tmp_path / "input-grid.xml.gz"),
        "--ile-args", str(tmp_path / "args_ile.txt"),
        "--cip-args", str(tmp_path / "args_cip.txt"),
        "--ile-exe", str(CODE / "bin/integrate_likelihood_extrinsic_batchmode"),
        "--n-iterations", "1", "--n-samples-per-job", "1", "--n-copies", "1",
        "--use-singularity", "--frames-dir", str(tmp_path / "frames"),
        "--request-gpu-ILE",
    ]
    built = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    assert built.returncode == 0, built.stdout + "\n" + built.stderr
    assert (tmp_path / "marginalize_intrinsic_parameters_BasicIterationWorkflow.dag").is_file()
    submission = (tmp_path / "ILE.sub").read_text()
    assert "universe = container" in submission
    assert "container_image = $$([" in submission
    assert "request_GPUs = 2" in submission
    assert "request_CPUs = 2" in submission
    assert "use_oauth_services = scitokens" in submission
    assert (tmp_path / "ile_pre.sh").is_file()
    assert "RIFT_ILE_GPU_FANOUT" in (tmp_path / "ile_pre.sh").read_text()
