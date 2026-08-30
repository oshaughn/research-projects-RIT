"""
Tests for container family manifest parsing and the expression-valued
SingularityImage / selective-transfer / require_gpus wiring.

These run without a real HTCondor pool: the parser + expression builders are
pure, and the integration test inspects the generated ``condor_cmds`` on the
job object returned by ``write_ILE_sub_simple`` (no .sub file or condor needed).

Run directly:  python test/test_container_manifest.py
Or via pytest: pytest test/test_container_manifest.py
"""

import os
import shutil
import stat
import subprocess
import sys
import textwrap

import pytest

yaml = pytest.importorskip("yaml")  # manifest parsing requires PyYAML

import RIFT.misc.container_manifest as cm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

MIXED_MANIFEST = textwrap.dedent(
    """
    version: 1
    fallback: ancient
    containers:
      - label: ancient
        image: /cvmfs/sw/rift_ancient_cuda11.sif
        cuda_capability_min: 3.0
        cuda_capability_max: 7.0
      - label: modern
        image: osdf:///igwn/rift_modern_cuda12.sif
        cuda_capability_min: 7.0
    """
)

ALL_CVMFS_MANIFEST = textwrap.dedent(
    """
    version: 1
    fallback: ancient
    containers:
      - label: ancient
        image: /cvmfs/sw/rift_ancient.sif
        cuda_capability_min: 3.0
      - label: modern
        image: /cvmfs/sw/rift_modern.sif
        cuda_capability_min: 7.0
    """
)


ALL_OSDF_MANIFEST = textwrap.dedent(
    """
    version: 1
    fallback: ancient
    containers:
      - label: ancient
        image: osdf:///igwn/sw/rift_ancient_cuda11.sif
        cuda_capability_min: 3.0
        cuda_capability_max: 7.0
      - label: modern
        image: osdf:///igwn/sw/rift_modern_cuda12.sif
        cuda_capability_min: 7.0
    """
)


def _write(tmp_path, text, name="fam.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# ---------------------------------------------------------------------------
# 1. parser
# ---------------------------------------------------------------------------

def test_parser_sorts_and_resolves_fallback(tmp_path):
    m = cm.load_container_manifest(_write(tmp_path, MIXED_MANIFEST))
    # sorted by capability descending
    assert [c["label"] for c in m["containers"]] == ["modern", "ancient"]
    assert m["fallback"] == "ancient"
    assert m["capability_attr"] == cm.DEFAULT_CAPABILITY_ATTR


def test_parser_default_fallback_is_lowest(tmp_path):
    # no explicit fallback -> most-compatible (lowest-min) container
    text = MIXED_MANIFEST.replace("fallback: ancient\n", "")
    m = cm.load_container_manifest(_write(tmp_path, text))
    assert m["fallback"] == "ancient"


def test_parser_rejects_unknown_fallback(tmp_path):
    text = MIXED_MANIFEST.replace("fallback: ancient", "fallback: nope")
    with pytest.raises(cm.ContainerManifestError):
        cm.load_container_manifest(_write(tmp_path, text))


def test_parser_rejects_empty(tmp_path):
    with pytest.raises(cm.ContainerManifestError):
        cm.load_container_manifest(_write(tmp_path, "version: 1\ncontainers: []\n"))


def test_parser_rejects_missing_image(tmp_path):
    text = "containers:\n  - label: x\n    cuda_capability_min: 5.0\n"
    with pytest.raises(cm.ContainerManifestError):
        cm.load_container_manifest(_write(tmp_path, text))


# ---------------------------------------------------------------------------
# 2. expressions
# ---------------------------------------------------------------------------

def test_image_expression(tmp_path):
    m = cm.load_container_manifest(_write(tmp_path, MIXED_MANIFEST))
    expr = cm.build_singularity_image_expr(m)
    # Bare capability selector (NOT undefined-guarded): a GPU job must instead add
    # the build_capability_defined_requirement Requirements clause so it never
    # matches a slot where this would be undefined.
    assert expr == (
        'ifThenElse(TARGET.GPUs_Capability >= 7.0, '
        '"./rift_modern_cuda12.sif", "/cvmfs/sw/rift_ancient_cuda11.sif")'
    )
    # an expression must NOT be a quoted string literal
    assert not expr.startswith('"')


def test_transfer_expression_is_comma_free_ternary(tmp_path):
    m = cm.load_container_manifest(_write(tmp_path, MIXED_MANIFEST))
    expr = cm.build_transfer_input_expr(m)
    assert expr == (
        '$$([ (TARGET.GPUs_Capability >= 7.0 ? '
        '"osdf:///igwn/rift_modern_cuda12.sif" : "") ])'
    )
    # the token sits inside a comma-separated transfer_input_files list, so it
    # must contain no commas of its own
    assert "," not in expr


def test_transfer_expression_none_when_all_in_place(tmp_path):
    m = cm.load_container_manifest(_write(tmp_path, ALL_CVMFS_MANIFEST))
    assert cm.build_transfer_input_expr(m) is None


def test_require_gpus_floor(tmp_path):
    m = cm.load_container_manifest(_write(tmp_path, MIXED_MANIFEST))
    assert cm.build_require_gpus_floor(m) == "Capability >= 3.0"


def test_selectors_are_not_undefined_guarded(tmp_path):
    # The capability selectors must NOT default an undefined-capability slot to the
    # fallback image: that slot could be a Blackwell that hard-fails on the older
    # fallback.  The safe fix is the Requirements exclusion below, not a guess.
    m = cm.load_container_manifest(_write(tmp_path, MIXED_MANIFEST))
    assert "=?= undefined" not in cm.build_singularity_image_expr(m)
    assert "=?= undefined" not in cm.build_transfer_input_expr(m)
    assert "=?= undefined" not in cm.build_container_image_select(
        cm.load_container_manifest(_write(tmp_path, ALL_OSDF_MANIFEST, "osdf.yaml")))


def test_capability_defined_requirement(tmp_path):
    m = cm.load_container_manifest(_write(tmp_path, MIXED_MANIFEST))
    # excludes slots that don't advertise the (machine-level) capability attr
    assert cm.build_capability_defined_requirement(m) == "TARGET.GPUs_Capability =!= undefined"


def test_capability_defined_requirement_respects_attr_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RIFT_GPU_CAPABILITY_ATTR", "CUDACapability")
    m = cm.load_container_manifest(_write(tmp_path, MIXED_MANIFEST))
    assert cm.build_capability_defined_requirement(m) == "TARGET.CUDACapability =!= undefined"


def test_fallback_single_image(tmp_path):
    # MIXED fallback (ancient) is a CVMFS image -> referenced in place, no transfer
    m = cm.load_container_manifest(_write(tmp_path, MIXED_MANIFEST))
    runtime, transfer = cm.build_fallback_single_image(m)
    assert runtime == "/cvmfs/sw/rift_ancient_cuda11.sif"
    assert transfer is None
    # an osdf fallback -> runtime is ./basename and it IS transferred
    osdf_text = MIXED_MANIFEST.replace("/cvmfs/sw/rift_ancient_cuda11.sif",
                                       "osdf:///igwn/rift_ancient_cuda11.sif")
    m2 = cm.load_container_manifest(_write(tmp_path, osdf_text, name="fam2.yaml"))
    runtime2, transfer2 = cm.build_fallback_single_image(m2)
    assert runtime2 == "./rift_ancient_cuda11.sif"
    assert transfer2 == "osdf:///igwn/rift_ancient_cuda11.sif"


def test_capability_attr_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RIFT_GPU_CAPABILITY_ATTR", "CUDACapability")
    m = cm.load_container_manifest(_write(tmp_path, MIXED_MANIFEST))
    assert "TARGET.CUDACapability >=" in cm.build_singularity_image_expr(m)


# ---------------------------------------------------------------------------
# 3-5. integration with write_ILE_sub_simple (inspect generated condor_cmds)
# ---------------------------------------------------------------------------

def _make_ile_job(tmp_path, monkeypatch, singularity_image):
    """Call write_ILE_sub_simple in an isolated cwd; return its condor_cmds dict.

    Skips if the dag_utils_generic backend cannot be imported in this env.
    """
    dag = pytest.importorskip("RIFT.misc.dag_utils_generic")
    monkeypatch.chdir(tmp_path)
    job, _ = dag.write_ILE_sub_simple(
        tag="ILE",
        log_dir=str(tmp_path) + "/",
        exe="/usr/bin/true",
        arg_str="--foo bar",
        transfer_files=["../all.net"],
        use_singularity=True,
        singularity_image=singularity_image,
        request_gpu=True,
        cache_file="local.cache",
    )
    return dict(job.condor_cmds)


def test_integration_family_mixed(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "RIFT_REQUIRE_GPUS", '(DeviceName=!="Tesla K10.G1.8GB")'
    )
    cmds = _make_ile_job(tmp_path, monkeypatch, _write(tmp_path, MIXED_MANIFEST))

    img = cmds["MY.SingularityImage"]
    assert img.startswith("ifThenElse(")          # expression, not a literal
    assert not img.startswith('"')

    # selective transfer: exactly one $$() token, whole family NOT dumped
    tif = cmds["transfer_input_files"]
    assert tif.count("$$([") == 1
    assert "/cvmfs/sw/rift_ancient_cuda11.sif" not in tif  # cvmfs image not transferred
    assert tif.count("osdf:///igwn/rift_modern_cuda12.sif") == 1

    # floor composed with (not replacing) the user's RIFT_REQUIRE_GPUS
    rg = cmds["require_gpus"]
    assert "Capability >= 3.0" in rg
    assert 'DeviceName=!="Tesla K10.G1.8GB"' in rg
    assert "&&" in rg

    # GPU family job: Requirements exclude slots that don't advertise the
    # machine-level capability attr (else the selection $$/ifThenElse holds).
    assert "TARGET.GPUs_Capability =!= undefined" in cmds["requirements"]


def test_integration_all_cvmfs_no_transfer_token(tmp_path, monkeypatch):
    cmds = _make_ile_job(tmp_path, monkeypatch, _write(tmp_path, ALL_CVMFS_MANIFEST))
    assert "$$([" not in cmds.get("transfer_input_files", "")
    # still an expression-valued image + a capability floor
    assert cmds["MY.SingularityImage"].startswith("ifThenElse(")
    assert "Capability >= 3.0" in cmds["require_gpus"]


def test_backward_compat_single_sif(tmp_path, monkeypatch):
    monkeypatch.delenv("RIFT_REQUIRE_GPUS", raising=False)
    cmds = _make_ile_job(tmp_path, monkeypatch, "./foo.sif")
    # byte-identical legacy behavior: quoted literal, no $$() token, no floor
    assert cmds["MY.SingularityImage"] == '"./foo.sif"'
    assert "$$([" not in cmds.get("transfer_input_files", "")
    assert "require_gpus" not in cmds


# ---------------------------------------------------------------------------
# 6. container universe: $$()-substituted container_image selection (OSG-safe)
# ---------------------------------------------------------------------------

def test_container_image_select_expression(tmp_path):
    m = cm.load_container_manifest(_write(tmp_path, ALL_OSDF_MANIFEST))
    expr = cm.build_container_image_select(m)
    # A $$() match-time substitution token over BASENAMES.  condor_submit derives the
    # job ad's ContainerImage as the text after the LAST '/', *before* any $$
    # expansion, so a selector containing a path is truncated and the job holds at the
    # execute point ("Unable to download or build singularity image ...sif\") ])",
    # observed live on an OSPool glidein).  With no '/' the token survives intact and
    # the schedd expands it at match time.
    assert expr.startswith("$$([ ") and expr.endswith(" ])")
    assert "/" not in expr                                        # THE invariant
    assert "=?= undefined" not in expr                            # not a guess-guard
    assert "ifThenElse(TARGET.GPUs_Capability >= 7.0," in expr
    assert '"rift_modern_cuda12.sif"' in expr                     # basename branch
    assert '"rift_ancient_cuda11.sif"' in expr                    # fallback basename


def test_container_image_select_rejects_in_place_images(tmp_path):
    # An in-place (CVMFS/local) image can only be named by its full path, which would
    # reintroduce the '/' truncation.  Refuse loudly rather than emit a submit file
    # that holds every job.
    m = cm.load_container_manifest(_write(tmp_path, MIXED_MANIFEST))
    with pytest.raises(cm.ContainerManifestError) as exc:
        cm.build_container_image_select(m)
    assert "ancient" in str(exc.value)
    # ... but the CPU-only single-image path is unaffected: it is a plain literal that
    # condor_submit handles correctly.
    assert cm.build_container_image_select(m, request_gpu=False) == "/cvmfs/sw/rift_ancient_cuda11.sif"


def test_integration_container_universe(tmp_path, monkeypatch):
    # Opt-in container-universe mode: per-machine image via $$()-substituted
    # container_image; no MY.SingularityImage / BindCVMFS / $$() transfer token;
    # universe=container; require_gpus floor still applied.
    monkeypatch.setenv("RIFT_CONTAINER_UNIVERSE", "1")
    monkeypatch.delenv("RIFT_REQUIRE_GPUS", raising=False)
    monkeypatch.chdir(tmp_path)
    dag = pytest.importorskip("RIFT.misc.dag_utils_generic")
    job, _ = dag.write_ILE_sub_simple(
        tag="ILE",
        log_dir=str(tmp_path) + "/",
        exe="/usr/bin/true",
        arg_str="--foo bar",
        transfer_files=["../all.net"],
        use_singularity=True,
        singularity_image=_write(tmp_path, ALL_OSDF_MANIFEST),
        request_gpu=True,
        cache_file="local.cache",
    )
    cmds = dict(job.condor_cmds)

    ci = cmds["container_image"]
    assert ci.startswith("$$([")               # match-time substitution, unquoted
    assert not ci.startswith('"')
    assert "/" not in ci                       # else condor_submit truncates it
    assert "MY.SingularityImage" not in cmds    # the OSG-breaking attr is gone
    assert "MY.SingularityBindCVMFS" not in cmds

    # container_image names only a basename, so the image must arrive by transfer:
    # exactly one comma-free $$() token carrying the full URLs.
    tif = cmds["transfer_input_files"]
    assert tif.count("$$([") == 1
    assert "osdf:///igwn/sw/rift_modern_cuda12.sif" in tif
    # ... and TransferInput is pinned, so condor_submit does not append the basename
    # selector to it as a bogus extra input file.
    assert cmds["MY.TransferInput"] == '"' + tif.replace('"', '\\"') + '"'

    assert "Capability >= 3.0" in cmds["require_gpus"]         # floor still steers GPUs
    # GPU family job: still excludes slots that don't advertise the capability attr
    assert "TARGET.GPUs_Capability =!= undefined" in cmds["requirements"]

    assert job.universe == "container"          # HTCondor container universe


def test_container_image_select_no_gpu_collapses_to_single(tmp_path):
    m = cm.load_container_manifest(_write(tmp_path, MIXED_MANIFEST))
    # A job that requests no GPU has no capability to key on -- a $$() expression
    # would not resolve on a CPU-only slot and would HOLD the job.  Collapse to
    # the single (CPU-safe) fallback image: a plain literal, no $$(), no ifThenElse.
    val = cm.build_container_image_select(m, request_gpu=False)
    assert val == "/cvmfs/sw/rift_ancient_cuda11.sif"   # the fallback image, verbatim
    assert "$$(" not in val
    assert "ifThenElse" not in val


def test_integration_cip_container_universe_single_image(tmp_path, monkeypatch):
    # CIP is CPU-only: under container universe it must use a SINGLE fixed
    # container (the fallback image), never the $$() capability selection.
    monkeypatch.setenv("RIFT_CONTAINER_UNIVERSE", "1")
    monkeypatch.chdir(tmp_path)
    dag = pytest.importorskip("RIFT.misc.dag_utils_generic")
    job, _ = dag.write_CIP_sub(
        tag="CIP",
        out_dir=str(tmp_path),
        log_dir=str(tmp_path) + "/",
        exe="/usr/bin/true",
        arg_str="--foo bar",
        transfer_files=["../all.net"],
        use_singularity=True,
        singularity_image=_write(tmp_path, MIXED_MANIFEST),
    )
    cmds = dict(job.condor_cmds)
    ci = cmds["container_image"]
    assert ci == "/cvmfs/sw/rift_ancient_cuda11.sif"   # single fixed fallback image
    assert "$$(" not in ci                              # NOT a capability $$() selection
    assert "MY.SingularityImage" not in cmds
    assert "$$([" not in cmds.get("transfer_input_files", "")
    assert "require_gpus" not in cmds                   # CPU job: no GPU floor
    assert job.universe == "container"


def test_integration_cip_legacy_single_image(tmp_path, monkeypatch):
    # CIP (CPU-only) on the LEGACY path: a single QUOTED fallback MY.SingularityImage
    # (a bare path is a ClassAd parse error), NOT the family $$()/ifThenElse selection
    # (a CPU slot can't resolve it), no $$() transfer token, and NO capability
    # Requirements exclusion (CIP requests no GPU, so it must not be GPU-constrained).
    monkeypatch.delenv("RIFT_CONTAINER_UNIVERSE", raising=False)
    monkeypatch.chdir(tmp_path)
    dag = pytest.importorskip("RIFT.misc.dag_utils_generic")
    job, _ = dag.write_CIP_sub(
        tag="CIP",
        out_dir=str(tmp_path),
        log_dir=str(tmp_path) + "/",
        exe="/usr/bin/true",
        arg_str="--foo bar",
        transfer_files=["../all.net"],
        use_singularity=True,
        singularity_image=_write(tmp_path, MIXED_MANIFEST),
    )
    cmds = dict(job.condor_cmds)
    assert cmds["MY.SingularityImage"] == '"/cvmfs/sw/rift_ancient_cuda11.sif"'  # single, quoted
    assert "ifThenElse" not in cmds["MY.SingularityImage"]
    assert "container_image" not in cmds
    assert "$$([" not in cmds.get("transfer_input_files", "")
    assert "=!= undefined" not in cmds.get("requirements", "")   # CPU job: no GPU exclusion
    assert "require_gpus" not in cmds


def _make_calibration_job(tmp_path, monkeypatch, manifest, container_universe):
    if container_universe:
        monkeypatch.setenv("RIFT_CONTAINER_UNIVERSE", "1")
    else:
        monkeypatch.delenv("RIFT_CONTAINER_UNIVERSE", raising=False)
    monkeypatch.chdir(tmp_path)
    dag = pytest.importorskip("RIFT.misc.dag_utils_generic")
    job, _ = dag.write_calibration_uncertainty_reweighting_sub(
        tag="Calib_reweight",
        log_dir=str(tmp_path) + "/",
        exe="/usr/bin/true",
        pickle_file=str(tmp_path / "event.pickle"),
        posterior_file=str(tmp_path / "posterior.dat"),
        transfer_files=[],
        use_osg=True,
        use_singularity=True,
        singularity_image=manifest,
    )
    return job, dict(job.condor_cmds)


def test_calibration_family_legacy_uses_fallback_not_manifest(tmp_path, monkeypatch):
    manifest = _write(tmp_path, ALL_OSDF_MANIFEST)
    job, cmds = _make_calibration_job(tmp_path, monkeypatch, manifest, False)
    assert job.universe == "vanilla"
    assert cmds["MY.SingularityImage"] == '"./rift_ancient_cuda11.sif"'
    assert manifest not in cmds["MY.SingularityImage"]
    assert "ifThenElse" not in cmds["MY.SingularityImage"]
    assert cmds["transfer_input_files"].count(
        "osdf:///igwn/sw/rift_ancient_cuda11.sif"
    ) == 1
    assert "osdf:///igwn/sw/rift_modern_cuda12.sif" not in cmds["transfer_input_files"]


def test_calibration_family_container_universe_uses_fallback_not_manifest(
    tmp_path, monkeypatch
):
    manifest = _write(tmp_path, ALL_OSDF_MANIFEST)
    job, cmds = _make_calibration_job(tmp_path, monkeypatch, manifest, True)
    assert job.universe == "container"
    assert cmds["container_image"] == "osdf:///igwn/sw/rift_ancient_cuda11.sif"
    assert manifest not in cmds["container_image"]
    assert "MY.SingularityImage" not in cmds
    assert "MY.SingularityBindCVMFS" not in cmds
    assert "$$(" not in cmds["container_image"]
    assert "rift_modern_cuda12.sif" not in cmds["transfer_input_files"]

    # Exercise the same emission path used by a build-only pseudo-pipe run.
    job.write_sub_file()
    submit = (tmp_path / "Calib_reweight.sub").read_text()
    assert "universe = container" in submit
    assert "container_image = osdf:///igwn/sw/rift_ancient_cuda11.sif" in submit
    assert "fam.yaml" not in submit


# ---------------------------------------------------------------------------
# 7. runtime-selection wrapper fallback
# ---------------------------------------------------------------------------

def test_runtime_wrapper_text_contents(tmp_path):
    m = cm.load_container_manifest(_write(tmp_path, MIXED_MANIFEST))
    text = cm.build_runtime_selection_wrapper(m, inner_command="./ile_pre.sh")
    assert text.startswith("#!/bin/bash")
    assert '"./rift_modern_cuda12.sif"' in text
    assert '"/cvmfs/sw/rift_ancient_cuda11.sif"' in text
    assert '"osdf:///igwn/rift_modern_cuda12.sif"' in text
    assert 'FALLBACK_LABEL="ancient"' in text
    assert 'INNER_COMMAND="./ile_pre.sh"' in text
    bash = shutil.which("bash")
    if bash:
        r = subprocess.run([bash, "-n", "-c", text], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not shutil.which("bash") or not shutil.which("awk"),
                    reason="needs bash + awk to exercise the wrapper")
def test_runtime_wrapper_selects_by_capability(tmp_path):
    anc = tmp_path / "anc.sif"; anc.write_text("x")
    mod = tmp_path / "mod.sif"; mod.write_text("x")
    manifest_text = textwrap.dedent(
        """
        version: 1
        fallback: ancient
        containers:
          - label: ancient
            image: {anc}
            cuda_capability_min: 3.0
            cuda_capability_max: 7.0
          - label: modern
            image: {mod}
            cuda_capability_min: 7.0
        """
    ).format(anc=anc, mod=mod)
    m = cm.load_container_manifest(_write(tmp_path, manifest_text))
    wrapper = tmp_path / "select.sh"
    wrapper.write_text(cm.build_runtime_selection_wrapper(m, inner_command="/bin/true"))
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)

    fakebin = tmp_path / "bin"; fakebin.mkdir()
    fake = fakebin / "apptainer"
    fake.write_text('#!/bin/bash\necho "APPTAINER $*"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH="{}:{}".format(fakebin, os.environ["PATH"]))

    def run(cap):
        env2 = dict(env, RIFT_CONTAINER_FORCE_CAP=cap)
        return subprocess.run([str(wrapper)], capture_output=True, text=True, env=env2)

    r = run("12.0")
    assert r.returncode == 0, r.stderr
    assert "selected: modern" in r.stderr
    r = run("5.0")
    assert "selected: ancient" in r.stderr
    r = run("2.0")
    assert "fallback" in r.stderr and "selected: ancient" in r.stderr


def test_integration_runtime_select(tmp_path, monkeypatch):
    monkeypatch.delenv("RIFT_CONTAINER_UNIVERSE", raising=False)
    monkeypatch.setenv("RIFT_CONTAINER_RUNTIME_SELECT", "1")
    monkeypatch.delenv("RIFT_REQUIRE_GPUS", raising=False)
    monkeypatch.setenv("SINGULARITY_BASE_EXE_DIR", "/opt/rift/bin/")
    cmds = _make_ile_job(tmp_path, monkeypatch, _write(tmp_path, MIXED_MANIFEST))

    assert "MY.SingularityImage" not in cmds
    assert "MY.SingularityBindCVMFS" not in cmds
    assert "transfer_executable" not in cmds
    assert "$$([" not in cmds.get("transfer_input_files", "")
    assert "Capability >= 3.0" in cmds["require_gpus"]

    wrapper = tmp_path / "rift_container_select.sh"
    assert wrapper.exists()
    body = wrapper.read_text()
    assert body.startswith("#!/bin/bash")
    assert 'INNER_COMMAND="/opt/rift/bin/true"' in body


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-v"]))
