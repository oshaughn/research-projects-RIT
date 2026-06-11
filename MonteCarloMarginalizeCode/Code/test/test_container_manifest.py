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
    m = cm.load_container_manifest(_write(tmp_path, MIXED_MANIFEST))
    expr = cm.build_container_image_select(m)
    # a $$() match-time substitution token, undefined-safe, with VERBATIM image
    # values (osdf URL fetched by container universe; cvmfs path used in place) --
    # NOT a ./basename rewrite
    assert expr.startswith("$$([ ") and expr.endswith(" ])")
    assert "TARGET.GPUs_Capability =?= undefined" in expr          # undefined -> fallback
    assert "ifThenElse(TARGET.GPUs_Capability >= 7.0," in expr
    assert '"osdf:///igwn/rift_modern_cuda12.sif"' in expr         # raw osdf URL
    assert '"/cvmfs/sw/rift_ancient_cuda11.sif"' in expr           # fallback verbatim
    assert "./rift_modern_cuda12.sif" not in expr                  # no basename rewrite


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
        singularity_image=_write(tmp_path, MIXED_MANIFEST),
        request_gpu=True,
        cache_file="local.cache",
    )
    cmds = dict(job.condor_cmds)

    ci = cmds["container_image"]
    assert ci.startswith("$$([")               # match-time substitution, unquoted
    assert not ci.startswith('"')
    assert "MY.SingularityImage" not in cmds    # the OSG-breaking attr is gone
    assert "MY.SingularityBindCVMFS" not in cmds
    assert "$$([" not in cmds.get("transfer_input_files", "")  # image via container_image, not transfer
    assert "Capability >= 3.0" in cmds["require_gpus"]         # floor still steers GPUs

    assert job.universe == "container"          # HTCondor container universe


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-v"]))
