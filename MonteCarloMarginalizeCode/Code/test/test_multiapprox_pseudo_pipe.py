"""The multi-approximant builder must satisfy pseudo_pipe's API, or refuse.

pseudo_pipe emits ONE command line, shaped for BasicIteration's option surface.
BasicMultiApproxIteration implements cross-model marginalization, not every
feature of the single-model pipeline, so it accepts the rest for compatibility
and refuses any that is actually used.

The failure this guards is drift: BasicIteration gains an option, pseudo_pipe
starts emitting it, and the multi-approximant builder dies on an unrecognized
argument -- or worse, someone "fixes" that by ignoring it, and runs quietly stop
matching their configuration.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
BIN = CODE / "bin"
BASIC = BIN / "create_event_parameter_pipeline_BasicIteration"
MULTI = BIN / "create_event_parameter_pipeline_BasicMultiApproxIteration"
PSEUDO = BIN / "util_RIFT_pseudo_pipe.py"

ADD_ARG = r'add_argument\(\s*["\'](--[a-z0-9\-]+)["\']'


def declared_options(path):
    return set(re.findall(ADD_ARG, path.read_text()))


def api_only_lists():
    """The two lists the builder hardcodes."""
    text = MULTI.read_text()
    out = {}
    for name in ("_API_ONLY_FLAGS", "_API_ONLY_VALUED"):
        m = re.search(re.escape(name) + r"\s*=\s*(\[[^\]]*\])", text, re.S)
        assert m, "{} not found in the builder".format(name)
        out[name] = set(eval(m.group(1)))          # a literal list of strings
    return out["_API_ONLY_FLAGS"], out["_API_ONLY_VALUED"]


def test_builder_covers_every_option_basiciteration_declares():
    """No option pseudo_pipe could emit may be unknown to the multi builder.

    Re-derived from source, so adding an option to BasicIteration without
    deciding what the multi-approximant path does with it fails here rather
    than at DAG-build time in someone's campaign.
    """
    basic = declared_options(BASIC)
    multi = declared_options(MULTI)
    flags, valued = api_only_lists()
    covered = multi | flags | valued
    missing = sorted(basic - covered)
    assert not missing, (
        "BasicIteration declares options the multi-approximant builder neither "
        "implements nor accepts for API compatibility: {}. Decide for each: "
        "implement it, or add it to _API_ONLY_FLAGS/_API_ONLY_VALUED so it is "
        "refused explicitly.".format(missing))


def test_api_only_lists_do_not_claim_implemented_options():
    """An option cannot be both implemented and refused."""
    multi = declared_options(MULTI)
    flags, valued = api_only_lists()
    overlap = sorted((flags | valued) & multi)
    assert not overlap, (
        "these are declared normally AND listed as API-only, so they would be "
        "refused despite being implemented: {}".format(overlap))


def test_api_only_flag_is_refused_when_used(tmp_path):
    """Accepted at parse time, refused when actually set -- never ignored."""
    out = subprocess.run(
        [sys.executable, str(MULTI), "--approx", "IMRPhenomD",
         "--n-iterations-subdag-max", "5"],
        cwd=str(tmp_path), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert out.returncode != 0, "an unimplemented option was accepted silently"
    assert "does" in out.stdout and "NOT implement" in out.stdout, out.stdout[-800:]
    assert "--n-iterations-subdag-max" in out.stdout


def test_pseudo_pipe_offers_and_routes_to_the_builder():
    text = PSEUDO.read_text()
    assert '"BasicMultiApproxIteration"' in text, (
        "--pipeline-builder does not offer the multi-approximant builder")
    assert "--approx-extra" in text, "pseudo_pipe has no --approx-extra"
    # asking for a second model must select the builder without a second flag
    assert re.search(r"approx_extra and not opts\.pipeline_builder", text), (
        "--approx-extra does not imply --pipeline-builder BasicMultiApproxIteration")
    # and the models must actually reach the builder's command line
    assert re.search(r'--approx \{\}\s*"\s*\.format\(_ap\)', text), (
        "pseudo_pipe never emits --approx per model")


def test_multiapprox_without_a_second_model_is_refused():
    """The cross-model builder with one model is a misconfiguration, not a run."""
    text = PSEUDO.read_text()
    assert re.search(r"use_multiapprox and not opts\.approx_extra", text), (
        "pseudo_pipe does not check that the multi builder has >1 model")
