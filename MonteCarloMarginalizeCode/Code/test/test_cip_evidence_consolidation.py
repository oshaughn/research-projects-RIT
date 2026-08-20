"""Regression tests for terminal CIP evidence and prior normalization.

The numerical consolidator is imported by path so these tests need only numpy.
Pipeline-driver checks are structural because importing either driver executes
its argparse and requires the full RIFT science stack.
"""

import importlib.util
import os

import numpy as np
import pytest


HERE = os.path.dirname(__file__)
CODE = os.path.abspath(os.path.join(HERE, os.pardir))
SUMMARIZER = os.path.join(CODE, "bin", "util_CIPDirSummarizeEvidence.py")
CIP = os.path.join(CODE, "bin", "util_ConstructIntrinsicPosterior_GenericCoordinates.py")
PIPELINES = [
    os.path.join(CODE, "bin", "create_event_parameter_pipeline_BasicIteration"),
    os.path.join(CODE, "bin", "cepp_basic_htcondor"),
]


spec = importlib.util.spec_from_file_location("cip_evidence", SUMMARIZER)
cip_evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cip_evidence)


def _write_annotation(path, ln_z, sigma, neff=100):
    np.savetxt(path, [[ln_z, sigma, neff]], header=" lnL sigmaL neff")


def _write_worker(cip_dir, worker, ln_z, sigma):
    base = cip_dir / ("overlap-grid-4-{}+annotation.dat".format(worker))
    alt = cip_dir / ("overlap-grid-4-{}_withpriorchange+annotation.dat".format(worker))
    _write_annotation(base, ln_z - 1, sigma)
    _write_annotation(alt, ln_z, sigma, neff=200 + worker)


def test_consolidation_preserves_established_weighting_and_scatter(tmp_path):
    cip_dir = tmp_path / "iteration_3_cip"
    cip_dir.mkdir()
    _write_worker(cip_dir, 0, 10.0, 0.5)
    _write_worker(cip_dir, 1, 12.0, 1.0)

    result = cip_evidence.consolidate_cip_directory(str(cip_dir), strict=True)

    assert result["lnZ"] == pytest.approx(10.4)
    # Historical prescription takes the larger of error-of-mean and worker scatter.
    assert result["sigma_lnZ"] == pytest.approx(1.0)
    assert result["n_workers"] == 2


def test_strict_terminal_mode_rejects_missing_or_unpaired_workers(tmp_path):
    with pytest.raises(ValueError, match="No files"):
        cip_evidence.consolidate_cip_directory(str(tmp_path), strict=True)

    base = tmp_path / "overlap-grid-4-0+annotation.dat"
    _write_annotation(base, 10, 0.2)
    with pytest.raises(ValueError, match="cannot read"):
        cip_evidence.consolidate_cip_directory(str(tmp_path), strict=True)


def test_normalized_output_is_log_ratio_with_propagated_mc_error(tmp_path):
    cip_dir = tmp_path / "iteration_2_cip"
    cip_dir.mkdir()
    _write_worker(cip_dir, 0, 14.0, 0.3)
    prior = tmp_path / "prior.dat"
    _write_annotation(prior, 2.0, 0.4, neff=321)
    raw = tmp_path / "evidence_2"
    normalized = tmp_path / "evidence_2_normalized"

    rc = cip_evidence.main([
        "--cip-dir", str(cip_dir), "--strict", "--output", str(raw),
        "--prior-integral", str(prior),
        "--normalized-output", str(normalized),
    ])

    assert rc == 0
    np.testing.assert_allclose(np.loadtxt(raw), [14.0, 0.3])
    row = np.loadtxt(normalized)
    np.testing.assert_allclose(row[:6], [14.0, 0.3, 2.0, 0.4, 12.0, 0.5])
    np.testing.assert_allclose(row[6:], [1.0, 321.0])


def test_non_exploded_cip_uses_its_top_level_output_prefix(tmp_path):
    prefix = tmp_path / "overlap-grid-3"
    _write_annotation(str(prefix) + "+annotation.dat", 8.0, 0.25)
    _write_annotation(str(prefix) + "_withpriorchange+annotation.dat", 9.0, 0.25)

    result = cip_evidence.consolidate_cip_directory(
        str(tmp_path), strict=True, cip_prefix=str(prefix))

    assert result["lnZ"] == pytest.approx(9.0)
    assert result["n_workers"] == 1


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_pipeline_has_terminal_prior_then_strict_final_evidence(pipeline):
    source = open(pipeline).read()
    loop = source.index("for it in np.arange(it_start,opts.n_iterations):")
    terminal = source.index("final_iteration = opts.n_iterations - 1", loop)
    export = source.index("# Create export stages", terminal)

    assert loop < terminal < export
    assert "prior_node.add_parent(parent_fit_node)" in source[terminal:export]
    assert "final_evidence_node.add_parent(prior_node)" in source[terminal:export]
    assert "--prior-integral prior-integral-$(macroiteration)_withpriorchange+annotation.dat" in source
    assert "--normalized-output evidence_$(macroiteration)_normalized" in source
    assert "--cip-prefix overlap-grid-$(macroiterationnext)" in source


def test_prior_mode_is_independent_and_reweighted_evidence_restores_shift():
    source = open(CIP).read()
    assert 'parser.add_argument("--integrate-prior"' in source
    assert "if opts.integrate_prior:" in source
    assert "replacing the fitted likelihood by L=1" in source
    assert "supplemental_ln_likelihood and not opts.integrate_prior" in source
    assert ("log_res_reweighted = lnLmax + np.log(np.mean(weights)) + "
            "supplemental_ln_likelihood_offset + lnL_shift") in source
