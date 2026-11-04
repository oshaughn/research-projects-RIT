"""Importance evidence must average over every draw from the proposal."""

import ast
import os

import numpy as np
import pytest

from RIFT.likelihood.jax_ile.samplers import evidence_from_logweights

_CODE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_DRIVER = os.path.join(_CODE, "bin", "integrate_likelihood_extrinsic_jax")


def test_out_of_support_draws_are_zero_weight_not_removed():
    # Two of four Gaussian proposal draws miss the prior.  Their weights are
    # zero, so Z is (2 + 2 + 0 + 0) / 4 = 1, not 2.
    logZ, sigma_over_Z, neff = evidence_from_logweights(
        [np.log(2.), np.log(2.), -np.inf, -np.inf])
    assert logZ == pytest.approx(0.)
    assert sigma_over_Z == pytest.approx(0.5)
    assert neff == pytest.approx(2.)


def test_constant_and_all_zero_weights():
    logZ, sigma_over_Z, neff = evidence_from_logweights([3., 3.])
    assert logZ == pytest.approx(3.)
    assert sigma_over_Z == pytest.approx(0.)
    assert neff == pytest.approx(2.)
    logZ, sigma_over_Z, neff = evidence_from_logweights([-np.inf, -np.inf])
    assert logZ == -np.inf
    assert sigma_over_Z == np.inf
    assert neff == 0.


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonzero_nonfinite_weights_fail_closed(bad):
    logZ, sigma_over_Z, neff = evidence_from_logweights([0., bad])
    assert np.isnan(logZ)
    assert np.isnan(sigma_over_Z)
    assert neff == 0.


def test_the_driver_uses_this_estimator_and_keeps_no_copy():
    """WIRING, and the reason the cases above gate the production CLI.

    The driver carried its OWN ``evidence_from_logweights`` -- the pre-fix one,
    which dropped ``-inf`` weights before averaging -- so fixing the library left
    ``--mode laplace-is`` (the default), the NUTS importance estimate and the
    prior pilot biased high.  A value test cannot see that: it imports the
    library copy, which is already correct.  So read the driver source instead,
    and pin BOTH halves -- the import is present AND no local definition shadows
    it -- because either one alone is satisfied by the broken arrangement.
    """
    with open(_DRIVER) as f:
        tree = ast.parse(f.read())

    local = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "evidence_from_logweights"]
    assert not local, (
        "the driver defines its own evidence_from_logweights; a second copy of the "
        "estimator is a copy that drifts, and the copy it had removed zero-weight "
        "(out-of-prior-support) draws before averaging, biasing lnZ upward")

    imported = [n for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom)
                and n.module == "RIFT.likelihood.jax_ile.samplers"
                and any(a.name == "evidence_from_logweights" and a.asname is None
                        for a in n.names)]
    assert len(imported) == 1, (
        "the driver must import evidence_from_logweights from "
        "RIFT.likelihood.jax_ile.samplers under that exact name, so its estimator "
        "is the one this file tests")
