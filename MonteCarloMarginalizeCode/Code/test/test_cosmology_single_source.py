#!/usr/bin/env python
"""
One named cosmology, from one helper, across the codes.

RO, 2026-08-16: *"move it to the helper, so it is consistent by default and changed in a
consistent fashion between codes; not hardcoded.  Agree minute effect, but the sort of random
complaint people do make in refereeing reports."*

The ILE driver used to build its own `FlatLambdaCDM` from `lal.H0_SI`/`lal.OMEGA_M`, with a
hardcoded pair as fallback.  That is a cosmology nobody can cite by name: the installed lal
gives H0=67.900, Om0=0.3065 while Planck15 is H0=67.740, Om0=0.3075.  The difference is
physically negligible (dL(z=5) 47756 vs 47732 Mpc, 0.05%) -- the point is answerability, and
that a change should happen in ONE place for every code that needs a cosmology.

These tests are cheap and general: they say "ask the helper, do not roll your own", which is
the property that keeps the two ILE drivers (and CIP, and anything else) from quietly
disagreeing about z.
"""

import ast
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.join(_HERE, '..')

# Files that legitimately need a cosmology.  The helper itself is excluded: it IS the source.
TARGETS = [
    'bin/integrate_likelihood_extrinsic_batchmode',
    'bin/integrate_likelihood_extrinsic_batchmode_lisa',
]


def _src(rel):
    with open(os.path.join(_CODE, rel)) as fh:
        return fh.read()


def test_the_framework_helper_exists_and_defaults_to_Planck15():
    import RIFT.likelihood.priors_utils as priors_utils
    import inspect
    sig = inspect.signature(priors_utils.get_astropy_cosmology)
    assert sig.parameters['name'].default == 'Planck15'
    cosmo = priors_utils.get_astropy_cosmology()
    assert abs(cosmo.H0.value - 67.74) < 0.01 and abs(cosmo.Om0 - 0.3075) < 0.001


@pytest.mark.parametrize("rel", TARGETS)
def test_no_driver_constructs_its_own_cosmology(rel):
    """FlatLambdaCDM(...) built by hand is the thing being removed."""
    try:
        src = _src(rel)
    except IOError:
        pytest.skip("%s not present" % rel)
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in ('FlatLambdaCDM', 'LambdaCDM', 'wCDM')]
    assert not calls, (
        "%s builds its own cosmology at line(s) %s; call "
        "priors_utils.get_astropy_cosmology() instead so every code shares one and a change "
        "is made once" % (rel, [c.lineno for c in calls]))


@pytest.mark.parametrize("rel", TARGETS)
def test_no_driver_hardcodes_the_lal_cosmology_constants(rel):
    try:
        src = _src(rel)
    except IOError:
        pytest.skip("%s not present" % rel)
    assert "2.200489137532724e-18" not in src, \
        "%s hardcodes an H0; that value cannot be cited by name in a paper" % rel


def test_the_ILE_driver_asks_the_helper():
    src = _src('bin/integrate_likelihood_extrinsic_batchmode')
    assert 'priors_utils.get_astropy_cosmology("Planck15")' in src
