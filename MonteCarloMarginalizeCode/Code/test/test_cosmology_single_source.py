#!/usr/bin/env python
"""
One named cosmology, from one helper, across every code in the tree.

RO, 2026-08-16: *"move it to the helper, so it is consistent by default and changed in a
consistent fashion between codes; not hardcoded.  Agree minute effect, but the sort of random
complaint people do make in refereeing reports."*

Three files used to build their own: the ILE driver and `util_InitMargTable` (both
`FlatLambdaCDM` from `lal.H0_SI`/`lal.OMEGA_M` = H0 67.900, Om0 0.3065, with a hardcoded
fallback), and `resample_uniform_comoving` (`LambdaCDM(H0=67.90, ...)`, named `Planck15_lal`
because it deliberately reproduced the ILE's cosmology to ~1e-12).  They all now ask
`priors_utils.get_astropy_cosmology("Planck15")`.

WHY THAT MATTERED MORE THAN 0.05%.  The three are coupled:
  * the ILE driver builds the distance prior for the UNmarginalized path and
    `util_InitMargTable` for the MARGINALIZED one, and `helper_LDG_Events` hands both the
    same `--d-prior` -- so a disagreement means identical CLI gives two different priors
    depending only on `--internal-marginalize-distance`;
  * `resample_uniform_comoving` DIVIDES OUT the prior the ILE imposed, so the two only cancel
    if they are the same object.
For one commit the ILE driver moved to the helper and the other two did not, which created
both defects at once.  An adversarial review found it.

WHY THIS FILE IS A SWEEP, NOT A LIST.  The first version of these tests parametrized over a
hand-written TARGETS list that named the LISA driver (which has no cosmology at all, so those
cases were vacuous and could never fail) and omitted the two files that actually violated the
property.  A second version discovered TARGETS by looking for files that MENTION a cosmology
class -- which goes vacuous the moment the last construction is removed.  So: sweep every
file, assert the construction count is zero, and assert the sweep itself saw a plausible
number of files.  A hand-maintained list of what to check is the same mistake as a
hand-maintained cosmology.
"""

import ast
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.abspath(os.path.join(_HERE, '..'))

# Cosmology classes it is a defect to instantiate outside the helper.
_COSMO_CLASSES = ('FlatLambdaCDM', 'LambdaCDM', 'wCDM', 'FlatwCDM', 'w0waCDM', 'w0wzCDM')

# The helper IS the source of truth, so it may name and return these freely.
_ALLOWED = ('RIFT/likelihood/priors_utils.py',)

_LAL_H0_LITERAL = '2.200489137532724e-18'


def _python_files():
    """Every python source under bin/ and RIFT/, tests excluded.

    bin/ holds extensionless executables, so selection is by successful parse rather than by
    suffix -- picking only *.py would skip util_InitMargTable, which is one of the files this
    exists to police.
    """
    out = []
    for sub in ('bin', 'RIFT'):
        for root, _dirs, files in os.walk(os.path.join(_CODE, sub)):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), _CODE)
                if rel in _ALLOWED:
                    continue
                if f.endswith(('.pyc', '.ipynb', '.txt', '.md', '.xml', '.dat', '.png')):
                    continue
                if os.sep + 'test' in os.sep + rel or rel.startswith('test'):
                    continue
                try:
                    with open(os.path.join(_CODE, rel)) as fh:
                        src = fh.read()
                    ast.parse(src)
                except (IOError, OSError, UnicodeDecodeError, SyntaxError, ValueError):
                    continue
                out.append((rel, src))
    return out


@pytest.fixture(scope="module")
def sources():
    return _python_files()


def test_the_sweep_covers_a_plausible_number_of_files(sources):
    """A broken walk returning [] would make every assertion below vacuous."""
    assert len(sources) > 50, (
        "the sweep parsed only %d files; it is not covering the tree" % len(sources))
    rels = {r for r, _ in sources}
    for expect in ('bin/integrate_likelihood_extrinsic_batchmode',
                   'bin/util_InitMargTable',
                   'bin/resample_uniform_comoving.py'):
        assert expect in rels, "the sweep missed %s, which it exists to police" % expect


def test_nothing_constructs_its_own_cosmology(sources):
    """Matches BOTH call forms: bare name and `astropy.cosmology.FlatLambdaCDM(...)`."""
    offenders = []
    for rel, src in sources:
        for n in ast.walk(ast.parse(src)):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            name = (f.id if isinstance(f, ast.Name)
                    else f.attr if isinstance(f, ast.Attribute) else None)
            if name in _COSMO_CLASSES:
                offenders.append("%s:%d (%s)" % (rel, n.lineno, name))
    assert not offenders, (
        "these build their own cosmology instead of calling "
        "priors_utils.get_astropy_cosmology():\n  " + "\n  ".join(offenders))


def test_nothing_hardcodes_the_lal_H0_constant(sources):
    offenders = ["%s" % rel for rel, src in sources if _LAL_H0_LITERAL in src]
    assert not offenders, (
        "these hardcode an H0 that cannot be cited by name in a paper: %s" % offenders)


def test_the_framework_helper_defaults_to_Planck15():
    import inspect

    import RIFT.likelihood.priors_utils as priors_utils
    assert inspect.signature(priors_utils.get_astropy_cosmology).parameters['name'].default \
        == 'Planck15'
    c = priors_utils.get_astropy_cosmology()
    assert abs(c.H0.value - 67.74) < 0.01 and abs(c.Om0 - 0.3075) < 0.001


@pytest.mark.parametrize("rel", ['bin/integrate_likelihood_extrinsic_batchmode',
                                 'bin/util_InitMargTable',
                                 'bin/resample_uniform_comoving.py'])
def test_the_coupled_three_all_ask_the_helper(rel):
    """Named explicitly because these three must agree with EACH OTHER, not merely avoid
    hardcoding: two build the distance prior for the two marginalization paths, and the third
    divides that prior out again."""
    with open(os.path.join(_CODE, rel)) as fh:
        src = fh.read()
    assert 'get_astropy_cosmology("Planck15")' in src, \
        "%s does not ask the helper for its cosmology" % rel
