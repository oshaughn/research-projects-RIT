"""The (phi_ref, psi) scheme default has exactly ONE definition, and one
look-alike literal that must NOT follow it.

Changed 2026-09-02: 'grid' -> 'exact'.  The previous default move on this path
(interp linear -> sinc) was bitten by the value being re-typed in many places,
and by two independent things that happened to be the same string.  Both hazards
are pinned here.
"""
import importlib.machinery
import importlib.util
import inspect
import os
import re

import pytest

from RIFT.likelihood.jax_ile.anglemarg import (ANGLE_MARG_CHOICES,
                                               ANGLE_MARG_DEFAULT,
                                               ANGLE_MARG_LEGACY)
from RIFT.likelihood.jax_ile.samplers import angle_marg_eval_chunk
from RIFT.likelihood.jax_ile.wrapper import JAXDistPhiPsiMargLikelihood

_CODE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_DRIVER = os.path.join(_CODE, "bin", "integrate_likelihood_extrinsic_jax")


def _driver():
    loader = importlib.machinery.SourceFileLoader("_amd_drv", _DRIVER)
    spec = importlib.util.spec_from_loader("_amd_drv", loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = "_amd_drv"          # keep the __main__ guard from firing
    loader.exec_module(mod)
    return mod


def _parse(argv):
    optp = _driver().build_parser()
    opts, _ = optp.parse_args(list(argv))
    return opts


def test_default_has_a_single_definition():
    """Driver flag, library argument and the constant must all agree."""
    parsed = _parse(["--inj-mode", "--mass1", "35", "--mass2", "30"])
    lib = inspect.signature(
        JAXDistPhiPsiMargLikelihood.__init__).parameters["angle_marg"].default
    assert parsed.angle_marg_scheme == ANGLE_MARG_DEFAULT
    assert lib == ANGLE_MARG_DEFAULT
    assert ANGLE_MARG_DEFAULT in ANGLE_MARG_CHOICES


def test_legacy_spelling_still_reachable():
    """The pre-change behaviour must remain reproducible by an explicit flag."""
    assert ANGLE_MARG_LEGACY == "grid"
    assert ANGLE_MARG_LEGACY in ANGLE_MARG_CHOICES
    assert ANGLE_MARG_LEGACY != ANGLE_MARG_DEFAULT
    opts = _parse(["--inj-mode", "--mass1", "35", "--mass2", "30",
                   "--angle-marg-scheme", ANGLE_MARG_LEGACY])
    assert opts.angle_marg_scheme == ANGLE_MARG_LEGACY


def test_unknown_scheme_is_rejected_at_parse_time():
    """A typo must die at argument parsing, not minutes later after precompute."""
    with pytest.raises(SystemExit):
        _parse(["--angle-marg-scheme", "definitely-not-a-scheme"])


def test_eval_chunk_sentinel_does_not_follow_the_default():
    """`angle_marg_eval_chunk`'s "grid" is a SENTINEL, not the default.

    A likelihood with no ``angle_marg_scheme`` at all (JAXDistanceMarginalized,
    JAXExtrinsic) runs no dense angle scheme and must pass its chunk through
    UNCHANGED.  If someone syncs that literal to ANGLE_MARG_DEFAULT the chunk
    gets capped for every such object -- a memory/throughput regression with no
    benefit.  This is the "two independent defaults, same string" trap.
    """
    class _Data:
        # npts must be large enough that the cap actually bites, or a SECOND
        # early return (npts <= 0) masks the scheme check and this test passes
        # for the wrong reason -- it did, and a mutation sweep caught it.
        npts = 65537

    class _NoScheme:            # no angle_marg_scheme attribute at all
        data = _Data()

    class _GridScheme:
        angle_marg_scheme = "grid"
        data = _Data()

    class _ExactScheme:
        angle_marg_scheme = "exact"
        data = _Data()

    # Positive control: the cap DOES bite for a dense scheme at this npts, so a
    # pass-through below is discrimination and not an inert code path.
    capped = angle_marg_eval_chunk(_ExactScheme(), 4096)
    assert capped < 4096, (
        "cap did not engage at npts=%d; this test cannot discriminate"
        % _Data.npts)

    assert angle_marg_eval_chunk(_NoScheme(), 4096) == 4096
    assert angle_marg_eval_chunk(_GridScheme(), 4096) == 4096
    assert ANGLE_MARG_DEFAULT == "exact", (
        "if the default is no longer 'exact', re-derive this test's premise")


def test_no_retyped_default_literal_in_driver_or_wrapper():
    """Neither entry point may spell the default as a bare literal."""
    for path in (_DRIVER,
                 os.path.join(_CODE, "RIFT", "likelihood", "jax_ile",
                              "wrapper.py")):
        with open(path) as fh:
            src = fh.read()
        assert not re.search(r'angle_marg_scheme"\s*,\s*"grid"', src), (
            "%s re-types the angle-marg default as a literal" % path)
        assert not re.search(r'angle_marg\s*=\s*"(grid|exact|laplace|auto)"',
                             src), (
            "%s re-types the angle-marg default as a literal" % path)
