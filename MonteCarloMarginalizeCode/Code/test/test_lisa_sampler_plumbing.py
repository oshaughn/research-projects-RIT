#!/usr/bin/env python
"""
Tests for the portfolio freeze/allocation policy ported into the LISA ILE driver
(bin/integrate_likelihood_extrinsic_batchmode_lisa).

This is pure PASS-THROUGH plumbing to samplers the LISA driver already wires -- it exposes
the same ``ok_lnL_methods`` as the main driver (``GMM, adaptive_cartesian,
adaptive_cartesian_gpu, AV, portfolio``, verified identical) and builds mcsamplerPortfolio
the same way.  Before this port the knobs were reachable only through
``--sampler-portfolio-args``, an eval-able dict; the pipeline passes the named flags.

WHAT CAN ACTUALLY GO WRONG HERE, and is therefore what these tests check:

  * a default that differs between the two drivers.  Worse than a missing option: the same
    command line then means two different things depending on which driver ran it.
  * an option that is UNSET leaking into the kwargs as ``None`` and overriding the sampler's
    own default with nothing.  The assembly's whole shape -- ``if opts.x is not None`` --
    exists for that, and a single dropped guard is invisible until a run behaves oddly.
  * the two mutually-exclusive VARAHA flags resolving the wrong way round.

The freeze-policy assembly is inline in both drivers (not a function), so it is exercised
here by extracting the block and exec'ing it against a fake ``opts``.  That tests the real
source, not a paraphrase of it.
"""

import ast
import os
import re
import textwrap

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_LISA = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode_lisa')
_MAIN = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')

PORTFOLIO_OPTS = [
    "--portfolio-adaptive-alloc", "--portfolio-alloc-exponent", "--portfolio-freeze-wt",
    "--portfolio-grace-iters", "--portfolio-probe-period", "--portfolio-quality-signal",
    "--portfolio-revive-period", "--portfolio-varaha-can-freeze",
    "--portfolio-varaha-max-frac", "--portfolio-varaha-min-frac",
    "--portfolio-varaha-never-freeze", "--portfolio-weight-clip",
]


def _src(path):
    with open(path) as fh:
        return fh.read()


def _option_nodes(path):
    """{'--foo': ast.Call} for every add_option in a driver."""
    out = {}
    for n in ast.walk(ast.parse(_src(path), filename=path)):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("add_option", "add_argument")):
            names = [a.value for a in n.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if names and names[0].startswith("--"):
                out[names[0]] = n
    return out


def _kwargs_of(node):
    out = {}
    for kw in node.keywords:
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            out[kw.arg] = ast.dump(kw.value)
    return out


@pytest.fixture(scope="module")
def opts_lisa():
    return _option_nodes(_LISA)


@pytest.fixture(scope="module")
def opts_main():
    return _option_nodes(_MAIN)


# ------------------------------------------------------------------------------ presence
@pytest.mark.parametrize("opt", PORTFOLIO_OPTS)
def test_option_is_present_in_the_lisa_driver(opt, opts_lisa):
    assert opt in opts_lisa


# ------------------------------------------------------------------------------- defaults
@pytest.mark.parametrize("opt", PORTFOLIO_OPTS)
def test_option_signature_matches_the_main_driver(opt, opts_lisa, opts_main):
    """Same default, same type, same action.

    A knob that means something different in the two drivers is worse than a missing one:
    the same pipeline command line would then produce two different integrations.
    """
    a, b = _kwargs_of(opts_lisa[opt]), _kwargs_of(opts_main[opt])
    for key in ("default", "type", "action", "choices"):
        assert a.get(key) == b.get(key), (
            "%s: %s differs (lisa=%r, main=%r)" % (opt, key, a.get(key), b.get(key)))


@pytest.mark.parametrize("opt", [
    "--portfolio-alloc-exponent", "--portfolio-freeze-wt", "--portfolio-grace-iters",
    "--portfolio-probe-period", "--portfolio-quality-signal", "--portfolio-revive-period",
    "--portfolio-varaha-max-frac", "--portfolio-varaha-min-frac", "--portfolio-weight-clip",
])
def test_tuning_options_default_to_none_so_the_sampler_keeps_its_own(opt, opts_lisa):
    """None is the sentinel the assembly keys on.  A default of 0/0.0 would silently
    override the sampler's built-in value for every run that never set the flag."""
    assert _kwargs_of(opts_lisa[opt]).get("default") is None


@pytest.mark.parametrize("opt", [
    "--portfolio-adaptive-alloc", "--portfolio-varaha-can-freeze",
    "--portfolio-varaha-never-freeze",
])
def test_flags_are_store_true_and_default_false(opt, opts_lisa):
    kw = _kwargs_of(opts_lisa[opt])
    assert kw.get("action") == "store_true" and kw.get("default") is False


# --------------------------------------------------------- the assembly block, executed
_START = "_freeze_policy_kwargs = {}"
_END = 'print(" PORTFOLIO freeze-policy overrides: "'


def _assembly_block(path):
    """The inline freeze-policy assembly, dedented so it can be exec'd on its own.

    Slice from the START OF THE LINE holding the sentinel, not from the sentinel itself:
    otherwise the first line carries no indentation while the rest do, and dedent finds no
    common prefix.
    """
    src = _src(path)
    i = src.rindex("\n", 0, src.index(_START)) + 1
    j = src.index(_END, i)
    j = src.rindex("\n", i, j) + 1
    return textwrap.dedent(src[i:j])


class _Opts(object):
    """Every portfolio option at its documented default."""
    portfolio_grace_iters = None
    portfolio_revive_period = None
    portfolio_freeze_wt = None
    portfolio_varaha_can_freeze = False
    portfolio_varaha_never_freeze = False
    portfolio_adaptive_alloc = False
    portfolio_varaha_min_frac = None
    portfolio_varaha_max_frac = None
    portfolio_weight_clip = None
    portfolio_quality_signal = None
    portfolio_alloc_exponent = None
    portfolio_probe_period = None

    def __init__(self, **kw):
        for k, v in kw.items():
            assert hasattr(type(self), k), "unknown option %s" % k
            setattr(self, k, v)


def _assemble(**kw):
    ns = {"opts": _Opts(**kw)}
    exec(compile(_assembly_block(_LISA), "freeze_policy", "exec"), ns)
    return ns["_freeze_policy_kwargs"]


def test_nothing_set_means_nothing_overridden():
    """The important one: an all-defaults run must not touch the sampler's policy at all."""
    assert _assemble() == {}


def test_each_tuning_option_passes_through_when_set():
    got = _assemble(portfolio_grace_iters=7, portfolio_revive_period=3,
                    portfolio_freeze_wt=0.25, portfolio_varaha_min_frac=0.2,
                    portfolio_varaha_max_frac=0.8, portfolio_weight_clip=1.0,
                    portfolio_quality_signal='credit', portfolio_alloc_exponent=2.0,
                    portfolio_probe_period=5)
    assert got == {'portfolio_grace_iters': 7, 'portfolio_revive_period': 3,
                   'portfolio_freeze_wt': 0.25, 'portfolio_varaha_min_frac': 0.2,
                   'portfolio_varaha_max_frac': 0.8, 'portfolio_weight_clip': 1.0,
                   'portfolio_quality_signal': 'credit', 'portfolio_alloc_exponent': 2.0,
                   'portfolio_probe_period': 5}


def test_zero_is_passed_through_not_treated_as_unset():
    """0 disables probing/reviving and is a REAL value; `if x:` would drop it."""
    got = _assemble(portfolio_probe_period=0, portfolio_revive_period=0)
    assert got == {'portfolio_probe_period': 0, 'portfolio_revive_period': 0}


def test_varaha_never_freeze_sets_true():
    assert _assemble(portfolio_varaha_never_freeze=True) == {'portfolio_varaha_never_freeze': True}


def test_varaha_can_freeze_sets_false():
    assert _assemble(portfolio_varaha_can_freeze=True) == {'portfolio_varaha_never_freeze': False}


def test_can_freeze_wins_when_both_are_given():
    """Documented precedence; the two flags are mutually exclusive."""
    got = _assemble(portfolio_varaha_can_freeze=True, portfolio_varaha_never_freeze=True)
    assert got == {'portfolio_varaha_never_freeze': False}


def test_adaptive_alloc_is_opt_in_only():
    assert 'portfolio_adaptive_alloc' not in _assemble()
    assert _assemble(portfolio_adaptive_alloc=True) == {'portfolio_adaptive_alloc': True}


def test_assembly_block_is_identical_to_the_main_drivers():
    """Deliberate copies in a deliberate fork.  Change one, change both."""
    def norm(s):
        return re.sub(r"\s+", " ", s).strip()
    assert norm(_assembly_block(_LISA)) == norm(_assembly_block(_MAIN))


def test_assembly_result_is_actually_handed_to_setup():
    """Building the dict and not passing it would be a silent no-op."""
    src = _src(_LISA)
    assert "sampler.setup(portfolio_args=opts.sampler_portfolio_args, **_freeze_policy_kwargs" in src
