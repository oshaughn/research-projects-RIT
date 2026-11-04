"""The ``--interp`` CLI must expose every gatherer the likelihood implements.

`RIFT/likelihood/jax_ile/core.py` registers three arrival-time interpolation
stencils in ``_GATHERERS``: ``nearest``, ``linear`` and ``cubic``.  The driver's
``--interp`` option listed only two, so ``cubic`` -- the stencil that mirrors the
production ``factored_likelihood`` one, and the only one `_gather_cubic`'s own
docstring recommends -- was unreachable from the command line and ``linear`` was
silently the only realistic choice.

Nothing failed.  The option parsed, the run completed, and the likelihood was
biased: linear undershoots the razor-sharp rholm peak by an amount that depends
on where the peak falls between samples, so it stamps a mass-dependent ripple
onto the surface and biases the recovered arrival time, hence the sky.  On a
zero-spin BNS at network amplitude 23.8, selecting cubic instead recovered
+4.34 nats at the peak.

The first test below is the durable one: it does not name ``cubic``, it asserts
that the CLI and the registry agree.  Adding a fourth gatherer without exposing
it fails here rather than silently shipping an unreachable code path.
"""
import importlib.machinery
import importlib.util
import os

import numpy as np
import pytest

import RIFT.likelihood.jax_ile.core as core

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_JAXDRIVER = os.path.join(_CODE, 'bin', 'integrate_likelihood_extrinsic_jax')


def _load_driver():
    """Import the driver by path.  It guards its entry point with __main__, so
    importing it defines build_parser() without running an analysis."""
    # The driver has NO .py extension, so spec_from_file_location cannot infer a
    # loader and returns a spec with loader=None.  Name the loader explicitly --
    # otherwise these tests skip, and a skipped test reads exactly like a passing
    # one in the summary line.
    assert os.path.exists(_JAXDRIVER), 'driver missing: %s' % _JAXDRIVER
    loader = importlib.machinery.SourceFileLoader('_ile_jax_driver', _JAXDRIVER)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _interp_choices():
    parser = _load_driver().build_parser()
    for opt in parser.option_list + [o for g in parser.option_groups for o in g.option_list]:
        if '--interp' in (opt._long_opts or []):
            return set(opt.choices or ())
    raise AssertionError('the driver no longer defines --interp')


def test_cli_exposes_every_registered_gatherer():
    """Every key of _GATHERERS must be selectable from --interp, and vice versa."""
    registered = set(core._GATHERERS)
    exposed = _interp_choices()
    assert registered, '_GATHERERS is empty; the registry moved'
    assert exposed == registered, (
        'CLI --interp choices %s disagree with the _GATHERERS registry %s. '
        'A stencil that is implemented but not exposed is dead code the user '
        'cannot reach; one that is exposed but not implemented is a KeyError at '
        'runtime.' % (sorted(exposed), sorted(registered)))


def test_cubic_is_reachable_and_is_not_linear():
    """Guard the specific regression: cubic selectable, and a DIFFERENT stencil.

    Equality of choices alone would still pass if someone aliased cubic to the
    linear implementation, so check that the two actually compute differently.
    """
    assert 'cubic' in _interp_choices()
    assert core._GATHERERS['cubic'] is not core._GATHERERS['linear']

    # A smooth, band-limited column sampled at integers; interpolate off-sample.
    n = 64
    idx = np.arange(n)
    col = np.exp(-0.5 * ((idx - 31.7) / 2.5) ** 2)
    pos = np.array([12.5, 20.25, 31.7, 44.75])

    lin = np.asarray(core._GATHERERS['linear'](col, pos))
    cub = np.asarray(core._GATHERERS['cubic'](col, pos))
    assert not np.allclose(lin, cub), 'cubic returns the linear result'

    exact = np.exp(-0.5 * ((pos - 31.7) / 2.5) ** 2)
    err_lin = np.abs(lin - exact).max()
    err_cub = np.abs(cub - exact).max()
    assert err_cub < err_lin, (
        'cubic (%.3e) should beat linear (%.3e) on a smooth peak' % (err_cub, err_lin))


def test_every_stencil_reproduces_the_samples_it_sits_on():
    """At integer positions every stencil must return the sample itself.

    This is what makes the comparison above meaningful: the stencils differ only
    between samples, not at them, so a difference at integer positions would mean
    an indexing bug rather than an interpolation choice.
    """
    n = 48
    idx = np.arange(n)
    col = np.sin(0.21 * idx) + 0.4 * np.cos(0.07 * idx)
    # stay clear of the edges: the cubic stencil zero-extends outside the buffer
    pos = np.arange(4, n - 4).astype(float)
    for name, gather in sorted(core._GATHERERS.items()):
        got = np.asarray(gather(col, pos))
        assert np.allclose(got, col[pos.astype(int)], atol=1e-10), (
            '%s does not reproduce the sample at integer positions' % name)
