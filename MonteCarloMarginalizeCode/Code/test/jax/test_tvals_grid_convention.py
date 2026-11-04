#!/usr/bin/env python
"""Both extrinsic drivers must build the SAME time-marginalization grid.

Issue #146: ``bin/integrate_likelihood_extrinsic_batchmode`` built
``linspace(-t_ref_wind, t_ref_wind, int(2*t_ref_wind/deltaT))`` at ten sites while
``RIFT/likelihood/jax_ile/wrapper.py`` (and hence
``bin/integrate_likelihood_extrinsic_jax``) built ``arange(-Nw, Nw)*deltaT``.  Both
likelihoods consume ONLY ``tvals[0]`` and ``len(tvals)`` -- each steps by ``deltaT``
and integrates with ``dx=deltaT`` regardless of the grid's own spacing -- so the two
grids differed in ORIGIN (0.2 samples at iwh=0.075 s, srate 4096), enough to round
``ifirst`` to a different integer sample and, since ``t_det`` carries the
per-detector delay, a different subset PER DETECTOR: up to 67.8 nats per sample.
They also differed in LENGTH, because ``2*int(x) != int(2*x)``, at srate 1024, 2048
and 16384 -- 16384 being the low-mass production rate.

WHY THIS FILE IS SHAPED THE WAY IT IS
-------------------------------------
The obvious test -- call the shared helper twice and compare -- is TAUTOLOGICAL: it
passes whether or not the drivers use the helper, which is the entire defect.  So
these tests read the ACTUAL DRIVER SOURCE, extract every window-grid construction by
AST, and evaluate the extracted expressions.  A driver that reverts one site to
``linspace`` fails ``test_all_driver_grid_sites_agree_by_value``; a driver that adds
an eleventh site by hand fails ``test_no_handrolled_window_grid_remains``.

The sample rates deliberately include 16384.  ``test_jax_endtoend.py`` runs at 4096,
one of only two rates where the two old conventions' LENGTHS coincidentally agreed,
so it structurally could not catch this even after #144.
"""

import ast
import os
import re

import numpy as np
import pytest

import RIFT.likelihood.factored_likelihood as factored_likelihood

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_BATCHMODE = os.path.join(_CODE, 'bin', 'integrate_likelihood_extrinsic_batchmode')
_WRAPPER = os.path.join(_CODE, 'RIFT', 'likelihood', 'jax_ile', 'wrapper.py')
_JAXDRIVER = os.path.join(_CODE, 'bin', 'integrate_likelihood_extrinsic_jax')

# Distinguishes a window grid from the many other linspace/arange calls in these
# files (distance grids, index ranges, the dense resampling grid).
_WINDOW_NAME = re.compile(r'\b(?:t_ref_wind|integration_window_half)\b')
_LEGACY_NW = re.compile(r'-\s*Nw\b')

# Sample rates to check.  16384 is the low-mass production rate and one of the three
# where the two pre-#146 conventions produced DIFFERENT LENGTHS (152/153, 306/307,
# 2456/2457); 4096 and 8192 are the two where they happened to agree.
SRATES = (1024, 2048, 4096, 8192, 16384)
IWH = 0.075   # --data-integration-window-half default, seconds

# The convention, written out independently of the implementation: npts, and the
# first and last grid sample as an EXACT rational multiple of deltaT.  If someone
# changes marginalization_time_grid(), these literals are what they have to argue
# with.  (npts = int(2*iwh/deltaT); first = -(npts//2); last = first + npts - 1.)
EXPECTED = {
    1024:  (153,   -76,   76),
    2048:  (307,  -153,  153),
    4096:  (614,  -307,  306),
    8192:  (1228, -614,  613),
    16384: (2457, -1228, 1228),
}


def _grid_call_sites(path):
    """Every window-grid construction in `path`, as (lineno, source_text) pairs.

    Matched by AST from the real file (the driver is a script and is never imported
    here).  THREE spellings are recognised, on purpose:

      * ``marginalization_time_grid(...)``  -- the shared helper, what must be there;
      * ``linspace(...)`` mentioning the window half-width -- batchmode's ten
        pre-#146 sites;
      * ``arange(-Nw, Nw)*deltaT`` -- the wrapper's three pre-#146 sites.

    Recognising the legacy spellings is what stops the comparison below from being a
    helper-presence check: run these tests against a pre-#146 tree and they extract
    the OLD grids from both drivers and fail on the actual 67.8-nat divergence,
    rather than passing vacuously because both sides now call one function.
    """
    with open(path) as f:
        src = f.read()
    tree = ast.parse(src, filename=path)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, 'id', None)
        if name not in ('marginalization_time_grid', 'linspace', 'arange'):
            continue
        text = ast.get_source_segment(src, node)
        assert text is not None, "could not recover source for call at line %d" % node.lineno
        if name == 'marginalization_time_grid':
            pass
        elif name == 'linspace' and _WINDOW_NAME.search(text):
            pass          # legacy batchmode spelling
        elif name == 'arange' and _LEGACY_NW.search(text):
            text += ' * deltaT'   # legacy wrapper spelling: arange(-Nw,Nw) is scaled
        else:
            continue      # an unrelated linspace/arange (distance grids, indices, ...)
        out.append((node.lineno, text))
    return out


class _P(object):
    """Stand-in for the driver's global ChooseWaveformParams: only deltaT is read."""
    def __init__(self, deltaT):
        self.deltaT = deltaT


def _eval_site(text, srate):
    """Evaluate one extracted grid expression at `srate`, as the driver would."""
    deltaT = 1.0 / srate
    ns = {
        'np': np, 'numpy': np, 'xpy_default': np,
        'factored_likelihood': factored_likelihood,
        'marginalization_time_grid': factored_likelihood.marginalization_time_grid,
        # batchmode names
        't_ref_wind': IWH, 'P': _P(deltaT),
        # wrapper names
        'integration_window_half': IWH, 'deltaT': deltaT,
        # The pre-#146 wrapper spelling was `Nw = int(iwh/deltaT); arange(-Nw,Nw)*deltaT`,
        # with Nw bound on the line above the call.  Bind it here so an un-migrated tree
        # is EVALUATED and fails on the grid values, rather than escaping the comparison.
        'Nw': int(IWH / deltaT),
    }
    return np.asarray(eval(compile(ast.Expression(ast.parse(text, mode='eval').body),
                                   '<site>', 'eval'), ns))


def test_extractor_actually_finds_the_sites():
    """Guard the guard: a broken extractor would make every test below vacuous."""
    bm = _grid_call_sites(_BATCHMODE)
    wr = _grid_call_sites(_WRAPPER)
    assert len(bm) >= 10, (
        "expected at least the 10 known window-grid sites in %s, found %d -- either "
        "sites were removed or the AST extractor broke" % (_BATCHMODE, len(bm)))
    assert len(wr) >= 3, (
        "expected at least the 3 known window-grid sites in %s, found %d" % (_WRAPPER, len(wr)))


@pytest.mark.parametrize('srate', SRATES)
def test_all_driver_grid_sites_agree_by_value(srate):
    """THE test for #146: every grid either driver builds is bit-identical.

    Before #146 this failed at every one of these rates: differing origin at all
    five, and differing length at 1024, 2048 and 16384.
    """
    bm = [('batchmode', l, t) for (l, t) in _grid_call_sites(_BATCHMODE)]
    wr = [('wrapper', l, t) for (l, t) in _grid_call_sites(_WRAPPER)]
    # Without this, the test degenerates: if one file contributed ZERO sites the loop
    # below would compare the other file against itself and pass, which is the single
    # -path-conjunct failure shape this whole file exists to avoid.  It is asserted
    # here, not only in test_extractor_actually_finds_the_sites, so that THIS test
    # cannot pass vacuously on its own.
    assert bm and wr, (
        "cross-driver comparison needs sites from BOTH files; got %d from batchmode "
        "and %d from the wrapper" % (len(bm), len(wr)))
    sites = bm + wr
    ref_tag, ref_line, ref_text = sites[0]
    ref = _eval_site(ref_text, srate)
    for tag, line, text in sites[1:]:
        got = _eval_site(text, srate)
        assert got.shape == ref.shape, (
            "srate %d: %s:%d builds %d grid points, %s:%d builds %d"
            % (srate, tag, line, got.size, ref_tag, ref_line, ref.size))
        assert np.array_equal(got, ref), (
            "srate %d: %s:%d differs from %s:%d by up to %g s (%g samples)"
            % (srate, tag, line, ref_tag, ref_line,
               np.max(np.abs(got - ref)), np.max(np.abs(got - ref)) * srate))


@pytest.mark.parametrize('srate', SRATES)
def test_grid_matches_the_pinned_convention(srate):
    """The shared helper's own values, against hand-written expectations."""
    deltaT = 1.0 / srate
    npts_expect, first_expect, last_expect = EXPECTED[srate]
    tvals = factored_likelihood.marginalization_time_grid(IWH, deltaT)

    assert tvals.size == npts_expect, (
        "srate %d: npts %d, expected int(2*%g/deltaT) = %d"
        % (srate, tvals.size, IWH, npts_expect))
    # Compare as integer sample indices: exact, and independent of float formatting.
    assert tvals[0] == first_expect * deltaT
    assert tvals[-1] == last_expect * deltaT
    # Spacing EXACTLY deltaT -- the property that makes tvals[k] a truthful label
    # for the sample the likelihood actually reads.  Not approximately: exactly.
    assert np.array_equal(np.diff(tvals), np.full(tvals.size - 1, deltaT))
    # The fiducial epoch is on the grid.
    assert (tvals == 0.0).sum() == 1
    # And the window stays inside the requested half-width.
    assert np.abs(tvals).max() <= IWH


def test_jax_driver_takes_the_wrapper_default_grid():
    """`integrate_likelihood_extrinsic_jax` must NOT build or pass its own grid.

    The cross-driver test above compares batchmode against ``jax_ile/wrapper.py``.  That
    is only a valid proxy for "the two DRIVERS agree" while the JAX driver actually
    inherits the wrapper's default -- i.e. calls ``build_data_from_precompute`` with no
    ``tvals=``.  If someone gives that driver its own grid, the wrapper comparison keeps
    passing while the drivers diverge again, which is precisely the #146 shape.
    """
    with open(_JAXDRIVER) as f:
        src = f.read()
    tree = ast.parse(src, filename=_JAXDRIVER)
    builders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, 'id', None)
        if name and name.startswith('build_') and name.endswith('_from_precompute'):
            builders.append(node)
    assert builders, (
        "no build_*_from_precompute call found in %s -- the driver was restructured and "
        "this pin no longer checks anything" % os.path.basename(_JAXDRIVER))
    for node in builders:
        passed = [kw.arg for kw in node.keywords if kw.arg == 'tvals']
        assert not passed, (
            "%s:%d passes its own tvals= to the builder; it must inherit the shared "
            "default so the two drivers cannot drift apart again (issue #146)"
            % (os.path.basename(_JAXDRIVER), node.lineno))


def test_no_handrolled_window_grid_remains():
    """No file may rebuild this grid by hand; #146 was ten copies drifting apart.

    ``#`` comments are skipped -- the historical notes left in place deliberately
    quote the old forms, and a test that forbade naming them would forbid explaining
    them.  Docstrings are NOT skipped, deliberately: a docstring that still describes
    the grid as ``arange(-Nw, Nw)`` is documentation that has gone stale, which is
    how #146 stayed invisible.  Put such prose in a ``#`` comment.
    """
    # The two pre-#146 spellings.  Whitespace-insensitive so a reformat cannot hide one.
    BANNED = (re.compile(r'linspace\(\s*-\s*t_ref_wind'),
              re.compile(r'arange\(\s*-\s*Nw'))
    offenders = []
    for path in (_BATCHMODE, _WRAPPER,
                 os.path.join(_CODE, 'bin', 'integrate_likelihood_extrinsic_jax')):
        with open(path) as f:
            for i, line in enumerate(f, 1):
                code = line.split('#', 1)[0]
                if any(rx.search(code) for rx in BANNED):
                    offenders.append('%s:%d: %s'
                                     % (os.path.basename(path), i, line.rstrip()))
    assert not offenders, (
        "hand-rolled window grid(s) reintroduced; call "
        "factored_likelihood.marginalization_time_grid() instead:\n  "
        + "\n  ".join(offenders))


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
