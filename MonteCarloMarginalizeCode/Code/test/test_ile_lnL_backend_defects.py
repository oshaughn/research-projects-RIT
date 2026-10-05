#!/usr/bin/env python
"""
Regression tests for three pre-existing ILE defects on the --internal-use-lnL backends
(RIFT/integrators/mcsamplerGPU.py, RIFT/integrators/mcsampler.py,
bin/integrate_likelihood_extrinsic_batchmode).

All three were found running the ILE-GPU-Paper demo with `--vectorized --gpu` on an A100
(cupy 12.0.0, IGWN CVMFS python 3.11) against an UNMODIFIED checkout, so none of them is
branch-specific.  They matter together: the .dgrid marginal-distance-grid export is gated on
`opts.internal_use_lnL`, and (1)+(2) took out BOTH linear-integrand samplers, leaving AV and
GMM as the only backends that could produce a .dgrid at all.

  1. adaptive_cartesian_gpu (the DEFAULT sampler) + --internal-use-lnL died in
     mcsamplerGPU.integrate_log with

         TypeError: Unsupported type <class 'numpy.ndarray'>
         ... eff_samp = xpy.exp(  outvals[0]+np.log(self.ntotal) - maxval)

     `maxval` tracks the running max of the LOG weights via

         maxval = max(maxval, identity_convert(self.xpy.max(log_integrand)))

     Under cupy, `self.xpy.max` returns a 0-d DEVICE array and `identity_convert` is
     `cupy.asnumpy`, so this hands Python's `max()` a 0-d numpy.ndarray -- and from the first
     chunk whose largest log weight exceeds the initializer, `maxval` IS a host array.  The
     next line mixes it into a device expression, which cupy refuses.  It is invisible on
     CPU because `numpy.max` returns a numpy SCALAR, which cupy accepts.

     Those same two lines carry a second, quieter defect that is DELIBERATELY NOT fixed here:
     `maxval` is a LOG-scale accumulator initialized at 0, a linear-weight idiom, so
     eff_samp = sum(w)/max(w) is floored whenever the largest weight is < 1.  Real GW lnL is
     large and positive so it does not bite in production, and the identical initializer is
     copied verbatim into mcsamplerPortfolio and mcsamplerNFlow -- changing one would split
     n_eff, and hence run lengths, across backends mid-campaign.  Recorded in the code.

  2. Any --sampler-method that lands on the driver's "original sampler" fallback -- e.g.
     `adaptive_cartesian` -- died AFTER a successful integration with

         AttributeError: 'MCSampler' object has no attribute 'identity_convert'
         ... _neff_val = None if neff is None else float(sampler.identity_convert(neff))

     mcsamplerGPU, mcsamplerAdaptiveVolume, mcsamplerPortfolio and mcsamplerEnsemble all
     define the host/device converters in __init__; plain mcsampler.MCSampler did not, and
     the driver only assigns them onto the samplers it recognizes BY NAME.  The result was
     thrown away at the last step of a run that had already done all the work.
     (Note this is a crash and not a wrong number: `adaptive_cartesian` never gets
     `use_lnL`, so `return_lnL` stays False and it integrates linear L, correctly.
     --internal-use-lnL is simply inert for it, apart from opening the .dgrid gate.)

  3. `--sampler-method portfolio` with no --sampler-portfolio member exited 1 with
         TypeError: 'NoneType' object is not iterable
     because the option defaults to None and the flattening comprehension iterated it
     directly.  Already fixed in-tree (a134af37); these tests pin it so it cannot come back
     when that comprehension is next edited.

Two more came out of checking that the fixes actually produced NUMBERS, not just exit code 0:

  4. Clearing (2) exposed a silent wrong answer behind it.  The post-processing asked
     `if not(opts.internal_use_lnL): log_res = numpy.log(res)` -- reading the OPTION where it
     had to read the convention the sampler was actually given.  They differ for exactly one
     pairing: `adaptive_cartesian` is in ok_lnL_methods, so the option is accepted, but
     mcsampler.py has no use_lnL handling, the driver never hands it one, and it returns a
     LINEAR integral.  The logarithm was skipped and the demo reported lnZ = 1.3e29 (= e^67)
     with sigma = inf, with the .dgrid built from it inheriting the same.  The zero-likelihood
     stand-in and the replica arm had the same substitution.  The honest predicate,
     `return_lnL`, is already derived in the driver and already used by the integrand.

  5. `--sampler-method portfolio --export-marginal-distance-grid` WITH members then reached
     the .dgrid exporter and died on `sampler.prior_pdf["distance"]`: the portfolio delegates
     prior_pdf to its members and kept no copy of its own.

  6. And the .dgrid gate itself required --internal-use-lnL, a flag that decides nothing this
     exporter reads.  All it did was make --export-marginal-distance-grid silently do nothing
     unless a second, unrelated flag happened to be set too.  Dropped from the gate.

The requirement is not "does not crash".  For (1) the integrator must also report the
eff_samp it claims to; for (2) the converters are an INTERFACE every sampler owes its
callers, so the check is over all of them, not just the one that was missing; and for (4)
the point is a lnZ that is right, which no exit code would have shown.
"""

import ast
import os
import textwrap

import numpy as np
import pytest
import scipy.special

import RIFT.integrators.mcsampler as mcsampler
import RIFT.integrators.mcsamplerGPU as mcsamplerGPU

_ILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')

# Captured before the autouse fixture below replaces it, so the real-GPU tests can put the
# genuine (cupyx) aggregator back and exercise the real device path end to end.
_REAL_XPY_SPECIAL = mcsamplerGPU.xpy_special_default


###
### helpers
###

class _Dev(np.ndarray):
    """Stand-in for a cupy array: numpy-backed, but a DISTINCT type whose ufuncs enforce
    cupy's operand rule -- python scalars and numpy SCALARS are accepted, a numpy.ndarray is
    not (`_preprocess_arg` -> "TypeError: Unsupported type <class 'numpy.ndarray'>").

    That rule is the whole mechanism of defect (1), and it is invisible under plain numpy,
    where `numpy.max` returns a numpy scalar and scalar-minus-0-d-array quietly succeeds.
    The real cupy path is covered by the GPU tests below; this lets the CPU lane see it too.
    """

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        for a in inputs:
            if isinstance(a, np.ndarray) and not isinstance(a, _Dev):
                raise TypeError("Unsupported type <class 'numpy.ndarray'>")
        strip = lambda v: np.asarray(v).view(np.ndarray) if isinstance(v, _Dev) else v
        raw = [strip(a) for a in inputs]
        if 'out' in kwargs:                       # in-place ops (`joint_p_s *= ...`)
            kwargs['out'] = tuple(strip(o) for o in kwargs['out'])
        return _to_dev(getattr(ufunc, method)(*raw, **kwargs))


def _to_dev(x):
    if isinstance(x, tuple):
        return tuple(_to_dev(y) for y in x)
    if isinstance(x, (np.ndarray, np.generic)):
        return np.asarray(x).view(_Dev)
    return x


def _to_host(x):
    """What cupy.asnumpy does, including the part that matters: a 0-d device value comes
    back as a 0-d numpy.ndarray, NOT as a numpy scalar."""
    return np.asarray(x).view(np.ndarray)


class _FakeCupy(object):
    """`sampler.xpy` for the device lane: forwards to numpy and re-labels every result."""

    ndarray = _Dev
    asnumpy = staticmethod(_to_host)

    def __getattr__(self, name):
        obj = getattr(np, name)
        if not callable(obj):
            return obj
        def wrapped(*a, **k):
            a = [np.asarray(x).view(np.ndarray) if isinstance(x, _Dev) else x for x in a]
            return _to_dev(obj(*a, **k))
        return wrapped


class _FakeCupySpecial(object):
    """Stands in for cupyx.scipy.special, so statutils stays on the device aggregate path
    instead of taking its host fallback -- which is what puts outvals[0] on the device."""

    @staticmethod
    def logsumexp(a, **kwargs):
        return _to_dev(scipy.special.logsumexp(np.asarray(a).view(np.ndarray), **kwargs))


def _sampler(names=('x',), lo=0.0, hi=1.0, device=False):
    """A minimal mcsamplerGPU sampler on the unit box: uniform pdf, uniform prior, no
    adaptation (so the sampling prior stays exactly uniform and the weights stay analytic).

    `device=True` binds it to the _FakeCupy stand-in.  Its pdf/prior must then return device
    arrays, exactly as the ILE's do when they are built out of xpy_default."""
    s = mcsamplerGPU.MCSampler()
    s.xpy = _FakeCupy() if device else np
    cast = _to_dev if device else (lambda x: x)
    span = hi - lo
    for nm in names:
        s.add_parameter(nm,
                        pdf=lambda x: cast(np.ones(np.shape(x)) / span),
                        cdf_inv=lambda p: lo + span * np.asarray(p),
                        left_limit=lo, right_limit=hi,
                        prior_pdf=lambda x: cast(np.ones(np.shape(x)) / span),
                        adaptive_sampling=False)
    return s


def _constant_lnL(c, device=False):
    """Every draw gets the same log weight, so sum(w)/max(w) is exactly ntotal.

    integrate_log calls the integrand BY KEYWORD (`lnF(**unpacked)`) unless
    'no_protect_names' is set, so take the draws from whichever it hands over."""
    cast = _to_dev if device else (lambda x: x)
    def lnL(*args, **kwargs):
        vals = args[0] if args else list(kwargs.values())[0]
        return cast(np.full(np.shape(vals), float(c)))
    return lnL


@pytest.fixture(autouse=True)
def _host_logsumexp(monkeypatch):
    """These tests drive integrate_log with xpy=numpy.  On a host that HAS cupy the module
    global xpy_special_default is cupyx.scipy.special, which cannot consume host arrays; a
    no-cupy host already has scipy.special here.  Pin it so the CPU tests mean the same
    thing on both.  `identity_convert` is deliberately NOT patched -- on a GPU host it stays
    cupy.asnumpy, which is the converter the defect runs through."""
    monkeypatch.setattr(mcsamplerGPU, 'xpy_special_default', scipy.special, raising=False)


def _driver_source():
    with open(_ILE) as f:
        return f.read()


def _driver_rhs(varname, occurrence=0):
    """Compile the RHS of the driver's OWN assignment to `varname` so a test can execute the
    real line out of the real file instead of a copy that can drift away from it."""
    hits = [node.value for node in ast.walk(ast.parse(_driver_source()))
            if isinstance(node, ast.Assign)
            for tgt in node.targets
            if isinstance(tgt, ast.Name) and tgt.id == varname]
    assert len(hits) > occurrence, 'no assignment to {} in the ILE'.format(varname)
    return compile(ast.Expression(hits[occurrence]), _ILE, 'eval')


class _Opts(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):     # anything the sliced driver code reads but we do not set
        return None


def _driver_if_test(marker):
    """Compile the CONDITION of the driver's own `if` statement containing `marker`."""
    tree = ast.parse(_driver_source())
    hits = [n for n in ast.walk(tree)
            if isinstance(n, ast.If) and marker in ast.dump(n.test)]
    assert len(hits) == 1, 'expected exactly one if-test mentioning {}, got {}'.format(
        marker, len(hits))
    return compile(ast.Expression(hits[0].test), _ILE, 'eval')


def _driver_slice(start, end):
    """Compile the driver's OWN source between two markers, so a test executes the real block.
    Snapped to line boundaries and dedented, since these blocks live inside analyze_event."""
    src = _driver_source()
    i = src.rindex('\n', 0, src.index(start)) + 1
    j = src.index(end, i)
    return compile(textwrap.dedent(src[i:j]), _ILE, 'exec')


def _derive_return_lnL(sampler_method, internal_use_lnL):
    """Run the ILE's real `return_lnL = ...` derivation for one option combination.

    This is the FIRST derivation block only.  A second, unconditional `return_lnL=True` for
    AV sits further down, past the sampler-setup code this cannot execute -- see
    test_the_av_override_cannot_disagree_with_the_option for why that one is consistent."""
    code = _driver_slice('return_lnL=False\n', 'if use_gmm_args:')
    ns = {'opts': _Opts(sampler_method=sampler_method, internal_use_lnL=internal_use_lnL),
          'pinned_params': {}, 'use_gmm_member': False, 'print': lambda *a, **k: None,
          'int': int, 'float': float}
    exec(code, ns)
    return ns['return_lnL'], ns['pinned_params']


###
### 1. mcsamplerGPU.integrate_log -- maxval must be a HOST SCALAR, on a LOG scale
###

def test_maxval_does_not_leak_a_host_array_into_the_device_expression(monkeypatch):
    """The crash itself, on the device stand-in.

    `maxval = max(maxval, identity_convert(self.xpy.max(log_integrand)))` sends the running
    max to the HOST as a 0-d array; the next line subtracts it from a DEVICE aggregate.
    Before the fix this raises "Unsupported type <class 'numpy.ndarray'>" on the first
    chunk -- and there is no later chance to recover, because eff_samp is the loop's own
    termination test.
    """
    monkeypatch.setattr(mcsamplerGPU, 'identity_convert', _to_host)
    monkeypatch.setattr(mcsamplerGPU, 'xpy_special_default', _FakeCupySpecial)
    xpy = _FakeCupy()
    s = _sampler(device=True)
    # verbose=True and a finite deltalnL on purpose: maxlnL leaks a host array the same way
    # maxval does, and its two victims are the per-iteration report (`outvals[0]-maxlnL`)
    # and the deltalnL cut.  The ILE demo runs with both on, which is why fixing maxval
    # alone left the demo failing at exactly the same TypeError one line further down.
    res, var, eff_samp, _ = s.integrate_log(_constant_lnL(+5.0, device=True), 'x', xpy=xpy,
                                            n=200, nmax=1000, neff=50, save_intg=True,
                                            verbose=True, igrand_threshold_deltalnL=15.0)
    assert float(eff_samp) == pytest.approx(s.ntotal, rel=1e-9)


def test_the_device_standin_reproduces_the_operand_rule_it_stands_for(monkeypatch):
    """Guards the test above: if _Dev stopped refusing host arrays it would pass either way."""
    dev = _to_dev(np.asarray(3.0))
    with pytest.raises(TypeError):
        dev - _to_host(np.asarray(1.0))          # 0-d HOST array: what identity_convert made
    assert float(dev - 1.0) == 2.0               # python float: what the fix makes
    assert float(dev - np.float64(1.0)) == 2.0   # numpy scalar: what plain numpy.max returns


def test_eff_samp_is_the_number_of_draws_for_a_constant_integrand():
    """Every weight equal => sum(w)/max(w) is exactly the number of draws.  Runs the same
    loop on plain numpy, where the host/device confusion cannot arise, so a change to the
    running max that silently altered the ESTIMATE (rather than only its type) is caught."""
    s = _sampler()
    res, var, eff_samp, _ = s.integrate_log(_constant_lnL(+5.0), 'x', xpy=np,
                                            n=100, nmax=1000, neff=50, save_intg=True)
    assert float(eff_samp) == pytest.approx(s.ntotal, rel=1e-9)
    assert s.ntotal == 100


def test_a_constant_integrand_integrates_to_that_constant():
    """Guards the obvious way to 'fix' the above by breaking the estimator: with a uniform
    prior and a uniform sampling prior, ln Z = c."""
    for c in (-5.0, +5.0):
        s = _sampler()
        res, var, eff_samp, _ = s.integrate_log(_constant_lnL(c), 'x', xpy=np,
                                                n=100, nmax=1000, neff=50, save_intg=True)
        assert float(res) == pytest.approx(c, abs=1e-9)


@pytest.mark.skipif(not mcsamplerGPU.cupy_ok, reason='needs a working cupy/GPU')
def test_cupy_really_does_reject_a_host_array_and_really_does_accept_a_float():
    """The premise the CPU tests stand in for, checked against the actual library."""
    import cupy
    dev = cupy.asarray(3.0)
    host_scalar_max = cupy.asnumpy(cupy.max(cupy.asarray([1.0, 2.0])))
    assert isinstance(host_scalar_max, np.ndarray), 'cupy stopped returning 0-d arrays'
    with pytest.raises(TypeError):
        cupy.exp(dev - host_scalar_max)
    assert float(cupy.exp(dev - float(host_scalar_max))) == pytest.approx(np.exp(1.0))


@pytest.mark.skipif(not mcsamplerGPU.cupy_ok, reason='needs a working cupy/GPU')
def test_integrate_log_runs_on_the_device_end_to_end(monkeypatch):
    """The failing configuration itself, on real hardware: adaptive_cartesian_gpu with a log
    integrand, the per-iteration report on and a finite deltalnL -- i.e. what the ILE-GPU-Paper
    demo runs.  Before the fix this raised TypeError: Unsupported type <class 'numpy.ndarray'>
    on the first chunk."""
    import cupy
    monkeypatch.setattr(mcsamplerGPU, 'xpy_special_default', _REAL_XPY_SPECIAL)
    s = mcsamplerGPU.MCSampler()
    s.xpy = cupy
    s.identity_convert = mcsamplerGPU.identity_convert
    s.identity_convert_togpu = mcsamplerGPU.identity_convert_togpu
    s.add_parameter('x',
                    pdf=lambda x: cupy.ones(cupy.shape(x)),
                    cdf_inv=lambda p: cupy.asarray(p),
                    left_limit=0.0, right_limit=1.0,
                    prior_pdf=lambda x: cupy.ones(cupy.shape(x)),
                    adaptive_sampling=False)

    def lnL(*args, **kwargs):
        vals = args[0] if args else list(kwargs.values())[0]
        return cupy.full(vals.shape, 5.0)

    res, var, eff_samp, _ = s.integrate_log(lnL, 'x', n=200, nmax=1000, neff=50,
                                            save_intg=True, verbose=True,
                                            igrand_threshold_deltalnL=15.0)
    host = mcsamplerGPU.identity_convert
    assert float(host(res)) == pytest.approx(5.0, abs=1e-9)
    assert float(host(eff_samp)) == pytest.approx(s.ntotal, rel=1e-9)


###
### 2. the host/device converters are part of the sampler INTERFACE
###

def _construct(modname):
    mod = pytest.importorskip('RIFT.integrators.' + modname)
    if modname == 'mcsamplerPortfolio':
        return mod.MCSampler(portfolio=[mcsampler.MCSampler()])
    return mod.MCSampler()


@pytest.mark.parametrize('modname', ['mcsampler', 'mcsamplerGPU', 'mcsamplerAdaptiveVolume',
                                     'mcsamplerEnsemble', 'mcsamplerPortfolio',
                                     'mcsamplerNFlow'])
def test_every_sampler_exposes_identity_convert_straight_out_of_the_constructor(modname):
    """`float(sampler.identity_convert(neff))` runs for EVERY sampler the ILE builds, and the
    driver only assigns the converter onto the ones it recognizes by name.  So a sampler that
    reaches a caller by any other route (the 'original sampler' fallback, a plugin, a test)
    must already carry it out of __init__.  Only mcsampler.MCSampler did not.

    (`identity_convert_togpu` is deliberately NOT required here: mcsamplerAdaptiveVolume has
    never defined one and never calls one.  It is checked below on the sampler this fix
    touches, whose sibling mcsamplerGPU sets both.)"""
    s = _construct(modname)
    assert hasattr(s, 'identity_convert'), '{}.MCSampler has no identity_convert'.format(modname)
    probe = np.asarray([1.0, 2.0])
    assert np.allclose(np.asarray(s.identity_convert(probe)), probe)


def test_the_original_sampler_carries_both_converters_like_its_gpu_sibling():
    s = mcsampler.MCSampler()
    probe = np.asarray([1.0, 2.0])
    assert np.allclose(s.identity_convert(probe), probe)
    assert np.allclose(s.identity_convert_togpu(probe), probe)


def test_the_drivers_own_neff_line_survives_the_original_sampler():
    """Executes the ILE's real `_neff_val = ...` source against a real mcsampler.MCSampler --
    the exact pairing that raised AttributeError after a completed integration."""
    code = _driver_rhs('_neff_val')
    s = mcsampler.MCSampler()
    for neff, expect in [(np.float64(3.0), 3.0), (np.asarray(7.5), 7.5), (2, 2.0)]:
        assert eval(code, {'neff': neff, 'sampler': s, 'float': float}) == pytest.approx(expect)
    assert eval(code, {'neff': None, 'sampler': s, 'float': float}) is None


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_driver_still_consumes_neff_through_the_converter():
    """If that line stops going through identity_convert, the test above stops testing it."""
    assert 'sampler.identity_convert(neff)' in _driver_source()


###
### 3. --sampler-method portfolio with no members: a message, not a TypeError
###

@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
@pytest.mark.parametrize('given,expect', [
    (None, []),                                  # the defect: option omitted entirely
    ([], []),
    (['AV'], ['AV']),
    (['AV,GMM'], ['AV', 'GMM']),                 # one comma-separated list
    (['AV', 'GMM'], ['AV', 'GMM']),              # repeated option
    (['AV, GMM', 'AC'], ['AV', 'GMM', 'AC']),    # both forms mixed, with whitespace
    ([' , '], []),
])
def test_the_drivers_own_portfolio_flattening_handles_a_missing_option(given, expect):
    """Executes the ILE's real `sampler_types = ...` source.  With opts.sampler_portfolio
    left at its None default this used to raise TypeError: 'NoneType' object is not
    iterable, several minutes into a run, instead of naming the missing option."""
    code = _driver_rhs('sampler_types')
    got = eval(code, {'opts': _Opts(sampler_portfolio=given), 'str': str})
    assert got == expect


def test_a_portfolio_reports_the_priors_it_integrated_against():
    """Found while checking that defect (3) really cleared: with members supplied, the same
    `--sampler-method portfolio ... --export-marginal-distance-grid` command reached the
    .dgrid exporter and died on `sampler.prior_pdf["distance"]`, because the portfolio
    delegates prior_pdf to its members and kept no copy.  The exporter divides the sampling
    prior back out, so a portfolio that cannot name its prior cannot export a .dgrid."""
    import RIFT.integrators.mcsamplerPortfolio as mcsamplerPortfolio
    prior = lambda x: np.ones(np.shape(x)) / 900.0
    s = mcsamplerPortfolio.MCSampler(portfolio=[mcsampler.MCSampler(), mcsampler.MCSampler()])
    s.add_parameter('distance', pdf=lambda x: np.ones(np.shape(x)) / 900.0,
                    cdf_inv=lambda p: 100.0 + 900.0 * np.asarray(p),
                    left_limit=100.0, right_limit=1000.0, prior_pdf=prior)
    assert 'distance' in s.prior_pdf, 'the portfolio cannot say what prior it integrated against'
    probe = np.asarray([200.0, 500.0])
    assert np.allclose(s.prior_pdf['distance'](probe), prior(probe))
    # and it must be the SAME callable the members were given, not a re-derived one
    assert s.prior_pdf['distance'] is s.portfolio_realizations[0].prior_pdf['distance']


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_an_empty_portfolio_is_refused_by_name():
    """Flattening to [] must not then build a zero-member portfolio and integrate with it."""
    src = _driver_source()
    i = src.index('sampler_types = [')
    block = src[i:i + 400]
    assert 'if not sampler_types:' in block, 'nothing checks that the portfolio has members'
    assert 'requires at least one --sampler-portfolio' in block


###
### 2b. the log_res gate must read the SAMPLER's convention, not the option
###

@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
@pytest.mark.parametrize('method', ['GMM', 'adaptive_cartesian_gpu', 'AV', 'portfolio'])
def test_return_lnL_tracks_the_option_for_every_sampler_that_honours_it(method):
    """These four get use_lnL, so the two predicates agree and nothing about them changes."""
    got, pinned = _derive_return_lnL(method, True)
    assert got is True
    assert pinned.get('use_lnL') is True


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_av_override_cannot_disagree_with_the_option():
    """AV gets a SECOND, unconditional `return_lnL=True` after the derivation block above, so
    reading only that block would model AV wrongly.  It is consistent anyway, but only because
    the AV branch of the sampler chain forces the option on -- if that ever stops, AV joins
    adaptive_cartesian as a place the two predicates part company."""
    src = _driver_source()
    i = src.index('if opts.sampler_method == "AV":')
    assert 'return_lnL=True' in src[i:i + 200], 'the AV override moved; recheck the model above'
    j = src.index("elif opts.sampler_method == 'AV':")          # the sampler-construction chain
    assert 'opts.internal_use_lnL=True' in src[j:j + 400], \
        'AV no longer forces --internal-use-lnL, so return_lnL and the option can now differ'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_adaptive_cartesian_is_the_one_place_the_two_predicates_disagree():
    """--internal-use-lnL is accepted for `adaptive_cartesian` (it is in ok_lnL_methods) but
    mcsampler.py has no use_lnL handling, so the driver never hands it one and it returns a
    LINEAR integral.  This disagreement is why the post-processing may not read the option."""
    got, pinned = _derive_return_lnL('adaptive_cartesian', True)
    assert got is False, 'if this ever becomes True, mcsampler.py must have grown a log integrator'
    assert 'use_lnL' not in pinned
    assert 'adaptive_cartesian' in _driver_source().split('ok_lnL_methods = ')[1].split(']')[0]


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_a_linear_sampler_result_gets_its_logarithm_taken():
    """The driver's real log_res block, run on the pairing that used to skip the logarithm.
    With `res` a linear integral, log_res must be ln(res) -- not res, which is what put
    lnL = 1.3e29 (= e^67) and sigma = inf in the ILE-GPU-Paper demo's output."""
    code = _driver_slice('if not(return_lnL):\n      log_res', '# ---------')
    for return_lnL, res, var, expect in [(False, np.exp(67.0), 1e56, 67.0),
                                         (True, 67.0, 0.5, 67.0)]:
        ns = {'numpy': np, 'res': res, 'var': var, 'return_lnL': return_lnL}
        exec(code, ns)
        assert ns['log_res'] == pytest.approx(expect, rel=1e-12)


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_no_post_processing_step_still_reads_the_option_as_the_convention():
    """The replica arm and the zero-likelihood stand-in had the same substitution."""
    src = _driver_source()
    # The zero-likelihood stand-in no longer has an `if` of its own: it is built by
    # make_zero_likelihood_standin, and the convention is an ARGUMENT.  Check that argument
    # instead, and that the construction is still there to check.
    _mark = 'like_to_integrate = make_zero_likelihood_standin('
    assert _mark in src, 'the zero-likelihood stand-in construction moved or was renamed'
    _call = src[src.index(_mark):]
    _call = _call[:_call.index(')\n')]
    assert 'return_lnL' in _call, \
        'the zero-likelihood stand-in is no longer told the convention: {!r}'.format(_call)
    assert 'internal_use_lnL' not in _call, \
        'the zero-likelihood stand-in reads the OPTION again: {!r}'.format(_call)
    for anchor in ('log_res = numpy.log(res)', '_lr2 = numpy.log(_res2)',
                   'res = numpy.exp(log_res); var'):
        before = src[:src.index(anchor)].split('\n')
        gate = next(ln for ln in reversed(before) if ln.strip().startswith(('if ', 'elif ')))
        assert 'internal_use_lnL' not in gate, \
            'the convention at {!r} is still read off the OPTION: {!r}'.format(anchor, gate.strip())
        assert 'return_lnL' in gate, \
            'the convention at {!r} is not read off return_lnL: {!r}'.format(anchor, gate.strip())


###
### 6. --export-marginal-distance-grid must mean what it says
###

@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
@pytest.mark.parametrize('internal_use_lnL', [True, False])
def test_a_requested_dgrid_is_written_whatever_the_lnL_flag_says(internal_use_lnL):
    """The gate also required --internal-use-lnL, which decides nothing this exporter reads:
    the weights go through ln_weights_for_posterior with the record's own convention, and
    log_res now comes through return_lnL.  All it did was make --export-marginal-distance-grid
    silently do nothing on any backend the second flag was not also set for."""
    code = _driver_if_test('export_marginal_distance_grid')
    opts = _Opts(output_file='o', export_marginal_distance_grid=True,
                 distance_marginalization=False, internal_use_lnL=internal_use_lnL)
    assert eval(code, {'opts': opts}) is True


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
@pytest.mark.parametrize('override,expect', [
    ({'export_marginal_distance_grid': False}, False),   # not requested
    ({'distance_marginalization': True}, False),         # d already marginalized away
    ({'output_file': None}, False),                      # nowhere to write it
])
def test_the_dgrid_gate_still_refuses_the_cases_it_should(override, expect):
    """Widening a gate is only safe if the conditions that carried meaning survive."""
    code = _driver_if_test('export_marginal_distance_grid')
    kw = dict(output_file='o', export_marginal_distance_grid=True,
              distance_marginalization=False, internal_use_lnL=False)
    kw.update(override)
    assert bool(eval(code, {'opts': _Opts(**kw)})) is expect


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_replica_pass_undoes_exactly_what_the_main_pass_did():
    """analyze_event RETURNS res, and the replica arm converts (log_res, sqrt_var_over_res)
    back into (res, var).  That conversion is the INVERSE of the main pass's log_res block,
    so the two must share a predicate: read one off the sampler's convention and the other
    off the option and the round trip stops being one.  They disagree for
    `--sampler-method adaptive_cartesian --internal-use-lnL`, where the main pass correctly
    logs a linear integral and this block has to exponentiate it back.

    Added after a rebase: this block did not exist when the other three sites were fixed, and
    it arrived keyed off the option.  Running the two real source blocks back to back is what
    pins them together, rather than each one separately looking defensible."""
    fwd = _driver_slice('if not(return_lnL):\n      log_res', '# ---------')
    back = _driver_slice('# keep the (res, var) pair consistent', '    # Calibration MC error')
    for return_lnL, res0, var0 in [(False, np.exp(67.0), 3.0e56), (True, 67.0, 0.5)]:
        ns = {'numpy': np, 'res': res0, 'var': var0, 'return_lnL': return_lnL}
        exec(fwd, ns)
        exec(back, ns)
        assert ns['res'] == pytest.approx(res0, rel=1e-9), \
            'round trip changed res for return_lnL={}'.format(return_lnL)
        assert ns['var'] == pytest.approx(var0, rel=1e-9), \
            'round trip changed var for return_lnL={}'.format(return_lnL)


def test_the_plugin_base_class_carries_the_converter_so_subclasses_inherit_it():
    """The parametrized check above can only name the samplers that exist today, and the
    plugin samplers (nflow, and everything under integrators/unreliable_oracle) are
    discovered at runtime.  What actually makes the rule hold for those is that they all
    subclass MCSamplerGeneric, so assert it THERE -- otherwise a new plugin silently reopens
    the hole that `float(sampler.identity_convert(neff))` fell into.

    mcsampler.MCSampler, the one that was missing it, is the class that does NOT inherit from
    this base; that is why it had to be fixed directly."""
    from RIFT.integrators.mcsampler_generic import MCSamplerGeneric
    s = MCSamplerGeneric()
    assert hasattr(s, 'identity_convert'), 'MCSamplerGeneric has no identity_convert'
    probe = np.asarray([1.0, 2.0])
    assert np.allclose(np.asarray(s.identity_convert(probe)), probe)
    assert not issubclass(mcsampler.MCSampler, MCSamplerGeneric), \
        'mcsampler.MCSampler now inherits the base; drop its own converters rather than duplicating'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_a_failed_dgrid_export_is_loud_but_not_fatal():
    """Widening the gate makes backends RUN an exporter they used to skip, and the block reads
    `sampler._rvs["distance"]` / `sampler.prior_pdf["distance"]` unguarded.  Turning a silent
    no-op into a job-killing KeyError at the very end -- after the integration is done and the
    .dat is written -- would be a worse bug than the one being fixed.  So the export is
    wrapped, and the handler must NAME the failure rather than swallow it."""
    # PARSED, not sliced: the 3000-character window this used to take stopped reaching
    # save_distance_grid the moment the exporter grew a retained-set branch, and failed
    # with the guard still fully in place.  The claim is about structure, so ask for it.
    src = _driver_source()
    export_if = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.If) and 'export_marginal_distance_grid' in ast.dump(node.test):
            names = set(n.func.id for n in ast.walk(node)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name))
            if 'save_distance_grid' in names:
                export_if = node
                break
    assert export_if is not None, 'no .dgrid export block found; this test is looking at nothing'

    tries = [n for n in ast.walk(export_if) if isinstance(n, ast.Try)
             and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                     and c.func.id == 'save_distance_grid' for c in ast.walk(n))]
    assert tries, 'the .dgrid exporter runs unguarded'
    handlers = [h for t in tries for h in t.handlers]
    assert any(isinstance(h.type, ast.Name) and h.type.id == 'Exception' for h in handlers), \
        'no `except Exception` around the export'
    # Asserted on the PRINT, not on the handler body.  The handler also writes a
    # .dgrid.skipped sidecar that mentions _e_dgrid, so `'_e_dgrid' in <handler>` stayed
    # true when the ERROR line stopped naming the exception -- caught by reverting it.
    prints = [n for h in handlers for n in ast.walk(h)
              if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'print']
    assert prints, 'the handler prints nothing at all'
    said = [ast.dump(n) for n in prints]
    assert any('ERROR' in d and 'could NOT be' in d for d in said), \
        'the handler does not say the .dgrid was not written'
    assert any('ERROR' in d and '_e_dgrid' in d for d in said), \
        'the ERROR line does not report which exception it caught'
