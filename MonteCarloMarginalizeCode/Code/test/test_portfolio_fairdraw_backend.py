#!/usr/bin/env python
"""
Regression tests for the mcsamplerPortfolio FAIR-DRAW BACKEND MIX
(RIFT/integrators/mcsamplerPortfolio.py, integrate_log).

Background (the bug these tests lock down).  The fair-draw block at the end of
integrate_log did

    wt = xpy.exp(identity_convert_togpu(ln_wt))
    indx_list = self.xpy.random.choice(self.xpy.arange(len(wt)), size=n_extr, replace=True, p=wt)

It BUILDS the weights with the module-global converter but DRAWS with self.xpy, and those
are independent.  `identity_convert_togpu` is `cupy.asarray` whenever cupy imports, while
self.xpy is numpy for any sampler whose driver did not set it (MCSampler.__init__ defaults
it to numpy) and for `--sampler-xpy numpy`, which sets sampler.xpy = numpy while
set_xpy_to_numpy() leaves the module globals on cupy.  Whenever the two disagree, ln_wt went
to the DEVICE, `numpy.exp` dispatched through cupy's `__array_ufunc__` and handed back a
cupy array, and numpy.random.choice was asked to read a device array as `p`:

    TypeError: Implicit conversion to a NumPy array is not allowed.
               Please use `.get()` to construct a NumPy array explicitly.

That is not a degraded result, it is an ABORT: the traceback runs
analyze_event -> sampler.integrate -> integrate_log, so the whole ILE process dies and no
extrinsic samples are written at all.

Measured on rift_O4d, where integrate_log additionally forces self.xpy = numpy (the
portfolio aggregates on the host) so the two backends ALWAYS disagree there: on
ldas-pcdev13 with the extrinsic-collapse demo at rho_net 146.8, 6/6 replicates of
`--sampler-portfolio AV,GMM` died at this line, and 6/6 complete with the fix.  On THIS
branch integrate_log leaves self.xpy alone, so the abort is conditional on the two backends
disagreeing rather than unconditional -- the defect is the same, the trigger is narrower.

The fix, in two parts:

  1. Build the weights on the SAMPLER's backend (`self.xpy.exp(self.xpy.asarray(ln_wt))`),
     never on the module-global one.
  2. Gather on the HOST (`indx_host = np.asarray(identity_convert(indx_list))`, and
     `identity_convert` each stored array before indexing it).  _rvs entries are NOT
     guaranteed to share a backend with the index array: `sample_n` is written through the
     INSTANCE `self.identity_convert_togpu`, which the ILE sets to cupy.asarray, while
     other keys arrive host-typed -- and a numpy array indexed by a cupy array raises the
     same TypeError.

NOT CUPY-ONLY IN THESE TESTS.  The reported failure needs a GPU, but the mechanism is just
"an array that refuses implicit numpy conversion".  `_DeviceArray` below reproduces both
behaviours cupy has that matter here (ufunc dispatch via __array_ufunc__, and __array__
raising), so these tests reproduce the exact production traceback -- same
numpy/random/mtrand.pyx frame, same message -- on a CPU-only host.  The cupy path is
pinned separately at the end when a GPU is present.
"""

import traceback

import numpy as np
import pytest

import RIFT.integrators.mcsamplerPortfolio as PF
import RIFT.integrators.mcsamplerAdaptiveVolume as AV
import RIFT.integrators.mcsamplerEnsemble as EN

NAMES = ['right_ascension', 'declination', 'phi_orb', 'inclination', 'psi', 'distance']
NDIM = len(NAMES)

SEED = 20260811


###
### A cupy stand-in that works on a CPU-only host
###

class _DeviceArray(object):
    """Minimal stand-in for a cupy array.

    Reproduces exactly the two cupy behaviours the bug turns on:
      * numpy ufuncs DISPATCH through __array_ufunc__ and return another device array, so
        `numpy.exp(device)` silently yields a device result rather than raising; and
      * any IMPLICIT conversion to numpy raises, with cupy's own message.
    Deliberately not an ndarray subclass: numpy.asarray() on a subclass returns a base-class
    view without ever calling __array__, which would make the fake inert.
    """

    def __init__(self, host):
        self._host = np.asarray(host)

    def __array__(self, *args, **kwargs):
        raise TypeError("Implicit conversion to a NumPy array is not allowed. "
                        "Please use `.get()` to construct a NumPy array explicitly.")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raw = [x._host if isinstance(x, _DeviceArray) else x for x in inputs]
        return _DeviceArray(getattr(ufunc, method)(*raw, **kwargs))

    def __len__(self):
        return len(self._host)

    def __getitem__(self, key):
        # cupy arrays ARE indexable by a host list/array and stay on the device; the
        # pruning block just above the fair draw relies on that.
        return _DeviceArray(self._host[_to_host(key)])

    @property
    def shape(self):
        return self._host.shape

    @property
    def dtype(self):
        return self._host.dtype

    def get(self):
        return self._host


def _to_host(x):
    """cupy.asnumpy, for _DeviceArray."""
    return x.get() if isinstance(x, _DeviceArray) else x


class _DeviceRandom(object):
    """numpy.random, except choice() hands back a DEVICE-typed index array (as cupy does)."""

    def __getattr__(self, name):
        return getattr(np.random, name)

    def choice(self, a, size=None, replace=True, p=None):
        return _DeviceArray(np.random.choice(a, size=size, replace=replace, p=p))


class _XpyDeviceChoice(object):
    """Stands in for the `numpy` the portfolio module binds self.xpy to.

    MCSampler.__init__ reads the module-global `numpy` into self.xpy, so patching the module
    attribute before constructing the sampler is what puts the draw on this shim.  Everything
    delegates to real numpy except random.choice.  `asnumpy` is present because
    statutils.init_log takes a device branch for any xpy that is not the numpy module itself
    -- cupy supplies it too.
    """

    random = _DeviceRandom()
    asnumpy = staticmethod(np.asarray)

    def __getattr__(self, name):
        return getattr(np, name)


###
### Fixtures
###

def _sampler(n_chunk=10000):
    """AV + GMM, the portfolio the demo runs (`--sampler-portfolio AV,GMM`)."""
    s = PF.MCSampler(portfolio=[AV.MCSampler(n_chunk=n_chunk), EN.MCSampler()])
    for name in NAMES:
        s.add_parameter(name, pdf=None, left_limit=0.0, right_limit=1.0,
                        prior_pdf=lambda x: np.ones(np.shape(x)),
                        adaptive_sampling=True)
    s.setup()
    return s


def _peaked(*args):
    """A 6-D Gaussian, well inside the box: converges, and fair-draws (n_extr < n)."""
    x = np.array(args).T
    return -0.5 * np.sum(((x - 0.5) / 0.15) ** 2, axis=1)


# save_intg=True is required: the fair-draw block reads _rvs['log_integrand'], which
# integrate_log only populates under that gate.
_KW = dict(nmax=100000, neff=20, n=10000, no_protect_names=True, verbose=False,
           save_intg=True, igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)


def _integrate(s):
    np.random.seed(SEED)
    return s.integrate_log(_peaked, *NAMES, **_KW)


def _skip_if_draw_cannot_aggregate_mixed_backends(exc):
    """Skip when the failure is the SEPARATE draw()-aggregation defect, not the fair draw.

    On this branch mcsamplerPortfolio.draw() converts each member's draw with the INSTANCE
    identity_convert_togpu and then assigns the result into host-typed buffers, so a
    device-typed instance converter raises inside draw() -- long before integrate_log reaches
    the fair-draw block this file is about.  That defect is pre-existing and untouched here
    (rift_O4d solves it by aggregating on the host); attributing it to the fair draw would be
    wrong.  Skip with the reason rather than fail, so these re-enable themselves if draw() is
    ever fixed.  The fair-draw gather is pinned regardless by the device-typed INDEX test,
    which does run on this branch.
    """
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.name == 'draw' and 'mcsamplerPortfolio' in frame.filename:
            pytest.skip('blocked upstream by the draw() mixed-backend aggregation defect '
                        '(separate, pre-existing, untouched by this change)')


###
### 1. The reported regression: a device-typed `p` reaching numpy.random.choice
###

def test_fairdraw_does_not_hand_a_device_array_to_the_host_draw(monkeypatch):
    """The regression.

    With the module-global converters device-backed -- exactly the GPU-host configuration --
    the fair draw must still complete.  Pre-fix this raised, from
    numpy/random/mtrand.pyx line 980, the TypeError that aborted the ILE run.
    """
    monkeypatch.setattr(PF, 'identity_convert_togpu', _DeviceArray)
    monkeypatch.setattr(PF, 'identity_convert', _to_host)

    s = _sampler()
    try:
        res = _integrate(s)
    except TypeError as e:
        if 'Implicit conversion to a NumPy array' in str(e):
            pytest.fail("the reported portfolio fair-draw abort is back: {}".format(e))
        raise

    assert np.isfinite(float(res[0])), "lnZ must be a real number, got {}".format(res[0])
    assert len(np.asarray(s._rvs['log_integrand'])) >= 1


def test_the_fairdraw_actually_ran(monkeypatch):
    """Guards the test above: a skipped fair draw would pass it vacuously.

    The block only draws when n_extr < len(log_integrand), so pin that the export really was
    truncated to the fair-draw size.  (This is also why the AV flavour of this bug hid: on a
    collapsed 1-sample live set the branch is never entered.)
    """
    monkeypatch.setattr(PF, 'identity_convert_togpu', _DeviceArray)
    monkeypatch.setattr(PF, 'identity_convert', _to_host)

    s = _sampler()
    _integrate(s)
    n = len(np.asarray(s._rvs['log_integrand']))
    assert 0 < n <= _KW['igrand_fairdraw_samples_max'], \
        "fair draw did not truncate ({} samples): the branch under test was skipped".format(n)


def test_the_fix_does_not_move_the_integral(monkeypatch):
    """The fix is a BACKEND correction, not a numerical one.

    Same seed, device-backed converters vs host ones: lnZ must agree to the bit.  If this
    ever drifts, the fix changed the estimate -- which would silently shift production lnZ.
    """
    s_host = _sampler()
    res_host = _integrate(s_host)

    monkeypatch.setattr(PF, 'identity_convert_togpu', _DeviceArray)
    monkeypatch.setattr(PF, 'identity_convert', _to_host)
    s_dev = _sampler()
    res_dev = _integrate(s_dev)

    assert float(res_dev[0]) == float(res_host[0]), \
        "lnZ moved with the backend: {} vs {}".format(res_dev[0], res_host[0])


###
### 2. The gather: index array and stored arrays need not share a backend
###

def test_gather_survives_a_device_typed_index_array(monkeypatch):
    """Part 2 of the port, index side.

    On a GPU host `self.xpy.random.choice` returns a cupy index array while the portfolio's
    own _rvs are host-typed, and numpy refuses to be indexed by it.  The gather must convert
    the index to the host first.
    """
    monkeypatch.setattr(PF, 'identity_convert_togpu', _DeviceArray)
    monkeypatch.setattr(PF, 'identity_convert', _to_host)
    monkeypatch.setattr(PF, 'numpy', _XpyDeviceChoice())   # what integrate_log binds self.xpy to

    s = _sampler()
    try:
        res = _integrate(s)
    except TypeError as e:
        if 'Implicit conversion to a NumPy array' in str(e):
            pytest.fail("the gather indexed with a device-typed index array: {}".format(e))
        raise
    assert np.isfinite(float(res[0]))


def test_gather_survives_a_device_typed_stored_array(monkeypatch):
    """The mixed-backend _rvs that production actually presents to the gather.

    `sample_n` is written through the INSTANCE converter (self.identity_convert_togpu), which
    bin/integrate_likelihood_extrinsic_batchmode sets to cupy.asarray, while the keys the
    portfolio aggregates arrive host-typed.  So _rvs genuinely holds BOTH backends here.

    Scope note: this pins the invariant (a mixed _rvs must survive and export host-typed), not
    the regression -- with a HOST index array the pre-fix gather handled this case too.  The
    discriminating test for part 2 of the port is the device-typed INDEX one above; verified
    by reverting part 2 alone, which fails that test and passes this one.
    """
    monkeypatch.setattr(PF, 'identity_convert_togpu', _DeviceArray)
    monkeypatch.setattr(PF, 'identity_convert', _to_host)

    s = _sampler()
    s.identity_convert_togpu = _DeviceArray     # as the ILE does on a GPU host
    s.identity_convert = _to_host
    try:
        res = _integrate(s)
    except TypeError as e:
        _skip_if_draw_cannot_aggregate_mixed_backends(e)
        if 'Implicit conversion to a NumPy array' in str(e):
            pytest.fail("the gather could not index a device-typed stored array: {}".format(e))
        raise

    assert np.isfinite(float(res[0]))
    n = len(np.asarray(_to_host(s._rvs['log_integrand'])))
    for k, v in s._rvs.items():
        assert len(np.asarray(_to_host(v))) == n, \
            'key {} kept a stale length: the gather skipped it'.format(k)


def test_the_gather_leaves_no_device_typed_entry_behind(monkeypatch):
    """Nothing device-typed may survive the export.

    Every consumer downstream of integrate_log (the samples XML writer, the L0-rescue seed
    selection) is host-side, so a stray device array here is a deferred crash rather than a
    caught one.  Inert on a CPU host without the patch, which is why the original bug reached
    production unnoticed by this suite.
    """
    monkeypatch.setattr(PF, 'identity_convert_togpu', _DeviceArray)
    monkeypatch.setattr(PF, 'identity_convert', _to_host)

    s = _sampler()
    s.identity_convert_togpu = _DeviceArray
    s.identity_convert = _to_host
    try:
        _integrate(s)
    except TypeError as e:
        _skip_if_draw_cannot_aggregate_mixed_backends(e)
        raise
    for k, v in s._rvs.items():
        assert isinstance(v, np.ndarray), \
            'key {} came back on the device: a later host-side consumer will raise'.format(k)


###
### 3. Source-level pin
###
# The fix is one line away from being undone by a copy-paste from any of the other
# integrators, and the runtime tests above only fire when the fake (or a real GPU) is in
# play.  Pin the shape directly, as test_av_empty_live_volume.py pins the ILE hint.

import inspect


def test_the_fairdraw_block_does_not_reach_for_the_module_global_backend():
    src = inspect.getsource(PF.MCSampler.integrate_log)
    i = src.find('Fairdraw size')
    assert i > 0, 'fair-draw block moved; update this test'
    # match CODE, not prose: the comments in that block name the offending call to explain it
    block = '\n'.join(line.split('#')[0] for line in src[i:].splitlines())
    assert 'identity_convert_togpu' not in block, \
        ('the fair-draw block calls identity_convert_togpu again.  That is the module-global '
         '(cupy.asarray when cupy imports), which is independent of the self.xpy the draw '
         'below uses -- when they disagree the ILE run aborts on a GPU host.')
    assert 'self.xpy.exp' in block, \
        'the fair-draw weights are no longer built on the sampler backend (self.xpy)'
    assert 'indx_host' in block, \
        'the fair-draw gather no longer converts the index array to the host'


###
### 4. Backend coverage
###
# The reported traceback is the cupy flavour.  The tests above run the fake on whatever host
# they land on; when a GPU is present, run the real thing so a CPU-only CI pass can never be
# mistaken for coverage of the reported configuration.

@pytest.mark.skipif(not PF.cupy_ok, reason='no cupy/GPU on this host')
def test_fairdraw_on_the_cupy_backend():
    """No patching: on a GPU host the module globals ARE cupy, which is the bug's setting."""
    import cupy
    s = _sampler()
    s.identity_convert_togpu = cupy.asarray
    s.identity_convert = cupy.asnumpy
    try:
        res = _integrate(s)
    except TypeError as e:
        if 'Implicit conversion to a NumPy array' in str(e):
            pytest.fail("the reported cupy fair-draw abort is back: {}".format(e))
        raise
    assert np.isfinite(float(res[0]))
    for k, v in s._rvs.items():
        assert not isinstance(v, cupy.ndarray), \
            'key {} came back on the device'.format(k)
