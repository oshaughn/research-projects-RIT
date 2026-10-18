"""mcsamplerGPU.integrate_log must aggregate on the INSTANCE backend, self.xpy.

THE DEFECT.  integrate_log's signature read `xpy=xpy_default`, bound when the module was
imported.  On a cupy host that is cupy, while self.xpy is numpy unless a caller sets it (the
constructor default, and ILE's --sampler-xpy numpy; set_xpy_to_numpy() only assigns locals).
The draws and the integrand were then host arrays and init_log fed them to cupy.exp:
"TypeError: Unsupported type <class 'numpy.ndarray'>".  A cupy-free runner binds numpy and
cannot see it.

The first test pins the wiring on any host.  The rest need a usable cupy device and assert
that they reached it; on a cupy-free runner they skip, and a skip there proves nothing.
"""
import numpy as np
import pytest
import scipy.special
from scipy.special import erf

import RIFT.integrators.mcsamplerGPU as mcsamplerGPU

REAL_CUPY = bool(getattr(mcsamplerGPU, 'cupy_ok', False))
needs_device = pytest.mark.skipif(not REAL_CUPY, reason="needs a usable cupy device")

LO, HI, MU, SIG = -1.0, 3.0, 0.2, 0.4
EXACT = np.log(SIG * np.sqrt(2 * np.pi) * 0.5 *
               (erf((HI - MU) / SIG / np.sqrt(2)) - erf((LO - MU) / SIG / np.sqrt(2))))


def _sampler(on_device):
    """On a device instance the caller supplies backend-agnostic pdf/cdf_inv, as ILE does;
    add_parameter's numerical cdf_inverse is a scipy interpolant and host-only."""
    s = mcsamplerGPU.MCSampler()
    kw = dict(pdf=np.vectorize(lambda x: 1.0))         # UNNORMALIZED -> _pdf_norm = 4
    if on_device:
        import cupy
        s.xpy = cupy
        s.identity_convert = cupy.asnumpy
        s.identity_convert_togpu = cupy.asarray
        kw = dict(pdf=lambda x: 0 * x + 1.0 / (HI - LO),
                  cdf_inv=lambda u: LO + (HI - LO) * u)
    s.add_parameter("xx", prior_pdf=lambda x: 0 * x + 1.0,
                    left_limit=LO, right_limit=HI, adaptive_sampling=True, **kw)
    return s


def _host_lnL(x):
    x = x.get() if hasattr(x, 'get') else np.asarray(x)
    return -0.5 * ((x - MU) / SIG) ** 2


def _device_lnL(x):
    import cupy
    x = cupy.asarray(x)
    return -0.5 * ((x - MU) / SIG) ** 2


def _run(s, lnL, seed=11, use_lnL=True, **kw):
    np.random.seed(seed)
    if REAL_CUPY:
        import cupy
        cupy.random.seed(seed)
    if not use_lnL:
        f = lnL
        lnL = lambda x: s.xpy.exp(s.xpy.asarray(f(x)) if s.xpy is not np else f(x))
    return s.integrate(lnL, "xx", n=1000, nmax=20000, neff=4000, use_lnL=use_lnL,
                       return_lnI=True, save_intg=True, no_protect_names=True,
                       verbose=False, **kw)


def _spy_draws(monkeypatch, s):
    """Record the array module of every joint_p_s draw_simplified hands to integrate_log."""
    seen = []
    orig = s.draw_simplified

    def spy(*a, **k):
        out = orig(*a, **k)
        seen.append(type(out[0]))
        return out
    monkeypatch.setattr(s, 'draw_simplified', spy)
    return seen


def test_aggregates_are_handed_the_instance_backend(monkeypatch):
    """Wiring: init_log/update_log/finalize_log receive self.xpy and a matching special.

    self.xpy is a stand-in object distinct from numpy and cupy, so the import-time default
    cannot satisfy this on either kind of host."""
    class _Backend(object):
        def __getattr__(self, name):
            return getattr(np, name)
    backend = _Backend()
    calls = []

    def recorder(fn):
        def inner(*a, xpy=None, special=None, **k):
            calls.append((fn.__name__, xpy, special))
            if special is not None:
                k['special'] = scipy.special
            return fn(*a, xpy=np, **k)
        return inner
    # host branch of cdf_inverse_from_hist even where cupy is importable
    monkeypatch.setattr(mcsamplerGPU, 'cupy_ok', False)
    for name in ('init_log', 'update_log', 'finalize_log'):
        monkeypatch.setattr(mcsamplerGPU, name, recorder(getattr(mcsamplerGPU, name)))
    s = _sampler(on_device=False)
    s.xpy = backend
    _run(s, _host_lnL)
    assert {c[0] for c in calls} == {'init_log', 'update_log', 'finalize_log'}, calls
    assert all(c[1] is backend for c in calls), \
        "aggregates got %r, not the instance backend" % ({type(c[1]) for c in calls},)
    # numpy instance -> scipy.special; anything else -> the module's special
    s2 = _sampler(on_device=False)
    calls.clear()
    _run(s2, _host_lnL)
    assert all(c[1] is np for c in calls)
    assert all(c[2] is scipy.special for c in calls if c[0] != 'finalize_log')


@needs_device
def test_host_instance_on_a_cupy_host(monkeypatch):
    """The reported failure: default (numpy) instance, cupy importable."""
    assert mcsamplerGPU.xpy_default is not np, "module did not bind cupy; lane not exercised"
    s = _sampler(on_device=False)
    seen = _spy_draws(monkeypatch, s)
    res = _run(s, _host_lnL)
    assert seen and all(t is np.ndarray for t in seen)
    assert s.pdf["xx"] is not s.pdf_initial["xx"], "adaptation never fired"
    assert abs(res[0] - EXACT) < 0.10


@needs_device
def test_host_instance_with_a_device_likelihood(monkeypatch):
    """--sampler-xpy numpy with a --gpu likelihood: lnL arrives on the device and must come
    back to the host.  identity_convert_togpu is the identity on this instance."""
    s = _sampler(on_device=False)
    res = _run(s, _device_lnL)
    assert abs(res[0] - EXACT) < 0.10


@needs_device
@pytest.mark.parametrize("lnL", [_host_lnL, _device_lnL], ids=["host_lnL", "device_lnL"])
def test_device_instance_runs_on_the_device(monkeypatch, lnL):
    """self.xpy = cupy.  Assert the draws really are device arrays, then the evidence."""
    import cupy
    s = _sampler(on_device=True)
    seen = _spy_draws(monkeypatch, s)
    res = _run(s, lnL)
    assert seen and all(t is cupy.ndarray for t in seen), \
        "draw_simplified returned %r, not cupy arrays" % (set(seen),)
    print("device:", cupy.cuda.Device().id,
          cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)['name'])
    assert s.pdf["xx"] is not s.pdf_initial["xx"], "adaptation never fired"
    assert abs(res[0] - EXACT) < 0.10


@needs_device
@pytest.mark.parametrize("use_lnL", [True, False], ids=["log", "linear"])
@pytest.mark.parametrize("on_device", [False, True], ids=["host_instance", "device_instance"])
def test_fairdraw_export_on_a_cupy_host(on_device, use_lnL):
    """The fair draw gathers every _rvs key with an index drawn on self.xpy.  sample_n was
    built with numpy.arange, so a device instance (ILE's default sampler on a GPU node, with
    --fairdraw-extrinsic-output) could not index it."""
    s = _sampler(on_device=on_device)
    _run(s, _host_lnL, use_lnL=use_lnL,
         igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=500)
    assert getattr(s, '_rvs_is_fairdraw', False), "the fair draw did not run"
    assert all(isinstance(v, np.ndarray) for v in s._rvs.values()), \
        {k: type(v) for k, v in s._rvs.items()}
    lengths = {k: len(v) for k, v in s._rvs.items()}
    assert len(set(lengths.values())) == 1, lengths


def test_no_signature_binds_the_import_time_backend():
    """`xpy=xpy_default` in a signature fixes the backend at import, whatever the caller holds.
    That was the integrate_log defect and the prior helpers' (uniform_samp_dec, cos_samp, ...).
    AST, so it runs on any host."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(mcsamplerGPU))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            args = node.args
            pairs = list(zip(args.args[len(args.args) - len(args.defaults):], args.defaults))
            pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
            for a, d in pairs:
                if isinstance(d, ast.Name) and d.id == 'xpy_default':
                    bad.append("%s(%s=xpy_default) line %d" % (
                        getattr(node, 'name', '<lambda>'), a.arg, node.lineno))
    assert not bad, bad


def test_prior_helpers_do_not_read_the_import_time_backend():
    """The signature pin above misses `xpy = xpy or xpy_default` in a body, or a lambda that
    reads xpy_default.  No reference at all inside the helpers."""
    import ast
    import inspect
    names = {h[0] for h in HELPERS} | {"ret_uniform_samp_vector_alt"}
    tree = ast.parse(inspect.getsource(mcsamplerGPU))
    bad = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id == 'xpy_default':
                    bad.append("%s line %d" % (node.name, sub.lineno))
    assert not bad, bad


HELPERS = [("uniform_samp_phase", ()), ("uniform_samp_psi", ()), ("uniform_samp_theta", ()),
           ("uniform_samp_dec", ()), ("cos_samp", ()), ("dec_samp", ()),
           ("cos_samp_cdf_inv_vector", ()), ("dec_samp_cdf_inv_vector", ()),
           ("uniform_samp_vector", (0.0, 1.0)), ("uniform_samp_withfloor_vector", (0.5, 1.0, 0.1))]


@needs_device
@pytest.mark.parametrize("name,lead", HELPERS, ids=[h[0] for h in HELPERS])
def test_prior_helpers_follow_their_argument(name, lead):
    """Host in, host out; device in, device out -- on a host where cupy is the module default."""
    import cupy
    fn = getattr(mcsamplerGPU, name)
    x = np.linspace(0.05, 0.95, 7)
    host = fn(*lead, x)
    dev = fn(*lead, cupy.asarray(x))
    assert isinstance(host, np.ndarray), type(host)
    assert isinstance(dev, cupy.ndarray), type(dev)
    assert np.allclose(host, cupy.asnumpy(dev))
    alt = mcsamplerGPU.ret_uniform_samp_vector_alt(0.0, 2.0)
    assert isinstance(alt(x), np.ndarray) and isinstance(alt(cupy.asarray(x)), cupy.ndarray)
