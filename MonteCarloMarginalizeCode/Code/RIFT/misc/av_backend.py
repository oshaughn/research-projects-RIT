"""Keep AV host sampling independent of deterministic GPU fit prediction."""
import inspect
import numpy
from scipy import special


def needs_host_av(fit_method, gp_predict_backend="sklearn", gp_torch_device="auto"):
    """True only for the opt-in GPU GP predictors; default AV keeps its backend."""
    if gp_predict_backend == "cupy":
        return True
    if fit_method != "gp-torch" or gp_torch_device == "cpu":
        return False
    if gp_torch_device != "auto":
        return True
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def configure_host_av(module):
    """Select NumPy after optional sampler discovery has finished.

    Native AV functions capture the import-time backend in Python defaults.
    Replacing only CuPy module defaults preserves all sampling arithmetic,
    priors, adaptation and stopping choices while keeping samples on the host.
    """
    module.cupy_ok = False
    module.identity_convert = lambda x: x
    module.identity_convert_togpu = lambda x: x
    module.cupy_pi = numpy.pi
    module.xpy_default = numpy
    module.xpy_special_default = special
    for value in list(vars(module).values()) + list(vars(module.MCSampler).values()):
        if not inspect.isfunction(value) or value.__module__ != module.__name__:
            continue
        # Optional discovery may retain functions whose globals belong to a
        # separately executed copy of the native module. Select both copies.
        for key in ('identity_convert', 'identity_convert_togpu'):
            if key in value.__globals__:value.__globals__[key] = lambda x: x
        if 'cupy_ok' in value.__globals__:value.__globals__['cupy_ok'] = False
        if 'cupy_pi' in value.__globals__:value.__globals__['cupy_pi'] = numpy.pi
        if 'xpy_default' in value.__globals__:
            value.__globals__['xpy_default'] = numpy
        if 'xpy_special_default' in value.__globals__:
            value.__globals__['xpy_special_default'] = special
        def host_backend(backend):
            if getattr(backend, '__name__', '') == 'cupy':return numpy
            if getattr(backend, '__name__', '') == 'cupyx.scipy.special':return special
            return backend
        if value.__defaults__:
            value.__defaults__ = tuple(host_backend(x) for x in value.__defaults__)
        if value.__kwdefaults__:
            value.__kwdefaults__ = {k: host_backend(x) for k, x in value.__kwdefaults__.items()}
