"""
jax_gp : robust, scalable, AD-compatible likelihood interpolators for RIFT/CIP.

This is an *optional* subpackage.  It is never imported by the production CIP
path unless the user explicitly selects a ``gp-jax-*`` fit method, so the JAX
dependency stack (jax, optax, equinox, tinygp, ...) is not required for normal
operation.

All interpolators share the contract defined in ``interface.BaseInterpolator``:

    model = SomeInterpolator(...).fit(X, y, y_errors=...)
    fn    = model.predict_callable()        # callable(np.ndarray[n,d]) -> np.ndarray[n]
    v, g  = model.lnL_and_grad(theta)        # differentiable lnL + gradient at one point

``predict_callable`` is the drop-in for the existing CIP fit dispatch (every
``fit_*`` there returns exactly such a callable).  ``lnL_and_grad`` -- and the
pure-JAX closure behind it -- is what makes the exported surrogate differentiable
for downstream users.

We enable 64-bit JAX on import: lnL spans a large dynamic range and single
precision is not adequate for faithful gradients.  This is process-global, but
only takes effect once this opt-in subpackage is imported.
"""
from __future__ import annotations

# The docstring above calls this subpackage OPTIONAL, but importing it used to require jax
# unconditionally -- so merely TOUCHING the package died with ModuleNotFoundError on a machine
# without the stack.  Pytest touches it: collecting any test module in this directory imports
# this __init__ first, which is why a skip guard inside test_interpolators.py could never fire.
#
# Only the ABSENCE is tolerated here.  When jax is present the sequence below is unchanged and
# stays EAGER on purpose: three callers import this package for no reason but its side effect
# (applications/compare.py, applications/jax_cip.py, applications/export_at_scale.py all say
# "enables float64"), and x64 must be set before any submodule builds a jax array.  Deferring it
# into get_interpolator() would leave those three silently in float32, which is a wrong-gradient
# bug that raises nothing.
try:
    import jax as _jax
except ImportError:  # pragma: no cover - exercised only where the jax stack is absent
    _jax = None
else:
    if not _jax.config.read("jax_enable_x64"):
        _jax.config.update("jax_enable_x64", True)

    from .interface import BaseInterpolator  # noqa: E402

__all__ = ["BaseInterpolator"]


def __getattr__(name):
    """Re-raise the real ImportError for the eager exports when jax is missing.

    Without this the jax-absent case reports a bare AttributeError, which reads like a typo
    rather than a missing dependency.  Unknown names still raise AttributeError, so
    ``from RIFT.interpolators.jax_gp import export`` (a SUBMODULE) keeps working -- the import
    machinery falls back to importing the submodule when this returns AttributeError.
    """
    if name in __all__:
        from . import interface  # raises ModuleNotFoundError naming the missing package
        return getattr(interface, name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))

# Method classes are imported lazily by name to avoid importing every backend
# (and its heavier deps, e.g. tinygp) when only one is needed.
def get_interpolator(name):
    """Return the interpolator class registered under ``name``.

    Names mirror the CIP ``--fit-method`` values (without the ``gp-jax-`` prefix):
    ``rff``, ``exact``, ``svgp``.
    """
    name = name.lower().replace("gp-jax-", "").replace("gp_jax_", "")
    if name in ("", "rff"):                 # RFF is the default jax method
        from .rff import RFFInterpolator
        return RFFInterpolator
    if name == "exact":
        from .exact import ExactGPInterpolator
        return ExactGPInterpolator
    if name == "svgp":
        from .svgp import SVGPInterpolator
        return SVGPInterpolator
    if name in ("quadgp", "quad", "quad-gp"):
        from .quad_gp import QuadraticPlusGPInterpolator
        return QuadraticPlusGPInterpolator
    raise ValueError("Unknown jax_gp interpolator: {!r}".format(name))
