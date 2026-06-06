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

import jax as _jax

if not _jax.config.read("jax_enable_x64"):
    _jax.config.update("jax_enable_x64", True)

from .interface import BaseInterpolator  # noqa: E402

__all__ = ["BaseInterpolator"]

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
