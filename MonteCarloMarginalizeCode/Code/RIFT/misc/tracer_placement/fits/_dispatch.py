"""build(method, X, Y, sigma=None) -> Fit."""
def build(method, X, Y, sigma=None, **kw):
    method = method.lower()
    if method == "rf":
        from ._rf import RandomForestFit
        return RandomForestFit(X, Y, sigma=sigma, **kw)
    if method == "rbf":
        from ._rbf import RBFFit
        return RBFFit(X, Y, sigma=sigma, **kw)
    if method == "quadratic":
        from ._quadratic import QuadraticFit
        return QuadraticFit(X, Y, sigma=sigma, **kw)
    if method == "polynomial":
        from ._polynomial import PolynomialFit
        return PolynomialFit(X, Y, sigma=sigma, **kw)
    raise ValueError(f"unknown fit method {method!r}")
