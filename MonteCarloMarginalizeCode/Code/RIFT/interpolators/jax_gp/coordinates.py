"""
Pure-JAX intrinsic-parameter coordinate transforms (differentiable).

This is a faithful JAX re-implementation of the *subset* of RIFT's
``lalsimutils.convert_waveform_coordinates`` machinery we need to fit/sample in
decorrelated coordinates while keeping gradients with respect to the *physical*
parameters (m1, m2, s1z, s2z, lambda1, lambda2).  The legacy NumPy path in
``lalsimutils`` / ``RIFT.misc.tools`` is the source of truth and is left
untouched; this module only mirrors it (and is validated against it in
``test_coordinates.py``).

Why: the exported GP lnL is a function of *fit* coordinates (e.g.
``mu1, mu2, delta_mc, LambdaTilde, DeltaLambdaTilde``).  Composing it with a
differentiable physical->fit map gives ``lnL(physical)`` whose ``jax.grad`` is the
gradient in physical parameters -- what a derivative-aware sampler / population
analysis actually wants.

Constants and formulas mirror ``RIFT/misc/tools.py`` (Morisaki mu1/mu2) and
``lalsimutils.tidal_lambda_tilde`` (Lackey et al. 2014, Eq. 5-6).
"""
from __future__ import annotations

import jax.numpy as jnp

# --- constants (must match RIFT/misc/tools.py exactly) --------------------- #
U = jnp.array([
    [0.97437198,  0.20868103,  0.08397302],
    [-0.22132704, 0.82273827,  0.52356096],
    [0.04016942, -0.52872863,  0.84783987],
])
FREF = 200.0
MSUN_TO_TIME = 4.92659e-6
_PI = jnp.pi


# --- basic mass relations -------------------------------------------------- #
def mc_of_m1m2(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def eta_of_m1m2(m1, m2):
    mt = m1 + m2
    return m1 * m2 / (mt * mt)


def q_of_m1m2(m1, m2):
    """Mass ratio in the tools.py convention q = m2/m1 (<= 1 for m1 >= m2)."""
    return m2 / m1


def delta_mc_of_m1m2(m1, m2):
    return (m1 - m2) / (m1 + m2)


def q_to_eta(q):
    return q / (1.0 + q) ** 2


# --- PN phase coefficients (Morisaki mu1/mu2) ------------------------------ #
def psi0(mc):
    return 0.75 * (8.0 * _PI * mc * MSUN_TO_TIME * FREF) ** (-5.0 / 3.0)


def psi2(mc, eta):
    return (psi0(mc) * (20.0 / 9.0) * (743.0 / 336.0 + 11.0 * eta / 4.0)
            * eta ** (-2.0 / 5.0) * (_PI * mc * MSUN_TO_TIME * FREF) ** (2.0 / 3.0))


def _beta_spin(q, a1z, a2z):
    return (((113.0 / 12.0 + 25.0 * q / 4.0) * a1z
             + q ** 2 * (113.0 / 12.0 + 25.0 / (4.0 * q)) * a2z)
            / (1.0 + q) ** 2)


def psi3(mc, q, a1z, a2z):
    eta = q_to_eta(q)
    return (psi0(mc) * (4.0 * _beta_spin(q, a1z, a2z) - 16.0 * _PI)
            * eta ** (-3.0 / 5.0) * _PI * mc * MSUN_TO_TIME * FREF)


def mu1mu2_of_mcq(mc, q, a1z, a2z):
    """(mc, q=m2/m1, s1z, s2z) -> (mu1, mu2) via U . [psi0, psi2, psi3]."""
    psis = jnp.stack([psi0(mc), psi2(mc, q_to_eta(q)), psi3(mc, q, a1z, a2z)])
    mu = U @ psis
    return mu[0], mu[1]


# --- tidal combinations (Lackey et al. 2014) ------------------------------- #
def lambda_tilde(m1, m2, lambda1, lambda2):
    """(m1, m2, lambda1, lambda2) -> (LambdaTilde, DeltaLambdaTilde)."""
    mt = m1 + m2
    eta = m1 * m2 / (mt * mt)
    # q here is sqrt(1-4eta) with the sign of (m1-m2); clip keeps grad finite at
    # the measure-zero equal-mass point.
    q = jnp.sqrt(jnp.clip(1.0 - 4.0 * eta, 1e-12, None)) * jnp.sign(m1 - m2)
    lt_sym = lambda1 + lambda2
    lt_asym = lambda1 - lambda2
    lam = ((1.0 + 7.0 * eta - 31.0 * eta ** 2) * lt_sym
           + q * (1.0 + 9.0 * eta - 11.0 * eta ** 2) * lt_asym)
    dlam = (q * (1.0 - 13272.0 * eta / 1319.0 + 8944.0 * eta ** 2 / 1319.0) * lt_sym
            + (1.0 - 15910.0 * eta / 1319.0 + 32850.0 * eta ** 2 / 1319.0
               + 3380.0 * eta ** 3 / 1319.0) * lt_asym)
    return (8.0 / 13.0) * lam, 0.5 * dlam


# --- canonical physical dict + coordinate registry ------------------------- #
def _canonical(theta, low_level_names):
    """Parse a low-level physical vector into a dict of canonical quantities.

    Supports low-level bases containing either {m1, m2} or {mc, delta_mc}
    (plus s1z, s2z, lambda1, lambda2; missing spin/tidal default to 0).
    """
    v = {name: theta[i] for i, name in enumerate(low_level_names)}
    if "m1" in v and "m2" in v:
        m1, m2 = v["m1"], v["m2"]
    elif "mc" in v and "delta_mc" in v:
        eta = 0.25 * (1.0 - v["delta_mc"] ** 2)
        mtot = v["mc"] * eta ** (-3.0 / 5.0)
        m1 = 0.5 * mtot * (1.0 + v["delta_mc"])
        m2 = 0.5 * mtot * (1.0 - v["delta_mc"])
    else:
        raise ValueError(
            "low_level_names must contain {m1,m2} or {mc,delta_mc}; got "
            + repr(low_level_names))
    return {
        "m1": m1, "m2": m2,
        "s1z": v.get("s1z", 0.0), "s2z": v.get("s2z", 0.0),
        "lambda1": v.get("lambda1", 0.0), "lambda2": v.get("lambda2", 0.0),
        "mc": mc_of_m1m2(m1, m2), "eta": eta_of_m1m2(m1, m2),
        "q": q_of_m1m2(m1, m2), "delta_mc": delta_mc_of_m1m2(m1, m2),
    }


#: fit-coordinate names this module can produce (subset of lalsimutils')
SUPPORTED = (
    "m1", "m2", "mc", "eta", "q", "delta_mc", "mtot",
    "s1z", "s2z", "xi", "chi_eff",
    "mu1", "mu2", "lambda1", "lambda2", "LambdaTilde", "DeltaLambdaTilde",
)


def _coord_value(name, P):
    if name in ("m1", "m2", "s1z", "s2z", "lambda1", "lambda2",
                "mc", "eta", "q", "delta_mc"):
        return P[name]
    if name == "mtot":
        return P["m1"] + P["m2"]
    if name in ("xi", "chi_eff"):
        return (P["m1"] * P["s1z"] + P["m2"] * P["s2z"]) / (P["m1"] + P["m2"])
    if name in ("mu1", "mu2"):
        mu1, mu2 = mu1mu2_of_mcq(P["mc"], P["q"], P["s1z"], P["s2z"])
        return mu1 if name == "mu1" else mu2
    if name in ("LambdaTilde", "DeltaLambdaTilde"):
        lt, dlt = lambda_tilde(P["m1"], P["m2"], P["lambda1"], P["lambda2"])
        return lt if name == "LambdaTilde" else dlt
    raise ValueError("Unsupported fit coordinate {!r}; supported: {}".format(
        name, SUPPORTED))


def make_transform(low_level_names, fit_coord_names):
    """Return a pure-JAX callable ``theta_phys[d_low] -> fit[d_fit]``.

    ``low_level_names`` label the input columns (physical params); the returned
    function is differentiable and vmappable, so ``lnL(fit(theta))`` differentiates
    with respect to the physical parameters.
    """
    low_level_names = list(low_level_names)
    fit_coord_names = list(fit_coord_names)
    for n in fit_coord_names:
        if n not in SUPPORTED:
            raise ValueError("Unsupported fit coordinate {!r}; supported: {}"
                             .format(n, SUPPORTED))

    def transform(theta):
        P = _canonical(theta, low_level_names)
        return jnp.stack([_coord_value(n, P) for n in fit_coord_names])

    return transform
