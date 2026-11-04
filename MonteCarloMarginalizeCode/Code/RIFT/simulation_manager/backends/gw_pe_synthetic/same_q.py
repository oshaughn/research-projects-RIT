"""same_q for the GW PE synthetic-targeted backend.

Two GW events are 'the same simulation' if their source-frame
parameters agree to a tolerance below typical PE precision. Tolerances
chosen to dedupe near-duplicates (re-runs with subtly different
floating-point values) without conflating distinct astrophysical
events. Adjust if your population calls for it.
"""

from __future__ import annotations


_TOL = {
    "mc":            1e-4,    # solar masses
    "eta":           1e-5,
    "distance":      1.0,     # Mpc
    "ra":            1e-4,    # rad
    "dec":           1e-4,    # rad
    "geocent_time":  1e-3,    # s
    "inclination":   1e-3,
    "polarization":  1e-3,
    "s1z":           1e-3,
    "s2z":           1e-3,
    "eccentricity":  1e-4,
}


def same_q(a, b):
    for k, tol in _TOL.items():
        va = a.get(k)
        vb = b.get(k)
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            return False
        try:
            if abs(float(va) - float(vb)) > tol:
                return False
        except (TypeError, ValueError):
            return False
    # Approximant must match exactly when set.
    if a.get("approximant") != b.get("approximant"):
        # treat None and 'IMRPhenomXPHM' as equal (the default)
        if (a.get("approximant") or "IMRPhenomXPHM") != \
           (b.get("approximant") or "IMRPhenomXPHM"):
            return False
    return True
