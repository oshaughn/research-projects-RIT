"""
Head-to-head marginal comparison: GP-surrogate posterior vs standard RIFT posterior.

Both inputs are named-column ``.dat`` sample files (the standard posterior from
``util_ConstructEOSPosterior.py`` / CIP, and the GP posterior from
``gp_from_grid.py``).  For each requested parameter we report the Jensen-Shannon
divergence (bits) between the two 1D marginals, with a bootstrap stderr so you can
tell when a small JS is statistics-limited rather than real agreement.

Derived parameters are computed on the fly when the primaries are present:
``mc`` and ``q`` (and ``eta``) from ``m1, m2``.  This lets a head-to-head over the
sampled coordinates (e.g. m1, m2, dist) also be read in the physically natural
chirp-mass / mass-ratio marginals.

Usage::

    python compare_marginals.py --standard joint_posterior.dat --gp gp_posterior.dat \\
        --param mc --param q --param dist
"""
from __future__ import annotations

import argparse

import numpy as np

from RIFT.interpolators.jax_gp.applications.compare import js_with_stderr


def _derive(cols):
    """Add derived columns (mc, eta, q; LambdaTilde) from primaries when present."""
    out = dict(cols)
    if "m1" in cols and "m2" in cols:
        m1 = np.asarray(cols["m1"], float); m2 = np.asarray(cols["m2"], float)
        hi = np.maximum(m1, m2); lo = np.minimum(m1, m2)
        out.setdefault("mc", (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)
        out.setdefault("eta", (m1 * m2) / (m1 + m2) ** 2)
        out.setdefault("q", lo / hi)
        if "lambda1" in cols and "lambda2" in cols and "LambdaTilde" not in out:
            l1 = np.asarray(cols["lambda1"], float); l2 = np.asarray(cols["lambda2"], float)
            mt = m1 + m2
            # standard mass-weighted combination (Favata/Wade)
            out["LambdaTilde"] = (16.0 / 13.0) * (
                (m1 + 12 * m2) * m1 ** 4 * l1 + (m2 + 12 * m1) * m2 ** 4 * l2) / mt ** 5
    return out


def _load(path):
    data = np.genfromtxt(path, names=True, comments="#")
    return _derive({n: np.asarray(data[n], float) for n in data.dtype.names})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--standard", required=True, help="standard RIFT posterior .dat")
    p.add_argument("--gp", required=True, help="GP-surrogate posterior .dat")
    p.add_argument("--param", action="append", required=True, dest="params",
                   help="parameter to compare (repeatable); derived mc/q/eta allowed")
    p.add_argument("--bins", type=int, default=80)
    a = p.parse_args(argv)

    std = _load(a.standard)
    gp = _load(a.gp)

    print("# JS divergence (bits) GP-vs-standard, per marginal "
          "(PE bar ~ few x 1e-3; report widths not just means)")
    print("# {:10s} {:>10s} {:>9s}   {:>22s} {:>22s}".format(
        "param", "JS", "+/-", "standard mean+/-std", "GP mean+/-std"))
    worst = 0.0
    for prm in a.params:
        if prm not in std or prm not in gp:
            print("  {:10s} (missing in {})".format(
                prm, "standard" if prm not in std else "GP"))
            continue
        js, err = js_with_stderr(std[prm], gp[prm], bins=a.bins)
        worst = max(worst, js)
        print("  {:10s} {:10.4f} {:9.4f}   {:11.5g} {:9.3g} {:11.5g} {:9.3g}".format(
            prm, js, err, std[prm].mean(), std[prm].std(),
            gp[prm].mean(), gp[prm].std()))
    print("# worst-marginal JS = {:.4f} bits".format(worst))


if __name__ == "__main__":
    main()
