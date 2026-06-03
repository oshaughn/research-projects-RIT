"""
Compare intrinsic-posterior 1D marginals via Jensen-Shannon divergence.

The success metric for jax_cip is NOT sampler ESS -- it is how close the recovered
posterior is to the production CIP+RF benchmark. We quantify that with the JS
divergence of each 1D marginal (start with mc, the best-measured parameter), in bits
(log base 2, so JS in [0, 1]).

Usage:
    python -m RIFT.interpolators.jax_gp.applications.compare \
        --a jax_out.xml.gz --b cip_rf_out.xml.gz --param mc

Both inputs are RIFT ChooseWaveformParams XML (what CIP and jax_cip both write).

Caveat (per design notes): a reliable JS needs enough *independent* samples in BOTH
posteriors. jax_cip's per-run effective sample count is modest, so accumulate over
many seeds/instances (Condor) before trusting a small JS. ``js_divergence_1d`` here
also returns a bootstrap stderr so you can see when you are statistics-limited.
"""
from __future__ import annotations

import argparse

import numpy as np

import RIFT.interpolators.jax_gp  # noqa: F401  (enable float64 / consistent env)


def js_divergence_1d(a, b, bins=80, value_range=None):
    """Jensen-Shannon divergence (bits) between two 1D sample sets via histograms."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if value_range is None:
        lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
        if hi <= lo:
            hi = lo + 1e-12
        value_range = (lo, hi)
    edges = np.linspace(value_range[0], value_range[1], bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    w = np.diff(edges)
    pa = pa * w; pb = pb * w                       # -> probabilities per bin
    pa = pa / pa.sum(); pb = pb / pb.sum()
    m = 0.5 * (pa + pb)

    def _kl(p, q):
        mask = p > 0
        return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

    return float(0.5 * _kl(pa, m) + 0.5 * _kl(pb, m))


def js_with_stderr(a, b, bins=80, n_boot=200, seed=0):
    """JS plus a bootstrap stderr (so you can tell when you're statistics-limited)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
    base = js_divergence_1d(a, b, bins=bins, value_range=(lo, hi))
    rng = np.random.default_rng(seed)
    boots = [js_divergence_1d(rng.choice(a, len(a)), rng.choice(b, len(b)),
                              bins=bins, value_range=(lo, hi)) for _ in range(n_boot)]
    return base, float(np.std(boots))


def load_param_from_xml(fname, param="mc"):
    """Load a 1D parameter array (default mc, in Msun) from a RIFT samples XML."""
    import lal
    import RIFT.lalsimutils as lalsimutils
    P_list = lalsimutils.xml_to_ChooseWaveformParams_array(fname)
    vals = []
    for P in P_list:
        if param == "mc":
            vals.append(lalsimutils.mchirp(P.m1, P.m2) / lal.MSUN_SI)
        elif param in ("m1", "m2"):
            vals.append(getattr(P, param) / lal.MSUN_SI)
        else:
            vals.append(P.extract_param(param))
    return np.asarray(vals, float)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="samples XML A (e.g. jax_cip output)")
    p.add_argument("--b", required=True, help="samples XML B (e.g. CIP+RF benchmark)")
    p.add_argument("--param", default="mc")
    p.add_argument("--bins", type=int, default=80)
    a = p.parse_args(argv)
    va = load_param_from_xml(a.a, a.param)
    vb = load_param_from_xml(a.b, a.param)
    js, se = js_with_stderr(va, vb, bins=a.bins)
    print("param={} : A n={} mean={:.6g}  B n={} mean={:.6g}".format(
        a.param, len(va), va.mean(), len(vb), vb.mean()))
    print("JS(A,B) = {:.4f} +/- {:.4f} bits".format(js, se))


if __name__ == "__main__":
    main()
