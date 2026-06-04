"""
Build the two head-to-head figures from the saved test arrays.

  Figure A (relerr_vs_lnL.png): relative error (lnL_a - lnL_b) vs lnL --
    (left)  GP - RF                     (surrogate vs surrogate)
    (right) GP - data and RF - data     (both vs leave-some-out held-out points)
  Figure B (corner_test.png): corner plot of the GP posterior (mu-frame NUTS)
    overlaid on the production RF+AV benchmark.

Writes PNGs to ``paper/figures/`` (override with --outdir). The benchmark glob for
Figure B defaults to $BENCH and is skipped gracefully if absent (GP-only corner).

    $PY make_figures.py
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
# head_to_head -> export_likelihoods -> rift -> demo -> Code -> .../MonteCarloMarginalizeCode -> repo root
REPO = os.path.abspath(os.path.join(HERE, *([os.pardir] * 6)))
DEFAULT_OUT = os.path.join(REPO, "paper", "figures")
BENCH = os.environ.get("BENCH", "/home/oshaughn/jaxcip_benchmark/out/cip_rf_*.xml.gz")
LOW = ["mc", "delta_mc", "s1z", "s2z", "lambda1", "lambda2"]


def fig_relerr(outdir):
    d = np.load(os.path.join(RES, "surface.npz"))
    lnL, gp, rf, data = d["lnL_data"], d["gp_pred"], d["rf_pred"], d["lnL_data"]
    dlnL = d["lnL_peak"] - lnL                      # depth below the peak
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    ax[0].scatter(lnL, gp - rf, s=4, alpha=0.3, color="C0")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("GP $-$ RF  (surrogate vs surrogate)")
    ax[0].set_xlabel(r"$\ln\mathcal{L}$"); ax[0].set_ylabel(r"$\Delta\ln\mathcal{L}$")
    ax[1].scatter(lnL, gp - data, s=4, alpha=0.3, color="C0", label="GP $-$ data")
    ax[1].scatter(lnL, rf - data, s=4, alpha=0.3, color="C3", label="RF $-$ data")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("vs held-out data (leave-some-out)")
    ax[1].set_xlabel(r"$\ln\mathcal{L}$"); ax[1].legend(markerscale=3, framealpha=0.9)
    fig.suptitle("Relative lnL error over the dynamic range (BNS GW170817 test)")
    fig.tight_layout()
    p = os.path.join(outdir, "relerr_vs_lnL.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)


def _load_bench():
    files = sorted(glob.glob(BENCH))
    if not files:
        return None
    from RIFT.interpolators.jax_gp.applications.compare import load_param_pooled
    cols = []
    for p in LOW:
        try:
            cols.append(load_param_pooled([BENCH], p))
        except Exception:
            return None
    n = min(len(c) for c in cols)
    return np.column_stack([c[:n] for c in cols])


def fig_corner(outdir):
    d = np.load(os.path.join(RES, "gp_posterior.npz"))
    gp = np.asarray(d["samples"])
    labels = [r"$\mathcal{M}_c$", r"$\delta\mathcal{M}_c$", r"$s_{1z}$",
              r"$s_{2z}$", r"$\Lambda_1$", r"$\Lambda_2$"]
    try:
        import corner
    except Exception:
        print("corner not available; skipping Figure B"); return
    bench = _load_bench()
    fig = corner.corner(gp, labels=labels, color="C0", hist_kwargs={"density": True},
                        plot_datapoints=False, levels=(0.5, 0.9))
    if bench is not None:
        corner.corner(bench, fig=fig, color="C3", hist_kwargs={"density": True},
                      plot_datapoints=False, levels=(0.5, 0.9))
        title = "GP (blue) vs production RF+AV benchmark (red)"
    else:
        title = "GP posterior (benchmark glob $BENCH not found; overlay skipped)"
    fig.suptitle(title, y=1.02)
    p = os.path.join(outdir, "corner_test.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--which", default="all", choices=["all", "relerr", "corner"])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.which in ("all", "relerr"):
        fig_relerr(a.outdir)
    if a.which in ("all", "corner"):
        fig_corner(a.outdir)


if __name__ == "__main__":
    main()
