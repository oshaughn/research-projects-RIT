"""
Run the GP<->RF head-to-head test stages, saving arrays for the figures.

Stages (pick one with --stage):
  surface   : fit RF and GP on a TRAIN split; evaluate on a held-out split
              (leave-some-out). Saves held-out lnL, GP/RF predictions, sigma ->
              results/surface.npz  (Figure A: relative error vs lnL).
  posterior : fit the GP on the on-support backbone; sample it with
              mu-frame-preconditioned NUTS. Saves low-level posterior samples ->
              results/gp_posterior.npz  (Figure B: corner vs the RF benchmark).

    $PY run_test.py --stage surface
    $PY run_test.py --stage posterior
(usually invoked via the Makefile, which sets PY/PYTHONPATH for you.)
"""
import argparse
import os

import numpy as np

import lib

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
D_ACC = 20.0
CAP = 9000
SEED = 0
# prior box (matches the production benchmark: narrow mc, small spin, BNS tides)
BOX = {"mc": (1.196, 1.199), "delta_mc": (0.0, 0.9), "s1z": (-0.05, 0.05),
       "s2z": (-0.05, 0.05), "lambda1": (0.01, 4000.0), "lambda2": (0.01, 4000.0)}


def stage_surface():
    Xfit, Xlow, y, yerr, _, dom = lib.load_oracle()
    idx = lib.backbone(Xfit, y, yerr, depth=D_ACC, cap=CAP, seed=SEED)
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(idx))
    nh = max(800, int(0.2 * len(idx)))
    te, tr = idx[perm[:nh]], idx[perm[nh:]]

    from RIFT.interpolators.jax_gp.benchmark.baselines import RFBaseline
    rf_tr = RFBaseline().fit(Xfit[tr], y[tr], y_errors=yerr[tr])
    gp_tr = lib.fit_gp(Xfit[tr], y[tr], yerr=yerr[tr])
    out = dict(lnL_data=y[te], sigma=yerr[te],
               gp_pred=np.asarray(gp_tr.predict(Xfit[te])),
               rf_pred=np.asarray(rf_tr.predict(Xfit[te])),
               lnL_peak=float(y.max()))
    np.savez(os.path.join(RES, "surface.npz"), **out)
    d = out["gp_pred"] - out["rf_pred"]
    w = np.exp(y[te] - y[te].max()); w /= w.sum()
    print("[surface] held-out {} pts; weighted RMS(GP-RF)={:.3f} nat; "
          "RMS(GP-data)={:.3f}, RMS(RF-data)={:.3f}".format(
              len(te), np.sqrt(np.sum(w * d ** 2)),
              np.sqrt(np.sum(w * (out["gp_pred"] - y[te]) ** 2)),
              np.sqrt(np.sum(w * (out["rf_pred"] - y[te]) ** 2))))
    print("  -> results/surface.npz")


def stage_posterior():
    import jax  # noqa
    from RIFT.interpolators.jax_gp import coordinates
    from RIFT.interpolators.jax_gp.applications.jax_cip import (
        _muframe_proposal, sample_nuts_muframe)

    Xfit, Xlow, y, yerr, _, dom = lib.load_oracle()
    idx = lib.backbone(Xfit, y, yerr, depth=D_ACC, cap=CAP, seed=SEED)
    model = lib.fit_gp(Xfit[idx], y[idx], yerr=yerr[idx])

    tf = coordinates.make_transform(lib.LOW, lib.FIT)
    def lnL_low(theta):
        return model.lnL_physical(tf(theta))

    box_lo = np.array([BOX[n][0] for n in lib.LOW])
    box_hi = np.array([BOX[n][1] for n in lib.LOW])
    in_prior = np.all((Xlow >= box_lo) & (Xlow <= box_hi), axis=1)
    gmean, gcov = _muframe_proposal(lib.LOW, lib.FIT, Xlow[in_prior], y[in_prior],
                                    box_lo, box_hi)
    res = sample_nuts_muframe(lnL_low, gmean, gcov, box_lo, box_hi,
                              num_warmup=800, num_samples=4000, num_chains=2, seed=SEED)
    np.savez(os.path.join(RES, "gp_posterior.npz"),
             samples=res["samples"], names=np.array(lib.LOW),
             ess=res["ess"], n_div=res["n_divergences"])
    print("[posterior] nuts-mu ESS(min) {:.0f} ({:.1%}), {} div -> "
          "results/gp_posterior.npz".format(res["ess"], res["ess_frac"], res["n_divergences"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=["surface", "posterior"])
    a = p.parse_args()
    os.makedirs(RES, exist_ok=True)
    (stage_surface if a.stage == "surface" else stage_posterior)()


if __name__ == "__main__":
    main()
