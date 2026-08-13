"""Numerically confirm (or refute) the three post-rebind claims, using the ILE's OWN
helpers rather than a reimplementation of them."""
import importlib.util
import os
import sys

import numpy as np

CODE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, CODE)

# The ILE is a script, not a module: load the two helpers by exec'ing just their source.
src = open(os.path.join(CODE, "bin", "integrate_likelihood_extrinsic_batchmode")).read()
ns = {"numpy": np, "np": np}
start = src.index("def ln_weights_from_rvs")
end = src.index("def _warm_seed_geometry")
body = src[start:end]
# _rvs_lnL_convention is defined elsewhere; stub it to "as passed"
ns["_rvs_lnL_convention"] = lambda x=None: bool(x)
exec(compile(body, "ile_helpers", "exec"), ns)
_lnZ_of_rvs = ns["_lnZ_of_rvs"]
_kish_neff_of_rvs = ns["_kish_neff_of_rvs"]

rng = np.random.default_rng(20260813)


def make_pass(n, spread):
    """A retained set whose log-weights have the given spread (nats).  Large spread =
    the collapsed, high-SNR regime; small spread = a healthy pass."""
    lnL = rng.normal(0.0, spread, size=n)
    return {
        "log_integrand": lnL,
        "log_joint_prior": np.zeros(n),
        "log_joint_s_prior": np.zeros(n),
        "x": rng.normal(size=n),
    }


def fair_draw(rvs, n_extr, rng):
    """The rebind, exactly as integrate_log performs it: n_extr rows WITH REPLACEMENT,
    proportional to weight."""
    lw = rvs["log_integrand"] + rvs["log_joint_prior"] - rvs["log_joint_s_prior"]
    lw = lw - np.max(lw)
    w = np.exp(lw)
    w = w / w.sum()
    idx = rng.choice(np.arange(len(w)), size=n_extr, replace=True, p=w)
    return {k: np.asarray(v)[idx] for k, v in rvs.items()}


def kish(rvs):
    lw = rvs["log_integrand"] + rvs["log_joint_prior"] - rvs["log_joint_s_prior"]
    lw = lw - lw.max()
    w = np.exp(lw)
    return w.sum() ** 2 / (w ** 2).sum()


print("=" * 78)
print("CLAIM 1: _lnZ_of_rvs(already_pooled=False) on a fair-drawn record is HIGH by")
print("         about log(n_retained / n_eff).")
print("=" * 78)
print("{:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "n", "spread", "n_eff", "n_extr", "lnZ_true", "lnZ_draw", "excess"))
for spread in (1.0, 3.0, 6.0, 9.0):
    n = 4000
    rvs = make_pass(n, spread)
    ne = kish(rvs)
    lnZ_true = _lnZ_of_rvs(rvs, already_pooled=False)
    n_extr = max(1, int(min(300, 1.5 * ne)))
    excess = []
    for _ in range(200):
        fd = fair_draw(rvs, n_extr, rng)
        excess.append(_lnZ_of_rvs(fd, already_pooled=False) - lnZ_true)
    print("{:>8} {:>8.1f} {:>10.1f} {:>10} {:>10.3f} {:>10.3f} {:>+10.3f}  (log(n/n_eff)={:+.3f})".format(
        n, spread, ne, n_extr, lnZ_true, lnZ_true + np.mean(excess),
        float(np.mean(excess)), float(np.log(n / ne))))

print()
print("=" * 78)
print("CLAIM 2: the gate's cold-vs-warm difference does NOT cancel, because the two")
print("         passes sit at very different n_eff.  Simulates the measured regime:")
print("         collapsed cold (n_eff~1) vs healthy warm (n_eff~20), SAME true lnZ.")
print("=" * 78)
cold = make_pass(1000, 9.0)
warm = make_pass(1000, 2.0)
# force identical true evidence so any gap the gate sees is pure artifact
lnZ_c = _lnZ_of_rvs(cold, already_pooled=False)
lnZ_w = _lnZ_of_rvs(warm, already_pooled=False)
warm["log_integrand"] = warm["log_integrand"] + (lnZ_c - lnZ_w)
print("  true lnZ cold {:.4f}  warm {:.4f}  (difference by construction {:+.4f})".format(
    _lnZ_of_rvs(cold, already_pooled=False), _lnZ_of_rvs(warm, already_pooled=False),
    _lnZ_of_rvs(cold, already_pooled=False) - _lnZ_of_rvs(warm, already_pooled=False)))
print("  n_eff cold {:.2f}   warm {:.2f}".format(kish(cold), kish(warm)))
gaps = []
for _ in range(400):
    c = fair_draw(cold, max(1, int(1.5 * kish(cold))), rng)
    w = fair_draw(warm, max(1, int(1.5 * kish(warm))), rng)
    gaps.append(_lnZ_of_rvs(c, already_pooled=False) - _lnZ_of_rvs(w, already_pooled=False))
gaps = np.array(gaps)
print("  gate reads cold-warm = {:+.3f} nats (median), IQR [{:+.3f}, {:+.3f}]".format(
    float(np.median(gaps)), float(np.percentile(gaps, 25)), float(np.percentile(gaps, 75))))
for thr in (0.5, 2.0, 5.0):
    print("    at --sampler-l0-rescue-reject-dlnZ {:.1f}: rejects the (equally good) warm pass"
          " {:.0f}% of the time".format(thr, 100.0 * np.mean(gaps > thr)))

print()
print("=" * 78)
print("CLAIM 3: _kish_neff_of_rvs of a fair-drawn record tracks the ROW COUNT, not the")
print("         pass's true n_eff.")
print("=" * 78)
print("{:>8} {:>12} {:>10} {:>14}".format("spread", "true n_eff", "n_extr", "kish(fairdraw)"))
for spread in (1.0, 3.0, 6.0, 9.0):
    rvs = make_pass(4000, spread)
    ne = kish(rvs)
    n_extr = max(1, int(min(300, 1.5 * ne)))
    vals = [_kish_neff_of_rvs(fair_draw(rvs, n_extr, rng)) for _ in range(100)]
    print("{:>8.1f} {:>12.1f} {:>10} {:>14.1f}".format(spread, ne, n_extr, float(np.mean(vals))))

print()
print("=" * 78)
print("CLAIM 4: re-weighting an already fair-drawn record (the .dgrid / .dslice path)")
print("         applies w twice.  Compare the weighted mean of a coordinate.")
print("=" * 78)
n = 4000
lnL = rng.normal(0.0, 4.0, size=n)
x = lnL * 0.5 + rng.normal(size=n) * 0.5          # x correlated with weight
rvs = {"log_integrand": lnL, "log_joint_prior": np.zeros(n),
       "log_joint_s_prior": np.zeros(n), "x": x}
w = np.exp(lnL - lnL.max()); w /= w.sum()
truth = float(np.sum(w * x))
ne = kish(rvs)
n_extr = max(1, int(min(300, 1.5 * ne)))
naive, correct = [], []
for _ in range(300):
    fd = fair_draw(rvs, n_extr, rng)
    lw = fd["log_integrand"] + fd["log_joint_prior"] - fd["log_joint_s_prior"]
    ww = np.exp(lw - lw.max()); ww /= ww.sum()
    naive.append(float(np.sum(ww * fd["x"])))       # what the exporter does
    correct.append(float(np.mean(fd["x"])))         # a fair draw is already equal-weight
print("  posterior mean of x, truth                  {:+.4f}".format(truth))
print("  fair draw, UNWEIGHTED (correct)             {:+.4f}".format(float(np.mean(correct))))
print("  fair draw, RE-WEIGHTED by w (the exporter)  {:+.4f}".format(float(np.mean(naive))))
print("  -> re-weighting shifts the estimate by      {:+.4f} ({:.0f}% of the truth)".format(
    float(np.mean(naive)) - truth, 100 * abs(float(np.mean(naive)) - truth) / abs(truth)))
