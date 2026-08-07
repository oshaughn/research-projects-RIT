#!/usr/bin/env python
"""
test_portfolio_balance_heuristic.py

Correctness test for mcsamplerPortfolio's SAFETY under a wrong ("decoy") member.

Setup (the failure mode the fix targets)
----------------------------------------
The portfolio pools draws from several member samplers and estimates
  I = \\int L(theta) prior(theta) dtheta.
Members are re-weighted by their per-member effective sample size (n_ess).  A
warm-started AdaptiveVolume (VARAHA) member seeded at a DECOY -- a wrong location
far from the true mode -- draws a tight, self-consistent cloud of LOW-likelihood
points.  Those points have nearly-uniform weights, so the decoy member reports a
HIGH per-member n_ess and gets driven to weight ~1, starving the broad covering
member (a GMM/mcsamplerEnsemble) down to the ~1% floor.

  * OLD (STRATIFIED) estimator: each pooled sample keeps its OWN member's
    sampling density p_s in L*prior/p_s.  Then
        E[I_hat] = sum_m w_m * Z_m,
    where Z_m is the true integral over member m's support.  The decoy member
    covers only the (empty) decoy region (Z_decoy ~ 0) and has w ~ 1, so the
    estimate is biased LOW by ~ ln(w_covering_floor).  A broad member CANNOT
    rescue it.  This is a real bias, not just variance.

  * NEW (BALANCE-HEURISTIC / deterministic-mixture) estimator: every pooled
    sample is weighted by the MIXTURE density
        q_mix(theta) = sum_m frac_m * q_m(theta),   frac_m = n_drawn_m / n,
    evaluated at that sample.  Then E[I_hat] = \\int q_mix * L*prior/q_mix = I,
    UNBIASED for any member weights, provided the mixture covers the peak.  The
    broad member's small-but-positive weight guarantees q_mix>0 at the true mode,
    so the wrongly-contracted decoy member can no longer bias the result.

This test builds AV(decoy) + GMM(broad) on a correlated-Gaussian target where a
cold AV converges, and checks:
  * OLD estimator -> badly biased low,
  * NEW estimator -> unbiased (matches true integral within a few percent),
  * and a no-regression control: a NORMAL portfolio (cold AV + GMM, both sane)
    stays unbiased under the NEW estimator.

Usage:
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
     python test_portfolio_balance_heuristic.py --as-test
"""
from __future__ import print_function
import argparse
import numpy as np

import benchmark_integrators as B
from RIFT.integrators import mcsamplerAdaptiveVolume as AVmod
from RIFT.integrators import mcsamplerEnsemble as Emod
from RIFT.integrators import mcsamplerPortfolio as Pmod


class PeakPlusPlateau(B.CorrelatedGaussian):
    """CorrelatedGaussian peak + a small CONSTANT likelihood floor over the whole
    box.  The floor makes any far-off-peak region a genuine FLAT plateau: an AV
    member that contracts onto a tight cloud out there sees an essentially
    constant likelihood, so its per-member weights L*prior/p_s are ~uniform and
    its Kish n_ess is near-maximal (n).  That is exactly the pathology the
    portfolio's n_ess re-weighting rewards -- it drives such a decoy member to
    weight ~1 and starves the broad covering member -- even though the plateau
    carries almost none of the integral.  The floor is set to contribute a tiny
    fraction of the total evidence so the true integral is essentially unchanged."""
    def __init__(self, floor_frac=1e-3, **kw):
        super(PeakPlusPlateau, self).__init__(**kw)
        self.name = "peakplateau_d{}".format(self.ndim)
        Vbox = float(np.prod(self.rlim - self.llim))
        # floor * Vbox = floor_frac * scale  ==> floor contributes floor_frac of Z
        self.floor = floor_frac * self.scale / Vbox
        self.true_lnZ = np.log(self.scale + self.floor * Vbox) - np.sum(np.log(self.rlim - self.llim))

    def lnL(self, X):
        return np.atleast_1d(np.log(self.scale * self._mvn.pdf(X) + self.floor))


def _host_lnfunc(target):
    """A cupy-tolerant wrapper around the benchmark's host integrand.

    The AV member's VARAHA self-update (update_sampling_prior_selfish) evaluates
    the integrand on its own DEVICE-native draws (cupy on GPU); the synthetic
    benchmark integrand is host/numpy-only and would choke on a cupy array.  In
    production the ILE likelihood is device-native so this never arises; for the
    synthetic target we simply move any device args to the host first, so the
    same test exercises the portfolio identically on CPU and GPU."""
    base = target.as_lnfunc()

    def ln_f(*cols):
        cols = [Emod.identity_convert(c) for c in cols]
        return base(*cols)
    return ln_f


def _seed_av_decoy(av, decoy):
    """Seed the AV member's live-volume state (binunique/dx/V) at the decoy cloud
    and apply it to the LIVE attributes draw_simplified() reads.  bootstrap_from_*
    only stashes self._warm (integrate_log applies it); here the member is driven
    through draw_simplified() by the portfolio, so we apply it ourselves.  VARAHA's
    live volume only ever contracts, so a seed at the decoy stays stuck there."""
    warm = av.bootstrap_from_samples(decoy)   # no cover_frac: deliberately wrong
    av.binunique = np.array(warm['binunique'])
    av.dx = np.array(warm['dx'])
    av.nbins = np.array(warm['nbins'])
    av.V = float(warm['V'])
    av.ninbin = ((av.n_chunk // av.binunique.shape[0] + 1)
                 * np.ones(av.binunique.shape[0])).astype(int)


def _seed_gmm_broad(gmm, target, broad_factor=3.0, n=8000, seed=7):
    """Make the GMM member a BROAD but peak-covering proposal: fit it (uniform
    weights) to a wide cloud N(mu, broad_factor^2 * cov) around the true mode.

    This is the covering member's job -- a reasonable, deliberately-wider-than-the
    -peak proposal (e.g. from a Fisher matrix or a previous posterior).  It matters
    for the TEST because when the flawed n_ess re-weighting starves this member to
    the ~1% floor, its few samples must still land near the peak for the q_mix
    estimate to have usable variance; a member left uniform over the whole box
    would be unbiased only in expectation but astronomically noisy (the peak is a
    ~1e-4 volume fraction).  The member still ADAPTS during the run and tightens
    further; the point being tested is the ESTIMATOR, given a sane covering member."""
    rng = np.random.RandomState(seed)
    cov = broad_factor ** 2 * np.atleast_2d(target.cov)
    cloud = rng.multivariate_normal(target.mu, cov, n)
    cloud = np.clip(cloud, target.llim + 1e-3, target.rlim - 1e-3)
    gmm.update_sampling_prior(np.zeros(len(cloud)), 2 * len(cloud),
                              external_rvs={p: cloud[:, i] for i, p in enumerate(gmm.params_ordered)},
                              log_scale_weights=True)


def build_portfolio(target, n_chunk, decoy=None, broad_gmm=True):
    """AV + GMM portfolio.  If `decoy` is given the AV member is seeded there
    (the failure case); otherwise AV starts cold (the no-regression control).
    `broad_gmm` pre-fits the GMM as a broad peak-covering proposal."""
    av = AVmod.MCSampler(n_chunk=n_chunk)
    gmm = Emod.MCSampler()
    members = [av, gmm]
    port = Pmod.MCSampler(portfolio=members, portfolio_freeze_wt=0.1, n_chunk=n_chunk)
    for d, p in enumerate(target.params):
        w = target.rlim[d] - target.llim[d]
        port.add_parameter(p, np.vectorize(lambda x, w=w: 1.0 / w),
                           prior_pdf=np.vectorize(lambda x, w=w: 1.0 / w),
                           left_limit=float(target.llim[d]), right_limit=float(target.rlim[d]),
                           adaptive_sampling=True)
    # propagate GMM configuration (full-covariance single component) through setup
    port.setup(portfolio_breakpoints=None, n_comp=1, correlate_all_dims=True, n=n_chunk)
    if broad_gmm:
        _seed_gmm_broad(gmm, target)
    if decoy is not None:
        _seed_av_decoy(av, decoy)
        # FREEZE the seeded AV member at the decoy: no-op its VARAHA self-update so
        # it keeps drawing from the seeded decoy grid every chunk.  This is a
        # faithful stand-in for "contracted and stuck" -- VARAHA's live volume only
        # ever contracts, so a real run seeded here stays near the decoy -- and it
        # sidesteps an unrelated numerical edge case (VARAHA's threshold search
        # empties the live set when the likelihood is perfectly flat).  The point
        # of the test is the ESTIMATOR (stratified vs q_mix), not VARAHA dynamics.
        av.update_sampling_prior_selfish = (lambda *a, **k: None)
    return port, members


def run(target, n_chunk, nmax, neff, use_mixture, decoy=None, seed=1234,
        tempering_exp=0.3, verbose=False):
    np.random.seed(seed)
    port, members = build_portfolio(target, n_chunk, decoy=decoy)
    ln_f = _host_lnfunc(target)
    lnI, logvar, eff, _ = port.integrate_log(
        ln_f, *target.params, no_protect_names=True,
        nmax=nmax, neff=neff, n=n_chunk, n_adapt=100,
        tempering_exp=tempering_exp, floor_level=0.0, use_lnL=True,
        save_intg=True, verbose=verbose,
        portfolio_use_mixture_density=use_mixture,
        # This test isolates the q_mix ESTIMATOR under a PINNED pathological allocation (the decoy
        # AV is frozen and, in the stratified case, dominates).  Adaptive-probe allocation would
        # dynamically re-allocate away from the decoy and change the scenario, so pin it off here;
        # the adaptive policy itself is exercised in test_portfolio_adaptive_alloc.py.
        portfolio_adaptive_alloc=False)
    lnI = float(B._asnumpy(lnI))
    ln_wt = B.log_weights_from_rvs(port._rvs)
    return dict(lnI=lnI, bias=lnI - float(target.true_lnZ),
                n_eval=int(getattr(port, "ntotal", 0)) or nmax,
                n_ess=B.n_ess_kish(ln_wt),
                final_weights=np.array(port.portfolio_weights))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndim", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=400000)
    ap.add_argument("--neff", type=int, default=2000)
    ap.add_argument("--n-chunk", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--as-test", action="store_true")
    args = ap.parse_args()

    target = PeakPlusPlateau(ndim=args.ndim)
    # a TIGHT decoy cloud on the far side of the box from the true mode.  On the
    # flat plateau its likelihood is ~constant -> maximal per-member n_ess.
    span = target.rlim - target.llim
    decoy_center = np.clip(target.mu - 0.55 * span * np.sign(target.mu + 1e-9),
                           target.llim + 0.08 * span, target.rlim - 0.08 * span)
    rng = np.random.RandomState(1)
    decoy = np.clip(rng.normal(decoy_center, 0.015 * span, size=(4000, args.ndim)),
                    target.llim + 1e-3, target.rlim - 1e-3)
    dist = np.linalg.norm(decoy_center - target.mu)
    print("# corrgauss ndim={}  true_lnZ={:.4f}  mode={}  decoy={}  |decoy-mode|={:.2f}".format(
        args.ndim, target.true_lnZ, np.round(target.mu, 2),
        np.round(decoy_center, 2), dist))

    kw = dict(n_chunk=args.n_chunk, nmax=args.nmax, neff=args.neff,
              seed=args.seed, verbose=args.verbose)

    old = run(target, use_mixture=False, decoy=decoy, **kw)   # legacy stratified
    new = run(target, use_mixture=True,  decoy=decoy, **kw)   # balance heuristic
    ctl = run(target, use_mixture=True,  decoy=None, **kw)    # no-regression control

    print("\nDECOY portfolio  (AV seeded at decoy + broad GMM):")
    print("  OLD stratified estimator : lnI-lnZ = {:+.3f}   n_ess={:8.1f}   wts={}".format(
        old["bias"], old["n_ess"], np.round(old["final_weights"], 3)))
    print("  NEW q_mix estimator      : lnI-lnZ = {:+.3f}   n_ess={:8.1f}   wts={}".format(
        new["bias"], new["n_ess"], np.round(new["final_weights"], 3)))
    print("NORMAL portfolio (cold AV + GMM), NEW q_mix estimator:")
    print("  control                  : lnI-lnZ = {:+.3f}   n_ess={:8.1f}".format(
        ctl["bias"], ctl["n_ess"]))
    print("\n# decoy bias improvement: old {:+.3f} -> new {:+.3f}  "
          "(factor exp = {:.1f}x closer to truth)".format(
              old["bias"], new["bias"],
              np.exp(abs(old["bias"]) - abs(new["bias"]))))

    if args.as_test:
        ok = True
        # 1. the OLD estimator MUST demonstrate the danger (biased low)
        if not (old["bias"] < -0.7):
            print(" FAIL: old stratified estimator not badly biased low ({:+.3f}); "
                  "decoy not exercised".format(old["bias"])); ok = False
        # 2. the NEW estimator must be unbiased within a few percent (a few % in
        #    the integral is ~0.03-0.20 in ln); allow a modest gate
        if abs(new["bias"]) > 0.20:
            print(" FAIL: new q_mix estimator biased ({:+.3f} > 0.20)".format(new["bias"])); ok = False
        # 3. the new estimator must be dramatically better than the old
        if not (abs(new["bias"]) < abs(old["bias"]) - 0.5):
            print(" FAIL: q_mix did not fix the decoy bias"); ok = False
        # 4. no regression: normal portfolio stays unbiased under q_mix
        if abs(ctl["bias"]) > 0.20:
            print(" FAIL: normal-portfolio control biased under q_mix "
                  "({:+.3f})".format(ctl["bias"])); ok = False
        if not ok:
            raise SystemExit(1)
        print("\n PASS: q_mix balance heuristic keeps the portfolio unbiased with a "
              "decoy member (old {:+.3f} -> new {:+.3f}); control unbiased "
              "({:+.3f}).".format(old["bias"], new["bias"], ctl["bias"]))


if __name__ == "__main__":
    main()
