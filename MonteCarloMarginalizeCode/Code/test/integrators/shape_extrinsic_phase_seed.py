#!/usr/bin/env python3
"""SHAPE and evidence recovery for the extrinsic (psi, phi_orb) proposal: None vs a seed.

HAND-RUN study, not a pytest target.  It is the evidence behind the removal of
mcsamplerEnsemble.create_wide_single_component_prior (see the block at the end of that
module) and behind test_extrinsic_phase_group_uniform.py.

n_eff and lnZ alone cannot answer "does seeding this group change the ANSWER": a proposal
can move the recovered posterior while leaving the evidence alone.  So this scores
KL(recovered || truth) per parameter against an EXACT quadrature reference, on a separable
6-D extrinsic-shaped integrand, in every configuration the drivers put this group in.

Every comparison carries a NULL arm -- the same proposal with its RNG stream nudged.  The
per-seed n_eff and KL of this integrator span two orders of magnitude, so an arm must beat
that null before a difference counts.  Without the null, 20 seeds of the same comparison
read as a 35% n_eff loss that 40 seeds and a null show to be nothing.

    export PYTHONPATH=$PWD/MonteCarloMarginalizeCode/Code OMP_NUM_THREADS=1
    /cvmfs/software.igwn.org/conda/envs/igwn/bin/python \
        MonteCarloMarginalizeCode/Code/test/integrators/shape_extrinsic_phase_seed.py [NSEED] [N] [ITERS]

Defaults 16 6000 60, about 25 min on one ldas-grid core.

ARMS
    none            what the drivers ship: gmm_dict[pair_phi_psi] = None (exact uniform)
    NULL(none)      none again, RNG stream nudged -- the noise floor
    seed_norm       a CORRECTED wide seed: normalized [-1,1] frame, array bounds
    seed_phys       the helper's own frame (physical units) with array bounds
    seed_asWritten  the helper verbatim, list-of-tuples bounds.  Before PR #347 score() raised
                    on those and integrate() reset every dim-group; score() now converts them,
                    so this arm is bit-identical to seed_phys and is kept as that check

CONFIGURATIONS
    frozen          gmm_adapt[pair_phi_psi] = False, the production default
    adapt           --internal-rotate-phase / --force-adapt-all
    adaptive-k      --internal-gmm-adaptive-components
    psi-only        phase marginalized, so the group is the 1-element (psi,) while the helper
                    still built a 2-D model for it.  The only arm where a seed changes the
                    answer: lnZ moves by ln(width of phi_orb) while the psi MARGINAL does
                    not, which is why neither n_eff nor shape can see it.  The integrator
                    now rejects that mismatch, so this arm reports the refusal.
"""
import os
import contextlib
import sys

import numpy as np
from scipy import integrate as si
from RIFT.integrators import mcsamplerEnsemble as ME
from RIFT.integrators import gaussian_mixture_model as GMM

TWOPI = 2*np.pi
LIM = {"psi":(0.,TWOPI), "phi_orb":(0.,TWOPI), "right_ascension":(0.,TWOPI),
       "declination":(-np.pi/2,np.pi/2), "distance":(10.,1000.), "inclination":(0.,np.pi)}
PARAMS6 = ["right_ascension","declination","distance","inclination","psi","phi_orb"]

RA0, DEC0, SIG = 2.1, 0.35, 0.12
KAPPA = 8.0            # phase contrast: exp(+-KAPPA) across the 4 quadrupolar modes

def f_sky(ra,dec): return np.exp(-0.5*(((ra-RA0)/SIG)**2+((dec-DEC0)/SIG)**2))
def f_di(d,iota):
    amp=(1+np.cos(iota)**2)/2.
    return np.exp(-0.5*((300.*amp/d-1.0)/0.12)**2 - 0.5*((np.cos(iota)-0.4)/0.9)**2)
def f_ph(psi,phi): return np.exp(KAPPA*(np.cos(2*psi-0.7)*np.cos(2*phi-1.3)))
def integrand6(ra,dec,d,iota,psi,phi): return f_sky(ra,dec)*f_di(d,iota)*f_ph(psi,phi)
# psi-only variant (the driver's branch when phi_orb is not a sampled parameter)
def f_ph1(psi): return np.exp(KAPPA*np.cos(2*psi-0.7))
def integrand5(ra,dec,d,iota,psi): return f_sky(ra,dec)*f_di(d,iota)*f_ph1(psi)

def _q2(fn,xl,xh,yl,yh):
    v,_=si.dblquad(lambda y,x: fn(x,y), xl,xh, lambda x:yl, lambda x:yh,
                   epsabs=1e-12, epsrel=1e-12)
    return v

# ---- exact truth marginals, by fine-grid integration of each separable block --------
NB  = 40     # histogram bins per axis for the KL estimator
SUB = 48     # sub-points per bin: trapezoid inside each bin, exact to ~1e-10 here

def _bin_grid(lim, nb=NB, sub=SUB):
    edges = np.linspace(lim[0], lim[1], nb+1)
    # (nb, sub) sample points, one trapezoid rule per bin
    t = np.linspace(0., 1., sub)
    pts = edges[:-1][:,None] + (edges[1:]-edges[:-1])[:,None]*t[None,:]
    return edges, pts, (edges[1]-edges[0])

def truth_block_2d(fn2, xlim, ylim, nb=NB):
    """(nb,nb) normalized mass of a separable 2-D block, and both bin-edge arrays."""
    ex, px, _ = _bin_grid(xlim, nb); ey, py, _ = _bin_grid(ylim, nb)
    X = px.reshape(-1)[:,None]; Y = py.reshape(-1)[None,:]
    Z = fn2(X, Y)                                   # (nb*SUB, nb*SUB)
    Z = Z.reshape(nb, SUB, nb, SUB)
    w = np.ones(SUB); w[0] = w[-1] = 0.5            # trapezoid weights
    M = np.einsum('aibj,i,j->ab', Z, w, w)
    return M/M.sum(), ex, ey

def truth_block_1d(fn1, lim, nb=NB):
    e, pts, _ = _bin_grid(lim, nb)
    Z = fn1(pts)
    w = np.ones(SUB); w[0] = w[-1] = 0.5
    M = (Z*w[None,:]).sum(axis=1)
    return M/M.sum(), e

def weighted_hist_1d(x, w, edges):
    h,_ = np.histogram(x, bins=edges, weights=w)
    return h/h.sum()

def weighted_hist_2d(x, y, w, ex, ey):
    h,_,_ = np.histogram2d(x, y, bins=[ex,ey], weights=w)
    return h/h.sum()

def kl(p, q, floor=1e-12):
    """KL(p||q) in nats over a shared binning; both normalized, zeros floored."""
    p=np.asarray(p,float); q=np.asarray(q,float)
    p=p/p.sum(); q=q/q.sum()
    m=p>0
    return float(np.sum(p[m]*np.log(np.maximum(p[m],floor)/np.maximum(q[m],floor))))

# ---- the wide seed, in the frame a gmm actually uses --------------------------------
def wide_seed(bounds_tuples, physical_units=False, list_bounds=False):
    bt=list(bounds_tuples)
    b = list(bt) if list_bounds else np.array(bt,dtype=float)
    m=GMM.gmm(1,b)
    d=len(bt)
    if physical_units:                      # verbatim helper: WRONG frame
        w=np.array([bt[k][1]-bt[k][0] for k in range(d)])
        m.means=[np.array([np.mean(bt[k]) for k in range(d)])]
        m.covariances=[np.diag(w**2)]
    else:                                   # corrected: normalized [-1,1] frame
        m.means=[np.zeros(d)]
        m.covariances=[np.diag(np.full(d,4.0))]   # sigma=2 = full normalized width
    m.weights=[1.0]; m.adapt=[False]*1; m.d=d; m.N=0
    return m

# ---- one run -------------------------------------------------------------------------
def run(seed, params, integrand, seeded, adapt_phase, n=6000, n_iters=60,
        adaptive_kmax=None, correlate_all=False, seed_kind="normalized", _prenudged=False):
    if not _prenudged:
        np.random.seed(seed)
    s=ME.MCSampler()
    for p in params:
        s.add_parameter(p,left_limit=LIM[p][0],right_limit=LIM[p][1],adaptive_sampling=True)
    idx={p:s.params_ordered.index(p) for p in params}
    prd=tuple(sorted((idx["right_ascension"],idx["declination"])))
    pdi=tuple(sorted((idx["distance"],idx["inclination"])))
    if "phi_orb" in params:
        ppp=tuple(sorted((idx["psi"],idx["phi_orb"]))); n_phase=4
        seed_bounds=[LIM['psi'],LIM['phi_orb']]
    else:
        ppp=(idx["psi"],); n_phase=2
        seed_bounds=[LIM['psi'],LIM['phi_orb']]     # driver builds 2-D regardless
    if correlate_all:
        pall=tuple(range(len(s.params_ordered)))
        gd={pall:None}; ga={pall:True}; nc={pall:4}
    else:
        sd = wide_seed(seed_bounds,
                       physical_units=seed_kind.startswith("physical"),
                       list_bounds=(seed_kind=="physical_listbounds")) if seeded else None
        gd={prd:None,pdi:None,ppp:sd}
        ga={prd:True,pdi:True,ppp:bool(adapt_phase)}
        nc={prd:4,pdi:2,ppp:n_phase}
    kw=dict(n_comp=nc,gmm_dict=gd,gmm_adapt=ga,n=n,min_iter=n_iters,max_iter=n_iters,
            verbose=False,super_verbose=False)
    if adaptive_kmax:
        kw['gmm_adaptive']={k:adaptive_kmax for k in gd if ga.get(k,True)}
    with open(os.devnull,'w') as dn, contextlib.redirect_stdout(dn):
        res=s.integrate(integrand,*params,**kw)
    ig=s.integrator
    X=np.asarray(ig.cumulative_samples)
    lnL=np.asarray(ig.cumulative_values)
    lp=np.log(np.asarray(ig.cumulative_p)); lps=np.log(np.asarray(ig.cumulative_p_s))
    lw=lnL+lp-lps
    lw-=lw.max()
    w=np.exp(lw)
    neff=w.sum()**2/np.sum(w**2)
    mk=gd.get(ppp) if not correlate_all else None
    kfin=(ig.gmm_dict[ppp].k if (not correlate_all and ig.gmm_dict[ppp] is not None) else 0)
    return dict(lnZ=float(np.log(res[0])), eff_samp=float(res[2]), X=X, w=w,
                neff=float(neff), idx=idx, k_phase=kfin, n_samples=len(w))


# --------------------------------------------------------------------------------------
from scipy import stats as st


def lnZ_truth(psi_only=False):
    """Exact ln of the volume-weighted integral, by quadrature on each separable block.

    integrate() returns the VOLUME-weighted integral (the default prior_pdf is 1), so this is
    ln(Is * Id * Ip) with no 1/volume factor.  Without it the per-arm lnZ column has nothing
    to compare against, which is the whole point of the psi-only configuration.
    """
    Is = _q2(f_sky, *LIM["right_ascension"], *LIM["declination"])
    Id = _q2(f_di,  *LIM["distance"],        *LIM["inclination"])
    if psi_only:
        Ip, _ = si.quad(f_ph1, *LIM["psi"], epsabs=1e-13, epsrel=1e-13)
    else:
        Ip = _q2(f_ph, *LIM["psi"], *LIM["phi_orb"])
    return np.log(Is * Id * Ip)


def main(argv):
    NSEED = int(argv[1]) if len(argv) > 1 else 16
    NPER  = int(argv[2]) if len(argv) > 2 else 6000
    NITER = int(argv[3]) if len(argv) > 3 else 60

    # ---------- exact truth, per separable block ----------
    T2_ph, EX_psi, EX_phi = truth_block_2d(f_ph, LIM['psi'], LIM['phi_orb'])
    T1_psi = T2_ph.sum(axis=1); T1_phi = T2_ph.sum(axis=0)
    T2_sky, EX_ra, EX_dec = truth_block_2d(f_sky, LIM['right_ascension'], LIM['declination'])
    T1_ra = T2_sky.sum(axis=1); T1_dec = T2_sky.sum(axis=0)
    T2_di, EX_d, EX_io = truth_block_2d(f_di, LIM['distance'], LIM['inclination'])
    T1_d = T2_di.sum(axis=1); T1_io = T2_di.sum(axis=0)
    T1_psi_only, EX_psi1 = truth_block_1d(f_ph1, LIM['psi'])

    def metrics(r, psi_only=False):
        X, w, idx = r['X'], r['w'], r['idx']
        g = lambda p: X[:, idx[p]]
        out = {}
        out['KL_ra']  = kl(weighted_hist_1d(g('right_ascension'), w, EX_ra),  T1_ra)
        out['KL_dec'] = kl(weighted_hist_1d(g('declination'),     w, EX_dec), T1_dec)
        out['KL_d']   = kl(weighted_hist_1d(g('distance'),        w, EX_d),   T1_d)
        out['KL_io']  = kl(weighted_hist_1d(g('inclination'),     w, EX_io),  T1_io)
        if psi_only:
            out['KL_psi'] = kl(weighted_hist_1d(g('psi'), w, EX_psi1), T1_psi_only)
            out['KL_2D']  = out['KL_psi']
        else:
            out['KL_psi'] = kl(weighted_hist_1d(g('psi'),     w, EX_psi), T1_psi)
            out['KL_phi'] = kl(weighted_hist_1d(g('phi_orb'), w, EX_phi), T1_phi)
            out['KL_2D']  = kl(weighted_hist_2d(g('psi'), g('phi_orb'), w, EX_psi, EX_phi), T2_ph)
        out['lnZ']=r['lnZ']; out['neff']=r['neff']; out['k_phase']=r['k_phase']
        return out

    CONFIGS = [
        # name          params        integrand        adapt  kmax  psi_only
        ("frozen  (production default)", PARAMS6, integrand6, False, None, False),
        # NOTE: the real option also rewrites param_limits['psi'] and ['phi_orb'] to (0, 4pi)
        # (driver:1737-1739).  This arm flips gmm_adapt only, so it does NOT exercise the
        # bounds-mismatch failure that widening enables -- that one is covered by
        # test_extrinsic_phase_group_uniform.py::test_seeded_model_bounds_must_match_the_group_box.
        ("adapt   (--internal-rotate-phase, gmm_adapt only)", PARAMS6, integrand6, True,  None, False),
        ("adaptive-k (--internal-gmm-adaptive-components)", PARAMS6, integrand6, True, 8, False),
        ("psi-only 1-D group (no phi_orb)", [p for p in PARAMS6 if p!="phi_orb"], integrand5, False, None, True),
    ]
    ARMS = [("none", dict(seeded=False, seed_kind="normalized")),
            ("NULL(none)", dict(seeded=False, seed_kind="normalized", nudge=True)),
            ("seed_norm", dict(seeded=True, seed_kind="normalized")),
            ("seed_phys", dict(seeded=True, seed_kind="physical_arraybounds")),
            ("seed_asWritten", dict(seeded=True, seed_kind="physical_listbounds"))]

    KEYS = ['KL_2D','KL_psi','KL_phi','KL_ra','KL_dec','KL_d','KL_io']

    for cname, params, integ, adapt, kmax, psi_only in CONFIGS:
        TRUTH = lnZ_truth(psi_only)
        print("\n" + "="*104)
        print("CONFIG: {}   ({} seeds, n={} x {} iters)   quadrature lnZ = {:.5f}".format(
            cname, NSEED, NPER, NITER, TRUTH))
        print("="*104)
        res={}
        for aname, aopt in ARMS:
            rows=[]; fails=[]
            for sd in range(400, 400+NSEED):
                s_eff = sd
                try:
                    if aopt.get('nudge'):
                        np.random.seed(sd); np.random.uniform(size=1)
                        r=run(None if False else sd, params, integ, False, adapt,
                                n=NPER, n_iters=NITER, adaptive_kmax=kmax,
                                seed_kind="normalized", _prenudged=True)
                    else:
                        r=run(sd, params, integ, aopt['seeded'], adapt, n=NPER, n_iters=NITER,
                                adaptive_kmax=kmax, seed_kind=aopt['seed_kind'])
                except Exception as e:
                    fails.append("{}: {}".format(type(e).__name__, str(e)[:70])); continue
                rows.append(metrics(r, psi_only))
            res[aname]=rows
            if fails:
                print("  {:12s} RAISED on {}/{} seeds. first: {}".format(aname,len(fails),NSEED,fails[0]))
            if not rows: continue
            med=lambda k: np.median([x[k] for x in rows])
            line="  {:12s} n={:2d}  lnZ {:+.4f} (bias {:+.4f})  neff {:7.0f}  k={:<3}".format(
                aname, len(rows), np.mean([x['lnZ'] for x in rows]),
                np.mean([x['lnZ'] for x in rows]) - TRUTH,
                np.median([x['neff'] for x in rows]), str(sorted(set(x['k_phase'] for x in rows))))
            for k in KEYS:
                if k in rows[0]: line += "  {} {:.4f}".format(k.replace('KL_',''), med(k))
            print(line)
        # paired tests against 'none'
        base=res.get('none',[])
        if base:
            print("  " + "-"*100)
            for aname in ['NULL(none)','seed_norm','seed_phys','seed_asWritten']:
                rows=res.get(aname,[])
                m=min(len(rows),len(base))
                if m<3: continue
                for k in (['KL_2D','KL_psi'] if psi_only is False else ['KL_2D']):
                    if k not in rows[0] or k not in base[0]: continue
                    a=np.array([x[k] for x in base[:m]]); b=np.array([x[k] for x in rows[:m]])
                    d=b-a
                    try: wp=st.wilcoxon(d).pvalue
                    except Exception: wp=float('nan')
                    sp=st.binomtest(int((d>0).sum()), m, 0.5).pvalue
                    print("    {:12s} {:6s}  median ratio {:6.3f}   worse-than-none {:2d}/{:2d}"
                          "   sign p={:.4f}  wilcoxon p={:.4f}".format(
                          aname, k, np.median(b)/np.median(a), int((d>0).sum()), m, sp, wp))


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
