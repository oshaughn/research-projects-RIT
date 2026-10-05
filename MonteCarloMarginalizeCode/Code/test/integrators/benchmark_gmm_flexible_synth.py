"""Data-free synthetic benchmark for the flexible GMM component allocation
(RIFT/integrators/DESIGN_flexible_gmm.md).  Mimics the S250114ax extrinsic
posterior's HARD feature: a CURVED degeneracy arc that no axis-aligned binning
(and no few-component Gaussian) can wrap.

Reference results, measured 2026-09-14 on ldas-pcdev2 (RTX 3080, cupy 12.0.0,
CUDA_VISIBLE_DEVICES=0, IGWN python), one process per run.  Three seeds: the
1234 seeded below, then 2 and 3.

  mode           N at n_eff>=100       final n_eff        wall
  corrall k=1    32k / 80k / 176k      346 / 112 / 171    10-18 s
  corrall k=2    72k / 172k / 260k     347 / 232 / 154    16-36 s
  corrall k=4    36k / 88k / 168k      407 /  45 / 240    38-52 s
  adaptive k<=8  never / 80k / 172k     35 / 398 /  16    38-45 s

Every run ends at N=400000 (nmax) and lnI lands in 3.00-3.11, so none of these
is biased.  READ THE SPREAD, NOT THE FIRST COLUMN: final n_eff moves by a factor
of 20 across three seeds, which is larger than any gap between the k settings.
Three seeds do not rank these configurations, and this benchmark as written
cannot; treat the table as the scale of the effect.

The table this replaces reported "corrall k=1 -> @76k, final ~50; k=2 -> @28k,
final ~312; k=4 -> @752k (over-allocation collapse); adaptive -> @220k, final
~135 (robust, unbiased, hands-free)".  None of it reproduces.  N=752k is not
reachable by this script: nmax has been 400_000 since the file was added, in the
same commit that recorded the number, and every run stops there.  k=4 did not
collapse, and adaptive was the WEAKEST of the four at the seed below, never
crossing n_eff=100.  Those runs also predate the seeding below reaching the GPU
backend, so they were not reproducible to begin with.

6D target on a broad box (needle-ish: peak is a small fraction of the prior):
  dims (2,3): a parabolic BANANA ridge  <-> distance-inclination arc (curved)
  dims (0,1): a strongly-correlated Gaussian <-> phase-polarization degeneracy
  dims (4,5): a tight isotropic Gaussian     <-> well-localized sky

We measure n_eff vs N (cumulative samples) for correlate-all GMM proposals with
varying component count, plus (later) the flexible-k prototype.

Usage:
  python synth_bench.py <mode> [k]
    mode = corrall  -> single full-dim GMM, fixed k components  (k from argv)
    mode = adaptive -> flexible-k prototype (n_comp='adaptive')
    mode = pairing  -> factored pairing {(0,1),(2,3),(4,5)} k each
"""
import sys, numpy as np
from RIFT.integrators import mcsamplerEnsemble
from RIFT.integrators.seeding import seed_everything

# The GMM member draws through its array backend -- gaussian_mixture_model's
# k-means++ initialization calls xpy.random.choice -- and that backend is cupy on
# a GPU host, where numpy.random.seed does not reach the generator.  Seeding numpy
# alone left this benchmark's fixed seed inert on the device it is meant to be run
# on.  Seed every backend instead.
#
# ONE RUN PER PROCESS.  Seeding fixes the RNG streams, not the sampler and library
# state a long-lived interpreter accumulates: calling run() twice in one process
# after seeding gave final n_eff 345.7 then 161.7.  In separate processes the same
# seed gives 345.7 both times, which is how the table above was measured.
seed_everything(1234, verbose=False)

# ---- box (broad prior) ----
LO = np.array([-6.,-6., -6.,-30., -6.,-6.])
HI = np.array([ 6., 6.,  6., 30.,  6., 6.])

# ---- target lnL ----
# banana in (x2,x3): ridge v = C*(u^2 - M); narrow across ridge, broad along it
C, M, SU, SV = 3.0, 3.0, 1.2, 1.5
# correlated pair (x0,x1)
RHO, SP = 0.92, 1.0
_cov = np.array([[SP**2, RHO*SP*SP],[RHO*SP*SP, SP**2]])
_covinv = np.linalg.inv(_cov)
# tight sky (x4,x5)
SK = 0.35
MU4, MU5 = 1.0, -1.0

def lnL_np(x):
    x = np.asarray(x)
    u, v = x[:,2], x[:,3]
    ban = -0.5*(u/SU)**2 - 0.5*((v - C*(u**2 - M))/SV)**2
    d0, d1 = x[:,0], x[:,1]
    q = _covinv[0,0]*d0*d0 + 2*_covinv[0,1]*d0*d1 + _covinv[1,1]*d1*d1
    corr = -0.5*q
    sky = -0.5*((x[:,4]-MU4)**2 + (x[:,5]-MU5)**2)/SK**2
    return ban + corr + sky

def like(*args):
    # args: one array per param, in params_ordered order
    X = np.array(args).T
    return lnL_np(X)   # returns lnL (we run with use_lnL)

def build_sampler():
    s = mcsamplerEnsemble.MCSampler()
    for i in range(6):
        s.add_parameter(str(i), left_limit=float(LO[i]), right_limit=float(HI[i]),
                        adaptive_sampling=True)
    return s

def run(mode, k):
    s = build_sampler()
    params = [str(i) for i in range(6)]
    traj = []
    def hook(integrator):
        traj.append((int(integrator.ntotal), float(integrator.identity_convert(integrator.eff_samp))))
    kw = dict(min_iter=5, max_iter=300, n=4000, nmax=400_000, neff=5000,
              use_lnL=True, return_lnI=True, integrator_func=hook,
              verbose=False, super_verbose=False)
    if mode == 'corrall':
        kw.update(correlate_all_dims=True, n_comp=int(k))
    elif mode == 'adaptive':
        # correlate-all single group with data-driven k (BIC), cap = k
        g = tuple(range(6))
        kw.update(gmm_dict={g:None}, n_comp={g:2}, gmm_adaptive={g:int(k)})
    elif mode == 'adaptpair':
        gd = {(0,1):None,(2,3):None,(4,5):None}
        kw.update(gmm_dict=gd, n_comp={(0,1):2,(2,3):2,(4,5):2},
                  gmm_adaptive={(0,1):int(k),(2,3):int(k),(4,5):int(k)})
    elif mode == 'pairing':
        gd = {(0,1):None,(2,3):None,(4,5):None}
        kw.update(gmm_dict=gd, n_comp={(0,1):int(k),(2,3):int(k),(4,5):int(k)})
    integral, err2, eff, _ = s.integrate(like, *params, **kw)
    return traj, float(s.identity_convert(eff)), float(s.identity_convert(integral))

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv)>1 else 'corrall'
    k = int(sys.argv[2]) if len(sys.argv)>2 else 2
    traj, eff, integral = run(mode, k)
    label = "{}{}".format(mode, k if mode!='adaptive' else '')
    print("MODE", label, "final_eff", eff, "lnI", integral)
    # print n_eff-vs-N crossings
    for target in [5,10,20,50,100,200,500,1000]:
        cross = next((N for (N,e) in traj if e>=target), None)
        print("  neff>={:<5} at N= {}".format(target, cross))
    # dump full trajectory sparsely
    for i,(N,e) in enumerate(traj):
        if i%10==0 or i==len(traj)-1:
            print("   traj N={:>8} eff={:.1f}".format(N,e))
