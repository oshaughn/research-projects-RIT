#!/usr/bin/env python3
"""How much memory would it cost to KEEP the retained set alongside the export?

Open question 2 of DESIGN_rvs_naming.md.  Holding the retained rows would close the last
BROKEN ledger entry (#79's cross-source lnZ fallback) and let .dslice reweight properly
instead of falling back to all-fresh -- but today the fair draw REPLACES `_rvs`, so the
pre-draw arrays become garbage and the peak is transient.  Keeping them makes the peak
persistent for the rest of analyze_event.

This is an operations question, so it is measured rather than argued.  Reported per sampler:
the retained row count, the column count, the implied bytes, and the process RSS actually
observed.

    OMP_NUM_THREADS=1 python3 measure_retained_set_memory.py
    OMP_NUM_THREADS=1 python3 measure_retained_set_memory.py --nmax 400000 1000000
"""
import argparse
import gc
import os
import resource
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, CODE)

import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAV  # noqa: E402

NAMES = ['right_ascension', 'declination', 'phi_orb', 'inclination', 'psi', 'distance']
NDIM = len(NAMES)


def _rss_mb():
    # ru_maxrss is KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _av(n_chunk):
    s = mcsamplerAV.MCSampler(n_chunk=n_chunk)
    s.xpy = mcsamplerAV.xpy_default
    s.identity_convert = mcsamplerAV.identity_convert
    for name in NAMES:
        s.add_parameter(name, pdf=None, left_limit=0.0, right_limit=1.0,
                        prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
    return s


def _portfolio(n_chunk):
    import RIFT.integrators.mcsamplerPortfolio as mcsamplerPF
    import RIFT.integrators.mcsamplerEnsemble as mcsamplerEnsemble
    members = [mcsamplerAV.MCSampler(n_chunk=n_chunk), mcsamplerEnsemble.MCSampler()]
    s = mcsamplerPF.MCSampler(portfolio=members)
    pdf = np.vectorize(lambda x: 1.0)
    for name in NAMES:
        s.add_parameter(name, pdf, prior_pdf=pdf, left_limit=0.0, right_limit=1.0,
                        adaptive_sampling=True)
    s.setup()
    return s


def _peaked(rho):
    x0 = 0.5 * np.ones(NDIM)
    w = (0.5 / rho) * np.ones(NDIM)
    lnLmax = 0.5 * rho ** 2

    def lnL(*args, **kwargs):
        x = np.array([np.asarray(a, dtype=float).ravel() for a in args]).T
        out = lnLmax - 0.5 * np.sum(((x - x0) / w) ** 2, axis=-1)
        return np.where(out > lnLmax - 745.0, out, -np.inf)
    return lnL


def _record_bytes(rvs):
    """Bytes actually held by the record's columns."""
    tot = 0
    for v in rvs.values():
        a = np.asarray(mcsamplerAV.identity_convert(v))
        tot += a.nbytes
    return tot


def measure(kind, nmax, rho=20.0, n_chunk=20000):
    """Run WITHOUT a fair draw, so _rvs IS the retained set, and weigh it."""
    gc.collect()
    rss0 = _rss_mb()
    s = _portfolio(n_chunk) if kind == 'portfolio' else _av(n_chunk)
    kw = dict(no_protect_names=True, verbose=False)
    if kind == 'portfolio':
        kw['save_intg'] = True
    try:
        s.integrate_log(_peaked(rho), *NAMES, nmax=nmax, neff=100, n=n_chunk, **kw)
    except Exception as e:
        return dict(kind=kind, nmax=nmax, error=str(e)[:70])
    rvs = s._rvs
    n_rows = len(np.atleast_1d(np.asarray(
        mcsamplerAV.identity_convert(rvs['log_integrand']))).ravel())
    out = dict(kind=kind, nmax=nmax, ntotal=int(getattr(s, 'ntotal', 0)),
               n_rows=n_rows, n_cols=len(rvs),
               mb=_record_bytes(rvs) / 1024.0 ** 2,
               rss_mb=_rss_mb(), rss_delta=_rss_mb() - rss0)
    del s
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, nargs='+', default=[200000, 400000, 800000])
    ap.add_argument("--rho", type=float, default=20.0)
    args = ap.parse_args()

    print("=" * 96)
    print("Retained-set size: what holding it alongside the export would cost")
    print("(no fair draw, so _rvs IS the retained set; rho={})".format(args.rho))
    print("=" * 96)
    print("{:<11} {:>10} {:>10} {:>10} {:>6} {:>10} {:>10} {:>10}".format(
        "sampler", "nmax", "ntotal", "rows", "cols", "record MB", "RSS MB", "dRSS MB"))
    rows = []
    for kind in ('AV', 'portfolio'):
        for nmax in args.nmax:
            r = measure(kind, nmax, rho=args.rho)
            rows.append(r)
            if 'error' in r:
                print("{:<11} {:>10}  FAILED: {}".format(kind, nmax, r['error']))
                continue
            print("{:<11} {:>10} {:>10} {:>10} {:>6} {:>10.1f} {:>10.1f} {:>10.1f}".format(
                kind, nmax, r['ntotal'], r['n_rows'], r['n_cols'],
                r['mb'], r['rss_mb'], r['rss_delta']))

    print()
    print("READING THIS.  `rows` is what the record would have to keep.  For AV it is the")
    print("RETAINED (in-volume) subset, so it grows far more slowly than ntotal.  For the")
    print("PORTFOLIO _rvs holds EVERY draw, so rows ~ ntotal and the cost is set by nmax.")
    ok = [r for r in rows if 'error' not in r and r['n_rows'] > 0]
    for kind in ('AV', 'portfolio'):
        sub = [r for r in ok if r['kind'] == kind]
        if len(sub) >= 2:
            per = (sub[-1]['mb'] - sub[0]['mb']) / max(1, sub[-1]['nmax'] - sub[0]['nmax'])
            print("  {:<10} ~{:.1f} MB per million nmax  ->  {:.0f} MB at nmax=4e6".format(
                kind, per * 1e6, per * 4e6 + sub[0]['mb']))
    print()
    print("Compare: _warm_seed_reserve already keeps a BOUNDED copy (n_max=20000 rows,")
    print("stratified by finite-ness), which is the affordable precedent.  The question is")
    print("whether the UNBOUNDED retained set is affordable too, per ILE process, alongside")
    print("the waveform and PSD working set.")


if __name__ == "__main__":
    sys.exit(main())
