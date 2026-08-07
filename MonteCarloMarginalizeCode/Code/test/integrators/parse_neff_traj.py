#!/usr/bin/env python3
"""Parse portfolio/AV ILE trajectory logs (lines beginning ' : N Neff ...') and report
the N (sample count) at which Neff first crosses a set of thresholds, plus the final
(N, Neff, sqrt(2 lnLmax)).  Usage: parse_neff_traj.py <log> [<log> ...]"""
import re, sys

THRESH = [5, 10, 20, 50, 100]
# Two trajectory formats:
#   portfolio (mcsamplerPortfolio): " : <N> <Neff> <sqrt(2lnLmax)> ..."
#   standalone AV (mcsamplerAdaptiveVolume): "<N> <Neff> <sqrt(2lnLmax)> - ..."
# Match an optional leading ' :', then N (integer-ish sample count), Neff, sqrt(2lnLmax).
row = re.compile(r"^\s*(?::\s*)?(\d[\d]*)\s+(nan|inf|[0-9.eE+-]+)\s+(nan|inf|-|[0-9.eE+-]+)\b")

def _f(tok):
    try:
        return float(tok)
    except ValueError:
        return float('nan')

def parse(path):
    Ns, Neffs, lmax = [], [], []
    with open(path, errors='replace') as f:
        for line in f:
            m = row.match(line)
            if not m:
                continue
            N = _f(m.group(1))
            if N < 1000:   # skip N=0 header/degenerate lines and non-trajectory numeric lines
                continue
            Ns.append(N); Neffs.append(_f(m.group(2))); lmax.append(_f(m.group(3)))
    return Ns, Neffs, lmax

def main():
    print(f"{'run':32s} {'Neff>=5':>9} {'>=10':>9} {'>=20':>9} {'>=50':>9} {'>=100':>9} {'finalN':>10} {'finalNeff':>9} {'sq2lnLmax':>9}")
    for path in sys.argv[1:]:
        Ns, Neffs, lmax = parse(path)
        name = path.split('/')[-1].replace('frz_', '').replace('.log', '')
        if not Ns:
            print(f"{name:32s}  (no trajectory lines)")
            continue
        cross = {}
        for t in THRESH:
            hit = next((Ns[i] for i in range(len(Ns)) if Neffs[i] >= t), None)
            cross[t] = hit
        def fmt(v):
            return f"{v/1e6:.3f}M" if v is not None else "  --  "
        print(f"{name:32s} {fmt(cross[5]):>9} {fmt(cross[10]):>9} {fmt(cross[20]):>9} "
              f"{fmt(cross[50]):>9} {fmt(cross[100]):>9} {Ns[-1]/1e6:>9.3f}M {Neffs[-1]:>9.1f} {lmax[-1]:>9.2f}")

if __name__ == '__main__':
    main()
