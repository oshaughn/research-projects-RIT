"""Tier-3 analysis.

The GPU ILE is NOT deterministic at fixed --seed (measured: base vs base, same code,
same seed, differs by dlnL 0.33 and neff 4.2 vs 22.5).  So the comparison is between
DISTRIBUTIONS, and the null has to be MEASURED rather than assumed:

  * permutation test on the base/cand labels -- exact, no normality assumption;
  * an A/A control that splits the BASE runs into two pseudo-arms and runs the same
    test, so we can see what |t| identical code produces on this hardware.
"""
import csv, math, sys, random, statistics as st

random.seed(20260818)
NPERM = 20000

def load(path):
    return [r for r in csv.DictReader(open(path))]

def vals(rows, cfg, arm, m):
    out = []
    for r in rows:
        if r['cfg'] == cfg and r['arm'] == arm:
            try: v = float(r[m])
            except (ValueError, TypeError): continue
            if not math.isnan(v): out.append(v)
    return out

def perm_p(b, c):
    """Two-sided permutation p-value on the difference of means."""
    obs = abs(st.mean(c) - st.mean(b))
    pool = b + c; nb = len(b)
    hits = 0
    for _ in range(NPERM):
        random.shuffle(pool)
        if abs(st.mean(pool[nb:]) - st.mean(pool[:nb])) >= obs - 1e-15:
            hits += 1
    return (hits + 1) / (NPERM + 1)

def aa_control(b):
    """Split base in half -> two pseudo-arms of IDENTICAL code."""
    x = list(b); random.shuffle(x)
    h = len(x) // 2
    return x[:h], x[h:2*h]

CFG = {'A':'GPU linear backend, plain',
       'B':'GPU linear backend + replica POOLING',
       'D':'cubic NoLoop time interpolation',
       'AV':'AV (lnL family) + pooling + .dgrid',
       'GMM':'GMM (lnL family) + pooling + .dgrid'}
METRICS = ('lnL','sigma_lnL','neff','dgrid_lnL_mean','dgrid_lnL_max')

rows = []
for p in sys.argv[1:]:
    rows += load(p)
bad = [r for r in rows if r['rc'] != '0' or r['failed'] != '0']
print("runs=%d  clean=%d  failed=%d\n" % (len(rows), len(rows)-len(bad), len(bad)))

results, aa = [], []
for cfg in ('A','B','D','AV','GMM'):
    if not any(r['cfg'] == cfg for r in rows): continue
    print("=== %s : %s ===" % (cfg, CFG[cfg]))
    for m in METRICS:
        b, c = vals(rows, cfg, 'base', m), vals(rows, cfg, 'cand', m)
        if len(b) < 3 or len(c) < 3: continue
        d = st.mean(c) - st.mean(b)
        se = math.sqrt(st.stdev(b)**2/len(b) + st.stdev(c)**2/len(c))
        t = d/se if se else float('nan')
        p = perm_p(b, c)
        results.append((cfg, m, d, t, p))
        print("  %-15s n=%2d/%2d  base %9.4f +-%7.4f  cand %9.4f +-%7.4f  d=%+8.4f  t=%+5.2f  p=%.3f%s"
              % (m, len(b), len(c), st.mean(b), st.stdev(b), st.mean(c), st.stdev(c),
                 d, t, p, '  <<<' if p < 0.05 else ''))
        # A/A control on the same data
        b1, b2 = aa_control(b)
        if len(b1) >= 3:
            aa.append((cfg, m, perm_p(b1, b2)))
    print()

n = len(results); sig = [r for r in results if r[4] < 0.05]
print("SUMMARY")
print("  %d comparisons; %d with p<0.05 (expected by chance at alpha=0.05: %.1f)"
      % (n, len(sig), 0.05*n))
for cfg, m, d, t, p in sig:
    print("    p<0.05: %s %s  d=%+.4f t=%+.2f p=%.3f" % (cfg, m, d, t, p))
naa = len(aa); saa = [a for a in aa if a[2] < 0.05]
print("  A/A CONTROL (base split against ITSELF, identical code):")
print("    %d comparisons; %d with p<0.05" % (naa, len(saa)))
for cfg, m, p in saa:
    print("      %s %s p=%.3f" % (cfg, m, p))
