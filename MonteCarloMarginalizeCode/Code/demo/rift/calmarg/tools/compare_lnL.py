#!/usr/bin/env python
"""
Compare the marginalized lnL produced by ILE runs in this demo.

Each ILE output .dat has one row per intrinsic point; the LAST column is the
extrinsic-marginalized lnL.  This prints that lnL for each supplied file and the
pairwise differences (in particular loop vs fused, which should agree to within
floating-point / kernel precision when run with the same --seed).
"""
import sys
import numpy as np

if len(sys.argv) < 2:
    print("usage: compare_lnL.py label=file.dat [label=file.dat ...]")
    sys.exit(1)

vals = {}
for arg in sys.argv[1:]:
    label, fname = arg.split("=", 1)
    a = np.atleast_2d(np.genfromtxt(fname))
    vals[label] = a[:, -1]   # last column = marginalized lnL
    print("{:10s}  lnL = {}".format(label, np.array2string(vals[label], precision=6)))

print("\npairwise max|delta lnL|:")
labels = list(vals)
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        a, b = vals[labels[i]], vals[labels[j]]
        n = min(len(a), len(b))
        d = float(np.max(np.abs(a[:n] - b[:n])))
        print("  {:10s} vs {:10s}  {:.3e}".format(labels[i], labels[j], d))
