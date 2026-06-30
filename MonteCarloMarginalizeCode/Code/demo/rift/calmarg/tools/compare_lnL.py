#!/usr/bin/env python
"""
Compare the marginalized lnL produced by ILE runs in this demo.

The ILE per-event .dat row ends with four columns:
    ... , lnL (=log_res), sqrt_var_over_res, ntotal, neff
so the marginalized lnL is column [-4], its error is [-3], and the LAST column is
neff (effective sample count) -- NOT lnL.  (A common foot-gun: grabbing [-1] gives
neff, which scatters run-to-run and is meaningless to compare.)

This prints lnL (with its sampling error) and neff for each file, and the pairwise
lnL differences (loop vs fused should agree to within ~the sampling error).
"""
import sys
import numpy as np

if len(sys.argv) < 2:
    print("usage: compare_lnL.py label=file.dat [label=file.dat ...]")
    sys.exit(1)

lnL, err, neff = {}, {}, {}
for arg in sys.argv[1:]:
    label, fname = arg.split("=", 1)
    a = np.atleast_2d(np.genfromtxt(fname))
    lnL[label] = a[:, -4]    # log_res = marginalized lnL
    err[label] = a[:, -3]    # sqrt_var_over_res (sampling error on the integral)
    neff[label] = a[:, -1]   # effective sample count (diagnostic, NOT a result)
    print("{:10s}  lnL = {}   +- {}   (neff = {})".format(
        label,
        np.array2string(lnL[label], precision=4),
        np.array2string(err[label], precision=4),
        np.array2string(neff[label], precision=0)))

labels = list(lnL)
print("\npairwise max|delta lnL|  (compare to the sampling errors above):")
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        a, b = lnL[labels[i]], lnL[labels[j]]
        n = min(len(a), len(b))
        d = float(np.max(np.abs(a[:n] - b[:n])))
        print("  {:10s} vs {:10s}  {:.4f}".format(labels[i], labels[j], d))
