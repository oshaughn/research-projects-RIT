#!/usr/bin/env python
"""
dat_to_compact.py  --  RIFT posterior .dat -> compact .npz for reconstruct_strain.py.

Converts a standard RIFT `extrinsic_posterior_samples.dat`
(from convert_output_format_ile2inference, header:
 m1 m2 a1x a1y a1z a2x a2y a2z mc eta ra dec time phiorb incl psi distance Npts lnL p ps neff ...)
into the compact .npz that reconstruct_strain.py consumes.  These rows are already
fair (equal-weight) posterior draws; use with --fair-draw.

If the run's final extrinsic stage used --resample-time-marginalization, the
'time' column varies per row (coherent coalescence times) -- exactly what the
reconstruction needs.

Usage:  dat_to_compact.py extrinsic_posterior_samples.dat out_compact.npz
"""
import sys
import numpy as np


def convert(src, out):
    S = np.genfromtxt(src, names=True, replace_space=None)
    n = S.dtype.names

    def col(name, default=0.0):
        return S[name].astype(float) if name in n else np.full(len(S), default)

    d = dict(
        m1=col("m1"), m2=col("m2"), a1z=col("a1z"), a2z=col("a2z"),
        ra=col("ra"), dec=col("dec"), incl=col("incl"), psi=col("psi"),
        phiorb=col("phiorb"), distance=col("distance"), time=col("time"),
        lnL=col("lnL"), p=col("p", 1.0), ps=col("ps", 1.0),
        eccentricity=col("eccentricity", 0.0),
    )
    np.savez(out, **d)
    print("dat_to_compact: %d rows -> %s (distinct times=%d)"
          % (len(S), out, len(np.unique(d["time"]))), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    convert(sys.argv[1], sys.argv[2])
