#!/usr/bin/env python
"""
extract_ile_samples.py  --  fast RIFT ILE sample extractor.

Reads a RIFT `integrate_likelihood_extrinsic_batchmode` output XML
(the sim_inspiral table written by --save-samples) and dumps a compact
.npz with exactly the columns the reconstruction needs.  This reads the
ligolw table directly, so it is ~30 s for a few-million-row file instead
of the ~25 min a full `convert_output_format_ile2inference` takes.

Column convention (RIFT xmlutils CMAP / ILE export block):
    alpha1 = lnL (per-sample log likelihood, at the drawn time if
             --resample-time-marginalization was used)
    alpha2 = p    (prior)
    alpha3 = ps   (sampling prior)
    alpha4 = eccentricity
    longitude=RA  latitude=dec  inclination  polarization=psi
    coa_phase=phi_orb  distance(Mpc)  mass1/2(Msun)  spin1z/2z
    geocent_end_time(+_ns) = coalescence time  -> stored here as 'time'

Usage:
    extract_ile_samples.py  ILE_output_0_.xml.gz  out_compact.npz
"""
import sys, time
import numpy as np
from igwn_ligolw import ligolw, lsctables, utils


class _CH(ligolw.LIGOLWContentHandler):
    pass
lsctables.use_in(_CH)


def extract(src, out):
    t0 = time.time()
    doc = utils.load_filename(src, contenthandler=_CH)
    tb = lsctables.SimInspiralTable.get_table(doc)
    n = len(tb)

    def col(attr):
        return np.fromiter((getattr(r, attr) for r in tb), float, n)

    d = dict(
        m1=col("mass1"), m2=col("mass2"),
        a1z=col("spin1z"), a2z=col("spin2z"),
        ra=col("longitude"), dec=col("latitude"),
        incl=col("inclination"), psi=col("polarization"),
        phiorb=col("coa_phase"), distance=col("distance"),
        lnL=col("alpha1"), p=col("alpha2"), ps=col("alpha3"),
        eccentricity=col("alpha4"),
        # geocent coalescence time (varies per row iff --resample-time-marginalization)
        time=np.fromiter((r.geocent_end_time + 1e-9 * r.geocent_end_time_ns
                          for r in tb), float, n),
    )
    np.savez(out, **d)
    print("extract_ile_samples: %d rows in %.1fs -> %s  (distinct times=%d)"
          % (n, time.time() - t0, out, len(np.unique(d["time"]))), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    extract(sys.argv[1], sys.argv[2])
