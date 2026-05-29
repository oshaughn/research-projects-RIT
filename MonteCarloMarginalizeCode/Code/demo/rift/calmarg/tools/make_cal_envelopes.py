#!/usr/bin/env python
"""
Generate synthetic per-IFO calibration envelope files for the in-loop
calibration-marginalization demo.

Output format (one file per IFO, named <IFO>.txt), matching what
RIFT.calmarg.generate_realizations.retrieve_envelope_from_file expects:

    frequency  median_mag  median_phase  16_mag  16_phase  84_mag  84_phase

Amplitude factors are centered on 1 (fractional), phase on 0 (radians).
The 16/84 columns are the 1-sigma band edges; RIFT turns (84-16)/2 into a 1-sigma
width per spline node.
"""
import argparse
import os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out-dir", required=True, help="directory to write <IFO>.txt files")
p.add_argument("--ifos", default="H1,L1,V1", help="comma-separated IFO names")
p.add_argument("--fmin", type=float, default=5.0)
p.add_argument("--fmax", type=float, default=2048.0)
p.add_argument("--n-freq", type=int, default=60)
p.add_argument("--amp-sigma", type=float, default=0.05,
               help="1-sigma fractional amplitude uncertainty (e.g. 0.05 = 5%%)")
p.add_argument("--phase-sigma-deg", type=float, default=3.0,
               help="1-sigma phase uncertainty in degrees")
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
f = np.geomspace(args.fmin, args.fmax, args.n_freq)
ph_sigma = np.deg2rad(args.phase_sigma_deg)

# Give each IFO a slightly different uncertainty so the demo is not degenerate.
ifos = [s.strip() for s in args.ifos.split(",") if s.strip()]
scale = {ifo: 1.0 + 0.3 * i for i, ifo in enumerate(ifos)}

for ifo in ifos:
    a_sig = args.amp_sigma * scale[ifo]
    p_sig = ph_sigma * scale[ifo]
    med_mag = np.ones_like(f)
    med_ph = np.zeros_like(f)
    out = np.column_stack([
        f,
        med_mag, med_ph,
        med_mag - a_sig, med_ph - p_sig,   # 16th percentile
        med_mag + a_sig, med_ph + p_sig,   # 84th percentile
    ])
    path = os.path.join(args.out_dir, ifo + ".txt")
    np.savetxt(path, out)
    print("wrote {}  (amp 1-sigma {:.1%}, phase 1-sigma {:.2f} deg)".format(
        path, a_sig, np.rad2deg(p_sig)))
