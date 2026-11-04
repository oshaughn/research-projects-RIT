#!/usr/bin/env python
"""
make_e2e_inputs.py -- generate a SELF-CONTAINED 2-detector injection (frames + PSD + grid + case.json)
so run_e2e_consistency.sh has NO external (paper-repo) dependency.  Pure RIFT + LAL.

Writes into <outdir> (argv[1], default ./e2e_case):
  <IFO>-FAKE-<t0>-<dur>.gwf   frames  (H1, L1; IMRPhenomD BNS via lsu.hoft, a fixed-seglen REAL8TimeSeries)
  data.cache                  absolute-path frame cache
  <IFO>-psd.xml.gz            analytic aLIGOZeroDetHighPower PSDs
  grid.xml.gz                 small (mc, delta_mc) intrinsic grid (util_ManualOverlapGrid)
  case.json                   ile_common + ile_finite_extra (40 km arm -> nontrivial finite-size response)

This is a GPU<->CPU PARITY fixture, not a physics demo: the injection just needs to be some real
2-detector signal that both the rotation and freqresponse likelihoods can evaluate; parity holds
regardless of whether the data literally carries the modeled effect.
"""
from __future__ import print_function, division
import os, sys, json, subprocess
import numpy as np
import lal
import lal.series
import lalsimulation as lalsim
import RIFT.lalsimutils as lsu

EVENT_TIME = 1000000000.0
FMIN, FMAX, SEGLEN, SRATE = 50., 1024., 32., 2048.
M1, M2 = 1.6, 1.4                                   # Msun
RA, DEC, PSI, INCL, PHIREF = 1.2, 0.3, 0.5, 0.4, 0.0
DIST_MPC = 300.
DETS = ["H1", "L1"]
ARM = 40000.0                                        # 40-km arm -> direction-dependent finite-size response
QMAX = 6
PSDFUNC = lalsim.SimNoisePSDaLIGOZeroDetHighPower


def base_params(det):
    P = lsu.ChooseWaveformParams(
        m1=M1 * lal.MSUN_SI, m2=M2 * lal.MSUN_SI, fmin=FMIN, radec=True,
        theta=DEC, phi=RA, psi=PSI, incl=INCL, phiref=PHIREF, detector=det,
        dist=DIST_MPC * 1e6 * lal.PC_SI, deltaT=1. / SRATE, tref=EVENT_TIME,
        deltaF=1. / SEGLEN)
    P.approx = lalsim.GetApproximantFromString("IMRPhenomD")
    P.taper = lsu.lsu_TAPER_START
    return P


def write_psd_xml(det, deltaF, fmax, path):
    n = int(fmax / deltaF) + 1
    s = lal.CreateREAL8FrequencySeries(det, lal.LIGOTimeGPS(0), 0., deltaF, lal.SecondUnit, n)
    f = np.arange(n) * deltaF
    vals = np.array([PSDFUNC(max(fi, 1.0)) for fi in f])
    vals[~np.isfinite(vals)] = 0.0
    s.data.data[:] = vals
    xmldoc = lal.series.make_psd_xmldoc({det: s})
    from ligo.lw import utils as ligolw_utils
    ligolw_utils.write_filename(xmldoc, path, compress="gz")


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    deltaF = 1. / SEGLEN

    # --- frames (fixed-seglen detector strain via lsu.hoft; deltaF set -> zero-padded to seglen*srate) ---
    frame_paths = []; t0 = None; dur = None
    for det in DETS:
        ht = lsu.hoft(base_params(det))                 # REAL8TimeSeries, length = SEGLEN*SRATE
        t0 = float(ht.epoch); dur = ht.data.length * ht.deltaT
        fname = os.path.join(outdir, "%s-FAKE-%d-%d.gwf" % (det, int(np.floor(t0)), int(np.ceil(dur))))
        lsu.hoft_to_frame_data(fname, det + ":FAKE-STRAIN", ht)
        frame_paths.append(os.path.abspath(fname))
    with open(os.path.join(outdir, "data.cache"), "w") as fc:
        for det, p in zip(DETS, frame_paths):
            base = os.path.basename(p)[:-4]; parts = base.split("-")
            fc.write("%s %s %s %s file://localhost%s\n" % (parts[0], "-".join(parts[1:-2]), parts[-2], parts[-1], p))

    # --- PSD xmls ---
    psd_paths = {}
    for det in DETS:
        p = os.path.join(outdir, "%s-psd.xml.gz" % det)
        write_psd_xml(det, deltaF, FMAX, p)
        psd_paths[det] = p

    # --- intrinsic (mc, delta_mc) grid via util_ManualOverlapGrid (invoke with THIS interpreter) ---
    mc0 = lsu.mchirp(M1, M2)
    grid_base = os.path.join(outdir, "grid")
    tool = os.path.join(os.environ.get('RIFT_CODE', ''), 'bin', 'util_ManualOverlapGrid.py')
    if not os.path.isfile(tool):
        tool = os.path.join(os.path.dirname(os.path.dirname(lsu.__file__)), 'bin', 'util_ManualOverlapGrid.py')
    subprocess.check_call([sys.executable, tool,
                           "--parameter", "mc", "--parameter-range", "[%f,%f]" % (mc0 * 0.98, mc0 * 1.02),
                           "--parameter", "delta_mc", "--parameter-range", "[0.0,0.25]",
                           "--grid-cartesian", "--grid-cartesian-npts", "1",
                           "--skip-overlap", "--fname", grid_base])
    grid_xml = grid_base + ".xml.gz"
    assert os.path.isfile(grid_xml)

    # --- data times: pad in from the segment edges; keep event_time strictly inside ---
    data_start = int(np.floor(t0)) + 2
    data_end = int(np.ceil(t0 + dur)) - 2
    assert data_start < EVENT_TIME < data_end, \
        "event_time %.1f not inside [%d,%d] (epoch=%.3f dur=%.3f)" % (EVENT_TIME, data_start, data_end, t0, dur)

    # --- box limits centered on the injected truth (so the AV sampler resolves the peak) ---
    box = dict(ra=[RA - 0.8, RA + 0.8], dec=[DEC - 0.8, DEC + 0.8],
               incl=[max(0.0, INCL - 0.7), INCL + 0.7], psi=[max(0.0, PSI - 0.6), PSI + 0.6],
               dist=[DIST_MPC * 0.3, DIST_MPC * 3.0])

    ile_common = ["--sim-xml", os.path.basename(grid_xml), "--cache-file", "data.cache",
                  "--event-time", repr(EVENT_TIME),
                  "--data-start-time", str(data_start), "--data-end-time", str(data_end),
                  "--fmin-template", str(FMIN), "--fmax", str(FMAX),
                  "--approximant", "IMRPhenomD", "--l-max", "2", "--srate", str(int(SRATE)),
                  "--vectorized", "--internal-use-lnL", "--time-marginalization",
                  "--sampler-method", "AV", "--n-eff", "300", "--n-max", "800000", "--n-chunk", "20000",
                  "--d-min", "%.4f" % box['dist'][0], "--d-max", "%.4f" % box['dist'][1],
                  "--limit-right-ascension", "%.6f,%.6f" % tuple(box['ra']),
                  "--limit-declination", "%.6f,%.6f" % tuple(box['dec']),
                  "--limit-inclination", "%.6f,%.6f" % tuple(box['incl']),
                  "--limit-psi", "%.6f,%.6f" % tuple(box['psi']),
                  "--fairdraw-extrinsic-output", "--fairdraw-extrinsic-output-n-max", "200", "--save-samples"]
    for det in DETS:
        ile_common += ["--channel-name", "%s=FAKE-STRAIN" % det,
                       "--psd-file", "%s=%s" % (det, os.path.basename(psd_paths[det]))]
    arm_str = ",".join("%s=%.1f" % (det, ARM) for det in DETS)
    ile_finite_extra = ["--freqresponse", "--freqresponse-qmax", str(QMAX), "--freqresponse-arm-length", arm_str]

    info = dict(network="H1L1", event_time=EVENT_TIME, fmin=FMIN, fmax=FMAX, seglen=SEGLEN,
                m1=M1, m2=M2, data_start=data_start, data_end=data_end, arm={d: ARM for d in DETS},
                grid=os.path.basename(grid_xml), box=box,
                frames=[os.path.basename(p) for p in frame_paths],
                ile_common=ile_common, ile_finite_extra=ile_finite_extra)
    with open(os.path.join(outdir, "case.json"), "w") as fj:
        json.dump(info, fj, indent=2)
    print("built self-contained e2e case in %s (H1L1 BNS %.1f+%.1f, data [%d,%d])"
          % (outdir, M1, M2, data_start, data_end))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "e2e_case")
