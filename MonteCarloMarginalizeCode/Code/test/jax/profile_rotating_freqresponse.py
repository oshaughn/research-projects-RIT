#!/usr/bin/env python
"""CPU usability profile for the compound conventional and JAX likelihoods.

The default waveform is IMRPhenomD with only the (2,+/-2) mode pair.  This is a
basis-scaling benchmark, not a science-accuracy benchmark: start cheap, identify
the usable Qmax range, then repeat with the intended long-duration source.
"""

import argparse
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.factored_likelihood_rotating_freqresponse as flrr
import RIFT.likelihood.slowrot_freqresponse as sfr
from RIFT.likelihood.jax_ile.banded import build_rotating_freqresponse_data
from RIFT.likelihood.jax_ile.core import fused_log_likelihood


def timed(call):
    start = time.perf_counter()
    value = call()
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qmax", default="0,1,2,3,4")
    parser.add_argument("--pmax", type=int, default=0)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--detectors", default="H1")
    parser.add_argument("--mass1", type=float, default=30.0)
    parser.add_argument("--mass2", type=float, default=25.0)
    parser.add_argument("--fmin", type=float, default=30.0)
    parser.add_argument("--fmax", type=float, default=512.0)
    parser.add_argument("--srate", type=float, default=1024.0)
    parser.add_argument("--delta-f", type=float, default=0.5)
    parser.add_argument("--arm-length", type=float, default=40000.0)
    args = parser.parse_args()

    event_time = 1.0e9
    deltaT = 1.0 / args.srate
    detectors = tuple(x.strip() for x in args.detectors.split(",") if x.strip())
    P = lsu.ChooseWaveformParams(
        fmin=args.fmin, radec=True, incl=0.3, phiref=0.0, theta=0.2,
        phi=1.0, psi=0.4, m1=args.mass1 * lal.MSUN_SI,
        m2=args.mass2 * lal.MSUN_SI, detector=detectors[0],
        dist=200e6 * lal.PC_SI, deltaT=deltaT, tref=event_time,
        deltaF=args.delta_f)
    P.approx = lalsim.IMRPhenomD
    data_dict = {}
    for det in detectors:
        Pd = P.manual_copy()
        Pd.detector = det
        data_dict[det] = lsu.non_herm_hoff(Pd)
    psd_dict = {det: lalsim.SimNoisePSDaLIGOZeroDetHighPower for det in detectors}
    tvals = fl.marginalization_time_grid(0.03, deltaT, xpy=np)

    rng = np.random.default_rng(20260909)
    S = args.samples
    Pv = P.manual_copy()
    Pv.phi = rng.uniform(0, 2 * np.pi, S)
    Pv.theta = np.arcsin(rng.uniform(-1, 1, S))
    Pv.psi = rng.uniform(0, np.pi, S)
    Pv.incl = np.arccos(rng.uniform(-1, 1, S))
    Pv.phiref = rng.uniform(0, 2 * np.pi, S)
    Pv.dist = rng.uniform(100, 800, S) * 1e6 * lal.PC_SI
    Pv.tref = event_time
    Pv.deltaT = deltaT
    dist_mpc = np.asarray(Pv.dist) / (1e6 * lal.PC_SI)

    print("model=IMRPhenomD modes=(2,+/-2) detectors=%s samples=%d pmax=%d" %
          (",".join(detectors), S, args.pmax))
    print("Qmax basis pairs bank_MB precompute_s numpy_s jax_compile_s jax_warm_s samples/s")
    for qmax in [int(x) for x in args.qmax.split(",")]:
        def precompute():
            bank = flrr.PrecomputeLikelihoodTermsRotatingFreqResponse(
                event_time, 0.1, P.manual_copy(), data_dict, psd_dict, 2,
                args.fmax, Qmax=qmax, L_arm=args.arm_length,
                p_max=args.pmax, analyticPSD_Q=True, verbose=False, quiet=True,
                skip_interpolation=True)
            packed = flrr.pack_rotating_freqresponse_arrays(
                bank[4], bank[3], bank[1], bank[2])
            return bank, packed
        (bank, packed), pre_s = timed(precompute)
        meta = bank[4]
        lk, rba, uba, vba, ep = packed
        det_geom = {d: sfr.detector_geometry(d, L_arm=args.arm_length)
                    for d in detectors}
        jdata = build_rotating_freqresponse_data(
            meta, lk, rba, uba, vba, ep, deltaT, tvals, det_geom)
        A = len(meta["a_list"])
        bank_bytes = sum(int(jdata.detectors[d][name].nbytes)
                         for d in detectors for name in ("Q_bank", "U_bank", "V_bank"))

        _, numpy_s = timed(lambda: flrr.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
            tvals, Pv, meta, lk, rba, uba, vba, ep, Lmax=2,
            time_interp="nearest", xpy=np))
        fn = jax.jit(lambda ra, dec, psi, incl, phiref, dist: fused_log_likelihood(
            jdata, ra, dec, psi, incl, phiref, dist, interp="nearest"))
        values = (Pv.phi, Pv.theta, Pv.psi, Pv.incl, Pv.phiref, dist_mpc)
        _, compile_s = timed(lambda: fn(*values))
        warm = []
        for _ in range(args.repeats):
            _, dt = timed(lambda: fn(*values))
            warm.append(dt)
        warm_s = min(warm)
        print("%d %d %d %.2f %.3f %.3f %.3f %.6f %.1f" %
              (qmax, A, A * A, bank_bytes / 2.0**20, pre_s, numpy_s,
               compile_s, warm_s, S / warm_s))


if __name__ == "__main__":
    main()
