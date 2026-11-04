"""Throughput of gpu_waveform_xphm modes versus single-threaded LALSimulation.

Usage (GPU host; CuPy needs CUDA headers):

    CUDA_PATH=/usr/local/cuda-12.8 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \\
        python benchmark_gpu_waveform_xphm.py [--backend cupy|numpy] [--lal]

Grids: (T, srate) = (8 s, 2048 Hz) and (32 s, 4096 Hz); f_min = f_ref = 20 Hz.
The XPHM reference is ChooseFDModes(IMRPhenomXPHM, PrecVersion 102,
co-precessing (2,+-2), ThresholdMband 0), timed per point.
"""

import argparse
import time

import numpy as np

from RIFT.likelihood import gpu_waveform_xphm as X

F_MIN = 20.0
GRIDS = ((8.0, 2048.0), (32.0, 4096.0))
NPOINTS = (1, 10, 100)
DIST = 500.0e6 * 3.085677581491367e16


def points(n, seed=1):
    rng = np.random.default_rng(seed)
    mtot = rng.uniform(10.0, 150.0, n)
    q = rng.uniform(0.1, 1.0, n)
    m1 = mtot / (1 + q)
    m2 = mtot - m1
    c1 = rng.normal(size=(n, 3)); c1 *= (rng.uniform(0, 0.95, n) / np.linalg.norm(c1, axis=1))[:, None]
    c2 = rng.normal(size=(n, 3)); c2 *= (rng.uniform(0, 0.95, n) / np.linalg.norm(c2, axis=1))[:, None]
    return m1, m2, c1, c2, rng.uniform(0, np.pi, n), rng.uniform(0, 2 * np.pi, n)


def time_backend(xp, sync, n, T, srate, repeats=5):
    df = 1.0 / T
    fmax = srate / 2.0
    nf = int(fmax / df) + 1
    m1, m2, c1, c2, incl, phi = [xp.asarray(a) for a in points(n)]
    X.imrphenomxp_nnlo_modes(m1, m2, c1, c2, DIST, phi, F_MIN, F_MIN, df, nf, f_max=fmax,
                             inclination=incl, xp=xp)
    sync()
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        X.imrphenomxp_nnlo_modes(m1, m2, c1, c2, DIST, phi, F_MIN, F_MIN, df, nf, f_max=fmax,
                                 inclination=incl, xp=xp)
        sync()
        best = min(best, time.perf_counter() - t0)
    return best


def time_lal(n, T, srate):
    import lal
    import lalsimulation as lalsim
    df = 1.0 / T
    fmax = srate / 2.0
    m1, m2, c1, c2, incl, phi = points(n)
    t0 = time.perf_counter()
    for i in range(n):
        d = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d, 102)
        lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(d, 0.0)
        ma = lalsim.SimInspiralCreateModeArray()
        lalsim.SimInspiralModeArrayActivateMode(ma, 2, 2)
        lalsim.SimInspiralModeArrayActivateMode(ma, 2, -2)
        lalsim.SimInspiralWaveformParamsInsertModeArray(d, ma)
        mj = lalsim.SimInspiralCreateModeArray()
        lalsim.SimInspiralModeArrayActivateAllModesAtL(mj, 2)
        lalsim.SimInspiralWaveformParamsInsertModeArrayJframe(d, mj)
        lalsim.SimInspiralChooseFDModes(m1[i] * lal.MSUN_SI, m2[i] * lal.MSUN_SI, *c1[i], *c2[i],
                                        df, F_MIN, fmax, F_MIN, phi[i], DIST, incl[i], d,
                                        lalsim.IMRPhenomXPHM)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cupy", choices=["cupy", "numpy"])
    ap.add_argument("--lal", action="store_true", help="also time LAL single-thread")
    args = ap.parse_args()
    if args.backend == "cupy":
        import cupy as xp
        sync = xp.cuda.Device().synchronize
    else:
        xp = np
        sync = lambda: None
    for T, srate in GRIDS:
        for n in NPOINTS:
            dt = time_backend(xp, sync, n, T, srate)
            line = "T=%4.0f s srate=%5.0f n=%4d  %s: %8.4f s  %9.1f pts/s" % (
                T, srate, n, args.backend, dt, n / dt)
            if args.lal:
                tl = time_lal(n, T, srate)
                line += "   lal: %8.4f s  %7.1f pts/s" % (tl, n / tl)
            print(line, flush=True)


if __name__ == "__main__":
    main()
