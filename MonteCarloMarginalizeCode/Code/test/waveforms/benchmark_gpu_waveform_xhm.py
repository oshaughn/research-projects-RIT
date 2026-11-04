"""Throughput of gpu_waveform_xhm modes versus single-threaded LALSimulation.

Usage (GPU host; see the design note for the container used on Blackwell cards):

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \\
        python benchmark_gpu_waveform_xhm.py [--backend cupy|numpy] [--lal] [--grids 8:2048,256:4096]

Times the five modes h_{l,-m}(f >= 0), (l,m) = (2,2),(2,1),(3,3),(3,2),(4,4), for
n = 1, 10, 100 aligned-spin BBH points (Mtot 10-150 Msun, f_min = f_ref = 20 Hz).
The LAL reference is ChooseFDModes(IMRPhenomXHM, ThresholdMband 0) with a
ModeArray of the five negative-m modes, timed per point.  For CuPy the peak
memory-pool size is reported.
"""

import argparse
import time

import numpy as np

from RIFT.likelihood import gpu_waveform_xhm as XHM

F_MIN = 20.0
NPOINTS = (1, 10, 100)
DIST = 500.0e6 * 3.085677581491367e16


def points(n, seed=1):
    rng = np.random.default_rng(seed)
    mtot = rng.uniform(10.0, 150.0, n)
    q = rng.uniform(0.1, 1.0, n)
    m1 = mtot / (1 + q)
    m2 = mtot - m1
    return m1, m2, rng.uniform(-0.9, 0.9, n), rng.uniform(-0.9, 0.9, n), rng.uniform(0, 2 * np.pi, n)


def time_backend(xp, sync, n, T, srate, repeats=3):
    df = 1.0 / T
    fmax = srate / 2.0
    nf = int(fmax / df) + 1
    m1, m2, c1, c2, phi = [xp.asarray(a) for a in points(n)]
    call = lambda: XHM.imrphenomxhm_modes(m1, m2, c1, c2, DIST, phi, F_MIN, F_MIN, df, nf,
                                          f_max=fmax, xp=xp, two_sided=False)
    out = call()
    sync()
    del out
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = call()
        sync()
        best = min(best, time.perf_counter() - t0)
        del out
    return best


def time_lal(n, T, srate):
    import lal
    import lalsimulation as lalsim
    df = 1.0 / T
    fmax = srate / 2.0
    m1, m2, c1, c2, phi = points(n + 1)
    t0 = None
    for i in range(n + 1):
        if i == 1:
            t0 = time.perf_counter()   # first call (one-time LAL setup) not timed
        d = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(d, 0.0)
        ma = lalsim.SimInspiralCreateModeArray()
        for ell, emm in XHM.XHM_MODES:
            lalsim.SimInspiralModeArrayActivateMode(ma, ell, -emm)
        lalsim.SimInspiralWaveformParamsInsertModeArray(d, ma)
        lalsim.SimInspiralChooseFDModes(m1[i] * lal.MSUN_SI, m2[i] * lal.MSUN_SI, 0., 0., c1[i], 0., 0., c2[i],
                                        df, F_MIN, fmax, F_MIN, phi[i], DIST, 0.0, d, lalsim.IMRPhenomXHM)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cupy", choices=["cupy", "numpy"])
    ap.add_argument("--lal", action="store_true", help="also time LAL single-thread")
    ap.add_argument("--lal-max-n", type=int, default=10, help="time LAL on at most this many points")
    ap.add_argument("--grids", default="8:2048,256:4096,2048:4096")
    ap.add_argument("--npoints", default="1,10,100")
    args = ap.parse_args()
    grids = [tuple(float(x) for x in g.split(":")) for g in args.grids.split(",")]
    npts = [int(x) for x in args.npoints.split(",")]
    pool = None
    if args.backend == "cupy":
        import cupy as xp
        sync = xp.cuda.Device().synchronize
        pool = xp.get_default_memory_pool()
    else:
        xp = np
        sync = lambda: None
    for T, srate in grids:
        for n in npts:
            if pool is not None:
                pool.free_all_blocks()
            try:
                dt = time_backend(xp, sync, n, T, srate)
            except Exception as exc:  # out of device memory
                print("T=%5.0f s srate=%5.0f n=%4d  %s: failed (%s)" % (T, srate, n, args.backend,
                                                                        type(exc).__name__), flush=True)
                continue
            line = "T=%5.0f s srate=%5.0f n=%4d  %s: %8.4f s  %9.1f pts/s" % (
                T, srate, n, args.backend, dt, n / dt)
            if pool is not None:
                line += "  pool %.2f GB" % (pool.total_bytes() / 1e9)
            if args.lal:
                nl = min(n, args.lal_max_n)
                tl = time_lal(nl, T, srate)
                line += "   lal: %7.1f pts/s" % (nl / tl)
            print(line, flush=True)


if __name__ == "__main__":
    main()
