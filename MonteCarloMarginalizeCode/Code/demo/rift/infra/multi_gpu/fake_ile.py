#!/usr/bin/env python3
"""Stand-in for integrate_likelihood_extrinsic_batchmode, for the multi-GPU
smoke test (`make smoke-local`).  It mimics ONLY the two behaviours the fan-out
launcher relies on:

  1. grid slicing:   evaluates grid indices [--event, --event+--n-events-to-analyze)
  2. output naming:  writes <--output-file>_<LOCALindex>_.dat   (local 0..n-1),
                     exactly like the real ILE (fname = output_file+"_"+str(indx)+"_.dat").

Each row records the GLOBAL grid index and the GPU the shard ran on
(CUDA_VISIBLE_DEVICES), so the test can prove the whole grid was covered exactly
once and spread across the GPUs.  No cupy / no real GPU work -- this validates
the launcher's partition + pinning, which is the only new logic.
"""
import os
import sys

GRID_N = int(os.environ.get("FAKE_ILE_GRID_N", "100"))   # pretend overlap-grid size


def getopt(argv, names, default=None):
    for i, a in enumerate(argv):
        for nm in names:
            if a == nm and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(nm + "="):
                return a.split("=", 1)[1]
    return default


def main():
    argv = sys.argv[1:]
    event = int(getopt(argv, ["--event", "-E"], "0"))
    ngroup = int(getopt(argv, ["--n-events-to-analyze"], "1"))
    outfile = getopt(argv, ["--output-file", "-o"], None)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    if outfile is None:
        sys.stderr.write("fake_ile: no --output-file\n")
        return 2
    n_event_max = min(GRID_N, event + ngroup)
    for local, gidx in enumerate(range(event, n_event_max)):
        with open("{}_{}_.dat".format(outfile, local), "w") as f:
            f.write("{} {} {}\n".format(gidx, gpu, -1.0))   # global_idx  gpu  (pretend lnL)
    sys.stderr.write("[fake_ile] event={} ngroup={} gpu={} wrote {} pts prefix={}\n".format(
        event, ngroup, gpu, n_event_max - event, outfile))
    return 0


if __name__ == "__main__":
    sys.exit(main())
