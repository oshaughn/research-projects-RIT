#!/usr/bin/env python3
"""Profile ILE wall-clock runtime by GPU type, from a run's condor logs.

calmarg ILE is very expensive, and runtime depends strongly on the GPU it lands on
(a 1050 Ti is ~2x slower than Blackwell and routinely blows the runtime wall).  This
reads each ILE-*.out (which prints the matched GPU's name + compute capability via the
container's cupy banner) and the matching ILE-*.log (condor execute/terminate events),
correlates runtime with GPU capability, and prints a per-capability summary so you can
pick the right `RIFT_REQUIRE_GPUS` floor and `ile-jobs-per-worker` BEFORE a big run.

Usage:
    profile_ile_runtimes.py <rundir> [<rundir> ...]
Each <rundir> is a top-level RIFT run dir (contains iteration_*_ile/logs/).
"""
import os, re, glob, sys, statistics, datetime

TS  = re.compile(r'\((\d+)\.\d+\.\d+\)\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)')
DEV = re.compile(r'Device 0 Name\s*:\s*(.+)')
CAP = re.compile(r'Device 0 Compute Capability\s*:\s*(\d+)')   # e.g. 120 -> 12.0, 61 -> 6.1

def _ts(line):
    m = TS.search(line)
    return datetime.datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S") if m else None

def collect(rundirs):
    rows = []          # (cap_float, device, runtime_min, ended)
    nout = 0
    for rd in rundirs:
        for o in glob.glob(os.path.join(rd, "iteration_*_ile/logs/ILE-*.out")):
            nout += 1
            try:
                txt = open(o, errors='ignore').read()
            except OSError:
                continue
            cm = CAP.search(txt)
            if not cm:
                continue                      # never reached the GPU banner
            cap = int(cm.group(1)) / 10.0
            dm  = DEV.search(txt)
            dev = dm.group(1).strip() if dm else "?"
            lg = o[:-4] + ".log"
            if not os.path.exists(lg):
                continue
            exes, term, ended = [], None, None
            for ln in open(lg, errors='ignore'):
                if "Job executing on host" in ln:
                    t = _ts(ln)
                    if t: exes.append(t)
                elif "Job terminated" in ln:
                    term, ended = _ts(ln), "done"
                elif "Job was aborted" in ln or "removed" in ln.lower():
                    t = _ts(ln)
                    if t: term, ended = t, (ended or "killed")
            if exes and term:
                rt = (term - exes[-1]).total_seconds() / 60.0
                if 0 < rt < 1200:
                    rows.append((cap, dev, rt, ended or "done"))
    return nout, rows

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    nout, rows = collect(sys.argv[1:])
    print(f"ILE .out scanned: {nout}   usable (gpu+runtime): {len(rows)}\n")
    by, devof = {}, {}
    for cap, dev, rt, _ in rows:
        by.setdefault(cap, []).append(rt)
        devof.setdefault(cap, dev)
    print(f"{'cc':>5} {'n':>5} {'median':>8} {'p90':>8} {'max':>8} {'>wall(115m)':>12}  device")
    for cap in sorted(by):
        v = sorted(by[cap]); n = len(v)
        p90 = v[int(0.9 * (n - 1))]
        wall = sum(1 for x in v if x > 115)
        print(f"{cap:>5.1f} {n:>5} {statistics.median(v):>7.1f}m {p90:>7.1f}m "
              f"{max(v):>7.1f}m {wall:>7}/{n:<4} {devof[cap][:34]}")

if __name__ == "__main__":
    main()
