#!/usr/bin/env python3
"""Rebuild a RIFT DAG's rescue file(s) from the dagman.out completion record.

When a DEV run's rescue DAG gets clobbered (a resubmit bypassed it, a cluster/
maintenance hiccup, an accidental fresh start), DAGMan restarts from scratch and
redoes work whose outputs are already on disk.  But the dagman.out -- which is
APPENDED across every invocation -- still logs each node that finished:

    Node <name> job proc (cluster.proc) completed successfully.

This reconstructs a partial rescue DAG (v2.1.0) marking those nodes DONE, and
recurses into nested `SUBDAG EXTERNAL` inner DAGs (the convergence subdags, where
most ILE shards live).  It also UNIONS in any DONE marks already present in the
current highest rescue, so manual fixes (e.g. a join run by hand) are preserved.

"Completed successfully" is the data signal: a node that exited 0 wrote its output.
(Assumes those outputs were not later deleted -- true for a resubmit, which does not
`rm` the run dir; NOT safe if you re-ran `make ... -build`, which starts with rm -rf.)

A SUBDAG node is only marked DONE in its parent when it appears in the PARENT's
completed set (DAGMan logs that only when the whole inner DAG finished).  A partially
complete subdag is left un-done in the parent (so it re-runs) and gets its own inner
rescue from the recursion -- so the inner DAGMan resumes mid-subdag.

Does NOT submit anything.  Inspect the dry-run, then re-run with --apply, then resubmit
the ORIGINAL dag by hand (condor_submit_dag).

Usage:
    rebuild_rescue_from_logs.py <top-level .dag path> [--apply]
"""
import os, re, sys, glob

DONE_LOG_RE = re.compile(r'Node\s+(\S+)\s+job proc\s+\(\S+\)\s+completed successfully')
RESCUE_DONE_RE = re.compile(r'^DONE\s+(\S+)', re.M)
NODE_RE = re.compile(r'^(JOB|SUBDAG EXTERNAL)\s+(\S+)\s+(\S+)', re.M)
RESCUE_NUM_RE = re.compile(r'\.rescue(\d+)$')


def parse_dag(dagpath):
    """Return (ordered node names, {subdag node name: inner dag path})."""
    nodes, subdags = [], {}
    for m in NODE_RE.finditer(open(dagpath, errors='ignore').read()):
        kind, name, target = m.group(1), m.group(2), m.group(3)
        nodes.append(name)
        if kind == 'SUBDAG EXTERNAL':
            subdags[name] = target
    return nodes, subdags


def completed_from_dagman(dagpath):
    """Unique node names logged 'completed successfully' across all runs."""
    do = dagpath + '.dagman.out'
    done = set()
    if os.path.exists(do):
        for ln in open(do, errors='ignore'):
            m = DONE_LOG_RE.search(ln)
            if m:
                done.add(m.group(1))
    return done


def _rescue_files(dagpath):
    return [r for r in glob.glob(dagpath + '.rescue*') if RESCUE_NUM_RE.search(r)]


def existing_rescue_done(dagpath):
    """DONE marks in the highest current rescueNNN (preserve manual fixes)."""
    files = sorted(_rescue_files(dagpath),
                   key=lambda r: int(RESCUE_NUM_RE.search(r).group(1)))
    if not files:
        return set()
    return set(RESCUE_DONE_RE.findall(open(files[-1], errors='ignore').read()))


def next_rescue_path(dagpath):
    nums = [int(RESCUE_NUM_RE.search(r).group(1)) for r in _rescue_files(dagpath)]
    n = (max(nums) + 1) if nums else 1
    return '%s.rescue%03d' % (dagpath, n), n


def write_rescue(dagpath, done, total, apply):
    path, n = next_rescue_path(dagpath)
    if apply:
        with open(path, 'w') as f:
            f.write("# Rescue DAG file, REBUILT from dagman.out completion records\n")
            f.write("#   by tools/rebuild_rescue_from_logs.py (DEV recovery, not production)\n")
            f.write("# Rescue DAG version: 2.1.0\n#\n")
            f.write("# Total number of Nodes: %d\n" % total)
            f.write("# Nodes premarked DONE: %d\n\n" % len(done))
            for nm in sorted(done):
                f.write("DONE %s\n" % nm)
    return n


def process(dagpath, apply, depth=0):
    dagpath = os.path.abspath(dagpath)
    nodes, subdags = parse_dag(dagpath)
    nodeset = set(nodes)
    done = (nodeset & completed_from_dagman(dagpath)) | (nodeset & existing_rescue_done(dagpath))
    # recurse into subdags that did NOT fully complete at this level
    for sname, inner in subdags.items():
        if sname in done:
            continue
        innerpath = inner if os.path.isabs(inner) else os.path.join(os.path.dirname(dagpath), inner)
        if os.path.exists(innerpath):
            process(innerpath, apply, depth + 1)
    n = write_rescue(dagpath, done, len(nodes), apply)
    print("%s%s : %d/%d DONE -> rescue%03d %s"
          % ('  ' * depth, os.path.basename(os.path.dirname(dagpath)) + '/' + os.path.basename(dagpath),
             len(done), len(nodes), n, '(WRITTEN)' if apply else '(dry-run)'))
    return done, nodes


if __name__ == '__main__':
    pos = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not pos:
        print(__doc__); sys.exit(2)
    process(pos[0], '--apply' in sys.argv)
