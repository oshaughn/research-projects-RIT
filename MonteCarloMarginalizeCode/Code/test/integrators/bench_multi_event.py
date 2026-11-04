#!/usr/bin/env python3
"""
Multi-event robustness check for the portfolio freeze-policy change.

For each event we take its REAL iteration-0 ILE worker args (real strain/PSD/grid, from the
event's ILE.sub), strip the gwsignal/SEOBNR *container-only* flags, and swap the waveform to
the bare-venv-native IMRPhenomXPHM (l-max 4, precessing).  The waveform is only the integrand;
the point of this test is the INTEGRATOR: does the AV+GMM portfolio (VARAHA never-freeze, the
new default) REPLICATE the standalone-AV integral (same ln Z within MC error) and converge to a
comparable n_eff, across a spread of real events?

Usage:
  bench_multi_event.py smoke <EVENT>                 # tiny-budget single AV run (plumbing check)
  bench_multi_event.py run <EVENT> <tag> <sampler...> # one run; sampler... e.g. AV  OR  portfolio --sampler-portfolio AV,GMM
Env: GPU (default 2), NEFF (default 40), NMAX (default 2000000)

ln Z / n_eff are read back from the ILE output <tag>.xml_0_.dat (lnZ = field[-4], neff=field[-1]).
"""
import re, shlex, subprocess, sys, os

EVENTS_BASE = "/home/richard.oshaughnessy/unixhome/Projects/LIGO-ScienceMode/O4_era/RIFT_roboto_paper/analyses/rerun_o4ab_distance_export/project/working"
WT   = "/home/richard.oshaughnessy/RIFT_develUWM/src/research-projects-RIT/.claude/worktrees/rift-adaptive-integrator/.claude/worktrees/gifted-herschel-caf99c"
CODE = WT + "/MonteCarloMarginalizeCode/Code"
BIN  = CODE + "/bin/integrate_likelihood_extrinsic_batchmode"
PY   = "/home/richard.oshaughnessy/RIFT_develUWM/bin/python"

def event_dir(event):
    return "{}/{}/rift-distexport-nocal".format(EVENTS_BASE, event)

# exact production container for these events (from the event ILE.sub SingularityImage line)
SIF = "/home/richard.oshaughnessy/rift_cit_build_container_family/built_containers/rift_o4d-calmarg_in_loop_cc60-90_cuda118_20260615b.sif"

def build_args(event, container=False):
    # Parse the pipeline-generated command-single.sh: it holds the fully-formed, correctly-quoted
    # single-worker ILE command (in particular the --internal-waveform-extra-kwargs dict), so we
    # avoid re-deriving condor's '""'/''-escaping from ILE.sub (which mangled the nested quotes).
    edir = event_dir(event)
    txt = open(edir + "/command-single.sh").read()
    m = re.search(r'^\S*integrate_likelihood_extrinsic_batchmode\s+(.*)$', txt, re.M)
    argv = shlex.split(m.group(1))
    if not container:
        # BARE-VENV fallback (unused for the robustness suite): swap to a native waveform and drop
        # gwsignal/cosmo-prior flags.  FAILS on grids with transverse spins (RIFT cannot recover
        # modes from H+/Hx for precessing configs) -- that is why the suite uses the container.
        drop, out, skip = {"--use-gwsignal", "--force-gpu-only"}, [], 0
        for tok in argv:
            if skip: skip = 0; continue
            if tok == "--internal-waveform-extra-kwargs": skip = 1; continue
            if tok in drop: continue
            out.append("IMRPhenomXHM" if tok == "SEOBNRv5PHM" else tok)
        argv = out
    return edir, argv

def filtered(argv, container=False):
    """Drop flags we override: sampler-method, n-eff, n-max, output-file, n-events-to-analyze,
    event (we re-add --event 0 --n-events-to-analyze 1 to integrate a single intrinsic point).
    In BARE mode also drop --d-prior (cosmo_* needs cupyx.scipy.interpolate, absent in the old
    venv cupy 10.6); in CONTAINER mode the cosmo prior works, so keep it (faithful integrand)."""
    drop_val = {"--sampler-method", "--n-eff", "--n-max", "--n-events-to-analyze", "--event"}
    if not container:
        drop_val.add("--d-prior")
    out, skip = [], 0
    for tok in argv:
        if skip:
            skip = 0; continue
        if tok in drop_val:
            skip = 1; continue
        if tok.startswith("--output-file") or tok.startswith("--event="):
            continue
        out.append(tok)
    return out

def run(event, tag, sampler_extra, neff, nmax, gpu, container=False, wrap=True):
    """container=True builds the FULL production args (gwsignal/SEOBNRv5PHM/cosmo prior).
    wrap=False runs python directly instead of nesting singularity -- use when the job is ALREADY
    inside the container (e.g. condor supplied it via MY.SingularityImage)."""
    edir, argv = build_args(event, container=container)
    argv = filtered(argv, container=container)
    out = "{}/mev_{}.xml".format(edir, tag)
    argv += ["--sampler-method"] + sampler_extra + ["--n-eff", str(int(neff)), "--n-max", str(int(nmax)),
                                                    "--n-events-to-analyze", "1", "--event", "0",
                                                    "--output-file", out]
    log = "{}/mev_{}.log".format(edir, tag)
    if container and wrap:
        # Run inside the event's production container (real SEOBNRv5PHM + gwsignal + cuda118 cupy)
        # but force MY worktree RIFT onto PYTHONPATH so the CONTAINER supplies the waveform/cupy
        # stack while the INTEGRATOR code under test is this branch's.  Bind ceph frames + the
        # worktree; $HOME is auto-mounted so edir/local.cache/PSDs resolve.
        inner = ("cd {edir} && PYTHONPATH={code}:$PYTHONPATH PATH={code}/bin:$PATH "
                 "CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 "
                 "python -u {bin} {args}").format(
                     edir=edir, code=CODE, gpu=gpu, bin=BIN,
                     args=" ".join(shlex.quote(a) for a in argv))
        cmd = ["singularity", "exec", "--nv", "--bind", "/ceph", "--bind", "/cvmfs",
               "--bind", WT, SIF, "bash", "-c", inner]
        env = dict(os.environ)
    else:
        # direct exec: either the bare-venv fallback, or we are already inside the container
        _py = "python" if (container and not wrap) else PY
        cmd = [_py, "-u", BIN] + argv
        env = dict(os.environ)
        env["PYTHONPATH"] = CODE + ":" + env.get("PYTHONPATH", "")
        env["PATH"] = CODE + "/bin:" + env["PATH"]
        env["PYTHONUNBUFFERED"] = "1"; env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["OMP_NUM_THREADS"] = "2"
    with open(log, "w") as lf:
        lf.write("# MULTIEVENT {} tag={} gpu={} neff={} nmax={} container={}\n# {}\n".format(
            event, tag, gpu, neff, nmax, container, " ".join(argv)))
        lf.flush()
        rc = subprocess.call(cmd, cwd=edir, stdout=lf, stderr=subprocess.STDOUT, env=env)
        lf.write("\n# EXIT {}\n".format(rc))
    return log

def read_result(event, tag):
    dat = "{}/mev_{}.xml_0_.dat".format(event_dir(event), tag)
    try:
        with open(dat) as f:
            line = f.readline().split()
        vals = [float(x) for x in line]
        return {"lnZ": vals[-4], "sigOverL": vals[-3], "ntot": vals[-2], "neff": vals[-1]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    cmd = sys.argv[1]
    gpu = int(os.environ.get("GPU", 2)); neff = float(os.environ.get("NEFF", 40)); nmax = int(os.environ.get("NMAX", 2000000))
    container = os.environ.get("CONTAINER", "1") == "1"   # default: faithful container path
    wrap = os.environ.get("NO_SINGULARITY", "0") != "1"  # 0 => nest singularity; 1 => already inside
    if cmd == "smoke":
        log = run(sys.argv[2], "smoke", ["AV"], neff=999, nmax=60000, gpu=gpu, container=container, wrap=wrap)
        print("smoke log:", log); print(read_result(sys.argv[2], "smoke"))
    elif cmd == "run":
        event, tag = sys.argv[2], sys.argv[3]; sampler_extra = sys.argv[4:]
        log = run(event, tag, sampler_extra, neff, nmax, gpu, container=container, wrap=wrap)
        print(tag, read_result(event, tag), "log:", log)
    elif cmd == "read":
        print(sys.argv[2], sys.argv[3], read_result(sys.argv[2], sys.argv[3]))
