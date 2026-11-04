"""End-to-end ILE on a case whose answer is known in CLOSED FORM.

WHY THIS EXISTS.  The sampler unit tests check samplers; the pipeline gates (pseudo_pipe,
asimov) check that the pipeline RUNS.  Nothing checked that it runs and is CORRECT, and that
gap hid the defects listed below, none of which four rounds of adversarial review of the
drivers found.

THE TECHNIQUE.  --zero-likelihood makes the signal term exactly 0, so ln Z is the log of the
prior mass, which is 0 for normalized priors.  Adding an analytic --supplementary-likelihood-
factor-* whose marginal can be written down moves ln Z to a known value.  See
analytic_supplement_for_e2e for the factor and its closed form.  That is the constant-integrand
trick of the sampler unit tests lifted to the pipeline: no fit error, no MC scatter in the
TARGET, and an answer you can write down.

ITS FAST COMPANION.  test_zero_likelihood_standin.py checks the same stand-in as WIRING --
argument order against the driver's real call sites, generated signature, array module -- in
about 4 s.  It exists because an end-to-end marginal is structurally unable to distinguish a
permutation among right_ascension, phi_orb and psi: the three are independent and identically
distributed, so any factor's marginal is the same under the swap.  Read both files together.

SAMPLERS.  AV, portfolio (AV + AC) and GMM run every factor lane, and adaptive_cartesian runs
the prior-only one and its own factor lane in section 3.  GMM was a recorded known-wrong lane
here until #359 ("GMM sampler: dim-group keys were in the wrong frame"), which moved it from
-23.1 to -0.26 sigma on the sharply peaked target; it is a first-class lane now, and this file
is what keeps it one.  adaptive_cartesian_gpu -- mcsamplerGPU standalone, and the driver's own
default -- runs on the DEVICE only.  mcsamplerGPU on numpy is already under test: the portfolio
lanes' "AC" member IS adaptive_cartesian_gpu, and they run on the host.  What the device lane
adds is a second route into the host/device conversion in compute_hist; see section 4.

WHAT IS DELIBERATELY ABSENT.  The high-SNR recipe in
~/rift-integrator-lore/coordinates-and-degeneracies.md (--force-adapt-all plus
--internal-rotate-phase and --internal-sky-network-coordinates) is the documented fix for a
collapsing n_eff, and it is NOT used here.  --internal-sky-network-coordinates needs two
detectors and this fixture is H1-only.  --internal-rotate-phase changes which coordinates the
supplementary factor is handed on the raw path, so the closed form would no longer describe the
integral being measured.  A lane that passes because the geometry was changed underneath it is
measuring something else.  If a lane will not converge, scope it or leave it out; do not add
these.

THE DEVICE.  Section 4 runs the prior-only and factor answers again with a GPU VISIBLE, and
asserts the child reached it.  RIFT binds its array module at import from whether cupy imports,
not from --gpu, so on a GPU node production is on the device path by default and the rest of
this file -- which pins CUDA_VISIBLE_DEVICES="" -- said nothing about it.  All five samplers
run there.  What section 4 does NOT cover is the GPU signal likelihood: --zero-likelihood
replaces likelihood_function outright, so the NoLoop path never runs.

WHAT EACH ARM COSTS.  About 8-12 s of one core per ILE arm, plus ~15 s once for the
distance-marginalization lookup table.  No network, no real event.  The CPU lanes need no GPU;
the device lanes skip without one, and a skip there is not a pass.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

lal = pytest.importorskip("lal")
lalsim = pytest.importorskip("lalsimulation")
_lal_series = pytest.importorskip("lal.series")
_ligolw_utils = pytest.importorskip("igwn_ligolw.utils")
import RIFT.lalsimutils as lalsimutils

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
BIN = os.path.join(CODE, "bin")
ILE = os.path.join(BIN, "integrate_likelihood_extrinsic_batchmode")
MARG_TABLE_TOOL = os.path.join(BIN, "util_InitMargTable")
SUPPLEMENT_MODULE = "analytic_supplement_for_e2e"

# TOLERANCES, MEASURED not guessed.  See CALIBRATION at the end of this file for the sweep
# these come from, what was varied, and how much headroom each one has.
Z_TOLERANCE = 5.0
# The informativeness floor, and the one that does the work: a z-test passes vacuously if sigma
# is large, so cap it.  5 * MAX_SIGMA is a 0.30-nat band.
MAX_SIGMA = 0.06
# A DEGENERACY TRIPWIRE, deliberately loose -- not a convergence criterion, which is MAX_SIGMA.
# n_eff is a noisy statistic at fixed accuracy: adaptive_cartesian reported 91 to 220 over eight
# seeds whose lnZ all landed inside 1.2 sigma, so a tight floor here buys a flake, not a check.
MIN_NEFF = 30.0

_AV = ["--sampler-method", "AV"]
_PORTFOLIO = ["--sampler-method", "portfolio",
              "--sampler-portfolio", "AV", "--sampler-portfolio", "AC"]
_GMM = ["--sampler-method", "GMM"]
_AC = ["--sampler-method", "adaptive_cartesian"]
# mcsamplerGPU standalone, and the ILE driver's OWN default --sampler-method.  A different
# sampler from _AC despite the name: _AC is mcsampler.
_ACG = ["--sampler-method", "adaptive_cartesian_gpu"]
# adaptive_cartesian stops on the sample BUDGET rather than on --n-eff at the 20000 the other
# lanes use; see test_adaptive_cartesian for the eight-seed measurement behind this number.
_AC_N_MAX = 60000
SAMPLER_ARGS = {"AV": _AV, "portfolio": _PORTFOLIO, "GMM": _GMM, "adaptive_cartesian": _AC,
                "adaptive_cartesian_gpu": _ACG}

# Per-lane _run_ile overrides, keyed by sampler and SHARED by the host and device lanes, so a
# device lane runs its host twin's configuration and not a nearby one.
#
# Both mcsampler and mcsamplerGPU stop on the sample BUDGET with the factor on at the 20000 the
# other lanes use, so both take _AC_N_MAX.  MEASURED for adaptive_cartesian_gpu rather than
# inherited from adaptive_cartesian, because they are different samplers: at A=8 B=2, seed 1000,
# ldas-pcdev2 CUDA slot 0 (A100, cc 8.0), --n-max 20000 stops at ntotal 20000 with n_eff 230
# against a target of 250, while --n-max 60000 reaches n_eff 342 at ntotal 30000 with sigma
# 0.0221 against 0.0282.
#
# Over eight seeds on ldas-pcdev12 slot 0, adaptive_cartesian_gpu at --n-max 60000 spends ntotal
# 20000 to 40000 and lands n_eff 261 to 384, so the cap is headroom with about a 1.5x margin at
# the worst seed and NOT a promise that the lane stops on --n-eff.  Nothing asserts the override
# is still applied: at --n-max 20000 the device lane still passes (seed 1000: z -0.40, sigma
# 0.0282, n_eff 230, all inside the thresholds).  A ntotal < n_max guard was measured and
# REJECTED -- adaptive_cartesian reaches ntotal 60000 on seed 1004, so the guard would flake on
# the lane it was meant to protect.  Re-derive with make_e2e_calibration.py instead.
#
# Applied to the PRIOR-ONLY lanes too, where it costs nothing and keeps one definition: there
# the integrand is flat and the run stops on --n-eff at ntotal 10000, with lnL, sigma, ntotal
# and n_eff identical at either cap.  Seed 1000: adaptive_cartesian on ldas-grid 10000/3329,
# adaptive_cartesian_gpu on ldas-pcdev2 slot 0 10000/3364.
_LANE_KW = {"adaptive_cartesian": dict(n_max=_AC_N_MAX),
            "adaptive_cartesian_gpu": dict(n_max=_AC_N_MAX)}


# ---------------------------------------------------------------------------------------
# fixtures

def build_event(out):
    """Zero-strain H1 frame + PSD + LAL cache in `out`.  Copied in shape from the fixture in
    test_psi_marginalization.py, which is already exercised in CI.

    A plain function, not only a fixture, so the calibration generator in
    expensive_before_merging/integrators/make_e2e_calibration.py builds the SAME fixture this
    gate runs on.  A calibration measured on a different fixture would not calibrate anything."""
    if shutil.which("lal_path2cache") is None:
        return None
    t0, srate = 1000000000.0, 2048.0
    dt = 1.0 / srate
    seg_start, seg_end = t0 - 6.0, t0 + 2.0
    dur = seg_end - seg_start
    npts = int(round(dur / dt))
    ht = lal.CreateREAL8TimeSeries("Zero strain", lal.LIGOTimeGPS(seg_start), 0.0, dt,
                                   lalsimutils.lsu_DimensionlessUnit, npts)
    ht.data.data = np.zeros(npts)
    frame = out / ("H-fake_strain-%d-%d.gwf" % (int(seg_start), int(dur)))
    lalsimutils.hoft_to_frame_data(str(frame), "H1:FAKE-STRAIN", ht)
    cache = out / "test.cache"
    if os.system("echo %s | lal_path2cache > %s" % (frame, cache)) != 0:
        return None
    psd = lal.CreateREAL8FrequencySeries("psd", lal.LIGOTimeGPS(0), 0, 1.0 / dur,
                                         lal.SecondUnit, npts // 2 + 1)
    f = psd.f0 + np.arange(psd.data.length) * psd.deltaF
    psd.data.data = np.where(f > 1.0,
                             [lalsim.SimNoisePSDaLIGOZeroDetHighPower(x) for x in f], 1.0)
    _ligolw_utils.write_filename(_lal_series.make_psd_xmldoc({"H1": psd}),
                                 str(out / "H1_psd.xml.gz"))
    return dict(dir=out, cache=cache, psd=out / "H1_psd.xml.gz",
                t0=t0, seg_start=seg_start, seg_end=seg_end)


@pytest.fixture(scope="module")
def event(tmp_path_factory):
    ev = build_event(tmp_path_factory.mktemp("e2e_analytic"))
    if ev is None:
        pytest.skip("lal_path2cache unavailable or failed")
    return ev


# The distance-marginalized likelihood reads bmax/bref/s_array/t_array/lnI_array out of this
# file at SETUP, before any --zero-likelihood substitution, so the lane needs a real table and
# not a stub.  Its CONTENTS never reach the answer here (the likelihood body never runs), so
# the cheapest table the tool will build is the right one: a narrow distance range and low
# quadrature degrees take ~15 s instead of ~47 s for the defaults.
DMARG_TABLE_ARGS = ["--d-min", "100", "--d-max", "1000", "--max-snr", "100",
                    "--hermgauss-degree", "20", "--laggauss-degree", "20"]


@pytest.fixture(scope="module")
def dmarg_table(event):
    out = event["dir"] / "marg_lookup.npz"
    proc = subprocess.run([sys.executable, MARG_TABLE_TOOL] + DMARG_TABLE_ARGS
                          + ["--out", str(out)],
                          cwd=str(event["dir"]), env=_child_env(), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=1200)
    if proc.returncode != 0 or not out.exists():
        pytest.skip("util_InitMargTable failed:\n%s" % proc.stdout.decode()[-1500:])
    return out


# ---------------------------------------------------------------------------------------
# the device

# Printed by the driver's own preamble only AFTER `cupy.array(5)` succeeds, so it cannot appear
# on a host.  That makes it the one cheap proof a "GPU lane" was not a CPU lane.
_DEVICE_MARKER = "cupy memory [total, available]"

# The production shape of a deliberate GPU invocation.  Note these are NOT what puts the run on
# a device: xpy_default is bound at IMPORT from whether cupy imports, so a device that is merely
# visible is already in use.  That is why the bare-flags lane below exists as well.
_GPU_FLAGS = ("--vectorized", "--gpu", "--force-xpy")

# Every verdict is ONE line beginning with a known word, and every message is flattened, because
# cupy's ImportError is a multi-line banner: reading "the last line of the probe's output" turned
# it into the skip reason "no usable GPU (If you installed CuPy via whee)".
_GPU_PROBE = r"""
import numpy as np
def _flat(e):
    return ("%s: %s" % (type(e).__name__, e)).replace("\n", " ")[:150]
try:
    import cupy
except Exception as e:
    print("VERDICT NOCUPY %s" % _flat(e)); raise SystemExit(0)
bad = []
for d in range(cupy.cuda.runtime.getDeviceCount()):
    try:
        with cupy.cuda.Device(d):
            cupy.asnumpy(cupy.cos(cupy.asarray(np.zeros(2), dtype=float)))
    except Exception as e:
        bad.append("%d:%s" % (d, type(e).__name__)); continue
    print("VERDICT SLOT %d" % d); raise SystemExit(0)
print("VERDICT NOSLOT %s" % (",".join(bad) or "no devices at all"))
"""


def _no_gpu(reason):
    """Skip, or FAIL when the environment promised a device.

    .travis/test-integrate.sh applies this rule too (with RIFT_CI_REQUIRE_GPU=1 any skip is
    fatal there), but a rule that lives only in the shell does not survive `pytest <this file>`
    on the GPU runner -- which is what someone runs to reproduce a CI failure, and it would
    report green with all 11 device lanes skipped.  RIFT_CI_REQUIRE_GPU is read from the ambient
    environment deliberately: _child_env strips RIFT_* from ILE CHILDREN, a different question
    from what this pytest process was promised."""
    if os.environ.get("RIFT_CI_REQUIRE_GPU", "0") == "1":
        # Reported as an ERROR rather than a FAILURE, because gpu_slot is a fixture and this
        # fires during setup.  Red either way, which is the point; measured on ldas-grid
        # (cupy absent) 2026-09-18: `pytest -k gpu` with RIFT_CI_REQUIRE_GPU=1 gives 11
        # errors and 18 deselected, i.e. every device lane and no host lane.
        pytest.fail("RIFT_CI_REQUIRE_GPU=1 promised a usable device and there is none: %s.  On "
                    "this runner a skipped device lane is a failure, not a pass." % reason)
    pytest.skip("%s  A skip is NOT a pass: pin CUDA_VISIBLE_DEVICES to a slot the installed "
                "cupy supports and rerun." % reason)


@pytest.fixture(scope="module")
def gpu_slot():
    """A CUDA slot this cupy can actually build a kernel for, as the child should see it.

    TWO failures that look alike, and conflating them is how a device lane quietly never runs.
    `import cupy` fails outright where there is no CUDA runtime (ldas-grid, CI runners).  It
    SUCCEEDS on the CIT GPU head nodes while the visible device is one this cupy cannot compile
    for: ldas-pcdev13 slots 0-2 and all of ldas-pcdev11 are Blackwell cc 12.0, and cupy 12.0.0
    answers `nvrtc: error: invalid value for --gpu-architecture`.  So the probe RUNS a kernel
    instead of trusting the import, and it probes at dispatch, because the slot map moves.

    It probes in a SUBPROCESS so this pytest process never holds a CUDA context while the ILE
    children run, and it re-indexes: the probe's index is into the CURRENTLY VISIBLE list, so
    when the parent already has CUDA_VISIBLE_DEVICES set, what the child is given is that
    list's d-th entry, not d.

    A SKIP HERE IS NOT A PASS."""
    proc = subprocess.run([sys.executable, "-c", _GPU_PROBE], env=_child_env(cuda=None),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    verdicts = [l for l in proc.stdout.decode().splitlines() if l.startswith("VERDICT ")]
    line = verdicts[-1][len("VERDICT "):] if verdicts else (
        "probe produced no verdict (rc=%d): %s" % (proc.returncode,
                                                   proc.stdout.decode()[-300:]))
    if not line.startswith("SLOT "):
        _no_gpu("no usable GPU for these lanes -- %s." % line)
    d = int(line.split()[1])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return visible.split(",")[d].strip()
    return str(d)


# ---------------------------------------------------------------------------------------
# running one ILE arm

def _child_env(cuda="", **extra):
    # Every RIFT_* variable is stripped, not just the one that bit.  RIFT_HYPERPIPELINE_FORMAT
    # moves lnL/sigma to columns 0/1 and adds a header, under which the tail-column read below
    # returns a spin component (0.0) as sigma and the z-test passes VACUOUSLY.  Others
    # (RIFT_LOWLATENCY, RIFT_NO_GWSIGNAL, RIFT_GPU_*) change the code path.  A gate whose
    # answer depends on the invoking shell's environment is not a gate.
    env = {k: v for k, v in os.environ.items() if not k.startswith("RIFT_")}
    # CUDA_VISIBLE_DEVICES is pinned so the answer does not depend on the invoking shell.  ""
    # is the default and what every CPU lane uses.  The device lanes pass a real slot; see
    # gpu_slot.  cuda=None leaves whatever the parent has, which only the device PROBE wants.
    #
    # The empty default used to carry "required, because on a host where cupy sees a GPU the
    # scalar AV path raises Unsupported dtype float128".  That is true where it was written
    # (test_psi_marginalization.py) and NOT true here, measured: this gate runs
    # --zero-likelihood, so make_zero_likelihood_standin builds lnL with xpy.zeros -- float64
    # on the device -- and the scalar likelihood that would have built a RiftFloat never runs.
    # Verified on ldas-pcdev2 (A100) 2026-09-17: --sampler-method AV with the device visible
    # completes and lands 1.5 sigma from the closed form.  Do not re-copy that sentence here.
    if cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda
    env["OMP_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    # THIS tree, not whatever RIFT is installed: the driver is run by absolute path out of the
    # checkout, but `import RIFT.integrators...` inside it would otherwise resolve to the
    # installed package, and half of what this gate covers lives there.  HERE is also on the
    # path so the child can import the analytic factor module.
    env["PYTHONPATH"] = os.pathsep.join(
        [HERE, CODE] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update(extra)
    return env


def _read_result(d, tag):
    """(lnL, sigma, n_eff) from the ILE output row.

    The row is read by POSITION, which is what every downstream consumer does, and then
    CROSS-CHECKED against out_*_integrator_status.json, which carries the same four numbers
    keyed BY NAME.  That is the point of reading both: the positional contract is pinned by a
    self-describing file rather than by a comment that can go stale."""
    row = d / ("%s_0_.dat" % tag)
    status = d / ("%s_0_integrator_status.json" % tag)
    first = row.read_text().lstrip()[:1]
    assert first != "#", (
        "%s has a header line, so it is not the legacy column layout this reads by position; "
        "some RIFT_* output-format variable survived into the child environment" % row)
    vals = np.atleast_2d(np.loadtxt(str(row)))[0]
    lnL, sigma, ntotal, neff = (float(vals[-4]), float(vals[-3]),
                                float(vals[-2]), float(vals[-1]))
    st = json.loads(status.read_text())
    for name, got in (("lnL", lnL), ("sigma_lnL", sigma), ("ntotal", ntotal), ("neff", neff)):
        want = float(st[name])
        assert got == pytest.approx(want, rel=1e-12, nan_ok=True), (
            "column contract broken: the row's tail columns give %s=%r, the status JSON says "
            "%r.  The .dat layout changed and this file is reading the wrong columns."
            % (name, got, want))
    return lnL, sigma, neff


def _invoke_ile(event, tag, sampler_args, *, a_coeff=None, b_coeff=0.0, incl_is_cosine=False,
                n_max=20000, n_eff=250, seed=1000, extra=(), cuda=""):
    """Run one ILE arm.  Returns (dir, returncode, stdout text, whether the result row exists).

    Kept separate from _run_ile, which is its only caller today, because the raw child log is
    what a lane pinning the SHAPE of a failure needs -- section 4 had two such lanes until the
    defects they recorded were fixed, and the next one should not have to re-split this."""
    d = event["dir"] / tag
    d.mkdir(exist_ok=True)
    env = _child_env(cuda=cuda)
    cmd = [sys.executable, ILE,
           "--cache-file", str(event["cache"]), "--channel-name", "H1=FAKE-STRAIN",
           "--psd-file", "H1=%s" % event["psd"],
           "--event-time", str(event["t0"]),
           "--data-start-time", str(event["seg_start"]),
           "--data-end-time", str(event["seg_end"]),
           "--mass1", "35.0", "--mass2", "30.0",
           "--approximant", "TaylorT4", "--l-max", "2",
           "--reference-freq", "40", "--fmin-template", "40",
           "--srate", "2048", "--inv-spec-trunc-time", "0",
           "--zero-likelihood", "--n-max", str(n_max), "--n-eff", str(n_eff),
           "--seed", str(seed), "--output-file", tag] + list(sampler_args) + list(extra)
    if a_coeff is not None:
        env["E2E_A_COEFF"] = str(a_coeff)
        env["E2E_B_COEFF"] = str(b_coeff)
        env["E2E_INCL_IS_COSINE"] = "1" if incl_is_cosine else "0"
        cmd += ["--supplementary-likelihood-factor-code", SUPPLEMENT_MODULE,
                "--supplementary-likelihood-factor-function", "ln_analytic_factor"]
    proc = subprocess.run(cmd, cwd=str(d), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=1800)
    # The driver CATCHES an exception from analyze_event, prints "FAILED ANALYSIS", skips the
    # point and EXITS 0 -- so in a DAG a crashed configuration is silent.  Absence of the
    # output row, not the exit code, is what says the run failed.
    return d, proc.returncode, proc.stdout.decode(), (d / ("%s_0_.dat" % tag)).exists()


def _run_ile(event, tag, sampler_args, *, expect_device=False, **kw):
    """One ILE job as a subprocess.  Returns (lnL, sigma_lnL, n_eff).

    KEYWORD-ONLY past sampler_args, deliberately: this function's fourth positional used to be
    a_coeff, so `_run_ile(event, tag, args, 8.0)` would now quietly mean expect_device=8.0 with
    a_coeff=None -- a prior-only run asserted to be on a device.  Every caller passes keywords
    today; the `*` keeps that true.

    expect_device asserts the child really reached a GPU.  Without it a "GPU lane" that
    silently fell back to the host is indistinguishable from one that ran, which is the whole
    failure mode these lanes exist to remove: the marker is printed only after
    `cupy.array(5)` succeeds in the driver's own preamble, so it cannot be true on a host."""
    d, rc, out, have_row = _invoke_ile(event, tag, sampler_args, **kw)
    if rc != 0 or not have_row:
        pytest.fail("ILE (%s) exited %d and wrote no result row; tail:\n%s"
                    % (tag, rc, out[-3000:]))
    if expect_device:
        assert _DEVICE_MARKER in out, (
            "%s was supposed to run on a device, but the child never printed %r, so cupy did "
            "not initialise there and this lane measured the HOST path under a GPU name.  "
            "CUDA_VISIBLE_DEVICES reached the child as %r."
            % (tag, _DEVICE_MARKER, kw.get("cuda")))
    return _read_result(d, tag)


if HERE not in sys.path:
    sys.path.insert(0, HERE)
import analytic_supplement_for_e2e as _supplement


def _exact(a, b=0.0):
    return _supplement.exact_ln_Z(a, b)


def _assert_converged(tag, lnL, sigma, neff):
    assert np.isfinite(lnL) and np.isfinite(sigma), \
        "%s: the run reported lnL=%r sigma=%r -- no usable estimate" % (tag, lnL, sigma)
    assert neff >= MIN_NEFF, "%s: n_eff=%r did not reach the floor %r" % (tag, neff, MIN_NEFF)
    assert sigma < MAX_SIGMA, \
        "%s: sigma_lnL=%r exceeds %r, so a %r-sigma band is too wide to say anything" % (
            tag, sigma, MAX_SIGMA, Z_TOLERANCE)


def _assert_lnZ(tag, lnL, sigma, neff, exact):
    _assert_converged(tag, lnL, sigma, neff)
    z = (lnL - exact) / sigma
    assert abs(z) < Z_TOLERANCE, \
        "%s: ln Z = %r +- %r, exact = %r (z = %.1f)" % (tag, lnL, sigma, exact, z)


# ---------------------------------------------------------------------------------------
# 1. the prior-only answer

# adaptive_cartesian runs HERE and not in section 2: its factor case is section 3's
# test_adaptive_cartesian, which exists for a different reason (co_argcount) and already runs
# A=8 B=2 on the host.  This lane is here so section 4's device prior-only adaptive_cartesian
# row has a host twin.  That row held the worst |z| in the calibration table (2.39) and was the
# only row with no twin, so the number sizing Z_TOLERANCE's margin was the one nothing could be
# read against.  It now returns the same three columns on the host.
@pytest.mark.parametrize("sampler", ["AV", "portfolio", "GMM", "adaptive_cartesian"])
def test_zero_likelihood_alone_gives_ln_Z_zero(event, sampler):
    """--zero-likelihood makes the signal term exactly 0, so ln Z is the log prior mass, which
    is 0 for normalized extrinsic priors.  This is the whole extrinsic integrator, the driver
    and the export path checked against an absolute answer, not a difference."""
    tag = "zero_%s" % sampler
    lnL, sigma, neff = _run_ile(event, tag, SAMPLER_ARGS[sampler], a_coeff=None,
                                **_LANE_KW.get(sampler, {}))
    _assert_lnZ(tag, lnL, sigma, neff, 0.0)


# ---------------------------------------------------------------------------------------
# 2. the analytic factor

# (A, B): A alone is a nearly flat / sharply peaked target in phi_orb.  B turns on the
# inclination term, which is what makes the factor able to detect a MIS-WIRING; see the module
# docstring of analytic_supplement_for_e2e.
FACTOR_CASES = [(0.75, 0.0), (8.0, 0.0), (0.75, 3.0)]
# (8.0, 2.0), peaked AND asymmetric, is the case every lane in section 3 runs, so it is not
# repeated here.


@pytest.mark.parametrize("sampler", ["AV", "portfolio", "GMM"])
@pytest.mark.parametrize("a_coeff,b_coeff", FACTOR_CASES)
def test_analytic_factor_marginal_is_recovered(event, sampler, a_coeff, b_coeff):
    """ln Z must equal ln I0(A) + ln(sinh(B)/B).  A sampler can be right on a flat target and
    wrong on a peaked one, so both are checked, with and without the inclination term."""
    tag = "supp_%s_%s_%s" % (sampler, a_coeff, b_coeff)
    lnL, sigma, neff = _run_ile(event, tag, SAMPLER_ARGS[sampler],
                                a_coeff=a_coeff, b_coeff=b_coeff)
    _assert_lnZ(tag, lnL, sigma, neff, _exact(a_coeff, b_coeff))


def test_supplementary_factor_survives_zero_likelihood(event):
    """THE REGRESSION.  --zero-likelihood swaps the likelihood function for a stand-in; the
    supplementary factor has to survive that, or the option pair is silently inert while the
    startup banner reports the factor as active.  Measured on rift_O4d at d1d7c7e84, these two
    runs returned ln Z bit-identical: -0.027110200401507356 with the factor and without it,
    where the exact answer with it is 6.653324.

    Asserted as a DIFFERENCE as well as an absolute value: the difference cancels every prior
    normalization constant, so it isolates the factor itself."""
    a, b = 8.0, 2.0
    with_factor, sig_w, neff_w = _run_ile(event, "surv_with", _AV, a_coeff=a, b_coeff=b)
    without, sig_wo, neff_wo = _run_ile(event, "surv_without", _AV, a_coeff=None)
    _assert_converged("surv_with", with_factor, sig_w, neff_w)
    _assert_converged("surv_without", without, sig_wo, neff_wo)
    assert with_factor != without, \
        "the supplementary factor changed nothing: ln Z = %r in both arms, so --zero-" \
        "likelihood is discarding it" % with_factor
    delta = with_factor - without
    sigma = float(np.hypot(sig_w, sig_wo))
    exact = _exact(a, b)
    assert abs(delta - exact) < Z_TOLERANCE * sigma, \
        "factor contributed %r +- %r; exact is %r" % (delta, sigma, exact)


def test_the_factor_receives_the_raw_sampled_inclination(event):
    """THE COORDINATE CONTRACT.  ILE hands a supplementary factor the RAW sampled inputs, so
    under --inclination-cosine-sampler the 'inclination' argument IS cos(iota), not iota.  The
    factor is told which convention to expect (E2E_INCL_IS_COSINE) and its closed form is the
    same either way, so this arm answers a question the other arms cannot: a stand-in that
    helpfully applied the arccos itself would move ln Z well off sinh(B)/B -- by 3.4 nats at
    B = 2, against a sigma of 0.034."""
    a, b = 8.0, 2.0
    tag = "raw_incl_cosine"
    lnL, sigma, neff = _run_ile(event, tag, _AV, a_coeff=a, b_coeff=b, incl_is_cosine=True,
                                extra=("--inclination-cosine-sampler",))
    _assert_lnZ(tag, lnL, sigma, neff, _exact(a, b))


# ---------------------------------------------------------------------------------------
# 3. the configurations that CHANGE likelihood_function's signature

def test_time_marginalized_portfolio(event):
    """--time-marginalization drops t_ref from likelihood_function's signature.  With an AV
    member in the portfolio the integrand is called through
    mcsamplerPortfolio.integrate_log -> AV.update_sampling_prior_selfish -> lnF(*samples.T),
    i.e. POSITIONALLY: a stand-in that unpacked a fixed 7-tuple died here with 'not enough
    values to unpack (expected 7, got 6)', and the driver swallowed it, printed FAILED ANALYSIS
    and exited 0.  This is a production setting."""
    a, b = 8.0, 2.0
    tag = "tmarg_portfolio"
    lnL, sigma, neff = _run_ile(event, tag, _PORTFOLIO, a_coeff=a, b_coeff=b,
                                extra=("--time-marginalization",))
    _assert_lnZ(tag, lnL, sigma, neff, _exact(a, b))


def test_distance_marginalized(event, dmarg_table):
    """--distance-marginalization removes 'distance' from the sampled parameters and from
    likelihood_function's signature, which becomes (right_ascension, declination, phi_orb,
    inclination, psi).  The factor still takes six arguments, so the stand-in has to supply a
    value for that one; it uses 0.0, which is what the two distance-marginalized real call
    sites already pass.  The factor here does not use distance, which is the advice --help
    gives for a portable factor.  Reading it out of the sampled values instead raised KeyError
    on this configuration."""
    a, b = 8.0, 2.0
    tag = "dmarg"
    lnL, sigma, neff = _run_ile(
        event, tag, _AV, a_coeff=a, b_coeff=b,
        extra=("--distance-marginalization",
               "--distance-marginalization-lookup-table", str(dmarg_table),
               "--d-min", "100", "--d-max", "1000",
               "--time-marginalization", "--vectorized", "--gpu", "--force-xpy"))
    _assert_lnZ(tag, lnL, sigma, neff, _exact(a, b))


def test_adaptive_cartesian(event):
    """mcsampler decides which of its parameters to pass by reading
    func.__code__.co_varnames[:co_argcount].  A `def zero_like(*args, **kwargs)` stand-in
    reports ZERO arguments there, so --zero-likelihood --sampler-method adaptive_cartesian died
    in mcsampler with 'cannot reshape array of size 70000 into shape (0,newaxis)' -- on the
    driver's own default sampler, and again behind FAILED ANALYSIS and exit 0."""
    a, b = 8.0, 2.0
    tag = "adaptive_cartesian"
    # A LARGER --n-max than the other lanes, because at 20000 this sampler stops on the budget
    # rather than on --n-eff: over eight seeds it reported n_eff 91 to 220 against a target of
    # 250, with sigma up to 0.0430 against a 0.06 cap.  60000 is HEADROOM, not a guarantee of
    # stopping on n_eff, and an earlier version of this comment claimed the latter from one
    # seed: over the same eight seeds, ntotal runs 30000, 30000, 40000, 50000, 60000, 40000,
    # 40000, 40000, so seed 1004 still spends the whole budget.  What the cap buys is the n_eff
    # column, 250 to 319 against a target of 250, instead of 91 to 220.  The device lane in
    # section 4, and the host prior-only lane in section 1, inherit it through _LANE_KW.
    lnL, sigma, neff = _run_ile(event, tag, _AC, a_coeff=a, b_coeff=b, n_max=_AC_N_MAX)
    _assert_lnZ(tag, lnL, sigma, neff, _exact(a, b))


# ---------------------------------------------------------------------------------------
# 4. the same answers, on a real GPU
#
# WHY THIS IS NOT THE SAME TEST TWICE.  Every lane above pins CUDA_VISIBLE_DEVICES="", so the
# whole file used to say nothing about the device path -- and RIFT picks that path up from
# whether cupy IMPORTS, not from --gpu, so on a GPU node production takes it by default.  These
# lanes run with a device visible and assert the child reached it.
#
# WHAT THEY COVER, precisely: the extrinsic sampler on device, the --zero-likelihood stand-in
# building its base array with xpy_default=cupy, and the supplementary factor evaluating on
# device.  They do NOT cover the GPU SIGNAL likelihood: --zero-likelihood replaces
# likelihood_function outright, so the NoLoop path never runs.  Measured, not assumed: AV with
# and without _GPU_FLAGS returns bit-identical lnZ, because those flags only choose a likelihood
# that this configuration does not evaluate.

# portfolio and adaptive_cartesian were RECORDED HERE AS BROKEN when this section was written:
# both mixed host and device arrays and died in a cupy ufunc with "TypeError: Unsupported type
# <class 'numpy.ndarray'>", invisible in production because the driver catches it, prints FAILED
# ANALYSIS and exits 0.  Fixed on OPPOSITE sides, because the two sides are not alike -- the
# samples go TO the device in mcsamplerGPU.compute_hist, the integrand comes BACK to the host in
# mcsampler.integrate, whose accumulators are deliberately RiftFloat and so cannot follow it.
# Read those two comments before moving either conversion.
#
# adaptive_cartesian_gpu is mcsamplerGPU STANDALONE, and the ILE driver's own default
# --sampler-method, so on a GPU node it is what production runs unless someone says otherwise.
# It reaches compute_hist by a DIFFERENT route from every other lane here: its own integrate /
# integrate_log adaptation blocks rather than a portfolio member's external_rvs.  Traced on
# ldas-pcdev2 slot 0 with compute_hist wrapped: caller=integrate, x=cupy.ndarray,
# weights=cupy, self.xpy=cupy.
#
# WHAT THIS LANE DOES AND DOES NOT ADD, measured as two mutations of that conversion, each run
# against the A=8 B=2 arm at seed 1000 on a device:
#
#   delete the conversion outright        lane PASSES, ln Z bit-identical (6.633716)
#   converge on the HOST instead          lane FAILS: "Unsupported type <class
#                                         'numpy.ndarray'>", no result row, exit 0
#
# So it does not re-cover what the portfolio lane covers: on this route the samples already
# arrive on device, the conversion is a no-op, and deleting it is invisible here.  The portfolio
# lane is what catches the deletion, which is the failure recorded above.  What this lane pins
# is the DIRECTION.  A future conversion that sends this histogram to the host breaks the
# standalone sampler, and nothing else in this file would have said so.
#
# expect_device IS WHAT MAKES IT A DEVICE LANE, and that was broken on purpose to check: the
# same arguments with cuda="" on ldas-grid COMPLETE and land a correct answer (ln Z 6.63129 +-
# 0.0278, n_eff 263, z -0.79), so a host run of this lane is a passing physics run.  Only the
# marker separates the two, and _run_ile raised on it.  Do not drop expect_device here on the
# grounds that the numbers look right.
_GPU_SAMPLERS = ["AV", "GMM", "portfolio", "adaptive_cartesian", "adaptive_cartesian_gpu"]

# Coefficients per device lane, and they are NOT the same pair for every sampler on purpose.
# Each mirrors the CPU lane that sampler already runs, so a device row is comparable with a host
# row in the table below, and none lands on a configuration this file records as unstable.
#
# GMM is A=0.75 B=3, not A=8 B=2.  The note at the end of the CALIBRATION section records that
# GMM at A=8 WITH the inclination term is an n_eff LOTTERY: over eight seeds, two collapsed to
# n_eff 73.5 and 14.5 with sigma 0.028 and 0.073, and 0.073 is over MAX_SIGMA.  The evidence was
# right on every seed; the error bar was not.  That is why no CPU lane runs it, and a device lane
# that ran it would be a flake with a ~1-in-4 seed.  Eight clean GPU seeds do not refute a
# lottery -- the CPU sweep that FOUND it was also eight seeds.  portfolio takes the same pair as
# its own A=0.75 B=3 host lane.  Every lane keeps a B term, so mis-routing inclination is still
# detectable; that is what the B term is for.
#
# adaptive_cartesian_gpu has no host lane to mirror, being a device sampler by name and the
# driver's GPU default.  It takes the pair its nearest neighbour runs, adaptive_cartesian's
# A=8 B=2.
_GPU_LANE_COEFFS = {"AV": (8.0, 2.0), "GMM": (0.75, 3.0),
                    "portfolio": (0.75, 3.0), "adaptive_cartesian": (8.0, 2.0),
                    "adaptive_cartesian_gpu": (8.0, 2.0)}


@pytest.mark.parametrize("sampler", _GPU_SAMPLERS)
def test_gpu_zero_likelihood_alone_gives_ln_Z_zero(event, gpu_slot, sampler):
    """The prior-only answer, on device.  ln Z = 0 exactly, for a normalized extrinsic prior."""
    tag = "gpu_zero_%s" % sampler
    lnL, sigma, neff = _run_ile(event, tag, SAMPLER_ARGS[sampler] + list(_GPU_FLAGS),
                                a_coeff=None, cuda=gpu_slot, expect_device=True,
                                **_LANE_KW.get(sampler, {}))
    _assert_lnZ(tag, lnL, sigma, neff, 0.0)


@pytest.mark.parametrize("sampler", _GPU_SAMPLERS)
def test_gpu_analytic_factor_marginal_is_exact(event, gpu_slot, sampler):
    """The factor's closed-form marginal, on device: ln Z = ln I0(A) + ln(sinh(B)/B).

    See _GPU_LANE_COEFFS for why the samplers get different coefficients."""
    a, b = _GPU_LANE_COEFFS[sampler]
    tag = "gpu_factor_%s" % sampler
    lnL, sigma, neff = _run_ile(event, tag, SAMPLER_ARGS[sampler] + list(_GPU_FLAGS),
                                a_coeff=a, b_coeff=b, cuda=gpu_slot, expect_device=True,
                                **_LANE_KW.get(sampler, {}))
    _assert_lnZ(tag, lnL, sigma, neff, _exact(a, b))


def test_the_gpu_flags_do_not_decide_the_backend(event, gpu_slot):
    """A VISIBLE DEVICE IS ALREADY A GPU RUN; --gpu does not opt in and its absence does not opt
    out.  xpy_default is bound at import from whether cupy imports, so someone who runs this
    driver on a GPU node without --gpu is on the device path anyway.

    Pinned because it is the assumption the lanes above rest on, and because it is the opposite
    of what the flag names suggest.  If this ever fails, the backend became flag-driven, which
    is a fix -- update these lanes rather than deleting the test."""
    kw = dict(a_coeff=8.0, b_coeff=2.0, cuda=gpu_slot, expect_device=True)
    bare = _run_ile(event, "gpu_backend_bare", SAMPLER_ARGS["AV"], **kw)
    flagged = _run_ile(event, "gpu_backend_flagged",
                       SAMPLER_ARGS["AV"] + list(_GPU_FLAGS), **kw)
    assert bare == flagged, (
        "--vectorized --gpu --force-xpy changed the answer (%r vs %r).  Under --zero-likelihood "
        "they select a signal likelihood that never runs, so they were expected to be inert; "
        "something else now depends on them." % (bare, flagged))


# ---------------------------------------------------------------------------------------
# CALIBRATION
#
# RE-DERIVE THIS TABLE, do not trust it:
#
#     cd test/expensive_before_merging/integrators && python make_e2e_calibration.py --seeds 8
#
# That generator drives this file's own _run_ile and build_event, so it measures the lanes this
# gate runs rather than a reimplementation of them.  ~20 minutes on one core, which is why the
# numbers live here as a comment and the measurement lives there.
#
# Where Z_TOLERANCE, MAX_SIGMA and MIN_NEFF come from.  Eight seeds (1000-1007) per lane on
# ldas-grid, IGWN CVMFS python 3.11, numpy 1.26.4, cupy absent, CUDA_VISIBLE_DEVICES="".
# Non-GMM lanes were measured at d1d7c7e84 and re-run after the rebase onto 0a5fdb3be: the
# prior-only AV lane came back BIT-IDENTICAL on all eight seeds, so those numbers carry over.
# GMM lanes were measured at 0a5fdb3be, i.e. after #359, and RE-DERIVED with the generator on
# 87780efce (after #360, #361, #362, one of which is a GMM dim-group follow-up): all four GMM
# rows came back identical, so that work does not move this fixture.  The prior-only
# adaptive_cartesian row was added on 2026-09-18, on the working tree that became this commit.
#
#   lane                                      max |z|   max sigma   min n_eff
#   prior-only,    AV                            2.03      0.0134        1349
#   prior-only,    portfolio                     1.68      0.0104        2316
#   prior-only,    GMM                           1.88      0.0134        1335
#   prior-only,    adaptive_cartesian            2.39      0.0091        3262
#   A=0.75 B=0,    AV                            1.37      0.0159         745
#   A=0.75 B=0,    portfolio                     1.14      0.0127        1287
#   A=0.75 B=0,    GMM                           1.66      0.0160         750
#   A=8    B=0,    AV                            1.73      0.0272         416
#   A=8    B=0,    portfolio                     1.31      0.0305         346
#   A=8    B=0,    GMM                           1.52      0.0349         267
#   A=0.75 B=3,    AV                            1.25      0.0234         363
#   A=0.75 B=3,    portfolio                     1.54      0.0227         392
#   A=0.75 B=3,    GMM                           1.34      0.0235         381
#   A=8    B=2,    AV (the survives-swap lanes)  1.06      0.0332         247
#   raw inclination contract (cosine sampler)    1.47      0.0353         163
#   time-marginalized portfolio                  1.65      0.0298         237
#   distance-marginalized                        2.15      0.0326         336
#   adaptive_cartesian, --n-max 60000            1.35      0.0305         250
#
# THE DEVICE LANES, measured separately because they need a GPU: eight seeds (1000-1007),
# cupy 12.0.0, with
#
#     python make_e2e_calibration.py --seeds 8 --lane "GPU " --gpu-slot 0
#
# run on the working tree that became THIS commit.  The generator lanes and --gpu-slot arrive in
# the same commit as these numbers, so there is no earlier commit at which that command exists;
# do not "correct" this to an ancestor SHA.  Splitting it into `--lane "GPU prior-only"` and
# `--lane "GPU A="` selects the same ten rows and lets two run at once, which is how these were
# taken: the table is about 40 minutes of device time.
#
# MEASURED ON ldas-pcdev12 CUDA slot 0 (A100-SXM4-80GB, cc 8.0) on 2026-09-18.  The eight rows
# that predate this commit came back IDENTICAL in all three columns to the numbers taken on
# ldas-pcdev2 CUDA slot 0 (A100-PCIE-40GB, cc 8.0) -- a different host and a different A100 --
# and two of them (GPU prior-only AV, GPU A=8 B=2 AV) were re-run on ldas-pcdev2 itself the
# same day, identical again.  So this fixture does not depend on which A100 it lands on.
#
# "slot 0" is CUDA's numbering, which is not nvidia-smi's, and the map differs per host: on
# ldas-pcdev12 all four cards are A100 and CUDA slot 0 is nvidia-smi index 0, while on
# ldas-pcdev2 `nvidia-smi` calls the A100 index 2 and an RTX 3080 index 0 and CUDA's default
# FASTEST_FIRST ordering puts the A100 at 0.  Probe at dispatch, with cupy's own
# getDeviceProperties and a kernel that actually runs, not with nvidia-smi.
#
# The GMM and portfolio rows are A=0.75 B=3 and not A=8 B=2 deliberately; see _GPU_LANE_COEFFS.
# Each device row sits on top of its host twin, which is the point: the device path is not a
# different answer.
#
#   lane                                      max |z|   max sigma   min n_eff
#   GPU prior-only, AV                           1.50      0.0134        1343
#   GPU prior-only, GMM                          0.93      0.0134        1352
#   GPU A=8    B=2, AV                           2.30      0.0325         257
#   GPU A=0.75 B=3, GMM                          1.49      0.0235         389
#   GPU prior-only, portfolio                    2.22      0.0104        2335
#   GPU A=0.75 B=3, portfolio                    2.32      0.0228         411
#   GPU prior-only, adaptive_cartesian           2.39      0.0091        3262
#   GPU A=8    B=2, adaptive_cartesian           1.35      0.0305         250
#   GPU prior-only, adaptive_cartesian_gpu       2.24      0.0091        3268
#   GPU A=8    B=2, adaptive_cartesian_gpu       2.54      0.0284         261
#
# Host twins, in the same order, from the table above: 2.03/0.0134/1349, 1.88/0.0134/1335,
# 1.06/0.0332/247, 1.34/0.0235/381, 1.68/0.0104/2316, 1.54/0.0227/392, 2.39/0.0091/3262,
# 1.35/0.0305/250, and none for the last two.  adaptive_cartesian_gpu is mcsamplerGPU
# standalone and has no host lane by design; see the SAMPLERS note at the top of this file.
# Sigma is identical to all four decimals on six of the eight twinned rows; the other two differ
# by 0.0007 (A=8 B=2 AV) and 0.0001 (A=0.75 B=3 portfolio).  The |z| values differ freely,
# because they are draws, not constants.
#
# THE TWO adaptive_cartesian ROWS PRINT AS THEIR HOST TWINS in all three columns, and the
# agreement is per-seed rather than only in the max: at A=8 B=2 the host and device arms return
# the same ntotal on all eight seeds (30000, 30000, 40000, 50000, 60000, 40000, 40000, 40000)
# and n_eff equal to within 0.2.  That is still agreement at this precision, NOT a bitwise
# identity -- do not read it as one.  An
# earlier version of this comment claimed the two arms run identical host arithmetic because
# "the only device work is xpy.zeros".  That is wrong and was disproved by measurement: for a
# sampler with return_lnL False the --zero-likelihood stand-in builds
# xpy.ones(n) * xpy.exp(supp), so cos and exp run on the DEVICE.  cupy 12.0.0 and numpy 1.26.4
# disagree bitwise on about a fifth of 200000 float64 draws (max relative 2e-15), and at seed
# 1000 the A=8 B=2 lane returns ln Z 6.646959838539267 on the host against 6.646957839585277 on
# the device -- a 2e-6 difference, four orders below the 0.0305 error bar.  The prior-only lane
# is closer and still not bitwise: at seed 1000 it returns -0.0017527695475497557 on the host
# against -0.001752769547549755 on the device, 7e-19 apart, while at seed 1005 the two ARE
# bit-identical.  So a last-digit move in either row is a rounding boundary, not a finding; a
# move you can see in the SECOND digit is worth reading.
#
# WHAT THE HOST TWIN OF THE PRIOR-ONLY adaptive_cartesian ROW BUYS, AND WHAT IT DOES NOT.  It
# answers whether 2.39 -- the worst |z| in the table until this commit -- is something the
# device path does.  It is not: the host arm returns the same eight draws.  It is not a second
# INDEPENDENT sample of that lane's |z|, because it is the same eight seeds producing the same
# numbers.  Read it as a cross-check on the path, not as more statistics.
#
# The four AV/GMM rows above were RE-DERIVED in the sweep that added the portfolio and
# adaptive_cartesian rows, i.e. after the two host/device fixes, and came back identical -- so
# those fixes do not move the lanes that already worked.
#
# Z_TOLERANCE = 5     is 2.0x the worst |z| seen, which is now 2.54 on the GPU A=8 B=2
#                     adaptive_cartesian_gpu lane; before this commit it was 2.39 (GPU
#                     prior-only adaptive_cartesian) and before that 2.30 (GPU A=8 B=2 AV).
#                     The worst host lane is 2.39 (prior-only adaptive_cartesian), then 2.15
#                     (distance-marginalized, re-measured at eight seeds after the stand-in
#                     started passing xpy= to factors that accept it: max |z| 2.15, max sigma
#                     0.0326, least n_eff 336, unchanged, seed 1000 bit-identical).
#
#                     THAT 2.54 IS A MAXIMUM OVER MANY DRAWS, not one lane's typical z.  The
#                     table is 28 lanes at 8 seeds, so 224 draws feed it, and the largest of
#                     224 |N(0,1)| has mean 3.00 and median 2.96 (20000 trials).  A worst |z|
#                     in the twos is what this table should look like, and it will creep up as
#                     lanes are added.  Judge the margin by how far the recorded DEFECTS sit
#                     above it -- the smallest is 6.3 sigma, below -- not by the ratio to the
#                     worst draw.
# MAX_SIGMA   = 0.06  is 1.7x the worst sigma seen (0.0353, host).  The worst device sigma is
#                     0.0325.  5 * MAX_SIGMA is a 0.30-nat band.
# MIN_NEFF    = 30    is 5.4x below the worst n_eff seen (163, host; 250 on device).  See its
#                     comment for why it is this loose.
#
# WHAT THE GATE HAS TO SEPARATE A CORRECT RUN FROM.  Each row was run on this fixture, not
# argued.  The smallest is 6.3 sigma, against a tolerance of 5 and a worst observed draw of 2.15:
#
#   defect                                                        lane              z
#   portfolio member p_s, rift_O4d @ 9e55f12b7 (without #356)     prior-only    -43.6
#   portfolio member p_s, same tree                               A=0.75 B=3    -25.9
#   ILE --sampler-method GMM before #359                          A=8 B=2      -684
#   phi_orb and inclination swapped in _SUPPLEMENT_ARG_ORDER      A=8 B=2       -19.5
#   inclination and psi swapped                                   A=8 B=2        +6.3
#   inclination and psi swapped, B term off                       A=8 B=0        -0.1
#   phi_orb and psi swapped                                       all            ~0
#
# The last two rows are why the B term and test_zero_likelihood_standin.py both exist.  With
# the B term off, mis-routing inclination is invisible.  And phi_orb, psi and right_ascension
# are independent and identically distributed, so NO closed-form lane can see a permutation of
# those three.  Run as a mutation, EVERY z-test in this file passes with phi_orb and psi
# swapped in the stand-in.  The module does go red, but on the adaptive_cartesian lane's sigma
# budget, because the mutated integrand happens to be harder for that one sampler -- an
# efficiency artifact, not a detection, and not something to rely on: a swap between phi_orb
# and right_ascension need not perturb any sampler at all.  The case is caught by reading the
# wiring, which is the companion file's job; it fails there immediately and for the right
# reason.
#
# ONE THING THE GMM LANES DO NOT COVER, measured while calibrating them.  GMM's EVIDENCE is
# correct on all three cases above, but at A=8 with the inclination term ON (B=2, which no lane
# here runs) its n_eff is a bimodal lottery: over eight seeds, six landed at n_eff 266-324 with
# sigma ~0.025 and two collapsed to 73.5 and 14.5 with sigma 0.028 and 0.073, while ln Z stayed
# right on every one (max |z| 1.49).  A=8 with B=0 does NOT collapse, so the inclination term is
# what drives it, not the phi_orb peak.  This is the shape recorded in
# ~/rift-integrator-lore/coordinates-and-degeneracies.md, section "What the n_eff lottery
# actually needs", whose diagnosis is extrinsic mode collapse rather than sample starvation --
# raising --n-max re-rolls the dice instead of fixing it.  Narrower than the production case it
# resembles: this fixture is H1-only, so neither the sky ring nor the two-detector
# phase-polarization degeneracy exists here.  Contrast adaptive_cartesian above, which DID get
# a bigger --n-max: there the answer was right on every seed and only the error bar was short,
# which is sample starvation.  Do not carry one precedent to the other case.
