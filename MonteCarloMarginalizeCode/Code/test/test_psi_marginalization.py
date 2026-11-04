"""
Tests for --psi-marginalization: analytic polarization-angle marginalization,
made reachable on the legacy scalar (non-vectorized, non-GPU, non-time-marginalized)
likelihood path via factored_likelihood.NetworkLogLikelihoodPolarizationMarginalized.

Before this change the function was dead code: its only tests are LEGACY (cannot
import: test_like_and_samp_margPsi.py, test_like_and_samp_noisydata_margPsi.py) and
no production driver called it.  It also had a live bug -- crossTermsV was indexed as
crossTermsV[(pair1,pair2)] instead of crossTermsV[det][(pair1,pair2)], which KeyErrors
immediately against the real precompute structure (crossTermsV is a dict keyed by
detector, then by mode pair) -- fixed alongside this PR.

1. test_matches_bruteforce_quadrature: the analytic marginal against a brute-force
   trapezoid over psi of exp(lnL(psi)) from the un-marginalized scalar likelihood, at
   two amplitudes, with a convergence sequence (the memory-known invariant: lnL(psi)
   is exactly two harmonics, so a coarse trapezoid on the LOG-likelihood would be
   exact, but exp(lnL) is not band-limited, so exp(lnL(psi)) needs many nodes).
2. test_matches_reference_above_overflow: the same comparison at max_psi lnL ~ 1100,
   ABOVE ln(DBL_MAX) = 709, where the un-subtracted integrand returned nan.  The
   trapezoid of exp(lnL) cannot be the reference there (it overflows too), so this uses
   an independent max-subtracted midpoint rule on the two-harmonic DFT fit of lnL(psi).
3. test_psi_prior_mass_matches_the_sampler: the analytic marginal is normalized, and so is
   the driver's sampled psi prior now that it is derived from its sampling range (it was
   1/pi over (0, 2 pi), mass 2, before that fix); the driver still derives the compensating
   ln(mass) from its own prior objects, as a guard, rather than assuming either value.
4. test_flag_reaches_help: the option is registered.
5. test_refuses_incompatible_combinations: every documented refusal actually fires,
   with no data files needed (the refusal runs before any data is read).
6. test_driver_runs_end_to_end: the driver actually completes a tiny run on
   synthetic zero-signal data, writes a result row, keeps psi out of the sampled
   dimensions, and writes the NaN (not fiducial) polarization column.
7. test_marginalized_and_sampled_lnZ_agree: the whole point of (3) end to end -- the
   flag must not move lnZ, so a marginalized and a sampled run of the same fixture
   agree within Monte Carlo error rather than differing by ln 2.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import lal
import lal.series
import lalsimulation as lalsim
from igwn_ligolw import utils as ligolw_utils
import RIFT.lalsimutils as lalsimutils
import RIFT.likelihood.factored_likelihood as FL

MSUN = lal.MSUN_SI
PC = lal.PC_SI

BIN = Path(__file__).resolve().parents[1] / "bin" / "integrate_likelihood_extrinsic_batchmode"


###
### 1. Correctness: analytic marginal vs brute-force quadrature
###

def _make_injection(detectors, dist_mpc):
    fiducial_epoch = 1126259462.0
    P = lalsimutils.ChooseWaveformParams()
    P.m1 = 35.0 * MSUN
    P.m2 = 30.0 * MSUN
    P.s1z = 0.1
    P.s2z = -0.2
    P.fmin = 30.0
    P.fref = 30.0
    P.deltaT = 1.0 / 4096
    P.deltaF = 1.0 / 4
    P.dist = dist_mpc * 1e6 * PC
    P.fmax = 0.0
    P.approx = lalsim.IMRPhenomD
    P.radec = True
    P.tref = fiducial_epoch
    P.phi = 1.2       # RA
    P.theta = -0.4    # DEC
    P.psi = 0.7
    P.incl = 0.9
    P.phiref = 2.1
    data_dict, psd_dict = {}, {}
    for det in detectors:
        Pdet = P.copy()
        Pdet.detector = det
        data_dict[det] = lalsimutils.non_herm_hoff(Pdet)
        psd_dict[det] = lalsim.SimNoisePSDaLIGOZeroDetHighPower
    return P, data_dict, psd_dict, fiducial_epoch


def _bruteforce_psi_marginal(rholms_intp, cross_terms, cross_terms_V, P, Lmax, nnodes):
    """(1/pi) int_0^pi exp(lnL(psi)) dpsi, by trapezoid, from the UN-marginalized
    scalar likelihood -- an independent route from NetworkLogLikelihoodPolarizationMarginalized:
    a full network sum of complex antenna-pattern products vs. per-psi calls into the
    ordinary FactoredLogLikelihood used by the scalar sampled-psi path.
    """
    psis = np.linspace(0.0, np.pi, nnodes)
    extr = lalsimutils.ChooseWaveformParams()
    extr.phi, extr.theta = P.phi, P.theta
    extr.incl, extr.phiref, extr.dist, extr.tref = P.incl, P.phiref, P.dist, P.tref
    lnLs = np.empty(nnodes)
    for i, psi in enumerate(psis):
        extr.psi = psi
        lnLs[i] = FL.FactoredLogLikelihood(
            extr, None, rholms_intp, cross_terms, cross_terms_V, Lmax, interpolate=True)
    integral = np.trapz(np.exp(lnLs), psis) / np.pi
    return np.log(integral)


@pytest.mark.parametrize("dist_mpc,label", [(1200.0, "low_amplitude"), (600.0, "moderate_amplitude")])
def test_matches_bruteforce_quadrature(dist_mpc, label):
    detectors = ["H1", "L1"]
    P, data_dict, psd_dict, fiducial_epoch = _make_injection(detectors, dist_mpc)
    Lmax, fMax, t_window = 2, 1000.0, 0.15

    rholms_intp, cross_terms, cross_terms_V, rholms, guess_snr, _rest = FL.PrecomputeLikelihoodTerms(
        fiducial_epoch, t_window, P, data_dict, psd_dict, Lmax, fMax,
        analyticPSD_Q=True, verbose=False, quiet=True, ignore_threshold=None)
    assert guess_snr > 0.5, "fixture should carry a real, detectable signal (%s)" % label

    lnL_analytic = FL.NetworkLogLikelihoodPolarizationMarginalized(
        fiducial_epoch, rholms_intp, cross_terms, cross_terms_V,
        P.tref, P.phi, P.theta, P.incl, P.phiref, P.psi, P.dist, Lmax, detectors)
    assert np.isfinite(lnL_analytic)

    # Convergence sequence.  A coarse (16-node) trapezoid must NOT already be at the
    # 1e-3 nat tolerance -- exp(lnL(psi)) is not the two-harmonic function itself, so a
    # rule that is exact for lnL(psi) is not exact here.  A dense (1024-node) trapezoid
    # must be.
    bf_coarse = _bruteforce_psi_marginal(rholms_intp, cross_terms, cross_terms_V, P, Lmax, 16)
    bf_dense = _bruteforce_psi_marginal(rholms_intp, cross_terms, cross_terms_V, P, Lmax, 1024)

    assert abs(lnL_analytic - bf_dense) < 1e-3, (
        "%s: analytic %.10f vs dense brute force %.10f (diff %.3e)"
        % (label, lnL_analytic, bf_dense, lnL_analytic - bf_dense))
    assert abs(lnL_analytic - bf_coarse) > 1e-3, (
        "%s: a 16-node trapezoid over exp(lnL(psi)) should NOT already be converged "
        "to 1e-3 nat -- if it is, this test's coarse/dense contrast proves nothing"
        % label)


###
### 1a. Correctness ABOVE the exp() overflow cliff
###

def _dft_reference_psi_marginal(rholms_intp, cross_terms, cross_terms_V, P, Lmax, nnodes=1 << 20):
    """An OVERFLOW-IMMUNE reference for (1/pi) int_0^pi exp(lnL(psi)) dpsi.

    lnL(psi) is exactly two harmonics (the memory-known invariant, and the review
    measured every harmonic above cos/sin(4 psi) at <= 2.4e-13 across seven
    configurations), so 16 calls into the ordinary FactoredLogLikelihood determine it
    exactly by DFT.  The marginal is then a midpoint rule -- exact to machine precision
    for a periodic analytic integrand -- taken on exp(lnL - max lnL) and shifted back.
    That subtraction is why this route stays finite where a trapezoid of exp(lnL) does
    not: it is an INDEPENDENT construction from
    NetworkLogLikelihoodPolarizationMarginalized, which never calls FactoredLogLikelihood.
    """
    nfit = 16
    psis = np.arange(nfit)*np.pi/nfit
    extr = lalsimutils.ChooseWaveformParams()
    extr.phi, extr.theta = P.phi, P.theta
    extr.incl, extr.phiref, extr.dist, extr.tref = P.incl, P.phiref, P.dist, P.tref
    y = np.empty(nfit)
    for i, psi in enumerate(psis):
        extr.psi = psi
        y[i] = FL.FactoredLogLikelihood(
            extr, None, rholms_intp, cross_terms, cross_terms_V, Lmax, interpolate=True)
    Y = np.fft.rfft(y)/nfit
    c, A, B = Y[0].real, 2*Y[1], 2*Y[2]
    residual_harmonics = float(np.max(np.abs(Y[3:])*2)) if len(Y) > 3 else 0.0
    assert residual_harmonics < 1e-9*max(1.0, np.max(np.abs(y))), (
        "lnL(psi) is not two harmonics (max |k>=3| = %.3e); this reference is invalid"
        % residual_harmonics)
    p = (np.arange(nnodes) + 0.5)*np.pi/nnodes
    v = c + (A*np.exp(-2j*p)).real + (B*np.exp(-4j*p)).real
    m = float(v.max())
    return m + np.log(np.mean(np.exp(v - m))), m


def test_matches_reference_above_overflow():
    """F1: exp() of the raw exponent overflows to inf, and log(inf) is nan, once
    max_psi lnL passes ln(DBL_MAX) = 709 -- a network SNR near 38, well inside the
    range this option is FOR.  Measured on this branch before the max subtraction:
    max lnL 663.80 returned 659.508, 725.42 and 1095.38 returned nan.  This fixture
    sits at max lnL ~ 1100, past the cliff, and is checked to 1e-6 nat.
    """
    detectors = ["H1", "L1"]
    P, data_dict, psd_dict, fiducial_epoch = _make_injection(detectors, 300.0)
    Lmax, fMax, t_window = 2, 1000.0, 0.15

    rholms_intp, cross_terms, cross_terms_V, rholms, guess_snr, _rest = FL.PrecomputeLikelihoodTerms(
        fiducial_epoch, t_window, P, data_dict, psd_dict, Lmax, fMax,
        analyticPSD_Q=True, verbose=False, quiet=True, ignore_threshold=None)

    reference, lnL_max = _dft_reference_psi_marginal(
        rholms_intp, cross_terms, cross_terms_V, P, Lmax)
    assert lnL_max > 709.0, (
        "fixture must sit ABOVE the overflow cliff to test anything: max lnL = %.2f" % lnL_max)

    lnL_analytic = FL.NetworkLogLikelihoodPolarizationMarginalized(
        fiducial_epoch, rholms_intp, cross_terms, cross_terms_V,
        P.tref, P.phi, P.theta, P.incl, P.phiref, P.psi, P.dist, Lmax, detectors)
    assert np.isfinite(lnL_analytic), (
        "analytic marginal returned %r at max lnL %.2f -- the max subtraction is gone"
        % (lnL_analytic, lnL_max))
    assert abs(lnL_analytic - reference) < 1e-6, (
        "max lnL %.2f: analytic %.10f vs independent reference %.10f (diff %.3e)"
        % (lnL_max, lnL_analytic, reference, lnL_analytic - reference))

    # the return value must not depend on the reference psi handed in, at this amplitude too
    vals = [FL.NetworkLogLikelihoodPolarizationMarginalized(
                fiducial_epoch, rholms_intp, cross_terms, cross_terms_V,
                P.tref, P.phi, P.theta, P.incl, P.phiref, psi_ref, P.dist, Lmax, detectors)
            for psi_ref in (0.0, 0.7, 2.0, 5.5)]
    assert np.ptp(vals) < 1e-6, vals


###
### 1b. Prior-mass normalization: the flag must not move lnZ
###

def test_psi_prior_mass_matches_the_sampler():
    """F2.  NetworkLogLikelihoodPolarizationMarginalized returns the NORMALIZED marginal,
    (1/pi) int_0^pi, mass 1.  The driver's sampled psi prior is derived from its sampling
    range, psi_prior_pdf = ret_uniform_samp_vector_alt(*psi_prior_range) with the range
    (0, 2 pi), so it is also mass 1 and the two paths agree with no offset.  Before the
    prior was derived from its range it was the fixed constant uniform_samp_psi = 1/pi
    over (0, 2 pi), mass 2, and the flag reported lnZ - 0.693 nat (measured 0.7027 and
    0.6602 +/- 0.036 on two fixtures).

    This pins BOTH halves: the sampled path's prior really integrates to 1 over the range
    the driver samples, and the driver derives the compensating mass from its own objects
    instead of assuming it (so a future change to either input moves the correction).
    """
    import RIFT.integrators.mcsampler as mcsampler

    # the sampled-path measure, built the way the driver builds it
    lo, hi = 0.0, 2*np.pi                       # psi_prior_range on the conventional path
    prior = mcsampler.ret_uniform_samp_vector_alt(lo, hi)
    density = float(np.atleast_1d(prior(np.atleast_1d(0.5*(lo+hi))))[0])
    mass = density*(hi - lo)
    assert density == pytest.approx(1.0/(2*np.pi))
    assert mass == pytest.approx(1.0), (
        "the conventional driver's psi prior no longer integrates to 1 (got %r); the "
        "compensation in integrate_likelihood_extrinsic_batchmode must move with it" % mass)
    # and the constant it replaced is NOT normalized over that range: the defect being guarded
    old = float(np.atleast_1d(mcsampler.uniform_samp_psi(np.atleast_1d(0.5*(lo+hi))))[0])
    assert old*(hi - lo) == pytest.approx(2.0)

    # The driver must DERIVE the mass from its own prior object and range.  At the default
    # operating point the derived value is 0 to the last bit, so no runtime observation can
    # separate a derivation from an omitted term.  What this pins is the expression, so that
    # changing either input (range or density) moves the correction.
    text = BIN.read_text()
    assert "psi_prior_pdf = mcsampler.ret_uniform_samp_vector_alt(psi_prior_range[0], psi_prior_range[1])" in text, \
        "the psi prior is no longer derived from the psi sampling range"
    assert "prior_pdf = psi_prior_pdf," in text, \
        "the sampled psi path no longer uses the range-derived prior"
    assert "psi_prior_pdf(numpy.atleast_1d(" in text, \
        "the psi prior density is no longer read from the driver's own prior object"
    assert "_psi_prior_mass = _psi_prior_density*(psi_prior_range[1]-psi_prior_range[0])" in text, \
        "the psi prior MASS is no longer derived from the driver's own range and density"
    assert "psi_marginalization_ln_prior_mass = float(numpy.log(_psi_prior_mass))" in text, \
        "the correction is not ln(the derived mass)"
    assert "lnL += _psi_marg_ln_prior_mass" in text, \
        "the derived prior mass is never applied to the marginalized likelihood"


###
### 2. The flag is registered
###

def test_flag_reaches_help():
    env = dict(os.environ)
    out = subprocess.run([sys.executable, str(BIN), "--help"], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=60)
    assert "--psi-marginalization" in out.stdout


###
### 3. Refusals fire (no data files needed -- the refusal runs before any data load)
###

# A minimal distance-marginalization lookup table, just complete enough (a
# "phase_marginalization" key) for the driver to get past its OWN np.load() and
# earliest phase-marginalization read -- both of which run before our refusal check
# -- so the --distance-marginalization case below actually reaches this option's
# refusal instead of failing on an unrelated missing-file error first.
import tempfile as _tempfile
_LOOKUP_TABLE_PATH = os.path.join(
    _tempfile.mkdtemp(prefix="psi_marg_dummy_lookup_"), "lookup.npz")
np.savez(_LOOKUP_TABLE_PATH, phase_marginalization=np.array(False))

_REFUSAL_CASES = [
    (["--time-marginalization"], "time-marginalization"),
    (["--vectorized"], "vectorized"),
    (["--distance-marginalization", "--distance-marginalization-lookup-table", _LOOKUP_TABLE_PATH],
     "distance-marginalization"),
    (["--rotation-slow", "--vectorized"], "rotation-slow"),
    (["--freqresponse", "--vectorized"], "freqresponse"),
    (["--calibration-envelope-directory", "/nonexistent"], "calibration"),
    (["--interpolate-time", "nearest"], "interpolate-time"),
    (["--sampler-method", "GMM"], "sampler-method"),
    (["--internal-rotate-phase"], "internal-rotate-phase"),
    (["--limit-psi", "0,1"], "limit-psi"),
    (["--zero-likelihood"], "zero-likelihood"),
]


@pytest.mark.parametrize("extra_args,label", _REFUSAL_CASES, ids=[c[1] for c in _REFUSAL_CASES])
def test_refuses_incompatible_combinations(extra_args, label):
    env = dict(os.environ)
    cmd = [sys.executable, str(BIN), "--event-time", "1000000000.0",
           "--psi-marginalization"] + extra_args
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=60)
    assert proc.returncode != 0, (
        "--psi-marginalization + %s should be REFUSED, not silently accepted" % label)
    assert "--psi-marginalization was requested, but this configuration cannot honour it" \
        in proc.stdout, proc.stdout[-2000:]


def test_gpu_prereq_present_in_source():
    """--gpu cannot be exercised as a subprocess refusal on a host without cupy: the
    driver's own (unrelated, pre-existing) '--gpu (not available)' downgrade clears
    opts.gpu back to False before this option's check ever runs, so the combination
    silently stops being a --gpu run at all rather than exercising the refusal.  Same
    story for a portfolio member carrying a GMM group, which is config-file-dependent
    and not reachable from a bare CLI probe.  Check the source directly instead, the
    same way test_ile_scalar_edge_cases.py checks call-site wiring it cannot run live.
    """
    text = BIN.read_text()
    start = text.index("_psi_marg_prereqs = (")
    end = text.index("_psi_marg_missing = [", start)
    block = text[start:end]
    assert "not bool(opts.gpu)" in block
    assert "'GMM', 'portfolio'" in block


###
### 4. End-to-end: the driver actually runs and writes a result row
###

@pytest.fixture(scope="module")
def synthetic_fixture(tmp_path_factory):
    """A tiny zero-signal H1 frame + PSD xml + LAL cache, so the driver can be run as
    a real subprocess without a network fetch or a real event.  Zero data (rather than
    an injected waveform) keeps the fixture simple and avoids sky/antenna-response
    bookkeeping that is irrelevant to what this test checks: that the WIRING for
    --psi-marginalization completes a run and writes output, not that the recovered
    parameters are accurate.
    """
    outdir = tmp_path_factory.mktemp("psi_marg_fixture")
    event_time = 1000000000.0
    srate = 2048.0
    deltaT = 1.0 / srate
    seg_start = event_time - 6.0
    seg_end = event_time + 2.0
    duration = seg_end - seg_start
    npts = int(round(duration / deltaT))

    channel = "H1:FAKE-STRAIN"
    ht = lal.CreateREAL8TimeSeries(
        "Zero strain", lal.LIGOTimeGPS(seg_start), 0.0, deltaT,
        lalsimutils.lsu_DimensionlessUnit, npts)
    ht.data.data = np.zeros(npts)

    fname = outdir / ("H-fake_strain-%d-%d.gwf" % (int(seg_start), int(duration)))
    lalsimutils.hoft_to_frame_data(str(fname), channel, ht)

    cache_path = outdir / "test.cache"
    os.system("echo %s | lal_path2cache > %s" % (fname, cache_path))

    psd_series = lal.CreateREAL8FrequencySeries(
        "psd", lal.LIGOTimeGPS(0), 0, 1.0 / duration, lal.SecondUnit, npts // 2 + 1)
    farr = psd_series.f0 + np.arange(psd_series.data.length) * psd_series.deltaF
    psd_vals = np.where(farr > 1.0, [lalsim.SimNoisePSDaLIGOZeroDetHighPower(f) for f in farr], 1.0)
    psd_series.data.data = psd_vals

    xmldoc = lal.series.make_psd_xmldoc({"H1": psd_series})
    psd_path = outdir / "H1_psd.xml.gz"
    ligolw_utils.write_filename(xmldoc, str(psd_path))

    return dict(outdir=outdir, cache=cache_path, psd=psd_path,
                event_time=event_time, seg_start=seg_start, seg_end=seg_end)


def _driver_env():
    """CUDA_VISIBLE_DEVICES="" is REQUIRED, and is not this option's problem (review F5):
    on any host where cupy sees a GPU, the AV scalar path raises
    "ValueError: Unsupported dtype float128" at mcsamplerAdaptiveVolume.py's
    identity_convert_togpu(lnL), because the scalar likelihood builds lnL as RiftFloat.
    Reproduced on the base tree with and without this flag.  Without this the driver
    tests here are green only on GPU-less runners.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def _driver_cmd(synthetic_fixture, *extra):
    return [
        sys.executable, str(BIN),
        "--cache-file", str(synthetic_fixture["cache"]),
        "--channel-name", "H1=FAKE-STRAIN",
        "--psd-file", "H1=%s" % synthetic_fixture["psd"],
        "--event-time", str(synthetic_fixture["event_time"]),
        "--data-start-time", str(synthetic_fixture["seg_start"]),
        "--data-end-time", str(synthetic_fixture["seg_end"]),
        "--mass1", "35.0", "--mass2", "30.0", "--approximant", "TaylorT4",
        "--l-max", "2", "--reference-freq", "40", "--fmin-template", "40",
        "--srate", "2048", "--inv-spec-trunc-time", "0",
        "--sampler-method", "AV",
    ] + list(extra)


def test_driver_runs_end_to_end(synthetic_fixture):
    outdir = synthetic_fixture["outdir"]
    cmd = _driver_cmd(
        synthetic_fixture,
        "--psi-marginalization", "--n-max", "20000", "--n-eff", "50",
        "--output-file", "out.xml.gz", "--save-samples", "--fairdraw-extrinsic-output",
    )
    proc = subprocess.run(cmd, cwd=str(outdir), env=_driver_env(),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-4000:]
    assert "ANALYTICALLY MARGINALIZED (--psi-marginalization)" in proc.stdout

    # F4/M5: psi must be gone from the integration dimensions, and the driver must SAY so
    # at the one place unpinned_params is built.  Reading the printed list rather than
    # trusting the absence of a crash: a psi left in would be integrated twice.
    dim_lines = [ln for ln in proc.stdout.splitlines()
                 if "psi is NOT a sampled dimension" in ln]
    assert len(dim_lines) == 1, proc.stdout[-4000:]
    assert "'psi'" not in dim_lines[0], dim_lines[0]
    for expected in ("right_ascension", "declination", "phi_orb", "inclination", "distance"):
        assert expected in dim_lines[0], dim_lines[0]

    # F2: the compensating prior mass is announced, and it is the sampler's own mass
    mass_lines = [ln for ln in proc.stdout.splitlines()
                  if "sampled-path psi prior mass is" in ln]
    assert len(mass_lines) == 1, proc.stdout[-4000:]
    # (mass 1: the sampled psi prior is derived from its (0, 2 pi) range; it was 2, +0.693147, before)
    assert "1.000000" in mass_lines[0], mass_lines[0]
    assert "+0.000000" in mass_lines[0], mass_lines[0]

    result_dat = outdir / "out.xml.gz_0_.dat"
    assert result_dat.exists(), "driver did not write a result row\n" + proc.stdout[-4000:]
    row = result_dat.read_text().split()
    # event_id, m1, m2, 6 spin components, lnL, sigma_lnL, ntotal, neff (13 columns);
    # verified against this run's own out.xml.gz_0_integrator_status.json (column 9 ==
    # "lnL" there) rather than assumed from another test's layout.
    assert len(row) == 13, row
    assert float(row[1]) == pytest.approx(35.0)
    assert float(row[2]) == pytest.approx(30.0)
    lnL_row = float(row[9])
    assert np.isfinite(lnL_row)

    status_path = outdir / "out.xml.gz_0_integrator_status.json"
    if status_path.exists():
        import json
        status = json.loads(status_path.read_text())
        assert status["lnL"] == pytest.approx(lnL_row)
        assert np.isfinite(status["lnL"])

    # F3/F6/M6/M7: --save-samples was requested, so the samples XML must EXIST (dropping the
    # placeholder used to KeyError past it, leaving the .dat this test already reads intact),
    # and its polarization column must be NaN.  It is NOT a psi sample -- psi was integrated
    # out -- and bin/convert_output_format_ile2inference copies this column straight into the
    # PE 'psi' column under a header that does not mark it, so a fiducial 0.0 would reach a
    # consumer as a delta function that reads like a polarization measurement.
    samples_xml = outdir / "out.xml.gz_0_.xml.gz"
    assert samples_xml.exists(), (
        "--save-samples was requested but no samples XML was written\n" + proc.stdout[-4000:])
    from igwn_ligolw import lsctables, ligolw
    from igwn_ligolw import utils as _ligolw_utils

    class _ContentHandler(ligolw.LIGOLWContentHandler):
        pass
    lsctables.use_in(_ContentHandler)
    xmldoc = _ligolw_utils.load_filename(str(samples_xml), contenthandler=_ContentHandler)
    tbl = lsctables.SimInspiralTable.get_table(xmldoc)
    psi_col = np.array([r.polarization for r in tbl], dtype=float)
    assert len(psi_col) > 0, "samples XML has no rows"
    assert np.all(np.isnan(psi_col)), (
        "exported polarization column must be NaN under --psi-marginalization, got %r"
        % np.unique(psi_col)[:5])
    # a genuinely sampled column, as a control that the file is not simply all NaN
    ra_col = np.array([r.longitude for r in tbl], dtype=float)
    assert np.all(np.isfinite(ra_col)) and len(np.unique(ra_col)) > 1, ra_col
    assert "exported 'psi'/'polarization' column is NaN" in proc.stdout


###
### 5. Evidence neutrality: the flag must not move lnZ (review F2)
###

def _run_driver_lnL(synthetic_fixture, tmpdir, marginalize):
    extra = ["--psi-marginalization"] if marginalize else []
    cmd = _driver_cmd(synthetic_fixture, "--n-max", "40000", "--n-eff", "200",
                      "--output-file", "out.xml.gz", *extra)
    proc = subprocess.run(cmd, cwd=str(tmpdir), env=_driver_env(),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:]
    row = (tmpdir / "out.xml.gz_0_.dat").read_text().split()
    return float(row[9]), float(row[10])          # lnL, sigma_lnL


def test_marginalized_and_sampled_lnZ_agree(synthetic_fixture, tmp_path):
    """F2, end to end.  --psi-marginalization must be evidence-NEUTRAL against the
    sampled-psi path it replaces, or its rows cannot share an all.net with ordinary rows.
    Before the prior-mass fix the marginalized run sat ln 2 = 0.693 nat low: the review
    measured 0.7027 (zero data, 3 seeds, n_eff > 4100) and 0.6602 +/- 0.036 (injected).

    Two replicates per mode, judged on the SPREAD rather than on one run: at this budget
    the per-run scatter is ~0.02 nat, and six replicates during development gave a
    difference of means of 0.005 nat.  The tolerance is set well inside ln 2 so the test
    cannot pass a reverted fix.
    """
    lnLs = {}
    for mode in ("marg", "samp"):
        vals, sigmas = [], []
        for seed in (1, 2):
            d = tmp_path / ("%s_%d" % (mode, seed))
            d.mkdir()
            v, sg = _run_driver_lnL(synthetic_fixture, d, mode == "marg")
            vals.append(v); sigmas.append(sg)
        lnLs[mode] = (np.array(vals), np.array(sigmas))

    diff = float(np.mean(lnLs["marg"][0]) - np.mean(lnLs["samp"][0]))
    se = float(np.sqrt(np.sum(lnLs["marg"][1]**2 + lnLs["samp"][1]**2))/2.0)
    tol = max(4.0*se, 0.15)
    assert tol < 0.5*np.log(2.0), (
        "tolerance %.4f is too loose to distinguish a missing ln 2" % tol)
    assert abs(diff) < tol, (
        "psi-marginalized lnZ %s vs sampled %s: difference %.4f nat exceeds %.4f "
        "(ln 2 = %.4f -- a difference that size means the psi prior mass is not matched)"
        % (lnLs["marg"][0], lnLs["samp"][0], diff, tol, np.log(2.0)))
