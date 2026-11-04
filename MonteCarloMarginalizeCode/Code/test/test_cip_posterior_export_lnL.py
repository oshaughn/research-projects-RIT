"""The lnL field the CIP posterior export reads, across every sampler convention.

util_ConstructIntrinsicPosterior_GenericCoordinates.py resolves the sampler's lnL field
ONCE, into dat_logL: 'log_integrand' if the integrator ran in log mode, else log of the
linear 'integrand'.  The posterior export then re-read the raw samples["integrand"]
instead, so --sampler-method adaptive_cartesian_gpu --internal-use-lnL died with
KeyError: 'integrand' AFTER a converged integral and never wrote the posterior samples.
mcsamplerGPU.integrate_log leaves only 'log_integrand'; AV, NFlow and portfolio alias
'integrand' to it as well, which is why one flag separated a working arm from a dead one.

Three things are asserted, because exit 0 alone does not distinguish a correct fix from a
wrong one:

* every sampler arm runs to a written samples file (the crash);
* the exported lnL equals the analytic likelihood evaluated at the masses in the SAME
  exported row.  The synthetic input is exactly quadratic in chirp mass and the arms fit
  with --fit-method quadratic, so the fit reproduces it to float32 (measured residual
  1e-5).  That pins the VALUE, which catches exporting exp(lnL), log(lnL) or a constant,
  and pins the ALIGNMENT, which catches indexing the unmasked raw field: permuting
  dat_logL before the export loop moves this residual to 5.0;
* every integrator leaves a field the driver's rule can resolve, so a new sampler that
  leaves neither key fails here rather than in a production export.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
DRIVER = os.path.join(CODE, "bin", "util_ConstructIntrinsicPosterior_GenericCoordinates.py")

# The synthetic likelihood, in closed form.  lnL(mc) = LNL_PEAK - ((mc-MC0)/MC_W)^2 / 2.
LNL_PEAK, MC0, MC_W = 20.0, 28.0, 1.5

# (id, extra driver args).  Each arm pins one leg of the convention matrix:
#   log-mode GPU  -- 'log_integrand' only; THE arm that used to crash
#   linear GPU    -- same sampler, other entry point, 'integrand' only
#   AV            -- log mode AND the 'integrand' alias
#   GMM           -- log mode via mcsamplerEnsemble
#   adaptive_cartesian -- the linear reference (mcsampler)
ARMS = [
    ("acgpu_lnL", ["--sampler-method", "adaptive_cartesian_gpu", "--internal-use-lnL"]),
    ("acgpu_linear", ["--sampler-method", "adaptive_cartesian_gpu"]),
    ("AV", ["--sampler-method", "AV"]),
    ("GMM", ["--sampler-method", "GMM", "--internal-use-lnL"]),
    ("adaptive_cartesian", ["--sampler-method", "adaptive_cartesian"]),
]


def _lnL_of_mc(mc):
    return LNL_PEAK - 0.5 * ((np.asarray(mc, dtype=float) - MC0) / MC_W) ** 2


def _write_ile(dirname, n=300):
    """A tiny ILE .dat in the standard 13-column layout."""
    rng = np.random.default_rng(7)
    m1 = rng.uniform(25.0, 45.0, n)
    m2 = np.minimum(rng.uniform(15.0, 30.0, n), m1)
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    z = np.zeros(n)
    path = os.path.join(dirname, "ile.dat")
    np.savetxt(path, np.column_stack([np.arange(n), m1, m2, z, z, z, z, z, z,
                                      _lnL_of_mc(mc), np.full(n, 0.01),
                                      np.full(n, 1000.0), np.full(n, 100.0)]))
    return path


@pytest.fixture(scope="module", params=[a[1] for a in ARMS], ids=[a[0] for a in ARMS])
def cip_export_run(request, tmp_path_factory):
    """One driver run per arm, shared by the tests below."""
    if not os.path.exists(DRIVER):
        pytest.fail("driver missing: %s" % DRIVER)
    tmp_path = tmp_path_factory.mktemp("cip_export")
    fname = _write_ile(str(tmp_path))
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["OMP_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["XDG_CACHE_HOME"] = os.path.join(str(tmp_path), "cache")
    env["MPLCONFIGDIR"] = os.path.join(str(tmp_path), "mpl")
    cmd = [sys.executable, DRIVER,
           "--fname", fname,
           "--parameter", "mc", "--parameter", "delta_mc",
           # quadratic, not rf: the input is exactly quadratic in mc, so the fit is exact
           # to float32 and the residual check below can be tight enough to see a
           # one-row misalignment.  Under rf the fit error is ~1 nat and swamps it.
           "--fit-method", "quadratic",
           "--n-max", "20000", "--n-eff", "30",
           "--n-output-samples", "50",
           "--fname-output-samples", "out",
           "--fname-output-integral", "out_int",
           "--no-plots"] + request.param
    # timeout MUST stay under the integration-check job's timeout, for the reason
    # test_cip_portfolio_members.py records: at a larger value GitHub cancels the job
    # first and the hung driver yields no test diagnostic.
    proc = subprocess.run(cmd, cwd=str(tmp_path), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=900)
    return proc, tmp_path


def test_posterior_export_completes(cip_export_run):
    """The regression proper.  Before the fix the acgpu_lnL arm reached this point with a
    converged integral and then exited 1 on KeyError: 'integrand'."""
    proc, tmp_path = cip_export_run
    assert "KeyError" not in proc.stdout, \
        "driver raised a KeyError in the export path:\n%s" % proc.stdout[-4000:]
    assert proc.returncode == 0, \
        "driver exited %d:\n%s" % (proc.returncode, proc.stdout[-4000:])
    # The integral alone is not the deliverable: the crash happened AFTER *_int.dat was
    # written, so a test that only checked the integral would have passed on the bug.
    assert os.path.exists(os.path.join(str(tmp_path), "out.xml.gz")), \
        "no posterior samples written:\n%s" % proc.stdout[-2000:]
    assert os.path.exists(os.path.join(str(tmp_path), "out_lnL.dat"))


def test_exported_lnL_matches_the_likelihood_at_the_exported_masses(cip_export_run):
    """Value AND row alignment, against ground truth outside the driver.

    out_lnL.dat[i] belongs to row i of out.xml.gz (both come from the same append loop
    over indx_list).  Recomputing lnL from that row's own masses is therefore an exact
    check, and it is the only one here that survives a fix which exports the right
    numbers against the wrong rows.  Measured separation: 1e-5 aligned, 5.0 permuted.
    """
    import lal
    import RIFT.lalsimutils as lalsimutils

    proc, tmp_path = cip_export_run
    assert proc.returncode == 0, proc.stdout[-2000:]
    P_list = lalsimutils.xml_to_ChooseWaveformParams_array(
        os.path.join(str(tmp_path), "out.xml.gz"))
    assert len(P_list) > 1, "need more than one exported row for alignment to mean anything"
    m1 = np.array([P.m1 for P in P_list]) / lal.MSUN_SI
    m2 = np.array([P.m2 for P in P_list]) / lal.MSUN_SI
    predicted = _lnL_of_mc((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)

    # out_lnL.dat covers every kept draw; the xml carries the first --n-output-samples of
    # them, in the same order.
    exported = np.atleast_1d(np.loadtxt(os.path.join(str(tmp_path), "out_lnL.dat")))
    assert len(exported) >= len(P_list)
    resid = exported[:len(P_list)] - predicted
    assert np.all(np.isfinite(resid)), "non-finite exported lnL"
    # 1e-2 is ~3 decades above the float32 storage floor this actually achieves and ~2
    # decades below the smallest convention or alignment error: exp/log/constant-offset
    # mutants land at >=2, a permutation at ~5.
    assert np.max(np.abs(resid)) < 1e-2, (
        "exported lnL does not match the likelihood at its own row's masses: "
        "max|resid|=%g rms=%g (exported %g..%g, predicted %g..%g)" % (
            np.max(np.abs(resid)), float(np.sqrt((resid ** 2).mean())),
            exported.min(), exported.max(), predicted.min(), predicted.max()))


# One case per sampler, parametrized rather than looped: inside a loop the first failure
# hides the rest, and this test exists to say WHICH sampler broke the contract.
_CONTRACT_SAMPLERS = ["mcsampler_linear", "mcsamplerGPU_linear", "mcsamplerGPU_log",
                      "mcsamplerAdaptiveVolume_log", "mcsamplerEnsemble_log",
                      "mcsamplerPortfolio_log"]


@pytest.mark.parametrize("which", _CONTRACT_SAMPLERS)
def test_every_sampler_supplies_a_resolvable_lnL_field(which):
    """The contract one level down: the driver's resolution rule must find a field.

    The export crash was a SECOND site reading the raw key after this rule had run.
    Removing that site is only safe while the rule itself always succeeds, so assert it on
    the integrators directly -- a new sampler that leaves neither key fails here in
    seconds instead of in a production export.  mcsamplerNFlow is omitted: it needs torch,
    which CI does not install, and it takes the same integrate_log path as AV.
    """
    from RIFT.integrators import (mcsampler, mcsamplerGPU, mcsamplerAdaptiveVolume,
                                  mcsamplerEnsemble, mcsamplerPortfolio)
    ctor = {
        "mcsampler_linear": mcsampler.MCSampler,
        "mcsamplerGPU_linear": mcsamplerGPU.MCSampler,
        "mcsamplerGPU_log": mcsamplerGPU.MCSampler,
        "mcsamplerAdaptiveVolume_log": mcsamplerAdaptiveVolume.MCSampler,
        "mcsamplerEnsemble_log": mcsamplerEnsemble.MCSampler,
        "mcsamplerPortfolio_log": mcsamplerPortfolio.MCSampler,
    }[which]
    use_lnL = which.endswith("_log")

    if which == "mcsamplerPortfolio_log":
        sampler = ctor(portfolio=[mcsamplerAdaptiveVolume.MCSampler()])
    else:
        sampler = ctor()
    sampler.add_parameter("x", pdf=lambda x: np.ones_like(x) / 2.0,
                          prior_pdf=lambda x: np.ones_like(x) / 2.0,
                          left_limit=-1.0, right_limit=1.0,
                          adaptive_sampling=True)
    # mcsampler hands the integrand an object-dtype array, so cast before any ufunc.
    lnf = lambda x: -0.5 * np.asarray(x, dtype=float) ** 2
    fn = lnf if use_lnL else (lambda x: np.exp(lnf(x)))
    kw = dict(n=200, nmax=2000, neff=10, save_intg=True, verbose=False)
    if use_lnL:
        kw.update(use_lnL=True, return_lnI=True)
    if which == "mcsamplerPortfolio_log":
        sampler.setup()
    sampler.integrate(fn, "x", **kw)
    rvs = sampler._rvs

    # The driver's rule, in the driver's order: log_integrand wins if present.
    if "log_integrand" in rvs:
        dat_logL = np.asarray(rvs["log_integrand"], dtype=float)
    elif "integrand" in rvs:
        raw = np.asarray(rvs["integrand"], dtype=float)
        dat_logL = raw if use_lnL else np.log(raw)
    else:
        pytest.fail("%s left neither 'log_integrand' nor 'integrand' in _rvs: %r"
                    % (which, sorted(str(k) for k in rvs)))
    # -0.5 x^2 on x in [-1,1]: lnL in [-0.5, 0].  A field stored under the wrong
    # convention lands outside this by orders of magnitude.
    x = np.asarray(rvs["x"], dtype=float).reshape(-1)
    np.testing.assert_allclose(dat_logL.reshape(-1), -0.5 * x ** 2,
                               rtol=1e-5, atol=1e-6,
                               err_msg="%s: resolved lnL is not lnL" % which)
