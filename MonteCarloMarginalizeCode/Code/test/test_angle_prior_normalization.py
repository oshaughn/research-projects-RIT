"""The ILE psi and phi_orb priors are derived from their sampling ranges, so they integrate to 1.

The defect: the drivers sampled psi on (0, 2 pi) (so that --internal-rotate-phase can use
(phi+psi, phi-psi) coordinates) but passed the fixed constant mcsampler.uniform_samp_psi =
1/pi, written for [0, pi), as its prior.  A prior defined independently of the range it is
integrated over has no reason to be normalized: this one had mass 2, mass 4 under
--internal-rotate-phase (range (0, 4 pi)), where phi_orb's fixed 1/(2 pi) also doubled.
Every reported lnZ carried +ln 2 (+ln 8 rotated), invisible in posteriors (the likelihood
is pi-periodic in psi) and equal to the size of AV-vs-JAX or RIFT-vs-bilby agreements people
quote.  Found by the 2026-09-08 marginalization audit (paper repo,
analyses/marginalization_audit_20260908, MATRIX X4).

Three checks, cheapest first:

1. test_range_derived_prior_integrates_to_one: the constructor the drivers now use,
   ret_uniform_samp_vector_alt(lo, hi), integrates to 1 over (lo, hi) on both the numpy and
   the GPU-module backends, and the constant it replaced does not.
2. test_driver_priors_are_built_from_their_ranges: source check on all three drivers that
   the prior objects are built from psi_prior_range / phi_orb_prior_range, captured AFTER the
   rotate-phase widening and BEFORE any --limit-* box, and that --limit-psi switches
   --internal-rotate-phase off (the box is on physical psi; the rotated coordinates cannot
   honour it).
3. test_zero_likelihood_evidence_is_the_prior_mass: the driver as a subprocess with
   --zero-likelihood, where lnZ is exactly ln(total prior mass).  Default: 0 (was ln 2).
   --internal-rotate-phase: 0 (was ln 8).  --limit-psi LO,HI: ln((HI-LO)/(2 pi)), the box's
   prior fraction, the same not-renormalized convention the sky and inclination boxes use.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# the synthetic zero-signal fixture and driver command builder live with the psi tests
from test_psi_marginalization import synthetic_fixture, _driver_cmd, _driver_env  # noqa: E402,F401

CODE = HERE.parent
DRIVERS = {
    "batchmode": CODE / "bin" / "integrate_likelihood_extrinsic_batchmode",
    "lisa": CODE / "bin" / "integrate_likelihood_extrinsic_batchmode_lisa",
    "legacy": CODE / "bin" / "integrate_likelihood_extrinsic",
}


###
### 1. The constructor is normalized over its own range; the constant it replaced is not
###

@pytest.mark.parametrize("lo,hi", [(0.0, 2*np.pi), (0.0, 4*np.pi), (0.3, 1.1)])
def test_range_derived_prior_integrates_to_one(lo, hi):
    import RIFT.integrators.mcsampler as mcsampler
    x = np.linspace(lo, hi, 2001)[1:-1]
    dens = np.asarray(mcsampler.ret_uniform_samp_vector_alt(lo, hi)(x), dtype=float)
    assert np.allclose(dens*(hi - lo), 1.0)
    # the fixed constant assumed [0, pi): over the driver's (0, 2 pi) it is mass 2
    old = np.asarray(mcsampler.uniform_samp_psi(x), dtype=float)
    assert np.allclose(old*2*np.pi, 2.0)

    # the GPU module's constructor, which the AV and GPU samplers bind as mcsampler
    try:
        import RIFT.integrators.mcsamplerGPU as G
    except Exception as exc:              # pragma: no cover  (module imports without cupy)
        pytest.skip("mcsamplerGPU unavailable: %r" % exc)
    val = G.ret_uniform_samp_vector_alt(lo, hi)(x)
    conv = getattr(G, "identity_convert", None)
    if conv is not None:
        val = conv(val)
    assert np.allclose(np.asarray(val, dtype=float)*(hi - lo), 1.0)


###
### 2. Wiring: every driver builds both angle priors from the captured ranges
###

@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_driver_priors_are_built_from_their_ranges(name):
    text = DRIVERS[name].read_text()
    assert "prior_pdf = mcsampler.uniform_samp_psi" not in text, \
        "%s still passes the fixed 1/pi constant as the psi prior" % name
    assert 'psi_prior_range = (param_limits["psi"][0], param_limits["psi"][1])' in text
    assert 'phi_orb_prior_range = (param_limits["phi_orb"][0], param_limits["phi_orb"][1])' in text
    assert "mcsampler.ret_uniform_samp_vector_alt(psi_prior_range[0], psi_prior_range[1])" in text
    assert "mcsampler.ret_uniform_samp_vector_alt(phi_orb_prior_range[0], phi_orb_prior_range[1])" in text
    # phi_orb's prior is the range-derived one, not the fixed phase constant
    i_phi = text.index('sampler.add_parameter("phi_orb"')
    phi_block = text[i_phi:text.index("adaptive_sampling", i_phi) if "adaptive_sampling" in text[i_phi:i_phi+800] else i_phi+800]
    assert "prior_pdf = mcsampler.ret_uniform_samp_vector_alt(phi_orb_prior_range[0], phi_orb_prior_range[1])" in phi_block
    assert "uniform_samp_phase" not in phi_block

    if name in ("batchmode", "lisa"):
        assert "prior_pdf = psi_prior_pdf," in text
        # captured after --internal-rotate-phase widens the ranges, so the rotated box is
        # what the prior normalizes over
        i_rot = text.index("param_limits['psi'] = (0, 4*numpy.pi)")
        i_cap = text.index("psi_prior_range = (")
        assert i_rot < i_cap, "%s captures the prior range before the rotate-phase widening" % name
    if name == "batchmode":
        # ... and before any --limit-* box narrows them
        i_box = text.index("for _optv, _k in [(opts.limit_psi, 'psi')")
        assert i_cap < i_box, "the prior range is captured after the --limit boxes are applied"
        # --limit-psi switches the rotation off, ahead of the widening it would otherwise apply
        i_guard = text.index("if opts.internal_rotate_phase and opts.limit_psi:")
        assert i_guard < i_rot
        guard = text[i_guard:i_guard + 500]
        assert "opts.internal_rotate_phase = False" in guard


###
### 3. End to end: with a unit likelihood, lnZ is ln(total prior mass)
###

CASES = [
    # extra args,                         expected lnZ,                     before this fix
    ([],                                  0.0,                              np.log(2.0)),
    (["--internal-rotate-phase"],         0.0,                              np.log(8.0)),
    (["--limit-psi", "0.2,0.9"],          np.log((0.9 - 0.2)/(2*np.pi)),    np.log((0.9 - 0.2)/np.pi)),
    (["--limit-psi", "0.2,0.9", "--internal-rotate-phase"],
                                          np.log((0.9 - 0.2)/(2*np.pi)),    None),
]


@pytest.mark.parametrize("extra,expected,before", CASES, ids=["default", "rotate", "box", "box_disables_rotate"])
def test_zero_likelihood_evidence_is_the_prior_mass(synthetic_fixture, tmp_path, extra, expected, before):
    """--zero-likelihood replaces the likelihood by 1, so the reported lnZ is the log of the
    total prior mass the sampler integrates: every normalized prior contributes 0.  The
    tolerance is set well inside ln 2 so a reverted fix cannot pass; Monte Carlo scatter on
    a constant integrand is far below it.
    """
    d = tmp_path / "run"
    d.mkdir()
    cmd = _driver_cmd(synthetic_fixture, "--zero-likelihood", "--n-max", "20000", "--n-eff", "100",
                      "--output-file", "out.xml.gz", *extra)
    proc = subprocess.run(cmd, cwd=str(d), env=_driver_env(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:]
    row = (d / "out.xml.gz_0_.dat").read_text().split()
    lnZ, sigma = float(row[9]), float(row[10])
    tol = max(5.0*sigma, 0.05)
    assert tol < 0.5*np.log(2.0), "tolerance %.3f cannot distinguish a missing ln 2" % tol
    assert abs(lnZ - expected) < tol, (
        "zero-likelihood lnZ %.4f, expected ln(prior mass) = %.4f (+/- %.4f); before the "
        "range-derived prior this configuration reported %s" % (lnZ, expected, tol, before))
    if "--limit-psi" in extra and "--internal-rotate-phase" in extra:
        assert "Disabling --internal-rotate-phase" in proc.stdout
