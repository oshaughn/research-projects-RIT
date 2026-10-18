SEOBNRv6EHM readiness
====================

Status as of 2026-10-18: awaiting a public v6 backend
------------------------------------------------

This change does not supply or certify the new model. Public pySEOBNR main
``32e5d6373e2e6db28588f0512c43eed4ad611b2e`` contains no SEOBNRv6EHM
implementation. Its public documentation lists the v5 models. The tested
LALSuite 7.26.16 registry does not expose v6 either. Installing RIFT alone is
insufficient. Do not alias v6 to SEOBNRv5EHM or silently fall back to LAL.

Target branches audited:

* usual rift_O4c: ``ee57834ebfafe2b93252664b0d1b31a6f681d052``;
* oshaughnessy-junior rift_O4d: ``0e47aa142798759978d528f5e65874f91b8d2712``.

The existing GWSignal interface passes eccentricity and meanPerAno, preserves
explicit backend options, and bypasses the LAL approximant enum when the caller
supplies approx_string. This does not establish the backend's parameter meaning.
In particular, the inspected LALSuite v5 eccentric wrapper translates
``meanPerAno`` to pySEOBNR ``rel_anomaly``. A changing waveform proves a parameter
is used; it does not prove it is a mean anomaly. Confirm v6's documented radial
phase convention, eccentricity definition and reference frequency against its
actual source before interpreting posterior columns or choosing priors.

Optional waveform smoke check
-----------------------------

Run in the exact production environment/container, using the candidate source
or an installed build containing the check::

    GW_SURROGATE='' PYTHONPATH=MonteCarloMarginalizeCode/Code \
      python MonteCarloMarginalizeCode/Code/bin/util_CheckSEOBNRv6EHM.py \
      --output seobnrv6ehm-readiness.json

A missing v6 registry entry produces status=skipped and exit 0 by default.
Use --require-available for production acceptance: a missing backend then returns
exit 1. Numerical failures always return exit 1. No replacement model is used.
``--backend-options`` accepts a JSON object for a documented review backend's
options. This gate requires condition=0 for its polarization convention check.
It records package versions, source commit/tree and waveform wrapper hash.

The bounded eccentric aligned-spin fixture checks all six positive/negative
mode pairs, finite series, consistent sampling, absolute polarization/mode
agreement with RIFT's phase convention (without optimizing away offsets), inverse
distance scaling, nonzero response to eccentricity and radial phase, circular
waveform generation and the likelihood-facing Fourier/conjugate mode interface.
A passed smoke check is necessary, not full parameter-estimation certification.

The mode-length fix in this change makes tapering and peak alignment safe without
deltaF, caps a taper at available data, and prevents FFT padding from changing the
physical taper. Mock-mode regressions establish this wrapper behavior only;
they do not validate SEOBNRv6EHM itself.

Production acceptance still required
------------------------------------

With the real backend available, pin both pySEOBNR and the GWSignal adapter to
exact commits/builds. Verify the radial phase and frequency conventions, supported
spin/mass/eccentricity domain, and declared parameter validity before setting
priors. Then run the check with --require-available on both branches and the target
execute nodes.

Generate an eccentric aligned-spin pipeline with --use-gwsignal and
--approx SEOBNRv6EHM, explicitly enabling eccentricity and radial-phase sampling
when wanted. Inspect generated ILE arguments and XML/sample round trips for both
parameters and inspect backend options reaching each job. Do not let the default
quasicircular/precessing fit strategy silently define this analysis.

Run a zero-noise injection/recovery and a finite likelihood comparison with a
trusted direct-backend reference. On O4d, all hand-written ILE smoke commands must
include --time-marginalization --vectorized --gpu --force-xpy, as required by
AGENTS.md; confirm the NoLoop path in the log. Validate actual container contents,
not just the submit host. No full ILE or posterior-recovery claim is made here.

Unbound/scattering initial conditions require a separate reviewed mapping of
energy, angular momentum and initial separation. They are not certified by this
bounded eccentric fixture or by simply forwarding a backend dictionary.

Upstream references
-------------------

* https://git.ligo.org/waveforms/software/pyseobnr
* https://waveforms.docs.ligo.org/software/pyseobnr/
* https://arxiv.org/abs/2605.28715

Tests while the public backend is unavailable
---------------------------------------------

The optional numerical test test_seobnrv6ehm_availability.py skips when the
installed public waveform stack does not expose SEOBNRv6EHM. It automatically
runs the numerical checks once that generator is available; unexpected factory
errors and numerical mismatches remain failures. Dependency-free tests cover
this skip policy. Wrapper and build-only pipeline regressions keep running now.

The build-only regression test test_seobnrv6ehm_pipeline.py uses the real
pseudo-pipe, helper and BasicIteration builders. It checks decoded ILE/ILE_extr
arguments, backend options, all four NoLoop flags, eccentric aligned-spin CIP
coordinates and nonzero eccentricity/radial-phase XML round trips. It does not
execute ILE or submit jobs. Preserve case in the INI option use-meanPerAno and
pass manual-extra-ile-args as raw shell argument text.
