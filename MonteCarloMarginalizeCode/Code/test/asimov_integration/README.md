# Asimov integration smoke test

This directory contains a CI-sized Asimov integration test for the RIFT plugin.

The test is intentionally small:

- create a fresh Asimov project
- apply current-style Asimov data blueprints
- add the public `GW190426_190642` event
- add the current O4b-style RIFT SEOBNRv5PHM analysis blueprint
- verify Asimov sees the RIFT pipeline plugin and writes project state

The companion frozen-input test exercises the RIFT-specific build contract
without requiring a live cluster or successful upstream productions.  It uses
the in-tree GW150914 `ini`/`coinc.xml` fixtures, constructs a minimal production
object, and intercepts the external `util_RIFT_pseudo_pipe.py` call after
checking that `RIFT.asimov.rift.Rift.build_dag` has assembled the expected
command-line interface.

The template-contract test renders the real `RIFT/asimov/rift.ini` Liquid
template against realistic ledger-shaped objects.  It checks baseline ledger
parsing, important optional blocks such as calibration/eccentricity/OSG/manual
ILE args, and a deterministic randomized sweep over key scalar options.

It does not submit jobs or require production frame/calibration storage.

The RIFT Asimov integration is currently developed against the Asimov `0.5`
series.  The pytest is ready to skip cleanly for `0.6` and `0.7` until the
integration is updated for those APIs.

The bundled blueprints are small snapshots of the current public Asimov data
repository (`https://git.ligo.org/asimov/data`) chosen to avoid live network
dependencies in CI.
