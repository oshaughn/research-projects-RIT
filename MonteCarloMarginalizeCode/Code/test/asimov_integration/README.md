# Asimov integration smoke test

This directory contains a CI-sized Asimov integration test for the RIFT plugin.

The test is intentionally small:

- create a fresh Asimov project
- apply current-style Asimov data blueprints
- add the public `GW190426_190642` event
- add the current O4b-style RIFT SEOBNRv5PHM analysis blueprint
- verify Asimov sees the RIFT pipeline plugin and writes project state

It does not submit jobs or require production frame/calibration storage.

The RIFT Asimov integration is currently developed against the Asimov `0.5`
series.  The pytest is ready to skip cleanly for `0.6` and `0.7` until the
integration is updated for those APIs.

The bundled blueprints are small snapshots of the current public Asimov data
repository (`https://git.ligo.org/asimov/data`) chosen to avoid live network
dependencies in CI.
