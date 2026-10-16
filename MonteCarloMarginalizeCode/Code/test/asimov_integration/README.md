# Asimov 0.8 integration smoke test

This directory exercises the operational boundary between the O4c RIFT
pipeline plugin and Asimov 0.8. It creates a real project and ledger from
frozen blueprints, verifies plugin discovery, and checks the command assembled
for `util_RIFT_pseudo_pipe.py` without submitting cluster work.

Live scheduler execution remains the responsibility of the production canary;
these tests deliberately avoid network data and Condor submission.
