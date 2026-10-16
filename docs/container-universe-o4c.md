# Container families on O4c

With `RIFT_CONTAINER_UNIVERSE=1`, a GPU ILE or calibration-pilot job selects
an image **basename** in `container_image`. The matching full URL is delivered
through `transfer_input_files`; `MY.TransferInput` pins that list so HTCondor
does not add a second, basename-only input. Full URLs inside the image selector
are unsafe: `condor_submit` takes the suffix after the last slash before
match-time expansion, corrupting the expression and holding worker jobs.

Every image in a GPU container-universe family must be a transferable URL
(such as `osdf://`). Local/CVMFS family entries fail at DAG generation with an
actionable error. Stage them at URLs or select a single image. O4c does not
implement the O4d runtime-selection wrapper.

CPU jobs, including calibration reweighting, use the declared fallback as a single literal image. They do not
require GPU-capability expansion. CPU ILE retains `--gpu --force-xpy --vectorized`
for the NoLoop likelihood path; scheduler GPU requests are independent of those flags. Plain single-image configurations keep their
existing behavior. The existing GPU capability-floor and defined-capability
requirements remain in force. This change does not alter capability-band
selection or add enforcement of `cuda_capability_max`.

Run the container tests on a host with HTCondor installed to include actual
submit-file parsing and effective job-ad assertions:

```sh
PYTHONPATH=MonteCarloMarginalizeCode/Code python -m pytest -q \
  MonteCarloMarginalizeCode/Code/test/test_container_manifest.py
```

The HTCondor checks are dry runs: they do not demonstrate image download or
execution on OSPool. The original basename-transfer mechanism was validated
with live OSPool jobs in upstream PR 165.
