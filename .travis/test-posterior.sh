#!/bin/bash

# Static guard on the posterior drivers, ahead of the end-to-end arms below: `indx_ok` keeps
# one meaning after the master post-integration mask is applied.  Pure ast, no RIFT import,
# ~2 s.  The defect is invisible to the runs below, because the rebindings it forbids were
# never read as the master mask -- it is a trap for the next edit, not a live bug.
python -m pytest -q MonteCarloMarginalizeCode/Code/test/test_cip_indx_ok_scope.py

python .travis/make_fake_composite.py
# Test default sampler (constant fit)
util_ConstructIntrinsicPosterior_GenericCoordinates.py  --fname fake.composite  --parameter mtot --parameter q --parameter s1z --parameter s2z  --use-precessing --no-plots
# Test alternative sampler (constant fit)
util_ConstructIntrinsicPosterior_GenericCoordinates.py  --fname fake.composite  --parameter mtot --parameter q --parameter s1z --parameter s2z  --use-precessing --no-plots --sampler-method GMM

# Test standard sampler (GP fit)
util_ConstructIntrinsicPosterior_GenericCoordinates.py  --fname fake.composite  --parameter mtot --parameter q --parameter s1z --parameter s2z  --use-precessing --no-plots  --fit-method gp

# Test standard sampler (GP fit)
util_ConstructIntrinsicPosterior_GenericCoordinates.py  --fname fake.composite  --parameter mtot --parameter q --parameter s1z --parameter s2z  --use-precessing --no-plots  --fit-method rf

# Plotting path.  Every arm above passes --no-plots, so the corner-plot block was
# never executed in CI and went on exiting 1 under matplotlib >= 3.10 unnoticed.
rm -f posterior_corner_fit_coords.png posterior_corner_extra_coords_*.png
util_ConstructIntrinsicPosterior_GenericCoordinates.py  --fname fake.composite  --parameter mc --parameter delta_mc --parameter s1z --parameter s2z  --n-output-samples 500
# Exit status covers this today, but the driver has dormant try/except scaffolding
# around these blocks: assert the deepest figure was actually written, so a future
# swallowed plotting failure is a CI failure rather than a missing file.
ls posterior_corner_extra_coords_*.png > /dev/null 2>&1 || { echo "ERROR: corner plots not written" 1>&2; false; }

# Plotting path, degenerate case.  The range each corner plot is drawn in comes from the
# input GRID; the largest data set drawn against it is the POSTERIOR, and nothing ties the
# two together.  corner RAISES ValueError on a 2-D panel with an empty histogram, so a
# posterior sitting off the grid used to end the whole job at the last step, after the
# samples were written -- intermittently, whenever the handful of posterior samples that
# happened to land in the m1-m2 box of the grid above came out as none (measured 0,1,2,2,3,5
# over six runs of the arm above, so a few percent of CI runs).
#
# --mc-range with --no-downselect-grid makes that disjointness deterministic instead of a
# lottery: the sampler is confined to mc in [1,5] while the grid stays at mc ~ 40-51.  Every
# mass panel is then unplottable and must be declined, and the run must still finish.
rm -f posterior_corner_fit_coords.png posterior_corner_extra_coords_*.png posterior_corner_nocut_beware.png
util_ConstructIntrinsicPosterior_GenericCoordinates.py  --fname fake.composite  --parameter mc --parameter delta_mc --parameter s1z --parameter s2z  --n-output-samples 500  --mc-range '[1,5]'  --no-downselect-grid > cip_disjoint_arm.log 2>&1 \
  || { echo "ERROR: CIP died on a posterior that does not overlap its grid" 1>&2; tail -40 cip_disjoint_arm.log 1>&2; false; }
# The trigger must still TRIGGER.  Without this the arm would keep passing after some
# unrelated change stopped producing a disjoint posterior, and would then be asserting
# nothing about the guard.
grep 'WARNING: skipping corner for' cip_disjoint_arm.log \
  || { echo "ERROR: no corner was declined, so this arm no longer reaches the degenerate case" 1>&2; false; }
# ... and plotting must still WORK, which is the distinction the guard has to preserve.  The
# s1z-s2z range is forced to the spin prior irrespective of the grid, so that panel can never
# be declined for want of samples: if corner or matplotlib is broken, it is missing here (and
# the driver exits nonzero, since nothing is caught).  posterior_corner_nocut_beware.png is
# the Corner 1 figure, which has to survive both of ITS overlays being declined.
for f in posterior_corner_extra_coords_s1z_s2z.png posterior_corner_nocut_beware.png ; do
  [ -s "$f" ] || { echo "ERROR: $f was not written; plotting is broken, not merely degenerate" 1>&2; false; }
done
