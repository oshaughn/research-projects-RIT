# demo/rift/slowrot -- verify-anywhere value demos for the slow-response likelihoods.
#
#   make demo              run BOTH quick-looks (no condor, no GPU); writes outputs/*.{txt,png}
#   make demo-rotation     Path A/B  : rotation-vs-static lnL gain grows with signal duration Omega*T
#   make demo-finite-size  Path D    : finite-size-vs-LWL lnL gain grows with detector arm length (fL/c)
#   make clean             remove outputs/
#
# Both show the maintained vectorized (NoLoop) likelihood RECOVERS SNR the standard analysis
# loses, on the SAME data at the true parameters.  They exercise the exact functions wired into
# integrate_likelihood_extrinsic_batchmode as --rotation-slow / --rotation-p-max / --freqresponse.
# Backing code: RIFT/likelihood/{factored_likelihood_with_rotation,factored_likelihood_freqresponse,
# slowrot_freqresponse}.py; validation RIFT/likelihood/test_slowrot_*.py; SLOWROT_HANDOFF.md.

PYTHON ?= python
# Code root is four levels up (demo/rift/slowrot -> demo/rift -> demo -> Code); put it on
# PYTHONPATH so `import RIFT...` resolves this worktree even without an editable install.
CODE_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
RUN = PYTHONPATH=$(CODE_ROOT):$$PYTHONPATH $(PYTHON)

.PHONY: help demo demo-rotation demo-finite-size clean
help:
	@echo "targets:"
	@echo "  demo              both verify-anywhere quick-looks (rotation + finite-size)"
	@echo "  demo-rotation     Path A/B: gain vs signal duration (Omega*T); null control at short T"
	@echo "  demo-finite-size  Path D:  gain vs arm length (fL/c); null control at LIGO 4 km"
	@echo "  clean             remove outputs/"
	@echo "(uses PYTHON=$(PYTHON); CODE_ROOT=$(CODE_ROOT))"

demo: demo-rotation demo-finite-size

demo-rotation:
	$(RUN) demo_rotation.py

demo-finite-size:
	$(RUN) demo_finite_size.py

clean:
	rm -rf outputs
