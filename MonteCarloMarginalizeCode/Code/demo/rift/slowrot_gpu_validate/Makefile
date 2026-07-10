# Validate the GPU (xpy=cupy) rotation + freqresponse likelihood on a real GPU.
#   make local-gpu  run the LIKELIHOOD parity unit tests in the container on THIS host's GPU  <-- fastest
#   make e2e        run the REAL ILE CPU-vs-GPU consistency check (rotation + freqresponse)   <-- end-to-end
#   make submit     tar the branch code + condor_submit gpu_test.sub  (source your env first!)
#   make local      run the wrapper locally (validates plumbing; SKIPS the GPU test if no cupy)
#   make show       print the job stdout once it finishes
#   make clean
# Condor path requires (from ~/setup_generic_env_vars.sh, sourced): SINGULARITY_RIFT_IMAGE,
# RIFT_REQUIRE_GPUS, LIGO_ACCOUNTING, LIGO_USER_NAME.
RIFT_CODE ?= /home/richard.oshaughnessy/RIFT_slowrot/MonteCarloMarginalizeCode/Code
PYTHON    ?= python3
CODE_PARENT := $(dir $(RIFT_CODE:/=))
# Direct-on-this-host GPU run (no Condor).  Point at a cc-matched CUDA-11.8 RIFT image; the
# A100/V100/Turing/Pascal boxes are all cc 6.0-9.0.  (setup_generic_env_vars.sh no longer exports
# SINGULARITY_RIFT_IMAGE, so `make submit` needs it set by hand; `make local-gpu` does not.)
RIFT_LOCAL_IMAGE ?= /home/richard.oshaughnessy/rift_cit_build_container_family/built_containers/rift_o4d-calmarg_in_loop_cc60-90_cuda118_20260615b.sif

# End-to-end ILE consistency case.  EMPTY = self-contained (make_e2e_inputs.py generates a throwaway
# 2-detector injection).  Override to reuse an existing ILE case dir: `make e2e RIFT_E2E_CASE=/path/to/case`.
RIFT_E2E_CASE ?=

.PHONY: tarball submit local local-gpu e2e show clean
tarball:
	# write OUTSIDE the Code tree (tar can't read a file it is growing), then move it in;
	# exclude this kit's own scratch so a stale tarball is never re-packed.
	tar --exclude='*/slowrot_gpu_validate/rift_code.tar.gz' --exclude='*/slowrot_gpu_validate/_code' \
	    --exclude='*/slowrot_gpu_validate/gpu_test.*' \
	    -czf $(CODE_PARENT)/.rift_code.tar.gz -C $(CODE_PARENT) $(notdir $(RIFT_CODE:/=))
	mv $(CODE_PARENT)/.rift_code.tar.gz rift_code.tar.gz
	@echo "tarball: $$(du -h rift_code.tar.gz | cut -f1)"

submit: tarball
	@test -n "$$SINGULARITY_RIFT_IMAGE" || (echo 'ERROR: source setup_generic_env_vars.sh first (SINGULARITY_RIFT_IMAGE unset)'; exit 1)
	condor_submit gpu_test.sub

local:
	RIFT_CODE=$(RIFT_CODE) PYTHON=$(PYTHON) ./run_gpu_test.sh

local-gpu:
	@test -f "$(RIFT_LOCAL_IMAGE)" || (echo "ERROR: RIFT_LOCAL_IMAGE not found: $(RIFT_LOCAL_IMAGE)"; exit 1)
	@nvidia-smi -L | head -1
	CUDA_VISIBLE_DEVICES=$${CUDA_VISIBLE_DEVICES:-0} singularity exec --nv "$(RIFT_LOCAL_IMAGE)" \
	    bash -c 'export RIFT_CODE=$(RIFT_CODE); export NUMBA_CACHE_DIR=$$(mktemp -d); RIFT_CODE=$(RIFT_CODE) PYTHON=$(PYTHON) ./run_gpu_test.sh' 2>&1 | grep -v 'underlay of'

# Real ILE CPU-vs-GPU consistency: runs {rotation,finite} x {cpu,gpu} on RIFT_E2E_CASE and asserts
# the marginalized lnL agrees to sampler noise.  Runs in the container on THIS host's GPU.
e2e:
	@test -f "$(RIFT_LOCAL_IMAGE)" || (echo "ERROR: RIFT_LOCAL_IMAGE not found: $(RIFT_LOCAL_IMAGE)"; exit 1)
	@nvidia-smi -L | head -1
	CUDA_VISIBLE_DEVICES=$${CUDA_VISIBLE_DEVICES:-0} singularity exec --nv "$(RIFT_LOCAL_IMAGE)" \
	    bash -c 'RIFT_CODE=$(RIFT_CODE) PYTHON=$(PYTHON) RIFT_E2E_CASE=$(RIFT_E2E_CASE) ./run_e2e_consistency.sh' 2>&1 | grep -v 'underlay of'

show:
	@echo '--- gpu_test.out ---'; cat gpu_test.out 2>/dev/null; echo '--- gpu_test.err (tail) ---'; tail -20 gpu_test.err 2>/dev/null

clean:
	rm -rf _code rift_code.tar.gz gpu_test.out gpu_test.err gpu_test.log
