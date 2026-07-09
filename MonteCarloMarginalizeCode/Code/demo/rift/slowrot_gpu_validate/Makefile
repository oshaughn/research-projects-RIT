# Validate the GPU (xpy=cupy) rotation likelihood on a real GPU via one Condor job.
#   make submit   tar the branch code + condor_submit gpu_test.sub  (source your env first!)
#   make local    run the wrapper locally (validates plumbing; SKIPS the GPU test if no cupy)
#   make show     print the job stdout once it finishes
#   make clean
# Requires (from ~/setup_generic_env_vars.sh, sourced): SINGULARITY_RIFT_IMAGE, RIFT_REQUIRE_GPUS,
# LIGO_ACCOUNTING, LIGO_USER_NAME.
RIFT_CODE ?= /home/richard.oshaughnessy/RIFT_slowrot/MonteCarloMarginalizeCode/Code
PYTHON    ?= python3
CODE_PARENT := $(dir $(RIFT_CODE:/=))

.PHONY: tarball submit local show clean
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

show:
	@echo '--- gpu_test.out ---'; cat gpu_test.out 2>/dev/null; echo '--- gpu_test.err (tail) ---'; tail -20 gpu_test.err 2>/dev/null

clean:
	rm -rf _code rift_code.tar.gz gpu_test.out gpu_test.err gpu_test.log
