RIFT_CODE_ROOT := $(abspath ../../..)
REPO_ROOT := $(abspath ../../../../..)
CI_DEMO := $(REPO_ROOT)/.travis/ILE-GPU-Paper/demos

INI ?= $(CURDIR)/add_distance_grids.ini
COINC ?= $(REPO_ROOT)/.travis/ref_ini/coinc.xml
CACHE ?= $(CI_DEMO)/zero_noise.cache
PSD ?= $(CI_DEMO)/HLV-ILIGO_PSD.xml.gz
INITIAL_GRID ?= $(CI_DEMO)/overlap-grid.xml.gz
RUN_DIR ?= $(CURDIR)/rundir

ENV = GW_SURROGATE='' PYTHONPATH=$(RIFT_CODE_ROOT):$${PYTHONPATH:-} PATH=$(RIFT_CODE_ROOT)/bin:$${PATH}

.PHONY: help inputs dag submit clean validate-args

help:
	@echo "Targets:"
	@echo "  make inputs        - verify CI fake-data inputs are available"
	@echo "  make dag           - build the RIFT DAG/run directory with distance-grid export enabled"
	@echo "  make submit        - submit the generated DAG with condor_submit_dag"
	@echo "  make validate-args - confirm generated ILE args contain the distance-grid flags"
	@echo "  make clean         - remove the generated run directory"

inputs:
	@test -s "$(INI)"
	@test -s "$(COINC)"
	@test -s "$(CACHE)"
	@test -s "$(PSD)"
	@test -s "$(INITIAL_GRID)"
	@echo "Using INI:          $(INI)"
	@echo "Using coinc:        $(COINC)"
	@echo "Using fake cache:   $(CACHE)"
	@echo "Using PSD:          $(PSD)"
	@echo "Using initial grid: $(INITIAL_GRID)"

dag: inputs
	rm -rf "$(RUN_DIR)"
	mkdir -p "$(RUN_DIR)"
	cp "$(INITIAL_GRID)" "$(RUN_DIR)/overlap-grid.xml.gz"
	printf '%s\n' 'X --mc-range [23,35] --eta-range [0.20,0.24999] --parameter mc --parameter-implied eta --parameter-nofit delta_mc --fit-method gp --verbose --lnL-offset 120 --cap-points 12000 --n-output-samples 1000 --no-plots --n-eff 1000' > "$(RUN_DIR)/args_cip.txt"
	printf '%s\n' 'X --always-succeed --method lame --parameter m1' > "$(RUN_DIR)/args_test.txt"
	printf '%s\n' 'X --parameter m1 --parameter m2' > "$(RUN_DIR)/args_plot.txt"
	printf '%s\n' 'X --n-chunk 10000 --time-marginalization --sim-xml overlap-grid.xml.gz --reference-freq 100.0 --adapt-weight-exponent 0.1 --event-time 1000000014.236547946 --save-P 0.1 --cache-file $(CACHE) --fmin-template 10 --n-max 50000 --fmax 1700.0 --save-deltalnL inf --l-max 2 --n-eff 50 --approximant SEOBNRv4 --adapt-floor-level 0.1 --force-xpy --d-max 1000 --psd-file H1=$(PSD) --psd-file L1=$(PSD) --channel-name H1=FAKE-STRAIN --channel-name L1=FAKE-STRAIN --inclination-cosine-sampler --declination-cosine-sampler --data-start-time 1000000008 --data-end-time 1000000016 --inv-spec-trunc-time 0 --no-adapt-after-first --no-adapt-distance --srate 4096 --sampler-method GMM --internal-use-lnL  ' > "$(RUN_DIR)/args_ile.txt"
	cd "$(RUN_DIR)" &&  create_event_parameter_pipeline_BasicIteration \
		--ile-n-events-to-analyze 20 \
		--input-grid "$(INITIAL_GRID)" \
		--ile-exe "$(RIFT_CODE_ROOT)/bin/integrate_likelihood_extrinsic_batchmode" \
		--ile-args args_ile.txt \
                --last-iteration-export-marginal-distance-grid \
		--cip-args args_cip.txt \
		--test-args args_test.txt \
		--plot-args args_plot.txt \
		--request-memory-CIP 4096 \
		--request-memory-ILE 4096 \
		--n-samples-per-job 20 \
		--working-directory "$(RUN_DIR)" \
		--n-iterations 1 \
		--ile-retries 1 \
		--general-retries 1
	$(MAKE) validate-args

validate-args:
	@test -s "$(RUN_DIR)/args_ile.txt"
	@grep -q -- "--export-marginal-distance-grid" "$(RUN_DIR)/args_ile.txt"
	@grep -q -- "--internal-use-lnL" "$(RUN_DIR)/args_ile.txt"
	@! grep -q -- "--distance-marginalization" "$(RUN_DIR)/args_ile.txt"
	@echo "Distance-grid ILE args are present in $(RUN_DIR)/args_ile.txt"

submit: validate-args
	cd "$(RUN_DIR)" && condor_submit_dag marginalize_intrinsic_parameters_BasicIterationWorkflow.dag

clean:
	rm -rf "$(RUN_DIR)"
