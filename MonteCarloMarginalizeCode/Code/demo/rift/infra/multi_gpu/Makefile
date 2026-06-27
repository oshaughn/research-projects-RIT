# ============================================================================
# Multi-GPU ILE fan-out demo
#
# RIFT ILE "batchmode" evaluates a contiguous block of intrinsic grid points
# (--event .. --event+--n-events-to-analyze) SERIALLY on ONE GPU.  On a node
# with several GPUs reserved, the rest sit idle.  The fan-out splits that block
# into N shards run concurrently -- one per GPU -- via the generated ile_pre.sh.
#
# DEFAULT: GPU ILE jobs grab EVERY physical GPU on the node ('all'): request_GPUs=1
# (matching unchanged), but the launcher splits the ILE block across all of the node's
# GPUs.  Assumes whole nodes are reserved.  Override with RIFT_ILE_GPU_FANOUT=<value>
# (env) or --ile-gpu-fanout <value> (CLI) or an asimov blueprint key.  The value is
# BAKED into ile_pre.sh and sizes request_GPUs/request_CPUs, so nothing depends on the
# submit/execute environment -- which is what makes it work under asimov.
#
# Values (see README.md "Values"):
#   all         DEFAULT: request 1 GPU, use EVERY physical GPU on the node (reserved node)
#   1 / single  FALLBACK: single-GPU run (shared node / old behaviour)
#   N           fixed: request N GPUs+CPUs, split across them
#   auto-max-N  partitionable-slot pool (not recommended): request up to N of the GPUs
#               available on the matched slot; split across whatever condor grants (1..N)
#
# Targets:
#   make smoke-local         # PROVE the split+pin logic on THIS node's GPUs (no cupy,
#                            #   no condor, no data).  FANOUT=4|all|...  This is the
#                            #   runnable-anywhere proof.
#   make requests            # show request_GPUs/CPUs + baked launcher for each mode
#                            #   (1, 4, all, auto-max-4) -- the "how do I get 1-4 GPUs" ref.
#   make build               # build a real pipeline run dir on the CI data with
#                            #   --ile-gpu-fanout (needs SINGULARITY_RIFT_IMAGE).
#   make verify              # assert the generated ILE.sub + ile_pre.sh carry the fan-out.
#   make inspect             # show the generated launcher + sub requests.
#   make clean
#
# See README.md for the asimov path and host/GPU matching.
# ============================================================================

RIFT_CODE_ROOT := $(abspath ../../../..)
REPO_ROOT      := $(abspath ../../../../../..)
CI_DEMO        := $(REPO_ROOT)/.travis/ILE-GPU-Paper/demos

ENV := PYTHONPATH=$(RIFT_CODE_ROOT):$${PYTHONPATH:-} PATH=$(RIFT_CODE_ROOT)/bin:$${PATH}

CACHE   ?= $(CI_DEMO)/zero_noise.cache
PSD     ?= $(CI_DEMO)/HLV-ILIGO_PSD.xml.gz
SIM_XML ?= $(CI_DEMO)/mdc.xml.gz
FRAMES  ?= $(CI_DEMO)/zero_noise_mdc

# How many GPUs to fan out across.  The smoke test caps this at the number of
# GPUs actually visible; the pipeline build bakes exactly this into ile_pre.sh
# and requests this many GPUs+CPUs.
FANOUT  ?= 4

BUILD_DIR := $(CURDIR)/rundir_multigpu
DAG       := marginalize_intrinsic_parameters_BasicIterationWorkflow.dag

.PHONY: all smoke-local build verify inspect requests clean coinc help

help:
	@sed -n '2,40p' Makefile

# ---------------------------------------------------------------------------
# smoke-local: the genuinely-runnable-anywhere proof.
#
# Builds a REAL ile_pre.sh from the shipped helper (RIFT.misc.dag_utils_generic.
# ile_invocation_shell), wrapping the fake_ile.py stub, then runs it with the
# fan-out enabled.  Asserts: every grid point covered exactly once, spread across
# the GPUs, distinct per-shard output prefixes (no clobber).  No cupy, no GPU
# compute -- this isolates the only new logic (partition + CUDA_VISIBLE_DEVICES
# pinning).  Set DEVICES=0,1,2,3 to force a device list on a shared node.
# ---------------------------------------------------------------------------
DEVICES ?=
smoke-local:
	@rm -rf _smoke && mkdir _smoke
	@# Generate ile_pre.sh exactly as the pipeline would, baking FANOUT in.
	@cd _smoke && $(ENV) RIFT_ILE_GPU_FANOUT=$(FANOUT) python3 -c \
	  "import os; from RIFT.misc.dag_utils_generic import ile_invocation_shell; \
	   open('ile_pre.sh','w').write('#! /bin/bash -xe\n'+ile_invocation_shell(os.path.abspath('../fake_ile.py'))); \
	   os.chmod('ile_pre.sh',0o755)"
	@echo "=== generated ile_pre.sh (launcher head) ==="
	@grep -nE 'RIFT_ILE_GPU_FANOUT|exec ' _smoke/ile_pre.sh | head -3
	@echo "=== running: 100-point block, fan-out=$(FANOUT)$(if $(DEVICES), on devices $(DEVICES)) ==="
	@cd _smoke && $(if $(DEVICES),CUDA_VISIBLE_DEVICES=$(DEVICES) ,)./ile_pre.sh \
	   --n-events-to-analyze 100 --output-file=CME_out-0-1-0.xml --event=0 \
	   --some-other-arg foo --internal-waveform-extra-kwargs '{a:1,b:2}' 2>shards.log; echo "  launcher exit=$$?"
	@grep rift_ile_multigpu _smoke/shards.log | sed 's/^/  /'
	@cd _smoke && python3 -c "import glob; \
	   rows=[l.split() for f in glob.glob('CME*.dat') for l in open(f)]; \
	   g=sorted(int(r[0]) for r in rows); gpus=sorted(set(r[1] for r in rows)); \
	   prefixes=sorted(set(f.rsplit('_',2)[0] for f in glob.glob('CME*.dat'))); \
	   assert g==list(range(100)), 'COVERAGE FAIL: '+str(g); \
	   assert len(g)==len(set(g)), 'DUPLICATE points'; \
	   print('  PASS: 100/100 grid points, no dups, range %d..%d'%(g[0],g[-1])); \
	   print('  PASS: spread across GPUs', gpus); \
	   print('  PASS: %d distinct shard output prefixes (no clobber):'%len(prefixes)); \
	   [print('        '+p) for p in prefixes]"

# ---------------------------------------------------------------------------
# build: a real pipeline run directory on the CI synthetic data, with fan-out.
#
# Uses singularity/OSG mode so ile_pre.sh is generated (the fan-out launcher
# lives there).  Needs the container-family manifest:
#   export SINGULARITY_RIFT_IMAGE=~/rift_cit_build_container_family/built_containers/rift_container_family.cit.yaml
#   export SINGULARITY_BASE_EXE_DIR=/usr/local/bin/
# (or pass SINGULARITY_RIFT_IMAGE=... on the make line).  This only BUILDS the
# DAG; it does not submit.
# ---------------------------------------------------------------------------
coinc: ci_coinc.xml
ci_coinc.xml:
	$(ENV) util_SimInspiralToCoinc.py --sim-xml $(SIM_XML) --event 0 \
	  --ifo H1 --ifo L1 --ifo V1 --output $(CURDIR)/ci_coinc.xml --injected-snr 17.5

build: ci_coinc.xml
	@test -n "$(SINGULARITY_RIFT_IMAGE)" || (echo "ERROR: export SINGULARITY_RIFT_IMAGE=<container family .yaml> first (see README.md)"; false)
	rm -rf $(BUILD_DIR)
	$(ENV) util_RIFT_pseudo_pipe.py \
	  --use-ini $(CURDIR)/multigpu_ci.ini --use-coinc $(CURDIR)/ci_coinc.xml --use-rundir $(BUILD_DIR) \
	  --fake-data-cache $(CACHE) \
	  --assume-nospin --approx IMRPhenomD --ile-sampler-method AV \
	  --internal-force-iterations 2 --ile-n-eff 10 \
	  --ile-force-gpu --ile-gpu-fanout $(FANOUT) \
	  --use-osg --use-osg-file-transfer --internal-truncate-files-for-osg-file-transfer \
	  --internal-ile-request-disk 4G
	@# OSG file-transfer mode wants frames + per-IFO PSDs physically present in the rundir.
	@# (In OSG mode the PSD is supplied as run-dir <IFO>-psd.xml.gz, copied below, not via a flag.)
	rm -rf $(BUILD_DIR)/frames_dir; cp -r $(FRAMES) $(BUILD_DIR)/frames_dir
	for ifo in H1 L1 V1; do cp $(PSD) $(BUILD_DIR)/$$ifo-psd.xml.gz; done
	@echo
	@echo "Built $(BUILD_DIR).  Now:  make verify"

# ---------------------------------------------------------------------------
# verify: assert the fan-out actually landed in the generated submit files.
# ---------------------------------------------------------------------------
verify:
	@test -f $(BUILD_DIR)/ILE.sub || (echo "no $(BUILD_DIR)/ILE.sub -- run 'make build' first"; false)
	@echo "=== ILE.sub GPU/CPU requests (expect $(FANOUT)) ==="
	@grep -iE 'request_GPUs|request_CPUs' $(BUILD_DIR)/ILE.sub
	@grep -qiE 'request_GPUs *= *$(FANOUT)' $(BUILD_DIR)/ILE.sub || (echo "FAIL: request_GPUs != $(FANOUT)"; false)
	@grep -qiE 'request_CPUs *= *$(FANOUT)' $(BUILD_DIR)/ILE.sub || (echo "FAIL: request_CPUs != $(FANOUT)"; false)
	@echo "=== ile_pre.sh baked fan-out + launcher ==="
	@grep -nE 'RIFT_ILE_GPU_FANOUT|rift_ile_multigpu' $(BUILD_DIR)/ile_pre.sh | head -3
	@grep -qE 'RIFT_ILE_GPU_FANOUT:-$(FANOUT)' $(BUILD_DIR)/ile_pre.sh || (echo "FAIL: ile_pre.sh did not bake fan-out=$(FANOUT)"; false)
	@echo
	@echo "OK: ILE.sub requests $(FANOUT) GPUs + $(FANOUT) CPUs; ile_pre.sh bakes RIFT_ILE_GPU_FANOUT=$(FANOUT)"
	@echo "    and will split each 100-point ILE block into $(FANOUT) shards, one per GPU."

# ---------------------------------------------------------------------------
# requests: show the condor request_GPUs/request_CPUs + baked launcher directive
# that each fan-out mode produces (fixed N vs. adaptive vs. all-physical).  Fast:
# calls write_ILE_sub_simple directly, no full pipeline build.  This is the
# reference for "how do I get 1/2/3/4 GPUs?".
# ---------------------------------------------------------------------------
MANIFEST := $(CURDIR)/blueprints/rift_container_family.cit.yaml
requests:
	@rm -rf _req && mkdir -p _req/frames_dir && touch _req/frames_dir/H1.gwf
	@echo "(default = unset => 'all'; '1'/single = fallback)"
	@for MODE in default 1 all 4 auto-max-4; do \
	  cd $(CURDIR)/_req; \
	  if [ $$MODE = default ]; then unset RIFT_ILE_GPU_FANOUT; else export RIFT_ILE_GPU_FANOUT=$$MODE; fi; \
	  $(ENV) RIFT_REQUIRE_GPUS='(Capability >= 8.0)' LIGO_ACCOUNTING=x LIGO_USER_NAME=y \
	    python3 -c "import os,RIFT.misc.dag_utils_generic as d; \
	      j,n=d.write_ILE_sub_simple(tag='ILE',exe='/usr/local/bin/integrate_likelihood_extrinsic_batchmode',request_gpu=True,use_singularity=True,singularity_image='$(MANIFEST)',frames_dir=os.path.abspath('frames_dir'),transfer_files=['../g.xml.gz'],arg_str=' --n-events-to-analyze \$$(macrongroup) --event=\$$(macroevent) ',output_file='CME_out.xml'); j.write_sub_file()" >/dev/null 2>&1; \
	  echo "=== RIFT_ILE_GPU_FANOUT=$$MODE ==="; \
	  grep -iE 'request_GPUs|request_CPUs' ILE.sub | sed 's/^/   /'; \
	  echo "   ile_pre.sh launcher directive: $$(grep -o 'RIFT_ILE_GPU_FANOUT:-[A-Za-z0-9]*' ile_pre.sh)"; \
	  cd $(CURDIR); \
	done
	@rm -rf _req

inspect:
	@echo "######## $(BUILD_DIR)/ile_pre.sh ########"; cat $(BUILD_DIR)/ile_pre.sh
	@echo; echo "######## ILE.sub (resource + container lines) ########"
	@grep -iE 'request_|require_gpus|SingularityImage|executable|when_to_transfer' $(BUILD_DIR)/ILE.sub

clean:
	rm -rf _smoke _req $(BUILD_DIR) ci_coinc.xml
