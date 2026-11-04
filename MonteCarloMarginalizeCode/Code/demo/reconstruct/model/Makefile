# model : self-contained GW150914 waveform-MODEL strain reconstruction (RIFT).
#
# Download GW150914 from GWOSC (O1, H1+L1), run a full RIFT parameter-estimation
# DAG (IMRPhenomD, aligned spin pinned to zero) to a posterior, then reconstruct
# the whitened strain band from that posterior.  See README.md.
#
# Order:
#   make data        # download O1 frames + build event.cache + detect channel
#   make psd         # estimate off-source PSDs (gwpy) -> H1/L1-psd.xml.gz
#   make coinc       # build coinc.xml (event time + IFOs)
#   make dag         # generate the RIFT DAG (rundir_gw150914_D)
#   make submit      # apply CIT-local fixes + condor_submit_dag
#   make status      # condor_q for this run
#   # ... the DAG runs on condor over hours ...
#   make reconstruct # once extrinsic_posterior_samples.dat exists -> PNG
#
# All config (container, GPS, paths, sizes) lives in config.sh.

SHELL := /bin/bash
CFG   := config.sh

.PHONY: help
help:
	@echo "targets: data psd coinc dag submit status reconstruct all clean"
	@echo "  data        download GWOSC O1 frames + cache + channel detection"
	@echo "  psd         off-source PSD estimation (gwpy) -> H1/L1-psd.xml.gz"
	@echo "  coinc       build coinc.xml (event time + IFOs)"
	@echo "  dag         generate the RIFT DAG (IMRPhenomD, --assume-nospin)"
	@echo "  submit      CIT-local fixes + condor_submit_dag"
	@echo "  status      condor_q for this run"
	@echo "  reconstruct posterior -> whitened strain band PNG (after the DAG finishes)"
	@echo "edit config.sh for container path / env / sizes."

# 'all' builds and SUBMITS everything up to the condor run (not reconstruct,
# which needs the finished posterior).
.PHONY: all
all: data psd coinc dag submit

.PHONY: data
data:
	source $(CFG); rift_env; \
	  $$PYBIN $$MODEL_DIR/fetch_gwosc_data.py --outdir $$DATA_DIR \
	    --event-time $$EVENT_TIME --ifos $$IFOS

.PHONY: psd
psd:
	source $(CFG); rift_env; \
	  $$PYBIN $$MODEL_DIR/estimate_psd.py --cache $$CACHE \
	    --channel H1=H1:$$CHANNEL_BARE --channel L1=L1:$$CHANNEL_BARE \
	    --seg-start $$PSD_SEG_START --seg-len $$PSD_SEG_LEN --fftlen $$PSD_FFTLEN \
	    --srate $$SRATE --outdir $$MODEL_DIR

.PHONY: coinc
coinc:
	bash make_coinc.sh

.PHONY: dag
dag:
	bash build_dag.sh

.PHONY: submit
submit:
	bash submit.sh

.PHONY: status
status:
	source $(CFG); \
	  condor_q -nobatch 2>/dev/null | grep -E "$${USER}|JOB|held" || condor_q 2>/dev/null; \
	  echo "--- held (if any) ---"; condor_q -held 2>/dev/null | tail -20

.PHONY: reconstruct
reconstruct:
	bash reconstruct_gw150914.sh

.PHONY: clean
clean:
	rm -rf rundir_gw150914_D data
	rm -f H1-psd.xml.gz L1-psd.xml.gz coinc.xml seed_inj.xml.gz \
	      gw150914_samples.npz GW150914_reconstruction.png
