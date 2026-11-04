# demo/reconstruct -- whitened NR strain reconstruction with a 90% confidence band.
#
# See README.md.  Each target reconstructs one event from a fixed NR simulation:
# it loops RIFT ILE fair-draw jobs until enough coherent samples accumulate, then
# builds the band vs the whitened data.
#
# Usage:
#   make gw190521      # full example (needs the GW190521 RIT-Five run dir + NR data)
#   make gw150914      # clean GWOSC-only example (estimates its own PSD)
#   make plot-only CONFIG=config_gw190521.sh   # re-plot from existing samples
#   make clean

SHELL := /bin/bash

.PHONY: help
help:
	@echo "targets: gw190521  gw150914  plot-only CONFIG=...  clean"
	@echo "edit a config_*.sh for your event; then 'make <event>'"

.PHONY: gw190521
gw190521:
	bash reconstruct.sh config_gw190521.sh

.PHONY: gw150914
gw150914:
	bash reconstruct.sh config_gw150914.sh

# Re-run only the reconstruction/plot step from whatever ile_fd_*_compact.npz exist
.PHONY: plot-only
plot-only:
	@test -n "$(CONFIG)" || { echo "set CONFIG=config_xxx.sh"; exit 1; }
	source $(CONFIG); \
	POOL=""; for f in $$WORKDIR/ile_fd_*_compact.npz; do POOL="$$POOL --samples $$f"; done; \
	$$PYBIN ../reconstruct_strain.py $$POOL --fair-draw \
	  --group $$NR_GROUP --nr-param $$NR_PARAM $$PLOT_PSD_ARGS \
	  --event-time $$EVENT_TIME --event-name $$EVENT_NAME --sim-id $$SIM_ID \
	  --srate $$SRATE --flow $$FLOW --nproc $${NPROC:-8} \
	  --tlo $${TLO:--0.10} --thi $${THI:-0.06} --out $$OUT_PNG

.PHONY: clean
clean:
	rm -f ile_fd_*_0_.xml.gz ile_fd_*_compact.npz ile_fd_*.dat *.png
