
SUBDIRS_SIM=$(wildcard sim_id_*)
SUBDIRS_COINC=$(wildcard coinc_id_*)

.PHONY: all_dirs all_dirs_coinc

all_dirs: $(SUBDIRS_SIM:%=%/marginalize_extrinsic_parameters.dag)
all_dirs_coinc: $(SUBDIRS_COINC:%=%/marginalize_extrinsic_parameters.dag)

all_dirs_plots: $(SUBDIRS_SIM:%=%/logevidence_hist.png) $(SUBDIRS_SIM:%=%/neff_evid.png)

%/marginalize_extrinsic_parameters.dag:
	$(MAKE) -C $* marginalize_extrinsic_parameters.dag

%/logevidence_hist.png:
	$(MAKE) -C $* logevidence_hist.png

%/neff_evid.png:
	$(MAKE) -C $* neff_evid.png
