
SUBDIRS=$(wildcard sim_id_*)

.PHONY: all_dirs

all_dirs: $(SUBDIRS:%=%/marginalize_extrinsic_parameters.dag)

all_dirs_plots: $(SUBDIRS:%=%/logevidence_hist.png) $(SUBDIRS:%=%/neff_evid.png)

%/marginalize_extrinsic_parameters.dag:
	$(MAKE) -C $* marginalize_extrinsic_parameters.dag

%/logevidence_hist.png:
	$(MAKE) -C $* logevidence_hist.png

%/neff_evid.png:
	$(MAKE) -C $* neff_evid.png
