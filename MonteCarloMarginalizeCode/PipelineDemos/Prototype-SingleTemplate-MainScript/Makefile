
# Pulled from Pankow's script in MonteCarloMarginalizeCode/data
# RETRIEVE='gracedb download'
# gracedb was not robust on ROS laptop

gid=G78339
ldf_server=ldr.ligo.caltech.edu:443

all: psds bayestar_map get_data local.cache setup
	find ${gid}/ -name "*.gwf" | lalapps_path2cache > local.cache  # rebuild at end

test:
	./integrate_likelihood_extrinsic --cache-file local.cache --channel-name H1=FAKE-STRAIN --channel-name L1=FAKE-STRAIN --channel-name V1=FAKE_h_16384Hz_4R --psd-file "H1=psd.xml.gz" --psd-file "L1=psd.xml.gz" --psd-file "V1=/psd.xml.gz" --coinc-xml coinc.xml

report:
	echo "Injection data: mc, eta, deff (per instrument)"
	ligolw_print coinc.xml  -t sngl_inspiral -c mchirp -c eta -c eff_distance

setup:
	ln -sf ../../Code/lalsimutils.py .
	ln -sf ../../Code/factored_likelihood.py .
	ln -sf ../../Code/ourio.py .
	ln -sf ../../Code/mcsampler.py .
	ln -sf ../../Code/statutils.py .
	ln -sf ../../Code/integrate_likelihood_extrinsic .


coinc.xml:
	gracedb download ${gid} coinc.xml

psds:
	gracedb download ${gid} psd.xml.gz

bayestar_map:
	gracedb download ${gid} skymap.fits.gz

H_remote.cache: coinc.xml
	ligo_data_find \
		-u file \
		-o H \
		-t H1_ER_C00_L1 \
		--server ${ldf_server}\
		-s $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` - 1300" | bc) \
		-e $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` + 100" | bc) > H_remote.cache

L_remote.cache: coinc.xml
	ligo_data_find \
		-u file \
		-o L \
		-t L1_ER_C00_L1 \
		--server ${ldf_server}\
		-s $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` - 4" | bc) \
		-e $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` + 4" | bc) > L_remote.cache

V_remote.cache: coinc.xml
	ligo_data_find \
		-u file \
		-o V \
		-t V1Online \
		--server ${ldf_server}\
		-s $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` - 4" | bc) \
		-e $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` + 4" | bc) > V_remote.cache

get_data: H_remote.cache L_remote.cache V_remote.cache
	mkdir -p ${gid}
	@while read line; do \
			gsiscp ldas-pcdev1.ligo.caltech.edu:$${line:16} ${gid}/; \
	done < H_remote.cache
	@while read line; do \
			gsiscp ldas-pcdev1.ligo.caltech.edu:$${line:16} ${gid}/; \
	done < L_remote.cache
	@while read line; do \
			gsiscp ldas-pcdev1.ligo.caltech.edu:$${line:16} ${gid}/; \
	done < V_remote.cache

local.cache: 
	find ${gid}/ -name "*.gwf" | lalapps_path2cache > local.cache
