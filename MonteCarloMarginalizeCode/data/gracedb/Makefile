
SHELL:=/bin/bash
gid=G78339
ldf_server=ldr.ligo.caltech.edu:443

all: psds bayestar_map get_data local.cache

coinc.xml:
	gracedb download ${gid} coinc.xml

psds:
	gracedb download ${gid} psd.xml.gz

bayestar_map:
	gracedb download ${gid} skymap.fits.gz

H_remote.cache: coinc.xml
	ligo_data_find \
		-u file \
		--gaps \
		-o H \
		-t H1_ER_C00_L1 \
		--server ${ldf_server}\
		-s $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` - 1300" | bc) \
		-e $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` + 100" | bc) > H_remote.cache

L_remote.cache: coinc.xml
	ligo_data_find \
		-u file \
		--gaps \
		-o L \
		-t L1_ER_C00_L1 \
		--server ${ldf_server}\
		-s $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` - 1300" | bc) \
		-e $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` + 100" | bc) > L_remote.cache

V_remote.cache: coinc.xml
	ligo_data_find \
		-u file \
		--gaps \
		-o V \
		-t V1Online \
		--server ${ldf_server}\
		-s $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` - 1300" | bc) \
		-e $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` + 100" | bc) > V_remote.cache

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

local_cit.cache: 
	ligo_data_find \
		-u file \
		-o H \
		-l \
		--gaps \
		-t H1_ER_C00_L1 \
		--server ${ldf_server}\
		-s $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` - 1300" | bc) \
		-e $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` + 100" | bc) > local_cit.cache
	ligo_data_find \
		-u file \
		-o L \
		-l \
		--gaps \
		-t L1_ER_C00_L1 \
		--server ${ldf_server}\
		-s $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` - 1300" | bc) \
		-e $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` + 100" | bc) >> local_cit.cache
	ligo_data_find \
		-u file \
		-o V \
		-l \
		--gaps \
		-t V1Online \
		--server ${ldf_server}\
		-s $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` - 1300" | bc) \
		-e $(shell echo "`ligolw_print -t coinc_inspiral -c end_time coinc.xml` + 100" | bc) >> local_cit.cache
