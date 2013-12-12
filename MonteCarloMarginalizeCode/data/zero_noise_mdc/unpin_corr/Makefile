#!/bin/bash

data_dir=/home/pankow/param_est/research-projects/MonteCarloMarginalizeCode/data/zero_noise_mdc/
psd_dir=/home/pankow/param_est/research-projects/MonteCarloMarginalizeCode/data/

EVENT_TIME=1000000014.236547946
DIST_M=`ligolw_print -c distance ${data_dir}mdc.xml.gz`
INJ_CHANNEL_NAME=FAKE-STRAIN

plots: geocent_end_time_latitude_fulllikelihood.png geocent_end_time_longitude_fulllikelihood.png latitude_longitude_fulllikelihood.png geocent_end_time_latitude_likelihood.png geocent_end_time_longitude_likelihood.png latitude_longitude_likelihood.png polarization_psi_likelihood.png distance_inclination_fulllikelihood.png distance_inclination_likelihood.png triplot_t_ra_dec_unpinned.png triplot_dist_inc_unpinned.png triplot_psi_phi_unpinned.png

zero_noise_t_ra_dec_unpinned.xml.gz: ${data_dir}zero_noise.cache ${data_dir}mdc.xml.gz
	integrate_likelihood_extrinsic --cache-file ${data_dir}zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${psd_dir}HLV-ILIGO_PSD.xml.gz" --psd-file "L1=${psd_dir}HLV-ILIGO_PSD.xml.gz" --psd-file "V1=${psd_dir}HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --reference-freq 0.0 --n-max 100000 --output-file zero_noise_t_ra_dec_unpinned.xml.gz --save-samples \
    --psi `ligolw_print -c polarization ${data_dir}mdc.xml.gz` \
    --phi-orb `ligolw_print -c coa_phase ${data_dir}mdc.xml.gz` \
    --inclination `ligolw_print -c inclination ${data_dir}mdc.xml.gz` \
    --distance ${DIST_M}

triplot_t_ra_dec_unpinned.png: zero_noise_t_ra_dec_unpinned.xml.gz
	make_triplot $< -o $@ -i ${data_dir}mdc.xml.gz

geocent_end_time_latitude_fulllikelihood.png: zero_noise_t_ra_dec_unpinned.xml.gz
	plot_like_contours --dimension1 geocent_end_time --dimension2 latitude zero_noise_t_ra_dec_unpinned.xml.gz --show-mdc ${data_dir}mdc.xml.gz --full-likelihood
	mv geocent_end_time_latitude_likelihood.png geocent_end_time_latitude_fulllikelihood.png

geocent_end_time_longitude_fulllikelihood.png: zero_noise_t_ra_dec_unpinned.xml.gz
	plot_like_contours --dimension1 geocent_end_time --dimension2 longitude zero_noise_t_ra_dec_unpinned.xml.gz --show-mdc ${data_dir}mdc.xml.gz --full-likelihood
	mv geocent_end_time_longitude_likelihood.png geocent_end_time_longitude_fulllikelihood.png

latitude_longitude_fulllikelihood.png: zero_noise_t_ra_dec_unpinned.xml.gz
	plot_like_contours --dimension1 latitude --dimension2 longitude zero_noise_t_ra_dec_unpinned.xml.gz --show-mdc ${data_dir}mdc.xml.gz --full-likelihood
	mv latitude_longitude_likelihood.png latitude_longitude_fulllikelihood.png

geocent_end_time_latitude_likelihood.png: zero_noise_t_ra_dec_unpinned.xml.gz
	plot_like_contours --dimension1 geocent_end_time --dimension2 latitude zero_noise_t_ra_dec_unpinned.xml.gz --show-mdc ${data_dir}mdc.xml.gz

geocent_end_time_longitude_likelihood.png: zero_noise_t_ra_dec_unpinned.xml.gz
	plot_like_contours --dimension1 geocent_end_time --dimension2 longitude zero_noise_t_ra_dec_unpinned.xml.gz --show-mdc ${data_dir}mdc.xml.gz

latitude_longitude_likelihood.png: zero_noise_t_ra_dec_unpinned.xml.gz
	plot_like_contours --dimension1 latitude --dimension2 longitude zero_noise_t_ra_dec_unpinned.xml.gz --show-mdc ${data_dir}mdc.xml.gz

zero_noise_psi_phi_unpinned.xml.gz: ${data_dir}zero_noise.cache ${data_dir}mdc.xml.gz
	integrate_likelihood_extrinsic --cache-file ${data_dir}zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${psd_dir}HLV-ILIGO_PSD.xml.gz" --psd-file "L1=${psd_dir}HLV-ILIGO_PSD.xml.gz" --psd-file "V1=${psd_dir}HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --reference-freq 0.0 --n-max 100000 --output-file zero_noise_psi_phi_unpinned.xml.gz --save-samples \
    --declination `ligolw_print -c latitude ${data_dir}mdc.xml.gz` \
    --right-ascension `ligolw_print -c longitude ${data_dir}mdc.xml.gz` \
    --inclination `ligolw_print -c inclination ${data_dir}mdc.xml.gz` \
    --distance ${DIST_M} \
    --t-ref `ligolw_print -c geocent_end_time -c geocent_end_time_ns -d "." ${data_dir}mdc.xml.gz`

triplot_psi_phi_unpinned.png: zero_noise_psi_phi_unpinned.xml.gz
	make_triplot $< -o $@ -i ${data_dir}mdc.xml.gz

polarization_psi_likelihood.png: zero_noise_psi_phi_unpinned.xml.gz
	plot_like_contours --dimension1 polarization --dimension2 coa_phase zero_noise_psi_phi_unpinned.xml.gz --show-mdc ${data_dir}mdc.xml.gz


zero_noise_dist_inc_unpinned.xml.gz: ${data_dir}zero_noise.cache ${data_dir}mdc.xml.gz
	integrate_likelihood_extrinsic --cache-file ${data_dir}zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${psd_dir}HLV-ILIGO_PSD.xml.gz" --psd-file "L1=${psd_dir}HLV-ILIGO_PSD.xml.gz" --psd-file "V1=${psd_dir}HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --reference-freq 0.0 --n-max 100000 --output-file zero_noise_dist_inc_unpinned.xml.gz --save-samples \
    --right-ascension `ligolw_print -c longitude ${data_dir}mdc.xml.gz` \
    --declination `ligolw_print -c latitude ${data_dir}mdc.xml.gz` \
    --psi `ligolw_print -c polarization ${data_dir}mdc.xml.gz` \
    --phi-orb `ligolw_print -c coa_phase ${data_dir}mdc.xml.gz` \
    --t-ref `ligolw_print -c geocent_end_time -c geocent_end_time_ns -d "." ${data_dir}mdc.xml.gz`

triplot_dist_inc_unpinned.png: zero_noise_dist_inc_unpinned.xml.gz
	make_triplot $< -o $@ -i ${data_dir}mdc.xml.gz

distance_inclination_fulllikelihood.png: zero_noise_dist_inc_unpinned.xml.gz
	plot_like_contours --dimension1 distance --dimension2 inclination zero_noise_dist_inc_unpinned.xml.gz --show-mdc ${data_dir}mdc.xml.gz --full-likelihood
	mv distance_inclination_likelihood.png distance_inclination_fulllikelihood.png

distance_inclination_likelihood.png: zero_noise_dist_inc_unpinned.xml.gz
	plot_like_contours --dimension1 distance --dimension2 inclination zero_noise_dist_inc_unpinned.xml.gz --show-mdc ${data_dir}mdc.xml.gz

clean:
	rm -rf *.png *.xml.gz
