START=999999000
STOP=1000001000
DUR=$(shell python -c "print ${STOP}-${START}")

INJ_CHANNEL_NAME=FAKE-STRAIN
INJ_XML=mdc.xml.gz
INJECTION_FRAME_TYPE=zero_noise_mdc

INJECTION_DEST=zero_noise_mdc/

LIGO_FRAME_DURATION=64 # s
LIGO_FRAME_DURATION_COMB=64 # s
LIGO_N_FRAMES=1

BASE_DIR=$(shell pwd)/
PSD_FILE=${BASE_DIR}/../HLV-ILIGO_PSD.xml.gz


EVENT_TIME=1000000014.236547946

plot_ts: H1_zero_noise.cache L1_zero_noise.cache V1_zero_noise.cache
	../utils/plot_ts zero_noise.cache

test_ile_dag: zero_noise.cache
	../../Code/compute_marginalized_likelihood --cache-file ${BASE_DIR}/zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --reference-freq 0.0 --save-samples --output-file zero_noise.xml.gz --n-copies 10

test_ile: zero_noise.cache
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --pin-to-sim mdc.xml.gz --reference-freq 0.0

test_ile_postprocessing: zero_noise.cache
	rm -f zero_noise.pdf
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0  --reference-freq 0.0  --save-samples  --data-start 1000000000 --data-end 1000000016 -i 2 --time-marginalization --n-max 60000 --output-file zero_noise.xml.gz --save-deltalnL 15 --save-P 0.0001  --adapt-floor-level 0.1 --adapt-weight-exponent 0.8 --n-chunk 2000 --convergence-tests-on --n-eff 100
	echo " Number of samples stored in this test (<number of samples performed): " `ligolw_print -t sim_inspiral -c alpha1 zero_noise.xml.gz | wc -l `
	../../Code/convert_output_format_ile2inference zero_noise.xml.gz > flatfile-points.dat
	../../Code/postprocess_1d_cumulative --save-sampler-file flatfile --inj mdc.xml.gz 
	../../Code/make_triplot zero_noise.xml.gz --output triplot.pdf
	../../Code/plot_integral zero_noise.xml.gz -o integral.pdf

test_ile_higher: zero_noise.cache
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0  --reference-freq 0.0  --save-samples  --data-start 1000000000 --data-end 1000000016 -i 2 --time-marginalization --n-max 60000 --output-file zero_noise_higher.xml.gz --save-deltalnL 15 --save-P 0.0001  --adapt-floor-level 0.1 --adapt-weight-exponent 0.8 --n-chunk 2000 --convergence-tests-on --n-eff 100  --amp-order -1 --l-max 4
	echo " Number of samples stored in this test (<number of samples performed): " `ligolw_print -t sim_inspiral -c alpha1 zero_noise.xml.gz | wc -l `
	../../Code/convert_output_format_ile2inference zero_noise_higher.xml.gz > flatfile_higher-points.dat
	../../Code/postprocess_1d_cumulative --save-sampler-file flatfile_higher --inj mdc.xml.gz 
	../../Code/make_triplot zero_noise_higher.xml.gz --output triplot.pdf
	../../Code/plot_integral zero_noise_higher.xml.gz -o integral.pdf


test_ile_postprocessing_2ifo: zero_noise.cache
	rm -f zero_noise.pdf
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME}  --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz"  --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0  --reference-freq 0.0  --save-samples  --data-start 1000000000 --data-end 1000000016 -i 2 --time-marginalization --n-max 10000 --output-file zero_noise_2ifo.xml.gz --save-deltalnL 15 --save-P 0.0001 --convergence-tests-on --n-eff 100
	echo " Number of samples stored in this test (<number of samples performed): " `ligolw_print -t sim_inspiral -c alpha1 zero_noise.xml.gz | wc -l `
	../../Code/convert_output_format_ile2inference zero_noise.xml.gz > flatfile_2ifo-points.dat
	../../Code/postprocess_1d_cumulative --save-sampler-file flatfile_2ifo --inj mdc.xml.gz 
	../../Code/plot_integral zero_noise_2ifo.xml.gz -o integral.pdf
	../../Code/make_triplot zero_noise_2ifo.xml.gz --output triplot.pdf


zero_noise.cache: H1_zero_noise.cache L1_zero_noise.cache V1_zero_noise.cache
	cat {H,L,V}1_zero_noise.cache > zero_noise.cache

H1_zero_noise.cache:
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --injections ${INJ_XML} --channel H1=${INJ_CHANNEL_NAME} --verbose --data-source silence --output-channel ${INJ_CHANNEL_NAME} --frame-type ${INJECTION_FRAME_TYPE} --output-path ${INJECTION_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES}
	find ${INJECTION_DEST}/ -name "H-*.gwf" | lalapps_path2cache > H1_zero_noise.cache

L1_zero_noise.cache:
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --injections ${INJ_XML} --channel L1=${INJ_CHANNEL_NAME} --verbose --data-source silence --output-channel ${INJ_CHANNEL_NAME} --frame-type ${INJECTION_FRAME_TYPE} --output-path ${INJECTION_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES}
	find ${INJECTION_DEST}/ -name "L-*.gwf" | lalapps_path2cache > L1_zero_noise.cache

V1_zero_noise.cache:
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --injections ${INJ_XML} --channel V1=${INJ_CHANNEL_NAME} --verbose --data-source silence --output-channel ${INJ_CHANNEL_NAME} --frame-type ${INJECTION_FRAME_TYPE} --output-path ${INJECTION_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES}
	find ${INJECTION_DEST}/ -name "V-*.gwf" | lalapps_path2cache > V1_zero_noise.cache

clean:
	rm *.cache
	rm -rf zero_noise_mdc/

ros_setup:
# Copy frame files from CIT: not everyone has gstlal installed.
	gsiscp -r ldas-grid.ligo.caltech.edu:/home/pankow/param_est/research-projects/MonteCarloMarginalizeCode/data/zero_noise_mdc/zero_noise_mdc .
	gsiscp -r ldas-grid.ligo.caltech.edu:/home/pankow/param_est/research-projects/MonteCarloMarginalizeCode/data/HLV-ILIGO_PSD.xml.gz .
	find zero_noise_mdc -name "*.gwf" | lalapps_path2cache > zero_noise.cache

ros_test_data:
	touch H1_zero_noise.cache
	touch L1_zero_noise.cache
	touch V1_zero_noise.cache
	find ${INJECTION_DEST}/ -name "*.gwf" | lalapps_path2cache > zero_noise.cache
	python ../../Code/test_data_vs_template.py --inj mdc.xml.gz --cache zero_noise.cache  # plot data and template.  Do they agree?

ros_test_ile:
	touch H1_zero_noise.cache
	touch L1_zero_noise.cache
	touch V1_zero_noise.cache
	find ${INJECTION_DEST}/ -name "*.gwf" | lalapps_path2cache > zero_noise.cache
	python -m cProfile -s cumtime ../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=HLV-ILIGO_PSD.xml.gz" --psd-file "L1=HLV-ILIGO_PSD.xml.gz" --psd-file "V1=HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0  --data-start-time 1000000000  --data-end-time  1000000016   --reference-freq 0.0  --output-file ile-ros-output.xml.gz --save-samples  -i 2 --n-max 10000 --time-marginalization #  --seglen 10  --padding 2

ros_test_ile_pinned:
	touch H1_zero_noise.cache
	touch L1_zero_noise.cache
	touch V1_zero_noise.cache
	find ${INJECTION_DEST}/ -name "*.gwf" | lalapps_path2cache > zero_noise.cache
#	make test_ile
# Pinning test 1: time
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --reference-freq 0.0 --output-file ile-ros-output.xml.gz --save-samples  `../../Code/xmlInspiralToILEPinned.py mdc.xml.gz 0 polarization inclination  phi_orb distance right_ascension declination `
	ligolw_print -t sngl_inspiral -c snr ile-ros-output.xml.gz  > ile-ros-pintest-time-result.dat
	../../Code/convert_output_format_ile2inference ile-ros-output.xml.gz > ile-ros-pintest-time-points.dat
	../../Code/plot_like_contours --dimension1=geocent_end_time  --show-points  ile-ros-output.xml.gz  --show-mdc=mdc.xml.gz  --fig-extension pdf
	mv ile-ros-output.xml.gz ile-ros-pintest-time.xml.gz

# Pinning test 2: distance
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --reference-freq 0.0 --output-file ile-ros-output.xml.gz --save-samples  `../../Code/xmlInspiralToILEPinned.py mdc.xml.gz 0 polarization inclination  phi_orb  right_ascension declination ` --t-ref 0
	ligolw_print -t sngl_inspiral -c snr ile-ros-output.xml.gz  > ile-ros-pintest-distance-result.dat
	../../Code/convert_output_format_ile2inference ile-ros-output.xml.gz > ile-ros-pintest-distance-points.dat
	../../Code/plot_like_contours --dimension1=distance  --show-points  ile-ros-output.xml.gz  --show-mdc=mdc.xml.gz --fig-extension pdf
	mv ile-ros-output.xml.gz ile-ros-pintest-distance.xml.gz

# Pinning test 2: distance and cos iota
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --reference-freq 0.0 --output-file ile-ros-output.xml.gz --save-samples  `../../Code/xmlInspiralToILEPinned.py mdc.xml.gz 0 polarization   phi_orb  right_ascension declination ` --t-ref 0
	ligolw_print -t sngl_inspiral -c snr ile-ros-output.xml.gz  > ile-ros-pintest-DCosI-result.dat
	../../Code/convert_output_format_ile2inference ile-ros-output.xml.gz > ile-ros-pintest-DCosI-points.dat
	../../Code/plot_like_contours --dimension1=distance --dimension2=inclination  --show-points  ile-ros-output.xml.gz --show-mdc=mdc.xml.gz  --fig-extension pdf
	mv ile-ros-output.xml.gz ile-ros-pintest-DCosI.xml.gz

ros_test_likelihood:
# TEST VERSION 1: Analytic (iLIGO) PSD model
#	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --force-read-full-frames --seglen 10  --mass1 4.0 --mass2 4.0
# TEST VERSION 2: Read PSD from file
	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME}  --psd-file "HLV-ILIGO_PSD.xml.gz"  --mass1 4.0 --mass2 4.0  --Lmax 2 --amporder -1 --show-likelihood-versus-time --show-sampler-inputs --show-psd   --seglen 16 # --sampling-prior-use-skymap skymap.fits.gz   --force-read-full-frames
	open FLT*


ros_test_likelihood_pinned:
# --psd-file "HLV-ILIGO_PSD.xml.gz"  # not working immediately
	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME}  --psd-file "HLV-ILIGO_PSD.xml.gz"  --mass1 4.0 --mass2 4.0 --fref 0 --Niter 100000 --Neff 10000 --fix-rightascension --fix-declination --fix-polarization --fix-time --fix-phase  --fix-inclination --save-sampler-file output-only-distance --show-sampler-results --show-sampler-inputs --show-psd --Lmax 4 --amporder -1 --seglen 16 
	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME}  --mass1 4.0 --mass2 4.0 --fref 0 --Niter 100000 --Neff 10000 --fix-rightascension --fix-declination --fix-polarization --fix-time --fix-phase  --save-sampler-file output-only-DIota --show-sampler-results --show-sampler-inputs --show-psd --Lmax 4 --amporder -1 --seglen 16
	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME}  --mass1 4.0 --mass2 4.0 --fref 0 --Niter 100000 --Neff 3000 --fix-rightascension --fix-declination --fix-polarization --fix-distance --fix-phase  --fix-inclination --save-sampler-file output-only-time --show-sampler-results --show-sampler-inputs --show-psd --Lmax 4 --amporder -1 --seglen 16


# Warning re master: lalsimutils frutils needs tmpdir.  Fix needed to implement longer seglents robustly
ros_test_response:
	rm -f test-Q-response*  FLT-*	
	python ../../Code/test_response_functions_Q.py --inj-xml mdc.xml.gz   --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME}  --psd-file "HLV-ILIGO_PSD.xml.gz"  --mass1 4.0 --mass2 4.0 --fref 0  --show-psd  --show-likelihood-versus-time --Lmax 2 --amporder 0 --seglen 16 --padding 2
	open test-Q-response* FLT*

#	python ../../Code/test_response_functions_Q.py --inj-xml mdc.xml.gz   --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME}  --psd-file "HLV-ILIGO_PSD.xml.gz"  --mass1 4.0 --mass2 4.0 --fref 0  --show-psd  --show-likelihood-versus-time --Lmax 2 --amporder 0 --seglen 16 --padding 2 --psd-truncate-inverse --psd-truncate-inverse-time 2
	open test-Q-response* FLT*

