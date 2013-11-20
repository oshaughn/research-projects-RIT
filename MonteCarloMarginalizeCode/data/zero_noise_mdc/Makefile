START=1000000000
STOP=1000000064
DUR=$(shell python -c "print ${STOP}-${START}")

INJ_CHANNEL_NAME=FAKE-STRAIN
INJ_XML=mdc.xml.gz
INJECTION_FRAME_TYPE=zero_noise_mdc

INJECTION_DEST=zero_noise_mdc/

LIGO_FRAME_DURATION=64 # s
LIGO_FRAME_DURATION_COMB=64 # s
LIGO_N_FRAMES=1

EVENT_TIME=1000000014.236547946

plot_ts: H1_zero_noise.cache L1_zero_noise.cache V1_zero_noise.cache
	../utils/plot_ts zero_noise.cache

test_ile_dag: zero_noise.cache
	../../Code/compute_marginalized_likelihood --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --seglen 10 --mass1 4.0 --mass2 4.0

test_ile: zero_noise.cache
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --seglen 10 --pin-to-sim mdc.xml.gz --reference-freq 0.0 --padding 2

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
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --seglen 10  --reference-freq 0.0 --padding 2 --output-file ile-ros-output.xml.gz --save-samples

ros_test_ile_pinned:
	touch H1_zero_noise.cache
	touch L1_zero_noise.cache
	touch V1_zero_noise.cache
	find ${INJECTION_DEST}/ -name "*.gwf" | lalapps_path2cache > zero_noise.cache
#	make test_ile
# Pinning test 1: time
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --seglen 10  --reference-freq 0.0 --padding 2 --output-file ile-ros-output.xml.gz --save-samples  `../../Code/xmlInspiralToILEPinned.py mdc.xml.gz 0 polarization inclination  phi_orb distance right_ascension declination `
	ligolw_print -t sngl_inspiral -c snr ile-ros-output.xml.gz  > ile-ros-pintest-time-result.dat
	../../Code/convert_output_format_ile2inference ile-ros-output.xml.gz > ile-ros-pintest-time-points.dat
	../../Code/plot_like_contours --dimension1=geocent_end_time  --show-points  ile-ros-output.xml.gz  --show-mdc=mdc.xml.gz  --fig-extension pdf
	mv ile-ros-output.xml.gz ile-ros-pintest-time.xml.gz

# Pinning test 2: distance
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --seglen 10  --reference-freq 0.0 --padding 2 --output-file ile-ros-output.xml.gz --save-samples  `../../Code/xmlInspiralToILEPinned.py mdc.xml.gz 0 polarization inclination  phi_orb  right_ascension declination ` --t-ref 0
	ligolw_print -t sngl_inspiral -c snr ile-ros-output.xml.gz  > ile-ros-pintest-distance-result.dat
	../../Code/convert_output_format_ile2inference ile-ros-output.xml.gz > ile-ros-pintest-distance-points.dat
	../../Code/plot_like_contours --dimension1=distance  --show-points  ile-ros-output.xml.gz  --show-mdc=mdc.xml.gz --fig-extension pdf
	mv ile-ros-output.xml.gz ile-ros-pintest-distance.xml.gz

# Pinning test 2: distance and cos iota
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --seglen 10  --reference-freq 0.0 --padding 2 --output-file ile-ros-output.xml.gz --save-samples  `../../Code/xmlInspiralToILEPinned.py mdc.xml.gz 0 polarization   phi_orb  right_ascension declination ` --t-ref 0
	ligolw_print -t sngl_inspiral -c snr ile-ros-output.xml.gz  > ile-ros-pintest-DCosI-result.dat
	../../Code/convert_output_format_ile2inference ile-ros-output.xml.gz > ile-ros-pintest-DCosI-points.dat
	../../Code/plot_like_contours --dimension1=distance --dimension2=inclination  --show-points  ile-ros-output.xml.gz --show-mdc=mdc.xml.gz  --fig-extension pdf
	mv ile-ros-output.xml.gz ile-ros-pintest-DCosI.xml.gz

ros_test_likelihood:
# Copy frame files from CIT: not everyone has gstlal installed.
	gsiscp -r ldas-grid.ligo.caltech.edu:/home/pankow/param_est/research-projects/MonteCarloMarginalizeCode/data/zero_noise_mdc/zero_noise_mdc .
	gsiscp -r ldas-grid.ligo.caltech.edu:/home/pankow/param_est/research-projects/MonteCarloMarginalizeCode/data/HLV-ILIGO_PSD.xml.gz .
	find zero_noise_mdc -name "*.gwf" | lalapps_path2cache > zero_noise.cache
# TEST VERSION 1: Analytic (iLIGO) PSD model
#	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --force-read-full-frames --seglen 10  --mass1 4.0 --mass2 4.0
# TEST VERSION 2: Read PSD from file
	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --force-read-full-frames --psd-file "HLV-ILIGO_PSD.xml.gz"  --mass1 4.0 --mass2 4.0


ros_test_likelihood_pinned:
# --psd-file "HLV-ILIGO_PSD.xml.gz"  # not working immediately
	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --force-read-full-frames   --mass1 4.0 --mass2 4.0 --fref 0 --Niter 100000 --Neff 10000 --fix-rightascension --fix-declination --fix-polarization --fix-time --fix-phase  --fix-inclination --save-sampler-file output-only-distance --show-sampler-results --show-sampler-inputs --show-psd
	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --force-read-full-frames   --mass1 4.0 --mass2 4.0 --fref 0 --Niter 100000 --Neff 10000 --fix-rightascension --fix-declination --fix-polarization --fix-time --fix-phase  --save-sampler-file output-only-DIota --show-sampler-results --show-sampler-inputs --show-psd
	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --force-read-full-frames   --mass1 4.0 --mass2 4.0 --fref 0 --Niter 100000 --Neff 3000 --fix-rightascension --fix-declination --fix-polarization --fix-distance --fix-phase  --fix-inclination --save-sampler-file output-only-time --show-sampler-results --show-sampler-inputs --show-psd
