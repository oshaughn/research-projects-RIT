START=999999900
STOP=1000000100
DUR=$(shell python -c "print ${STOP}-${START}")

INJ_CHANNEL_NAME=FAKE-STRAIN
INJ_XML=mdc.xml.gz
INJECTION_FRAME_TYPE=zero_noise_mdc

INJECTION_DEST=zero_noise_mdc/

LIGO_FRAME_DURATION=64 # s
LIGO_FRAME_DURATION_COMB=64 # s
LIGO_N_FRAMES=1

BASE_DIR=$(shell pwd)/
PSD_FILE=${BASE_DIR}/HLV-ILIGO_PSD.xml.gz


EVENT_TIME=1000000014.236547946

# Make the data
zero_noise.cache: H1_zero_noise.cache L1_zero_noise.cache V1_zero_noise.cache
	cat H1_zero_noise.cache L1_zero_noise.cache V1_zero_noise.cache > zero_noise.cache

H1_zero_noise.cache:
	mkdir ${INJECTION_DEST}; exit 0
	(cd ${INJECTION_DEST};	util_LALWriteFrame.py --start ${START} --stop ${STOP} --inj ../${INJ_XML} --event 0  --single-ifo --instrument H1 --seglen 64)
	find ${INJECTION_DEST}/ -name "H-*.gwf" | lalapps_path2cache > H1_zero_noise.cache

L1_zero_noise.cache:
	mkdir ${INJECTION_DEST}; exit 0
	(cd ${INJECTION_DEST};	util_LALWriteFrame.py --start ${START} --stop ${STOP} --inj ../${INJ_XML} --event 0  --single-ifo --instrument L1 --seglen 64)
	find ${INJECTION_DEST}/ -name "L-*.gwf" | lalapps_path2cache > L1_zero_noise.cache

V1_zero_noise.cache:
	mkdir ${INJECTION_DEST}; exit 0
	(cd ${INJECTION_DEST};	util_LALWriteFrame.py --start ${START} --stop ${STOP} --inj ../${INJ_XML} --event 0  --single-ifo --instrument V1 --seglen 64)
	find ${INJECTION_DEST}/ -name "V-*.gwf" | lalapps_path2cache > V1_zero_noise.cache

clean:
	rm *.cache
	rm -rf zero_noise_mdc/
	rm -f ts.*

# Plot the data you are using.
plot_ts: zero_noise.cache
	../utils/plot_ts zero_noise.cache

# Create a full DAG with many mass points, using the automated gridding code
test_ile_dag: zero_noise.cache
	../../Code/compute_marginalized_likelihood --cache-file ${BASE_DIR}/zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --reference-freq 0.0 --save-samples --output-file zero_noise.xml.gz --n-copies 2 --time-marginalization --save-P 0.001 --adapt-floor-level 0.1 --adapt-weight-exponent 0.8 --n-chunk 2000 --n-eff 200

# Evaluate the likelihood by MC integral on *one* mass point.  Make a bunch of plots to followup
test_ile_postprocessing: zero_noise.cache
	rm -f zero_noise.pdf
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0  --reference-freq 0.0  --save-samples  --data-start 1000000000 --data-end 1000000016 -i 2 --time-marginalization --n-max 60000 --output-file zero_noise.xml.gz --save-deltalnL 15 --save-P 0.0001  --adapt-floor-level 0.1 --adapt-weight-exponent 0.8 --n-chunk 2000  --n-eff 100  --d-max 500  # --adapt-adapt  --convergence-tests-on
	echo " Number of samples stored in this test (<number of samples performed): " `ligolw_print -t sim_inspiral -c alpha1 zero_noise.xml.gz | wc -l `
	../../Code/convert_output_format_ile2inference zero_noise.xml.gz > flatfile-points.dat
	../../Code/postprocess_1d_cumulative --save-sampler-file flatfile --inj mdc.xml.gz 
	../../Code/make_triplot zero_noise.xml.gz --output triplot.pdf
	../../Code/plot_integral zero_noise.xml.gz -o integral.pdf


# As above, but fix one or more of the parameters
test_ile_pinned: zero_noise.cache
# Pinning test 1: time
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --reference-freq 0.0 --output-file ile-ros-output.xml.gz --save-samples  `../../Code/xmlInspiralToILEPinned.py mdc.xml.gz 0 polarization inclination  phi_orb distance right_ascension declination `
	ligolw_print -t sngl_inspiral -c snr ile-ros-output.xml.gz  > ile-ros-pintest-time-result.dat
	../../Code/convert_output_format_ile2inference ile-ros-output.xml.gz > ile-ros-pintest-time-points.dat
	../../Code/plot_like_contours --dimension1=geocent_end_time  --show-points  ile-ros-output.xml.gz  --show-mdc=mdc.xml.gz  --fig-extension pdf
	mv ile-ros-output.xml.gz ile-ros-pintest-time.xml.gz

# Pinning test 2: distance
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --reference-freq 0.0 --output-file ile-ros-output.xml.gz --save-samples  `../../Code/xmlInspiralToILEPinned.py mdc.xml.gz 0 polarization inclination  phi_orb  right_ascension declination ` --t-ref 0
	ligolw_print -t sngl_inspiral -c snr ile-ros-output.xml.gz  > ile-ros-pintest-distance-result.dat
	../../Code/convert_output_format_ile2inference ile-ros-output.xml.gz > ile-ros-pintest-distance-points.dat
	../../Code/plot_like_contours --dimension1=distance  --show-points  ile-ros-output.xml.gz  --show-mdc=mdc.xml.gz --fig-extension pdf
	mv ile-ros-output.xml.gz ile-ros-pintest-distance.xml.gz

# Pinning test 2: distance and cos iota
	../../Code/integrate_likelihood_extrinsic --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --reference-freq 0.0 --output-file ile-ros-output.xml.gz --save-samples  `../../Code/xmlInspiralToILEPinned.py mdc.xml.gz 0 polarization   phi_orb  right_ascension declination ` --t-ref 0
	ligolw_print -t sngl_inspiral -c snr ile-ros-output.xml.gz  > ile-ros-pintest-DCosI-result.dat
	../../Code/convert_output_format_ile2inference ile-ros-output.xml.gz > ile-ros-pintest-DCosI-points.dat
	../../Code/plot_like_contours --dimension1=distance --dimension2=inclination  --show-points  ile-ros-output.xml.gz --show-mdc=mdc.xml.gz  --fig-extension pdf
	mv ile-ros-output.xml.gz ile-ros-pintest-DCosI.xml.gz

