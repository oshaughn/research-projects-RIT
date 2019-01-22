START=999999400
STOP=1000000600
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
ALIGO_PSD_FILE=${BASE_DIR}/HLV-aLIGO_PSD.xml.gz


EVENT_TIME=1000000014.236547946

# Make the data
zero_noise.cache: mdc.xml.gz
	util_WriteFrameAndCacheFromXML.sh mdc.xml.gz 0 zero_noise

bns_frames.cache: mdc-bns.xml.gz
	util_WriteFrameAndCacheFromXML.sh mdc-bns.xml.gz 0 bns_frames 

# appropriate for aLIGO
bns_frames_lowf.cache: mdc-bns.xml.gz
	util_WriteFrameAndCacheFromXML.sh mdc-bns-aligo.xml.gz 0 bns_frames_lowf


bns_eob_frames.cache: mdc-bns.xml.gz
	util_WriteFrameAndCacheFromXML.sh mdc-bns.xml.gz 0 bns_eob_frames MATLAB


# H1_zero_noise.cache:
# 	mkdir ${INJECTION_DEST}; exit 0
# 	(cd ${INJECTION_DEST};	util_LALWriteFrame.py --start ${START} --stop ${STOP} --inj ../${INJ_XML} --event 0  --single-ifo --instrument H1 --seglen 64)
# 	find ${INJECTION_DEST}/ -name "H-*.gwf" | lalapps_path2cache > H1_zero_noise.cache
# L1_zero_noise.cache:
# 	mkdir ${INJECTION_DEST}; exit 0
# 	(cd ${INJECTION_DEST};	util_LALWriteFrame.py --start ${START} --stop ${STOP} --inj ../${INJ_XML} --event 0  --single-ifo --instrument L1 --seglen 64)
# 	find ${INJECTION_DEST}/ -name "L-*.gwf" | lalapps_path2cache > L1_zero_noise.cache
# V1_zero_noise.cache:
# 	mkdir ${INJECTION_DEST}; exit 0
# 	(cd ${INJECTION_DEST};	util_LALWriteFrame.py --start ${START} --stop ${STOP} --inj ../${INJ_XML} --event 0  --single-ifo --instrument V1 --seglen 64)
# 	find ${INJECTION_DEST}/ -name "V-*.gwf" | lalapps_path2cache > V1_zero_noise.cache

clean:
	rm *.cache
	rm -rf zero_noise_mdc/
	rm -f ts.*

# Plot the data you are using.
plot_ts: zero_noise.cache
	../utils/plot_ts zero_noise.cache

# Create a full DAG with many mass points, using the automated gridding code
test_ile_dag: zero_noise.cache
	../../Code/compute_marginalized_likelihood --cache-file ${BASE_DIR}/zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --reference-freq 0.0 --save-samples --output-file zero_noise.xml.gz --n-copies 2 --time-marginalization --save-P 0.01 --adapt-floor-level 0.1 --adapt-weight-exponent 0.8 --n-chunk 2000 --n-eff 200 --d-max 500 --linear-spoked --approx EOBNRv2HM

# Create a DAG using the grid code
test_ile_dag_grid: zero_noise.cache
	# create a 1d grid example (in eta)
#	util_ManualOverlapGrid.py  --mass1 4 --mass2 4 --lambda1 0 --lambda2 0  --parameter 'eta' --parameter-range '[0.24,0.25]' --verbose --grid-cartesian-npts 10 #-inj mdc.xml.gz  --event 0
	# Run the code	
	create_event_dag_via_grid --sim-xml ${BASE_DIR}/overlap-grid.xml.gz --cache-file ${BASE_DIR}/zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME}  --reference-freq 0.0 --save-samples --output-file zero_noise.xml.gz --n-copies 2 --time-marginalization --save-P 0.01 --adapt-floor-level 0.1 --adapt-weight-exponent 0.8 --n-chunk 2000 --n-eff 200 --d-max 500 


test_ile_intrinsic:  #zero_noise.cache
	rm -f zero_noise.pdf
	../../Code/integrate_likelihood_extrinsic  --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME} --mass1 30.0 --mass2 29.0 --fmin 30 --fmin-template 30  --reference-freq 0.0  --save-samples  --n-max 60000 --output-file zero_noise.xml.gz --save-deltalnL 15 --save-P 0.0001  --time-marginalization  --adapt-floor-level 0.1 --adapt-weight-exponent 0.8 --n-chunk 2000  --n-eff 100  --d-max 500  --time-marginalization  --rom-group tutorial/TutorialSurrogate --rom-param /EOB_q1_2_NoSpin_Mode22/ --rom-use-basis --rom-integrate-intrinsic --parameter 'q' --parameter-range '[0.9,1]'    # --adapt-adapt  --convergence-tests-on

# Evaluate the likelihood by MC integral on *one* mass point specified by the command line.  Make a bunch of plots to followup
test_ile_postprocessing:  #zero_noise.cache
	rm -f zero_noise.pdf
	../../Code/integrate_likelihood_extrinsic --maximize-only --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0  --reference-freq 0.0  --save-samples  --n-max 60000 --output-file zero_noise.xml.gz --save-deltalnL 15 --save-P 0.0001  --time-marginalization --adapt-floor-level 0.1 --adapt-weight-exponent 0.8 --n-chunk 2000  --n-eff 100  --d-max 500  --declination-cosine-sampler --inclination-cosine-sampler --no-adapt-distance  --vectorized # --adapt-adapt  --convergence-tests-on
#	echo " Number of samples stored in this test (<number of samples performed): " `ligolw_print -t sim_inspiral -c alpha1 zero_noise.xml.gz | wc -l `
#	../../Code/convert_output_format_ile2inference zero_noise.xml.gz > flatfile-points.dat
#	../../Code/postprocess_1d_cumulative --save-sampler-file flatfile --inj mdc.xml.gz 
#	../../Code/make_triplot zero_noise.xml.gz --output triplot.pdf
#	../../Code/plot_integral zero_noise.xml.gz -o integral.pdf

# As above, but specify the point using the XML file
test_ile_postprocessing_grid: zero_noise.cache
	rm -f zero_noise.pdf
	../../Code/integrate_likelihood_extrinsic --sim-xml mdc.xml.gz --event 0 --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME}  --reference-freq 0.0  --save-samples  --time-marginalization --n-max 60000 --output-file zero_noise.xml.gz --save-deltalnL 15 --save-P 0.0001  --adapt-floor-level 0.1 --adapt-weight-exponent 0.8 --n-chunk 2000  --n-eff 100  --d-max 500  --approx EOBNRv2 # --adapt-adapt  --convergence-tests-on
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

ros_test_likelihood: zero_noise.cache
# TEST VERSION 1: Analytic (iLIGO) PSD model
	test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file zero_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME}  --psd-file "HLV-ILIGO_PSD.xml.gz"  --mass1 4 --mass2 4  --Lmax 2 --amporder -1 --show-likelihood-versus-time  --show-input-h  --show-psd  --d-max 500  # --sampling-prior-use-skymap skymap.fits.gz   --force-read-full-frames

ros_test_likelihood_aligo: bns_frames_lowf.cache
# TEST VERSION 1: Analytic (aLIGO) PSD model, with source scaled to sensible low frequency
	test_like_and_samp.py --inj-xml mdc-bns-aligo.xml.gz   --cache-file bns_frames_lowf.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME}  --psd-file "HLV-aLIGO_PSD.xml.gz"  --mass1 1.35 --mass2 1.35  --Lmax 2 --amporder -1 --show-likelihood-versus-time  --d-max 500 --seglen 1024  # --sampling-prior-use-skymap skymap.fits.gz   --force-read-full-frames


# Confirm EOB interface works
ros_test_likelihood_eob: 
# EOB with test_like_and_samp.py, both as signal and as template
	../../Code/test_like_and_samp.py --signal-fmin 70 --fmin-template 70 --mass1 1.35 --mass2 1.35 --signal-mass1 1.35 --signal-mass2 1.35 --eff-lambda 590   --Lmax 2 --amporder -1 --use-external-EOB --show-likelihood-versus-time  --show-input-h   --seglen 128 --d-max 500 --no-interactive-plots


ros_test_likelihood_eob_direct: bns_frames.cache
# EOB with test_like_and_samp.py, as template, using frame files as signal
	../../Code/test_like_and_samp.py  --inj mdc-bns.xml.gz --cache-file bns_frames.cache --eff-lambda 500  --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME}  --fmin-template 70   --Lmax 2 --amporder -1 --use-external-EOB --show-likelihood-versus-time  --show-input-h --seglen 128 --d-max 25 --Nskip 10000 --convergence-tests-on


ros_test_ile_eob: bns_frames.cache
	../../Code/integrate_likelihood_extrinsic --fmin-template 40 --mass1 1.35 --mass2 1.35  --eff-lambda 500 --cache-file bns_frames.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME}  --reference-freq 0.0  --save-samples  --time-marginalization --n-max 60000 --output-file zero_noise.xml.gz --save-deltalnL 15 --save-P 0.0001  --adapt-floor-level 0.1 --adapt-weight-exponent 0.8 --n-chunk 10000  --n-eff 100  --d-max 25 --use-external-EOB --convergence-tests-on   # --adapt-adapt  --convergence-tests-on


bns_grid.xml.gz:
	util_ManualOverlapGrid.py  --mass1 1.4 --mass2 1.4   --parameter 'eta' --parameter-range '[0.24,0.25]' --grid-cartesian-npts 3  --inj mdc-bns.xml.gz  --event 0
	mv overlap-grid.xml.gz bns_grid.xml.gz

ros_test_ile_eob_dag: bns_frames.cache bns_grid.xml.gz
	create_event_dag_via_grid --sim-xml ${BASE_DIR}/bns_grid.xml.gz  --fmin-template 10  --cache-file bns_frames.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME}  --reference-freq 0.0  --save-samples  --time-marginalization --n-max 60000 --output-file zero_noise.xml.gz --save-deltalnL 15 --save-P 0.0001 --n-chunk 10000  --n-eff 100  --d-max 25 --use-external-EOB --convergence-tests-on     --adapt-adapt 


ros_test_ile_eob_aligo: bns_frames_lowf.cache
	../../Code/integrate_likelihood_extrinsic --fmin-template 35 --mass1 1.35 --mass2 1.35  --eff-lambda 500 --cache-file bns_frames_lowf.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${ALIGO_PSD_FILE}" --psd-file "L1=${ALIGO_PSD_FILE}" --psd-file "V1=${ALIGO_PSD_FILE}" --event-time ${EVENT_TIME}  --reference-freq 0.0  --save-samples  --time-marginalization --n-max 60000 --output-file zero_noise.xml.gz --save-deltalnL 15 --save-P 0.0001  --adapt-floor-level 0.1 --adapt-weight-exponent 0.8 --n-chunk 10000  --n-eff 100  --d-max 25 --use-external-EOB --convergence-tests-on   # --adapt-adapt  --convergence-tests-on



