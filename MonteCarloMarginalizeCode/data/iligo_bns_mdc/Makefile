START=999999000
STOP=1000001000
DUR=$(shell python -c "print ${STOP}-${START}")

INJ_CHANNEL_NAME=FAKE-STRAIN
INJ_XML=mdc.xml.gz
INJECTION_FRAME_TYPE=iligo_bns_mdc
NOISE_FRAME_TYPE=iligo_noise_mdc

INJECTION_DEST=iligo_bns_mdc/
NOISE_DEST=iligo_noise_mdc/

LIGO_FRAME_DURATION=64 # s
LIGO_FRAME_DURATION_COMB=64 # s
LIGO_N_FRAMES=1

PSD_FILE=/home/pankow/param_est/research-projects/MonteCarloMarginalizeCode/data/HLV-ILIGO_PSD.xml.gz

EVENT_TIME=1000000034.236547946

plot_ts: iligo_bns_combined.cache
	../utils/plot_ts iligo_bns_combined.cache

plot_noise_ts: iligo_noise.cache
	../utils/plot_ts iligo_noise.cache

plots: H1_PSD_measured.png L1_PSD_measured.png V1_PSD_measured.png H1_PSD_signal_measured.png L1_PSD_signal_measured.png V1_PSD_signal_measured.png
	../utils/plot_psds {H,L,V}1_PSD_measured.xml.gz

test_ile_dag: iligo_bns.cache
	../../Code/compute_marginalized_likelihood --cache-file /home/pankow/param_est/research-projects/MonteCarloMarginalizeCode/data/iligo_bns_mdc/iligo_bns.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME} --mass1 1.4 --mass2 1.4 --reference-freq 0.0 --save-samples --output-file iligo_bns.xml.gz --n-copies 10

test_ile_sim_zero_noise: iligo_bns.cache
	../../Code/integrate_likelihood_extrinsic --cache-file iligo_bns.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 1.4 --mass2 1.4 --pin-to-sim mdc.xml.gz --reference-freq 0.0

test_ile_sim_with_noise: iligo_bns_combined.cache
	../../Code/integrate_likelihood_extrinsic --cache-file iligo_bns_combined.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 1.4 --mass2 1.4 --pin-to-sim mdc.xml.gz --reference-freq 0.0

test_ile_sim_only_noise: iligo_noise.cache
	../../Code/integrate_likelihood_extrinsic --cache-file iligo_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 1.4 --mass2 1.4 --pin-to-sim mdc.xml.gz --reference-freq 0.0

iligo_bns.cache: H1_iligo_bns.cache L1_iligo_bns.cache V1_iligo_bns.cache
	cat {H,L,V}1_iligo_bns.cache > iligo_bns.cache

H1_iligo_bns.cache:
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --injections ${INJ_XML} --channel H1=${INJ_CHANNEL_NAME} --verbose --data-source silence --output-channel ${INJ_CHANNEL_NAME} --frame-type H1_${INJECTION_FRAME_TYPE} --output-path ${INJECTION_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES}
	find ${INJECTION_DEST}/ -name "H-*.gwf" | lalapps_path2cache > H1_iligo_bns.cache

L1_iligo_bns.cache:
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --injections ${INJ_XML} --channel L1=${INJ_CHANNEL_NAME} --verbose --data-source silence --output-channel ${INJ_CHANNEL_NAME} --frame-type L1_${INJECTION_FRAME_TYPE} --output-path ${INJECTION_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES}
	find ${INJECTION_DEST}/ -name "L-*.gwf" | lalapps_path2cache > L1_iligo_bns.cache

V1_iligo_bns.cache:
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --injections ${INJ_XML} --channel V1=${INJ_CHANNEL_NAME} --verbose --data-source silence --output-channel ${INJ_CHANNEL_NAME} --frame-type V1_${INJECTION_FRAME_TYPE} --output-path ${INJECTION_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES}
	find ${INJECTION_DEST}/ -name "V-*.gwf" | lalapps_path2cache > V1_iligo_bns.cache

H1_iligo_noise.cache:
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --channel H1=${INJ_CHANNEL_NAME} --verbose --data-source LIGO --output-channel ${INJ_CHANNEL_NAME} --frame-type H1_${NOISE_FRAME_TYPE} --output-path ${NOISE_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES}
	find ${NOISE_DEST}/ -name "H-*.gwf" | lalapps_path2cache > H1_iligo_noise.cache

L1_iligo_noise.cache:
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --channel L1=${INJ_CHANNEL_NAME} --verbose --data-source LIGO --output-channel ${INJ_CHANNEL_NAME} --frame-type L1_${NOISE_FRAME_TYPE} --output-path ${NOISE_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES}
	find ${NOISE_DEST}/ -name "L-*.gwf" | lalapps_path2cache > L1_iligo_noise.cache

V1_iligo_noise.cache:
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --channel V1=${INJ_CHANNEL_NAME} --verbose --data-source LIGO --output-channel ${INJ_CHANNEL_NAME} --frame-type V1_${NOISE_FRAME_TYPE} --output-path ${NOISE_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES}
	find ${NOISE_DEST}/ -name "V-*.gwf" | lalapps_path2cache > V1_iligo_noise.cache

H1_combined.cache: H1_iligo_noise.cache H1_iligo_bns.cache
	../utils/add_frames H1_iligo_noise.cache H1_iligo_bns.cache H1
	mkdir -p iligo_bns_combined/
	mv H-*.gwf iligo_bns_combined
	find iligo_bns_combined/ -name "H-*.gwf" | lalapps_path2cache > H1_combined.cache

L1_combined.cache: L1_iligo_noise.cache L1_iligo_bns.cache
	../utils/add_frames L1_iligo_noise.cache L1_iligo_bns.cache L1
	mkdir -p iligo_bns_combined/
	mv L-*.gwf iligo_bns_combined
	find iligo_bns_combined/ -name "L-*.gwf" | lalapps_path2cache > L1_combined.cache

V1_combined.cache: V1_iligo_noise.cache V1_iligo_bns.cache
	../utils/add_frames V1_iligo_noise.cache V1_iligo_bns.cache V1
	mkdir -p iligo_bns_combined/
	mv V-*.gwf iligo_bns_combined
	find iligo_bns_combined/ -name "V-*.gwf" | lalapps_path2cache > V1_combined.cache

iligo_bns_combined.cache: H1_combined.cache L1_combined.cache V1_combined.cache
	cat {H,L,V}1_combined.cache > iligo_bns_combined.cache

iligo_noise.cache: H1_combined.cache L1_combined.cache V1_combined.cache
	cat {H,L,V}1_iligo_noise.cache > iligo_noise.cache

H1_PSD_signal_measured.xml.gz: H1_combined.cache
	gstlal_reference_psd --gps-start-time ${START} --gps-end-time ${STOP} --channel H1=${INJ_CHANNEL_NAME} --verbose --data-source frames --frame-cache H1_combined.cache --write-psd H1_PSD_signal_measured.xml.gz

H1_PSD_signal_measured.png: H1_PSD_signal_measured.xml.gz
	../utils/plot_psds_single H1 H1_PSD_measured.xml.gz H1_PSD_signal_measured.xml.gz ../HLV-ILIGO_PSD.xml.gz
	mv psd_single.png H1_PSD_signal_measured.png

L1_PSD_signal_measured.xml.gz: L1_combined.cache
	gstlal_reference_psd --gps-start-time ${START} --gps-end-time ${STOP} --channel L1=${INJ_CHANNEL_NAME} --verbose --data-source frames --frame-cache L1_combined.cache --write-psd L1_PSD_signal_measured.xml.gz

L1_PSD_signal_measured.png: L1_PSD_signal_measured.xml.gz
	../utils/plot_psds_single L1 L1_PSD_measured.xml.gz L1_PSD_signal_measured.xml.gz ../HLV-ILIGO_PSD.xml.gz
	mv psd_single.png L1_PSD_signal_measured.png

V1_PSD_signal_measured.xml.gz: V1_combined.cache
	gstlal_reference_psd --gps-start-time ${START} --gps-end-time ${STOP} --channel V1=${INJ_CHANNEL_NAME} --verbose --data-source frames --frame-cache V1_combined.cache --write-psd V1_PSD_signal_measured.xml.gz

V1_PSD_signal_measured.png: V1_PSD_signal_measured.xml.gz
	../utils/plot_psds_single V1 V1_PSD_measured.xml.gz V1_PSD_signal_measured.xml.gz ../HLV-ILIGO_PSD.xml.gz
	mv psd_single.png V1_PSD_signal_measured.png

H1_PSD_measured.xml.gz: H1_iligo_noise.cache
	gstlal_reference_psd --gps-start-time ${START} --gps-end-time ${STOP} --channel H1=${INJ_CHANNEL_NAME} --verbose --data-source frames --frame-cache H1_iligo_noise.cache --write-psd H1_PSD_measured.xml.gz

H1_PSD_measured.png: H1_PSD_measured.xml.gz
	../utils/plot_psds_single H1 H1_PSD_measured.xml.gz ../HLV-ILIGO_PSD.xml.gz
	mv psd_single.png H1_PSD_measured.png

L1_PSD_measured.xml.gz: L1_iligo_noise.cache
	gstlal_reference_psd --gps-start-time ${START} --gps-end-time ${STOP} --channel L1=${INJ_CHANNEL_NAME} --verbose --data-source frames --frame-cache L1_iligo_noise.cache --write-psd L1_PSD_measured.xml.gz

L1_PSD_measured.png: L1_PSD_measured.xml.gz
	../utils/plot_psds_single L1 L1_PSD_measured.xml.gz ../HLV-ILIGO_PSD.xml.gz
	mv psd_single.png L1_PSD_measured.png

V1_PSD_measured.xml.gz: V1_iligo_noise.cache
	gstlal_reference_psd --gps-start-time ${START} --gps-end-time ${STOP} --channel V1=${INJ_CHANNEL_NAME} --verbose --data-source frames --frame-cache V1_iligo_noise.cache --write-psd V1_PSD_measured.xml.gz

V1_PSD_measured.png: V1_PSD_measured.xml.gz
	../utils/plot_psds_single V1 V1_PSD_measured.xml.gz ../HLV-ILIGO_PSD.xml.gz
	mv psd_single.png V1_PSD_measured.png

clean:
	rm *.cache
	rm -rf iligo_bns_mdc/

ros_test_ile:
	touch H1_iligo_bns.cache
	touch L1_iligo_bns.cache
	touch V1_iligo_bns.cache
	find ${INJECTION_DEST}/ -name "*.gwf" | lalapps_path2cache > iligo_bns.cache
#	make test_ile
# Pinning test 1: time
	../../Code/integrate_likelihood_extrinsic --cache-file iligo_bns.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --seglen 10  --reference-freq 0.0 --padding 2 --output-file ile-ros-output.xml.gz --save-samples  `../../Code/xmlInspiralToILEPinned.py mdc.xml.gz 0 polarization inclination  phi_orb distance right_ascension declination `
	ligolw_print -t sngl_inspiral -c snr ile-ros-output.xml.gz  > ile-ros-pintest-time-result.dat
	../../Code/convert_output_format_ile2inference ile-ros-output.xml.gz > ile-ros-pintest-time-points.dat

# Pinning test 2: distance
	../../Code/integrate_likelihood_extrinsic --cache-file iligo_bns.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --seglen 10  --reference-freq 0.0 --padding 2 --output-file ile-ros-output.xml.gz --save-samples  `../../Code/xmlInspiralToILEPinned.py mdc.xml.gz 0 polarization inclination  phi_orb  right_ascension declination ` --t-ref 0
	ligolw_print -t sngl_inspiral -c snr ile-ros-output.xml.gz  > ile-ros-pintest-distance-result.dat
	../../Code/convert_output_format_ile2inference ile-ros-output.xml.gz > ile-ros-pintest-distance-points.dat

# Pinning test 2: distance and cos iota
	../../Code/integrate_likelihood_extrinsic --cache-file iligo_bns.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --seglen 10  --reference-freq 0.0 --padding 2 --output-file ile-ros-output.xml.gz --save-samples  `../../Code/xmlInspiralToILEPinned.py mdc.xml.gz 0 polarization   phi_orb  right_ascension declination ` --t-ref 0
	ligolw_print -t sngl_inspiral -c snr ile-ros-output.xml.gz  > ile-ros-pintest-DCosI-result.dat
	../../Code/convert_output_format_ile2inference ile-ros-output.xml.gz > ile-ros-pintest-DCosI-points.dat

ros_test_likelihood:
# Copy frame files from CIT: not everyone has gstlal installed.
	gsiscp -r ldas-grid.ligo.caltech.edu:/home/pankow/param_est/research-projects/MonteCarloMarginalizeCode/data/iligo_bns_mdc/iligo_bns_mdc .
	gsiscp -r ldas-grid.ligo.caltech.edu:/home/pankow/param_est/research-projects/MonteCarloMarginalizeCode/data/HLV-ILIGO_PSD.xml.gz .
	find iligo_bns_mdc -name "*.gwf" | lalapps_path2cache > iligo_bns.cache
# TEST VERSION 1: Analytic (iLIGO) PSD model
#	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file iligo_bns.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --force-read-full-frames --seglen 10  --mass1 4.0 --mass2 4.0
# TEST VERSION 2: Read PSD from file
	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file iligo_bns.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --force-read-full-frames --psd-file "HLV-ILIGO_PSD.xml.gz"  --mass1 4.0 --mass2 4.0


ros_test_likelihood_pinned:
# --psd-file "HLV-ILIGO_PSD.xml.gz"  # not working immediately
	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file iligo_bns.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --force-read-full-frames   --mass1 4.0 --mass2 4.0 --fref 0 --Niter 100000 --Neff 10000 --fix-rightascension --fix-declination --fix-polarization --fix-time --fix-phase  --fix-inclination --save-sampler-file output-only-distance --show-sampler-results --show-sampler-inputs --show-psd
	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file iligo_bns.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --force-read-full-frames   --mass1 4.0 --mass2 4.0 --fref 0 --Niter 100000 --Neff 10000 --fix-rightascension --fix-declination --fix-polarization --fix-time --fix-phase  --save-sampler-file output-only-DIota --show-sampler-results --show-sampler-inputs --show-psd
	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file iligo_bns.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --force-read-full-frames   --mass1 4.0 --mass2 4.0 --fref 0 --Niter 100000 --Neff 3000 --fix-rightascension --fix-declination --fix-polarization --fix-distance --fix-phase  --fix-inclination --save-sampler-file output-only-time --show-sampler-results --show-sampler-inputs --show-psd
