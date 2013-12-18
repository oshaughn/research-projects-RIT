START=999999000
STOP=1000001000
DUR=$(shell python -c "print ${STOP}-${START}")

INJ_CHANNEL_NAME=FAKE-STRAIN
INJ_XML=mdc.xml.gz
INJECTION_FRAME_TYPE=iligo_noise_mdc

INJECTION_DEST=iligo_noise_mdc/

LIGO_FRAME_DURATION=64 # s
LIGO_FRAME_DURATION_COMB=64 # s
LIGO_N_FRAMES=1

BASE_DIR=$(shell pwd)/
PSD_FILE=${BASE_DIR}/../HLV-ILIGO_PSD.xml.gz

EVENT_TIME=1000000014.236547946
#DATA_START_TIME=$(shell python -c "print ${EVENT_TIME}-20")
#DATA_END_TIME=$(shell python -c "print ${EVENT_TIME}+20")

test_ile_dag: iligo_noise.cache
	../../Code/compute_marginalized_likelihood --cache-file ${BASE_DIR}/iligo_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --reference-freq 0.0 --save-samples --output-file iligo_noise.xml.gz --n-copies 10

test_like_and_samp: iligo_noise.cache
	../../Code/test_like_and_samp.py --cache-file iligo_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file ../HLV-ILIGO_PSD.xml.gz --mass1 4.0 --mass2 4.0 --seglen 10 --inj-xml mdc.xml.gz

test_ile: iligo_noise.cache
	../../Code/integrate_likelihood_extrinsic --cache-file iligo_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --pin-to-sim mdc.xml.gz --reference-freq 0.0 

iligo_noise.cache: H1_iligo_noise.cache L1_iligo_noise.cache V1_iligo_noise.cache
	cat {H,L,V}1_iligo_noise.cache > iligo_noise.cache

psds: H1_PSD_measured.xml.gz L1_PSD_measured.xml.gz V1_PSD_measured.xml.gz

plots: H1_PSD_measured.png L1_PSD_measured.png V1_PSD_measured.png
	../utils/plot_psds {H,L,V}1_PSD_measured.xml.gz

plot_ts: H1_iligo_noise.cache L1_iligo_noise.cache V1_iligo_noise.cache
	../utils/plot_ts iligo_noise.cache 

H1_iligo_noise.cache:
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --injections ${INJ_XML} --channel H1=${INJ_CHANNEL_NAME} --verbose --data-source LIGO --output-channel ${INJ_CHANNEL_NAME} --frame-type H1_${INJECTION_FRAME_TYPE} --output-path ${INJECTION_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES}
	find ${INJECTION_DEST}/ -name "H-*.gwf" | lalapps_path2cache > H1_iligo_noise.cache

L1_iligo_noise.cache:
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --injections ${INJ_XML} --channel L1=${INJ_CHANNEL_NAME} --verbose --data-source LIGO --output-channel ${INJ_CHANNEL_NAME} --frame-type L1_${INJECTION_FRAME_TYPE} --output-path ${INJECTION_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES}
	find ${INJECTION_DEST}/ -name "L-*.gwf" | lalapps_path2cache > L1_iligo_noise.cache

V1_iligo_noise.cache:
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --injections ${INJ_XML} --channel V1=${INJ_CHANNEL_NAME} --verbose --data-source LIGO --output-channel ${INJ_CHANNEL_NAME} --frame-type V1_${INJECTION_FRAME_TYPE} --output-path ${INJECTION_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES}
	find ${INJECTION_DEST}/ -name "V-*.gwf" | lalapps_path2cache > V1_iligo_noise.cache

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
	rm -rf ${INJECTION_DEST}
	rm *.cache
	rm *.png


###
### ROS tests
###

ros_setup:
# Copy frame files from CIT: not everyone has gstlal installed.
	gsiscp -r ldas-grid.ligo.caltech.edu:/home/pankow/param_est/research-projects/MonteCarloMarginalizeCode/data/iligo_noise_mdc/iligo_noise_mdc .
	gsiscp -r ldas-grid.ligo.caltech.edu:/home/pankow/param_est/research-projects/MonteCarloMarginalizeCode/data/HLV-ILIGO_PSD.xml.gz .
	find iligo_noise_mdc -name "*.gwf" | lalapps_path2cache > iligo_noise.cache


ros_test_ile:
	../../Code/integrate_likelihood_extrinsic --cache-file iligo_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=HLV-ILIGO_PSD.xml.gz" --psd-file "L1=HLV-ILIGO_PSD.xml.gz" --psd-file "V1=HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 4.0 --mass2 4.0 --data-start-time 1000000000  --data-end-time  1000000016   --reference-freq 0.0  --output-file ile-ros-output.xml.gz --save-samples

ros_test_likelihood:
	python ../../Code/test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file iligo_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME}  --psd-file "HLV-ILIGO_PSD.xml.gz"  --mass1 4.0 --mass2 4.0  --Lmax 2 --amporder -1 --show-likelihood-versus-time --show-sampler-inputs --show-psd   --seglen 16 # --sampling-prior-use-skymap skymap.fits.gz  --force-read-full-frames


ros_test_response:
	rm -f test-Q-response*  FLT-*
	python ../../Code/test_response_functions_Q.py --inj-xml mdc.xml.gz   --cache-file iligo_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME}  --psd-file "HLV-ILIGO_PSD.xml.gz"  --mass1 4.0 --mass2 4.0 --fref 0  --show-psd  --show-likelihood-versus-time --Lmax 2 --amporder 0 --seglen 16 --padding 2 --psd-truncate-inverse --psd-truncate-inverse-time 2
	open test-Q-response* FLT*
