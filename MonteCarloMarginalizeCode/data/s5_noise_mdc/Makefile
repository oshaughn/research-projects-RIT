# from: /home/pankow/staging/seglists/L1H1H2V1/cat2/S5full-cat2-l1h1h2v1.txt
# 39670 875145998.0 875148265.0 2267.0
START=875146880
STOP=875148080
DUR=$(shell python -c "print ${STOP}-${START}")

LIGO_CHANNEL_NAME=LSC-STRAIN
VIRGO_CHANNEL_NAME=h_16384Hz
INJ_CHANNEL_NAME=FAKE-STRAIN
INJ_XML=mdc.xml.gz
INJECTION_FRAME_TYPE=s5_noise_mdc

INJECTION_DEST=s5_noise_mdc/
EVENT_TIME=

LIGO_FRAME_DURATION=64 # s
LIGO_FRAME_DURATION_COMB=64 # s
LIGO_N_FRAMES=1

PSD_FILE=/home/pankow/param_est/research-projects/MonteCarloMarginalizeCode/data/HLV-ILIGO_PSD.xml.gz

test_ile_dag: s5_noise.cache
	../../Code/compute_marginalized_likelihood --cache-file /home/pankow/param_est/research-projects/MonteCarloMarginalizeCode/data/s5_noise_mdc/s5_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=${PSD_FILE}" --psd-file "L1=${PSD_FILE}" --psd-file "V1=${PSD_FILE}" --event-time ${EVENT_TIME} --seglen 10 --padding 2.0 --mass1 4.0 --mass2 4.0 --reference-freq 0.0 --save-samples --output-file s5_noise.xml.gz

test_ile_sim_only_noise: s5_noise.cache
	../../Code/integrate_likelihood_extrinsic --cache-file s5_noise.cache --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${INJ_CHANNEL_NAME} --psd-file "H1=../HLV-ILIGO_PSD.xml.gz" --psd-file "L1=../HLV-ILIGO_PSD.xml.gz" --psd-file "V1=../HLV-ILIGO_PSD.xml.gz" --event-time ${EVENT_TIME} --mass1 1.4 --mass2 1.4 --seglen 30 --pin-to-sim mdc.xml.gz --reference-freq 0.0 --padding 2

psds: H1_PSD_measured.xml.gz L1_PSD_measured.xml.gz V1_PSD_measured.xml.gz

plots: H1_PSD_measured.png L1_PSD_measured.png V1_PSD_measured.png
	../utils/plot_psds {H,L,V}1_PSD_measured.xml.gz

s5_mdc.cache: H1_combined.cache L1_combined.cache V1_combined.cache
	cat {H,L,V}1_combined.cache > s5_mdc.cache

H1_combined.cache: s5_data.cache
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --channel H1=${LIGO_CHANNEL_NAME} --verbose --data-source frames --frame-cache s5_data.cache --output-channel ${INJ_CHANNEL_NAME} --frame-type ${INJECTION_FRAME_TYPE} --output-path ${INJECTION_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES} --injections mdc.xml.gz
	find ${INJECTION_DEST}/ -name "H-*.gwf" | lalapps_path2cache > H1_combined.cache

L1_combined.cache: s5_data.cache
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --channel L1=${LIGO_CHANNEL_NAME} --verbose --data-source frames --frame-cache s5_data.cache --output-channel ${INJ_CHANNEL_NAME} --frame-type ${INJECTION_FRAME_TYPE} --output-path ${INJECTION_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES} --injections mdc.xml.gz
	find ${INJECTION_DEST}/ -name "L-*.gwf" | lalapps_path2cache > L1_combined.cache

V1_combined.cache: s5_data.cache
	gstlal_fake_frames --gps-start-time ${START} --gps-end-time ${STOP} --channel V1=${VIRGO_CHANNEL_NAME} --verbose --data-source frames --frame-cache s5_data.cache --output-channel ${INJ_CHANNEL_NAME} --frame-type ${INJECTION_FRAME_TYPE} --output-path ${INJECTION_DEST} --frame-duration ${LIGO_FRAME_DURATION} --frames-per-file ${LIGO_N_FRAMES} --injections mdc.xml.gz
	find ${INJECTION_DEST}/ -name "V-*.gwf" | lalapps_path2cache > V1_combined.cache

H1_PSD_measured.xml.gz: s5_data.cache
	gstlal_reference_psd --gps-start-time ${START} --gps-end-time ${STOP} --channel H1=${LIGO_CHANNEL_NAME} --verbose --data-source frames --frame-cache s5_data.cache --write-psd H1_PSD_measured.xml.gz

H1_PSD_measured.png: H1_PSD_measured.xml.gz
	../utils/plot_psds_single H1 H1_PSD_measured.xml.gz ../HLV-ILIGO_PSD.xml.gz
	mv psd_single.png H1_PSD_measured.png

L1_PSD_measured.xml.gz: s5_data.cache
	gstlal_reference_psd --gps-start-time ${START} --gps-end-time ${STOP} --channel L1=${LIGO_CHANNEL_NAME} --verbose --data-source frames --frame-cache s5_data.cache --write-psd L1_PSD_measured.xml.gz

L1_PSD_measured.png: L1_PSD_measured.xml.gz
	../utils/plot_psds_single L1 L1_PSD_measured.xml.gz ../HLV-ILIGO_PSD.xml.gz
	mv psd_single.png L1_PSD_measured.png

V1_PSD_measured.xml.gz: s5_data.cache
	gstlal_reference_psd --gps-start-time ${START} --gps-end-time ${STOP} --channel V1=${VIRGO_CHANNEL_NAME} --verbose --data-source frames --frame-cache s5_data.cache --write-psd V1_PSD_measured.xml.gz

V1_PSD_measured.png: V1_PSD_measured.xml.gz
	../utils/plot_psds_single V1 V1_PSD_measured.xml.gz ../HLV-ILIGO_PSD.xml.gz
	mv psd_single.png V1_PSD_measured.png
clean:
	rm {H,L,V}1*.xml.gz
	rm -rf ${INJECTION_DEST}
	rm *.cache
	rm *.png

s5_data.cache:
	ligo_data_find --gaps -o L -l -t L1_RDS_C03_L1 -u file -s ${START} -e ${STOP} > s5_data.cache
	ligo_data_find --gaps -o H -l -t H1_RDS_C03_L1 -u file -s ${START} -e ${STOP} >> s5_data.cache
	ligo_data_find --gaps -o V -l -t HrecOnline -u file -s ${START} -e ${STOP} | sed "s| HrecOnline| V1_HrecOnline|g" >> s5_data.cache
