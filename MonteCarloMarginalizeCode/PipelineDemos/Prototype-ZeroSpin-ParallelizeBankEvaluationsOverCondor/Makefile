
TBUFFER=1024
PSD_GPS_START=1063119021
PSD_GPS_END=$(shell "echo ${PSD_GPS_START} -1024 | bc -l")


setup:
# Retrieve Pankow 'blind injection challenge'
	gsiscp -r ldas-grid.ligo.caltech.edu:/home/pankow/richard_mdc/signal_hoft .
	gsiscp  ldas-grid.ligo.caltech.edu:/home/pankow/richard_mdc/mdc.xml.gz .
# Generate template bank using some fake times
	echo " BARELY EVEN STARTED "
	lalapps_tmpltbank  --approximant TaylorF2 --spectrum-type iLIGOSRD --ifo H1 --gps-start-time ${PSD_GPS_START} --gps-end-time ${PSD_GPS_END} --max-total-mass 20 --min-total-mass 15 --min-eta 0.2
