setup:
# Retrieve Pankow 'blind injection challenge'
	gsiscp -r ldas-grid.ligo.caltech.edu:/home/pankow/richard_mdc/signal_hoft .
	gsiscp -r ldas-grid.ligo.caltech.edu:/home/pankow/richard_mdc/signal_hoft_no_noise .
	gsiscp  ldas-grid.ligo.caltech.edu:/home/pankow/richard_mdc/mdc.xml.gz .
	find -L signal_hoft -name '*.gwf' -print | lalapps_path2cache > test1.cache
	find -L signal_hoft_no_noise -name '*.gwf' -print | lalapps_path2cache > test1-0noise.cache


test:
# test hlm generation, overlaps
#	python test_hlm.py   # shows that hlm's generate via our wrapper.  SHOULD BE TESTED with other approximants.
	python test_rhotimeseries.py  # inner product with data
# test likelihood evaluation
	python test_precompute_singleifo.py
	python test_precompute.py
	python test_precompute_noisydata.py
# test sampler 
	test_mcsamp.py
#



injection-info:
# Parameters
	ligolw_print mdc.xml.gz -t sim_inspiral -c mass1 -c mass2 -c longitude -c latitude
# Frame files
	FrDump -i signal_hoft/H-ER_signal_hoft-10000/H-ER_signal_hoft-1000000000-64.gwf 
	FrDump -i signal_hoft/L-ER_signal_hoft-10000/L-ER_signal_hoft-1000000000-64.gwf 
	FrDump -i signal_hoft/V-ER_signal_hoft-10000/V-ER_signal_hoft-1000000000-64.gwf 
