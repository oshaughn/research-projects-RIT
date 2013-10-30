setup:
# Retrieve Pankow 'blind injection challenge'
	gsiscp -r ldas-grid.ligo.caltech.edu:/home/pankow/richard_mdc/signal_hoft .
	gsiscp -r ldas-grid.ligo.caltech.edu:/home/pankow/richard_mdc/signal_hoft_no_noise .
	gsiscp  ldas-grid.ligo.caltech.edu:/home/pankow/richard_mdc/mdc.xml.gz .
	find -L signal_hoft -name '*.gwf' -print | lalapps_path2cache > test1.cache
	find -L signal_hoft_no_noise -name '*.gwf' -print | lalapps_path2cache > test1-0noise.cache


# iLIGO noise
test-fiducial-data:
	python test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file test1-0noise.cache --channel-name H1=FAKE-STRAIN --channel-name L1=FAKE-STRAIN --channel-name V1=FAKE-STRAIN --force-read-full-frames --show-psd --show-likelihood-versus-time --show-sampler-inputs --show-sampler-results --Niter 1000000 --save-sampler-file fiducial-0noise --save-threshold-fraction 0.95

# iLIGO noise
test-fiducial-data-noisy:
	python test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file test1.cache --channel-name H1=FAKE-STRAIN --channel-name L1=FAKE-STRAIN --channel-name V1=FAKE-STRAIN --force-read-full-frames --show-psd --show-likelihood-versus-time --show-sampler-inputs --show-sampler-results --Niter 1000000 --save-sampler-file fiducial-0noise --save-threshold-fraction 0.95


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
# test marginalization (single template).  User may need to interact 
	test_like_and_samp_singleifo.py
	test_like_and_samp.py
# test Fisher matrix estimator
	test_effectiveFisher.py



injection-info:
# Parameters
	ligolw_print mdc.xml.gz -t sim_inspiral -c mass1 -c mass2 -c longitude -c latitude
# Frame files
	FrDump -i signal_hoft/H-ER_signal_hoft-10000/H-ER_signal_hoft-1000000000-64.gwf 
	FrDump -i signal_hoft/L-ER_signal_hoft-10000/L-ER_signal_hoft-1000000000-64.gwf 
	FrDump -i signal_hoft/V-ER_signal_hoft-10000/V-ER_signal_hoft-1000000000-64.gwf 
