setup:
# Retrieve Pankow 'blind injection challenge'
	gsiscp -r ldas-grid.ligo.caltech.edu:/home/pankow/richard_mdc/signal_hoft .
	gsiscp  ldas-grid.ligo.caltech.edu:/home/pankow/richard_mdc/mdc.xml.gz .


test:
# test likelihood evaluation
	python test_precompute_singleifo.py
	python test_precompute.py
# test access
