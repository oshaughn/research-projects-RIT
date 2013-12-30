

target_list: Makefile
# Yes, this leaves one line on top to be removed manually
	gracedb search "ER4 LowMass EM_READY FAR<1e-11" | awk '{print $$1}'  > target_list

prepare_dags:
	for i in `cat target_list`; ../../Code/ConstructGraceDB_pipeline_wrapper.py $$i; done

launch_dags:
	for i in `cat target_list`; cd $$i; condor_submit_dag -config ~/condor_config.local marginalize_extrinsic_parameters.dag; done

# Summarize results
#  - done manually, generate 'massgrid.txt' file

# Bundle results
bundle:
	tar cvzf test-events.tgz G*/massgrid.txt G*/coinc.xml

# Report on results
#   - remind me what the search found
#   - 
report:
	for i in G*; do echo $$i; ligolw_print -t sngl_inspiral -c ifo -c mass1 -c mass2 -c snr $$i/coinc.xml; done
