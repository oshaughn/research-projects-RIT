

target_list: Makefile
# Yes, this leaves one line on top to be removed manually
	gracedb search "ER4 LowMass EM_READY FAR<1e-11" | awk '{print $$1}'  > target_list

prepare_dags:
	for i in `cat target_list`; ../../Code/ConstructGraceDB_pipeline_wrapper.py $$i; done

launch_dags:
	for i in `cat target_list`; cd $$i; condor_submit_dag -config ~/condor_config.local marginalize_extrinsic_parameters.dag; done
