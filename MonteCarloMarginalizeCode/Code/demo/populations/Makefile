PWD=$(shell pwd)
ENV_NAME=pop-ecc-inj


USER=katelyn.wagner
# note this *MUST* match dirname in your ini file
RUNDIR='${PWD}/ecc_injections'
PARAM_FILE='${PWD}/inj_demo.dat'
INI_FILE='${PWD}/pop-example.ini'

WRITE_MDC='/home/$(USER)/.conda/envs/$(ENV_NAME)/src/research-projects-RIT/MonteCarloMarginalizeCode/Code/demo/populations/write_mdc.py'
PP_RIFT_DIR='/home/$(USER)/.conda/envs/$(ENV_NAME)/src/research-projects-RIT/MonteCarloMarginalizeCode/Code/test/pp'

setup-env:
	(bash -c "mamba create --clone igwn-py310 --name $(ENV_NAME)" && \
	source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh && \
	conda activate $(ENV_NAME) && \
	cd ~/.conda/envs/$(ENV_NAME) && mkdir src && cd src && \
	git clone https://github.com/oshaughn/research-projects-RIT.git && \
	cd research-projects-RIT && git checkout rift_O4c && \
	pip install -e . && \
	pip install --upgrade scipy && \
	pip install --upgrade pyseobnr)

injections:
	(mkdir $(RUNDIR) && \
	source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh && \
	conda activate $(ENV_NAME) && \
	cd $(RUNDIR); python $(WRITE_MDC) --param-file $(PARAM_FILE) --ini-file $(INI_FILE);)

rundir:
	(bash -c 'source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh && \
        conda activate $(ENV_NAME) && \
	export PATH="$$PATH:$(PP_RIFT_DIR)" && \
	which pp_RIFT_with_ini && \
	source ./setup.sh && pp_RIFT_with_ini --use-ini ${INI_FILE} --use-gwsignal --use-osg')

submit:
	(bash -c 'source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh && \
	conda activate $(ENV_NAME) && \
	cd $(RUNDIR) && \
	condor_submit_dag -import_env master.dag')
