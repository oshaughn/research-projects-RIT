#Here are the steps and instructions on using the HyperPipe code in RIFT.
# The goal of this code is to arrive at the posterior distribution for a given likelihood calculation via an executable code. The user provides an initial grid of points they are able to generate for usable indices, and a [min,max] range for each parameter. And the code adaptively explores the parameter range to resolve the posterior distribution.
#The example used in this Makefile is a 3D Gaussian distribution which can be made to be centered at some region.
#The distribution in this example is a Gaussian in parameters x, y, z, and the associated executable is 'example_gaussian.py'.
#The user is expected to use this Makefile as a template for running their executable, parameters, etc.
# Each of these make commands can be run in a bash terminal as 'make command_name', for example 'make install_RIFT'.

### STEP 0 : Install RIFT and install condor if not on LIGO-CIT ####

#Start by installing RIFT if you already don't have it.
install_RIFT:
	pip install RIFT


### STEP 1 : Create an intial grid ####

# Truly (uniform) random grid over prior range
# The command below creates a random uniform grid of indices needed for initiating the parameter search. For this 3D gaussian example there are three parameters x, y, z and we want a 1000 points grid to be made within certain ranges in each parameter with the 'blind_gaussian_3d_xy_plus.dat'.
# Use the parameters as follows
# --random-parameter : defines the parameters, x, y, z
# --random-parameter-range : sets the parameter range needed [min,max]
# --npts number of points needed in the initial grid.
# --fname-out : name of the grid file. Here we chose to make it the same name as the command itself 'blind_gaussian_3d_xy_plus.dat'
blind_gaussian_3d_xy_plus.dat:
	util_HyperparameterGrid.py --random-parameter x --random-parameter-range [-5,-2] --random-parameter y --random-parameter-range [2,5] --random-parameter z --random-parameter-range [2,5] --npts 1000 --fname-out $@

### STEP 1.8 : Run executable non-adaptively ####

#This is NOT a command. This name is given to the inital grid file and it is used later. Simply change the file address on the right hand side if running the pipeline for a different initial grid. ${SAMPLE_INPUT_GRID_FOR_GAUSSIAN} gets used later in various run commands
HERE=$(shell pwd)
SAMPLE_INPUT_GRID_FOR_GAUSSIAN = ${HERE}/blind_gaussian_3d_xy_plus.dat

# You may select the location of the gaussian and it's covariance using this command. Just set the location and covariance as a different value if you wish.
change_center_location:
	sed -i "s/rv = multivariate.*/rv = multivariate_normal([-5,0,0], [[2,0,0], [0,2,0], [0,0,2]])/g" example_gaussian.py


# Likelihoods of all points on the initial grid prepared above can be obtained by this command. Here:
# --using-eos : can be altered to point to a different file
# --outdir : a different output directory name can be chosen as desired.
# --fname-output-integral : is the name of the output file made by this executable (you may choose any name for this file for this serial example, but for the adaptive runs, a specific format is assigned for this parameter.)
# --eos_start_index : The executable is designed such that it can be return the likelihood of even a subset of the input grid. In the grid list, the starting point from which you want the executable to evaluate likelihood is identified as eos_start_index.
# --eos_end_index : Similar to eos_start_index, eos_end_index determines the endpoint of likelihood evaluations.
###!!IMPORTANT NOTE: These arguments are all NECESSARY for running this pipeline adaptively. That is to say, even though in this single core example it doesn't matter what argument names are chosen ('--using-file', '--eos_end_index', etc. and '--using-eos-index' not utilized in this example), for the multi-core adaptive solver you do need these exact argument names!! That will ensure that HyperPipe is not crashing. In other words, make your executable take the same argument names: --using-eos, --outdir, --fname-output-integral, and so on, by looking at the example executable 'example_gaussian.py'.
example_gaussian:
	python example_gaussian.py --using-eos file:${SAMPLE_INPUT_GRID_FOR_GAUSSIAN} --outdir Gaussian_example --fname-output-integral f_out_name_1_0-1000.txt --eos_start_index 0 --eos_end_index 1000


### STEP 1.9 : Visualize the non-adaptively likelihoods ####

#Visualize the likelihood by making 2D posteriors using 'plot_posterior_corner.py' which is a RIFT routine. This command will generate a corner plot in .pdf and save it into the directory made in the previous command.
plot_gaussian_single:
	plot_posterior_corner.py --composite-file Gaussian_example/f_out_name_1_0-1000.txt --composite-file-has-labels --lnL-cut 15 --use-all-composite-but-grayscale --parameter x --parameter y --parameter z --quantiles None --ci-list [0.9] --pdf --posterior-file Gaussian_example/f_out_name_1_0-1000.txt --posterior-label Initial_grid --posterior-color k --posterior-linestyle 'solid' --use-legend --bind-param x --param-bound [-6,6] --bind-param y --param-bound [-6,6] --bind-param z --param-bound [-6,6]; \
		mv corner_*.pdf Gaussian_example/

### STEP 2 : Run executable Adaptively!! ####

#This next command submits an adaptive solver on condor using the 'condor_submit_dag' command as is seen in the second last line in this set of commands.
# Going line by line:
# first line : leave as it is.
# second line : leave as it is.
# third line : leave as it is.
# fourth line : can change name of outdir (but this is not used by HyperPipe) and set any non-RIFT-required arguments needed for the executable here.
# fifth line : set the parameters you'd like the posterior to be evaluated for. --force-away and --puff-factor are additional parameters that can be adjusted to specify how much puff exploration you need the code to do; nominally leave as it is.
# sixth line : set ranges [min,max] where you want these parameters to be explored.
# seventh line : Here you could alter the following agruments based on need: --n-samples-per-job, --n-iterations, --eos-post-explode-jobs. n-samples-per-job is the number of new points you want the pipeline to produce for each iteration. n-iteration is the number of iteration.
# eighth line : leave as it is. This is the job run command line.
# ninth line : leave as it is.
### NOTE: Before running this you need to do 'source setup.sh' in your bash terminal to set the right python path.
Gaussian_adaptive_unimodal:
	mkdir $@
	echo "empty_event_file" > $@/my_event_A.txt
	echo ${HERE}/example_gaussian.py  > $@/args_marg_eos_exe.txt
	echo --outdir Gaussian_example --conforming-output-name  > $@/args_marg_eos.txt
	echo " --parameter x --parameter y --parameter z --force-away 0.03 --puff-factor 0.5" > $@/args_puff.txt
	echo " --parameter x --parameter y --parameter z --integration-parameter-range x:[-7,7] --integration-parameter-range y:[-7,7] --integration-parameter-range z:[-7,7] " > $@/args_eos_post.txt
	(cd $@; create_eos_posterior_pipeline --marg-event-exe-list-file `pwd`/args_marg_eos_exe.txt --marg-event-args-list-file  `pwd`/args_marg_eos.txt   --eos-post-args `pwd`/args_eos_post.txt --eos-post-exe `which util_ConstructEOSPosterior.py`  --puff-exe `which util_HyperparameterPuffball.py` --puff-args `pwd`/args_puff.txt --input-grid ${SAMPLE_INPUT_GRID_FOR_GAUSSIAN} --n-samples-per-job 1000 --use-full-submit-paths --working-dir `pwd` --event-file `pwd`/my_event_A.txt --n-iterations 5 --eos-post-explode-jobs 5)
	(source setup.sh; cd $@; condor_submit_dag -batch-name $@ marginalize_hyperparameters.dag)
	echo " Done doing source setup.sh, and cd $@, and then condor_submit_dag -batch-name $@ marginalize_hyperparameters.dag "

# At this stage condor is working on your pipeline request.
# CHECK STATUS of your run by doing 'watch condor_q'. There should be a job in queue.
# Typically for Gaussian it takes less than 20 mins, but sometimes it could take 2-3 hours depending on how busy CIT is and/or if something gets stuck.
# If runs are incomplete run the eighth line of the above command again on your bash terminal manually., i.e. 'source setup.sh' and then cd into the run directory and submit the dag with or without a batch-name.

### STEP 3 : Plot the adaptive run ####

# Run this command to use RIFT routine for making a corner plot with all iterations.
plot_gaussian_adaptive:
	./plot_last_iterations_with.sh Gaussian_adaptive_unimodal --parameter x --parameter y --parameter z --composite-file Gaussian_adaptive_unimodal/all.marg_net --composite-file-has-labels --lnL-cut 15 --use-all-composite-but-grayscale --quantiles None --ci-list [0.5] --use-legend --sigma-cut 10
	mv corner_*.png Gaussian_adaptive_unimodal/corner.png



##################################################
### STEP 4 : Adaptive run for multiple events ####
##################################################

# You can run this pipeline for multiple observable constraints. For instance, you may have some observational constraints on your model parameters but there might be terrestrial experiments that put constraints on some parameters (nuclear physics experiments, for instance). In this case you may have more than one executable that you'd like to provide for parameter inference, and this HyperPipe code can utilize both for running.

#The directory contains two python scripts 'example_gaussian.py' which was used above, and 'example_gaussian2.py' which we will use now as a second constraint providing executable. You should ensure the two gaussians are separated by at least some distance so that the effects of the two would be more easily visible in the posterior.
change_center_location_2:
	sed -i "s/rv = multivariate.*/rv = multivariate_normal([3,0,0], [[2,0,0], [0,2,0], [0,0,2]])/g" example_gaussian2.py


#Here we submit the two event HyperPipe run. The submit script is very similar to the unimodal case with the difference that you now need to specify the second executable and its arguments and parameters. Notice there are two lines in the beginning that mention the executable names one after the other. Then there are two lines for setting the arguments for each of the two executables. The arguments may be different for the two executables when you supply your own executables.
Gaussian_adaptive_bimodal:
	mkdir $@
	echo "empty_event_file" > $@/my_event_A.txt
	echo "empty_event_file" > $@/my_event_B.txt
	echo ${HERE}/example_gaussian.py > $@/args_marg_eos_exe.txt
	echo ${HERE}/example_gaussian2.py >> $@/args_marg_eos_exe.txt
	echo --outdir Gaussian_example --conforming-output-name > $@/args_marg_eos.txt
	echo --outdir Gaussian_example --conforming-output-name >> $@/args_marg_eos.txt
	echo " --parameter x --parameter y --parameter z --force-away 0.03 --puff-factor 0.5" > $@/args_puff.txt
	echo " --parameter x --parameter y --parameter z --integration-parameter-range x:[-7,7] --integration-parameter-range y:[-7,7] --integration-parameter-range z:[-7,7] " > $@/args_eos_post.txt
	(cd $@; create_eos_posterior_pipeline --marg-event-exe-list-file `pwd`/args_marg_eos_exe.txt --marg-event-args-list-file  `pwd`/args_marg_eos.txt   --eos-post-args `pwd`/args_eos_post.txt --eos-post-exe `which util_ConstructEOSPosterior.py`  --puff-exe `which util_HyperparameterPuffball.py` --puff-args `pwd`/args_puff.txt --input-grid ${SAMPLE_INPUT_GRID_FOR_GAUSSIAN} --n-samples-per-job 1000 --use-full-submit-paths --working-dir `pwd` --event-file `pwd`/my_event_A.txt --event-file `pwd`/my_event_B.txt --n-iterations 5 --eos-post-explode-jobs 5)
	(source setup.sh; cd $@; condor_submit_dag -batch-name $@ marginalize_hyperparameters.dag)
	echo " Done doing source setup.sh, and cd $@, and then condor_submit_dag -batch-name $@ marginalize_hyperparameters.dag "


#Plotting the result for this multi-event case can be done with the same plotting command above by changing the directories linked.


#Requirements from the executable (for e.g. `example_gaussian.py`)
#Our pipeline is currently made to be used in with python codes, however, one can simply write a wrapper script to execute codes in other languages/formats (discussed further below). Some essential arguments are specifically necessary for the wrapper or executable to be able to read and are named here:
#1. --fname, type = str : Dummy argument required by API, do not leave this blank. Can be anything arbitrary and is currently not used (One could use this to supply any additional files needed for running the code).
#2. --using-eos, type = str : Should point to the (absolute) directory of the EoS parameter file in RIFT format (e.g. blind_gaussian_3d.dat)
#3. --using-eos-index, type = int : This would be the specific line number when executing for a single set of parameters of 'using-eos'. Likelihood for this specific row in 'using-eos' should be evaluated.
#4. --fname-output-integral, type = str : The likelihoods evaluated by the Executable
#5. --fname-output-samples, type = str : Dummy argument required by API, do not leave this blank. Can be anything arbitrary and is currently not used (One could use this to supply any additional files needed for running the code).

#other arguments such as --eos_start_index and --eos_end_index are not necessary for the adaptive code, but might be useful for running a single grid in a non-adaptive way. Hence recommended to have.

# Refer to the technical_document for a complete description of how the pipeline works
