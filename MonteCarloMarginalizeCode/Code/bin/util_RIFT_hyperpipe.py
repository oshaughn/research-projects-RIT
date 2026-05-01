#! /usr/bin/env python
#
# EXAMPLE
#   

# ARGUMENT SPECIFICATION
#   arch
#     method:  <string>
#     n-iterations
#     n-samples-per-job
#     explode marg jobs: 
#   post:
#     coord-module: <string>
#     coords-fit:
#     coords-sample:
#     exe:
#     settings:
#        n-max
#        n-step
#        sampler-method
#        sigma-cut
#   marg-list:
#      - name:
#        exe:
#        args:
#      - name:
#        ....
#    puff:
#       exe:
#       settings:
#          puff-factor: 
#    test:
#    init:
#       file:
#       generation:
#            placement method: <string> 
#            params_and_ranges:
#            npts:
#            external_code:
#            external arguments:
#    general:
#      retries
#      request-disk
#      request-memory
#      use-osg
#    
#

import numpy as np
import os, shutil
import logging
logger = logging.getLogger(__name__)

# install reuires hydra-core
from omegaconf import DictConfig, OmegaConf
import hydra
import RIFT.misc.dag_utils_generic as dag_utils


base_dir =None
config_here = None

@hydra.main(version_base=None, config_path=".",config_name='hyperpipe_conf')
def my_app(cfg: DictConfig):
    global config_here
    logging.basicConfig()
    logger.info(" ---- INPUT CONFIG --- ")
    print(OmegaConf.to_yaml(cfg))
    config_here=cfg

if __name__ == "__main__":
    my_app()

    base_dir = os.getcwd()
    # make run directory
    if config_here['general']['rundir'] :
        if not(os.path.exists(config_here['general']['rundir'])):
            os.mkdir(config_here['general']['rundir'])
        os.chdir(config_here['general']['rundir'])
    run_dir = os.getcwd()

    # Create necessary files (note event files will need to be copied in place)
    #   - MARG
    lines_marg = ''
    lines_marg_exe = ''
    fnames_marg_exe_long = []
    lines_event_list = ''
    nchunk_list =[]
    for indx in range(len(config_here['marg-list'])):
        prefix=base_dir
        lines_marg += config_here['marg-list'][indx]['args'] + "\n"

        fname_orig = config_here['marg-list'][indx]['exe'] + "\n"
        fname_clean = os.path.basename(fname_orig.strip())
        lines_marg_exe += fname_orig
        loc = dag_utils.which(fname_clean)
        if loc is None:
            loc = base_dir + "/" +fname_clean
        print(fname_orig, loc)
        if not('MonteCarloMarginalizeCode/Code/bin' in loc): # don't copy core rift executables
            shutil.copyfile(loc, './'+fname_clean) # copy file
            fnames_marg_exe_long += [run_dir + "/" + fname_clean]

        # copy event files
        if 'event file' in config_here['marg-list'][indx]:
            prefix = base_dir
            fname_orig = config_here['marg-list'][indx]['event file']
            if fname_orig[0]  != '/':
                fname_orig = base_dir +'/'  +fname_orig
            shutil.copy(fname_orig, 'event_file_{}.dat'.format(indx))
        else:
            lines_event_list += "empty_event_file\n"
        if 'n-chunk' in config_here['marg-list'][indx]:
            nchunk_list.append(int(config_here['marg-list'][indx]['n-chunk']))
        else:
            nchunk_list.append(1)
    with open("args_marg_eos.txt", "w") as f:
        f.write(lines_marg)
    with open("args_marg_eos_exe.txt", "w") as f:
        f.write(lines_marg_exe)
    # Create event file.  Use empty file if empty
    with open("event_files.txt", "w") as f:
        f.write(lines_event_list)
    np.savetxt("event_nchunk.txt", np.array(nchunk_list,dtype=int),fmt="%i")
    
    #   - POST
    line_post =''
    line_post_exe = None
    coord_names = config_here['post']['coords-fit'].split()
    coord_range_blocks = config_here['post']['coords-sample'].split()
    for name in coord_names:
        line_post += " --parameter {} ".format(name) # no matching : fit coords in principle independent, may even use converter
    for block in coord_range_blocks:
        line_post += " --integration-parameter-range {} ".format(block)
    with open("args_eos_post.txt", "w") as f:
        f.write(line_post)
        
    #  - PUFF
    force_away_val =0.1
    if 'puff' in config_here:
        if 'puff factor' in config_here['puff']:
            force_away_val = config_here['puff']['puff factor']
    line_puff = ' --force-away {} '.format(force_away_val)
    for name in coord_names:
        line_puff += " --parameter {} ".format(name)  # default: puff in all parameters
    with open("args_puff.txt", "w") as f:
        f.write(line_puff)
        
        
    # Create initialization file
    if 'file' in config_here['init']:
        # default: copy from base directory, unless absolute
        prefix = base_dir
        target_file = config_here["init"]['file']
        if target_file[0]  != '/':
            target_file = prefix+ "/" + target_file
        shutil.copy(target_file, 'initial_grid.dat')

    # Extract architecture arguments
    n_iterations = config_here['arch']['n-iterations']
    n_samples_per_job = config_here['arch']['n-samples-per-job']
    
    
    files_transferred=False
    fname_transfer_files = 'transfer_file_list.txt'
    # Create command line
    cmd = "create_eos_posterior_pipeline  --n-samples-per-job {} --n-iterations {}  --working-dir `pwd` ".format(n_samples_per_job, n_iterations)
    # MARG
    cmd += " --marg-event-nchunk-list-file `pwd`/event_nchunk.txt  --event-file `pwd`/event_files.txt "
    cmd += " --marg-event-exe-list-file `pwd`/args_marg_eos_exe.txt --marg-event-args-list-file  `pwd`/args_marg_eos.txt  "
    # POST
    cmd += " --eos-post-args `pwd`/args_eos_post.txt --eos-post-exe `which util_ConstructEOSPosterior.py`  "
    # PUFF
    cmd += " --puff-exe `which util_HyperparameterPuffball.py` --puff-args `pwd`/args_puff.txt "
    # init
    cmd += " --input-grid `pwd`/initial_grid.dat "  # somewhat redundant to copy it, but we might be building in place
    # general
    if 'use-osg' in config_here['general']:
        check = config_here['general']['use-osg']
        if (isinstance(check, bool) and check) or (isinstance(check, str) and (check == 'True' or check == 'true') ):
          cmd += " --use-osg --use-singularity "
          # Transfer executables
          with open('transfer_file_list.txt', 'a') as f:
              for line in fnames_marg_exe_long:
                  f.write(line)
          files_transferred=True
    if 'condor-local-nonworker' in config_here['general']:
        cmd += " --condor-local-nonworker "
        if 'condor-local-nonworker-igwn-prefix' in config_here['general']:
            cmd += " --condor-local-nonworker-igwn-prefix "
    if files_transferred: # should embed this as control to transfer each executable ?
        cmd += " --transfer-file-list `pwd`/transfer_file_list.txt " # redundant - transfers all things every time for all MARG, need to improve
    print(cmd)
    os.system(cmd)
