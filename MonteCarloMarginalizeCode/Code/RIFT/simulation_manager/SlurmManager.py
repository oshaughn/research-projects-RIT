import sys, os
from pathlib import Path
import numpy as np
import RIFT.simulation_manager.BaseManager as bm
import logging
logger = logging.getLogger(__name__)


# OTHER INTERFACES
#      # https://pypi.org/project/pyslurm/
#    simple-slurm

has_slurm_pipeline_interactive = False
try:
    # https://pyslurmutils.readthedocs.io/en/stable/
    # interactive API
    import pyslurmutils
except:
    logger.info(" SlurmManager: pyslurmutils not available")

has_slurm_pipeline_static = False
try:
    # https://pypi.org/project/simple-slurm/
    import simple_slurm
    has_slurm_pipeline_static = True
except:
    logger.info(" SlurmManager: simple_slurm not available")
    

if has_slurm_pipeline_static:
    def default_slurm_build_job(tag='job_tag_default_slurm', exe=None, arg_str=None, sim_path=None,slurm_args=None, **kwargs):
        slurm_job = None
        if slurm_args:
            slurm_job = simple_slurm.Slurm(**slurm_args)
        else:
            slurm_job = simple_slurm.Slurm()
        slurm_job.add_cmd(f"{exe} {arg_str}")
        return slurm_job, tag+".slurm_sub"
    class SimulationArchiveOnLocalDiskIntegratedSlurmQueueSimple(bm.SimulationArchiveOnLocalDiskExternalQueue):
        """
        User can specify a
        """
        def __init__(self,**kwargs):
            self._internal_build_submit= default_slurm_build_job
            self._internal_exe = 'echo'
            self._internal_job = None
            super().__init__(**kwargs)
            if not os.path.exists(self.base_location+"/logs"):
                os.mkdir(self.base_location + '/logs')
            # workspace for dags which are building the simulations
            if not os.path.exists(self.base_location+"/slurm_submit_files/"):
                os.mkdir(self.base_location + '/slurm_submit_files/')

        def generate_simulation(self, sim_params,**kwargs):
            self._internal_simulations_have_sub_directories = True 
            # Create filesystem space, etc
            super().__init__(**kwargs)
            print(" NOT YET IMPLEMENTED TO INTERFACE AUTOMATICALLY")

        def build_single_job(self, tag=None, **kwargs):
            # Create slurm job to submit *one* simulation - different than what we do otherwise
            log_dir = self.base_location + "/logs"
            build_args = {}
            # defaults, required
            # updates
            build_args.update(kwargs)
            # build submit master job
            ile_job, ile_job_name  = self._internal_build_submit(tag=tag,log_dir=log_dir,**build_args)
            # write master job
            self._internal_job = ile_job
            print(str(ile_job))
            with open(self.base_location+"/slurm_submit_files/" + ile_job_name, 'w') as f:
                f.write(str(ile_job)) # should work


if __name__ == "__main__":
    def my_generator(k, **kwargs):
        return k*np.sqrt(2)  # argument
    
    archive = SimulationArchiveOnLocalDiskIntegratedSlurmQueueSimple(name="test", base_location="slurm_test", _internal_annotator=bm.append_queue_default)
    archive.generator = my_generator
    archive.build_single_job(tag="me", exe='echo', arg_str='hello') # Create condor submit file prototype
