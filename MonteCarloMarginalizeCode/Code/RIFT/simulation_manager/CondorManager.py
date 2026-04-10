
import sys, os
from pathlib import Path
import numpy as np
import RIFT.simulation_manager.BaseManager as bm
import logging
logger = logging.getLogger(__name__)

has_glue_pipeline  = False
try:
    from glue import pipeline
    has_glue_pipeline=True
except:
    logger.info(" CondorManager: glue.pipeline not available ")
has_htcondor_pipeline = False
try:
    import htcondor
except:
    logger.info(" CondorManager: htcondor not available ")

default_getenv_value='True'
default_getenv_osg_value='True'
if 'RIFT_GETENV' in os.environ:
    default_getenv_value = os.environ['RIFT_GETENV']
if 'RIFT_GETENV_OSG' in os.environ:
    default_getenv_osg_value = os.environ['RIFT_GETENV_OSG']


    

if has_glue_pipeline:
    def default_condor_build_job(tag='job_tag_default',exe=None,arg_str=None,sim_path=None,universe='vanilla', use_singularity=False, use_osg=False, singularity_image_has_exe=False,singularity_image=False,use_oauth_files=False,transfer_files=None,transfer_output_files=None,request_gpus=False,request_disk=False,request_memory=False,condor_commands=None,log_dir=None,**kwargs):
        # PREP: Container management in file list
        if use_singularity and (singularity_image == None)  :
            print(" FAIL : Need to specify singularity_image to use singularity ")
            sys.exit(0)

        singularity_image_used = "{}".format(singularity_image) # make copy
        extra_files = []
        if singularity_image:
            if 'osdf:' in singularity_image:
                singularity_image_used  = "./{}".format(singularity_image.split('/')[-1])
                extra_files += [singularity_image]
        # if using container, need to decide if we are transferring the executable or assuming it is in the container (in path)
        if use_singularity:
            if singularity_image_has_exe:
                # change the executable list
                exe_base = os.path.basename(exe)
                if 'SINGULARITY_BASE_EXE_DIR' in list(os.environ.keys()) :
                    singularity_base_exe_path = os.environ['SINGULARITY_BASE_EXE_DIR']
                else:
                    singularity_base_exe_path = "/usr/bin/"  # should not hardcode this ...!
                exe=singularity_base_exe_path + exe_base
            else:
                # Must transfer executable, AND change pathname for later
                extra_files += [str(exe)]            
                exe_base = os.path.basename(exe)
                exe = "./" + exe_base


        # BUILD JOB
        ile_job = pipeline.CondorDAGJob(universe=universe, executable=exe)
        ile_sub_name = tag + '.sub'
        ile_job.set_sub_file(ile_sub_name)

        if arg_str:
            arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
            arg_str = arg_str.lstrip('-')
            if '"' in arg_str:
                arg_str = safely_quote_arg_str(arg_str)
            ile_job.add_arg(arg_str)  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

        # Standard arguments
        for opt, param in list(kwargs.items()):
            if isinstance(param, list) or isinstance(param, tuple):
                # NOTE: Hack to get around multiple instances of the same option
                for p in param:
                    ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
            elif param is True:
                ile_job.add_opt(opt.replace("_", "-"), None)
            elif param is None or param is False:
                continue
            else:
                ile_job.add_opt(opt.replace("_", "-"), str(param))

        # Requirements
        requirements = []
        ile_job.add_condor_cmd('getenv', default_getenv_value)
        ile_job.add_condor_cmd('request_memory', str(request_memory)+"M") 
        if not(request_disk is False):
            ile_job.add_condor_cmd('request_disk', str(request_disk))
        # - External requirements
        if  ( 'RIFT_BOOLEAN_LIST' in os.environ):
            extra_requirements = [ "{} =?= TRUE".format(x) for x in os.environ['RIFT_BOOLEAN_LIST'].split(',')]
            requirements += extra_requirements
        # - avoid hosts
        if 'RIFT_AVOID_HOSTS' in os.environ:
            line = os.environ['RIFT_AVOID_HOSTS']
            line = line.rstrip()
            if line:
                name_list = line.split(',')
                for name in name_list:
                    requirements.append('TARGET.Machine =!= "{}" '.format(name))

        if use_singularity:
            # Compare to https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
            ile_job.add_condor_cmd('request_CPUs', str(1))
            ile_job.add_condor_cmd('transfer_executable', 'False')
            ile_job.add_condor_cmd("MY.SingularityBindCVMFS", 'True')
            ile_job.add_condor_cmd("MY.SingularityImage", '"' + singularity_image_used + '"')
            requirements.append("HAS_SINGULARITY=?=TRUE")
            ile_job.add_condor_cmd("when_to_transfer_output",'ON_EXIT')

        if use_oauth_files:
            # we are using some authentication to retrieve files from the file transfer list, for example, from distributed hosts, not just submit. eg urls provided
            ile_job.add_condor_cmd('use_oauth_services',use_oauth_files)
        if "OSG_DESIRED_SITES" in os.environ:
            ile_job.add_condor_cmd('+DESIRED_SITES',os.environ["OSG_DESIRED_SITES"])
        if "OSG_UNDESIRED_SITES" in os.environ:
            ile_job.add_condor_cmd('+UNDESIRED_SITES',os.environ["OSG_UNDESIRED_SITES"])

            
        
        # Logging
        uniq_str = "$(cluster)-$(process)"
        ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
        ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
        ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))
        # Stream log info
        if not ('RIFT_NOSTREAM_LOG' in os.environ):
            ile_job.add_condor_cmd("stream_error",'True')
            ile_job.add_condor_cmd("stream_output",'True')

        # Write requirements
        # From https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
        ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))
        try:
            ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
            ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
        except:
            logger.info(" LIGO accounting information not available")

        if not transfer_files is None:
            if not isinstance(transfer_files, list):
                fname_str=transfer_files + ' '.join(extra_files)
            else:
                fname_str = ','.join(transfer_files+extra_files)
            fname_str=fname_str.strip()
            ile_job.add_condor_cmd('transfer_input_files', fname_str)
            ile_job.add_condor_cmd('should_transfer_files','YES')

        if not transfer_output_files is None:
            if not isinstance(transfer_output_files, list):
                fname_str=transfer_output_files
            else:
                fname_str = ','.join(transfer_output_files)
            fname_str=fname_str.strip()
            ile_job.add_condor_cmd('transfer_output_files', fname_str)
            
        ###
        ### SUGGESTION FROM STUART (for later)
        # request_memory = ifthenelse( (LastHoldReasonCode=!=34 && LastHoldReasonCode=!=26), InitialRequestMemory, int(1.5 * NumJobStarts * MemoryUsage) )
        # periodic_release = ((HoldReasonCode =?= 34) || (HoldReasonCode =?= 26))
        # This will automatically release a job that is put on hold for using too much memory with a 50% increased memory request each tim.e
        if condor_commands is not None:
            for cmd, value in condor_commands.iteritems():
                ile_job.add_condor_cmd(cmd, value)
            
        if request_gpus:
                    ile_job.add_condor_cmd('require_gpus',str(request_gpus))

        return ile_job, ile_sub_name
        
    class SimulationArchiveOnLocalDiskIntegratedCondorQueue(bm.SimulationArchiveOnLocalDiskExternalQueue):
        """
        User can specify a
        """
        def __init__(self,**kwargs):
            self._internal_build_submit= default_condor_build_job
            self._internal_exe = 'echo'
            self._internal_job = None
            super().__init__(**kwargs)
            if not os.path.exists(self.base_location+"/logs"):
                os.mkdir(self.base_location + '/logs')
            # workspace for dags which are building the simulations
            if not os.path.exists(self.base_location+"/dags"):
                os.mkdir(self.base_location + '/dag')

        def generate_simulation(self, sim_params,**kwargs):
            self._internal_simulations_have_sub_directories = True 
            # Create filesystem space, etc
            super().generate_simulation(sim_params,**kwargs)
            print(" NOT YET IMPLEMENTED TO INTERFACE AUTOMATICALLY")

        def build_master_job(self, tag=None, **kwargs):
            # Create condor job for making simulations, assume a SINGLE JOB PATTERN for all, with DAG arguments/etc to pass patterns that localize
            log_dir = self.base_location + "/logs"
            build_args = {}
            # defaults, required
            build_args['exe'] = 'echo'
            build_args['request_memory'] = 4
            build_args['request_disk']       ="4G"
            build_args['sim_path']            =" $(macro_sim_path) "
            build_args['arg_str']               =" $(macro_sim_params) "
            # updates
            build_args.update(kwargs)
            # build submit master job
            ile_job, ile_job_sub = self._internal_build_submit(tag=tag,log_dir=log_dir,**build_args)
            # write master job
            fname = self.base_location+"/" + ile_job.get_sub_file()
            ile_job.set_sub_file(fname)
            ile_job.write_sub_file()
            self._internal_job = ile_job

        def get_node_for_dag(self,sim_id_internal=None,**kwargs):
            ile_node = pipeline.CondorDAGNode(ile_job)
            ile_node.add_macro("macro_sim_id", sim_id_internal)
            sim_path  = Path(self.simulations[sim_id_internal][1]).stem # path to directory 
            ile_node.add_macro("macro_sim_path", sim_path)
            return ile_node


        # NEXT STEP
        #   - make a dag for all simulations which are 'ready', and set their status to 'submit_ready'
    

if __name__ == "__main__":
    def my_generator(k, **kwargs):
        return k*np.sqrt(2)  # argument
    
    archive = SimulationArchiveOnLocalDiskIntegratedCondorQueue(name="test", base_location="foo", _internal_annotator=bm.append_queue_default)
    archive.generator = my_generator
    archive.build_master_job(tag="me") # Create condor submit file prototype

    # dag management
    dag = pipeline.CondorDAG(log=os.getcwd())
    # initialization node? None
    
    sim_param = 0.5
    archive.generate_simulation(sim_param+1)
    sim_node = acrhive.get_node_for_dag('1') # get simulation first from the table
    dag.set_dag_file("dags/my_dag.dag") # do not be this hardcoded
    dag.write_concrete_dag()
