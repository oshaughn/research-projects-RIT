
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

# HTCondor python bindings: try htcondor2 first (current upstream), fall
# back to legacy htcondor. The two share the surface we depend on
# (Schedd().query(constraint=..., projection=[...])), so callers should
# use the cached `_htcondor_module` rather than `import htcondor` directly.
_htcondor_module = None
_htcondor_module_name = None
for _candidate in ('htcondor2', 'htcondor'):
    try:
        _htcondor_module = __import__(_candidate)
        _htcondor_module_name = _candidate
        break
    except ImportError:
        continue
has_htcondor = _htcondor_module is not None
# Backward-compat alias (legacy code may have checked this symbol).
has_htcondor_pipeline = has_htcondor
if not has_htcondor:
    logger.info(" CondorManager: neither htcondor2 nor htcondor bindings available ")
else:
    logger.info(" CondorManager: using %s python bindings ", _htcondor_module_name)

default_getenv_value='True'
default_getenv_osg_value='True'
if 'RIFT_GETENV' in os.environ:
    default_getenv_value = os.environ['RIFT_GETENV']
if 'RIFT_GETENV_OSG' in os.environ:
    default_getenv_osg_value = os.environ['RIFT_GETENV_OSG']


    

def safely_quote_arg_str(s):
    if s.startswith('"') and s.endswith('"'):
        return s
    return '"' + s + '"'

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
        
    def condor_check_complete(params=None, sim_path=None, sim_meta_path=None,
                              sim_annotation=None):
        """`_internal_check_complete` for condor-backed archives:
        a sim is complete iff its output file exists and is non-empty.
        We deliberately do NOT consult the schedd here — the schedd tells us
        about *running* jobs, but completion is defined by output on disk
        (i.e. the worker actually finished and wrote its result).

        TODO: Implement more robust failure detection by scanning .err logs
        for common crash signatures (e.g., segfaults, OOM) to distinguish
        between 'stuck' (slow) and 'failed' jobs.
        """
        try:
            return os.path.exists(sim_path) and os.path.getsize(sim_path) > 0
        except OSError:
            return False

    class SimulationArchiveOnLocalDiskIntegratedCondorQueue(bm.SimulationArchiveOnLocalDiskExternalQueue):
        """
        Simulation archive whose entries are *queued* and *run* via HTCondor.

        Lifecycle:
          register_simulation(sim_params) -> entry created on disk, status='ready'
          generate_dag_for_all_ready_simulations() -> DAG built, status flips to 'submit_ready'
          submit_dag(dag_path)             -> condor_submit_dag invoked
          (optional) refresh_status_from_condor() -> scrape condor_q, mark 'running'
          refresh_status_from_disk()       -> output present? mark 'complete'
        """
        def __init__(self, **kwargs):
            self._internal_build_submit = default_condor_build_job
            self._internal_exe = 'echo'
            self._internal_job = None
            self._internal_dag_cluster_id = None  # condor cluster id of submitted DAGman
            # Each sim gets its own subdirectory so per-job output paths don't collide.
            self._internal_simulations_have_sub_directories = True
            super().__init__(**kwargs)
            # Default completion check: the worker writes the output file.
            self._internal_check_complete = condor_check_complete
            if not os.path.exists(self.base_location+"/logs"):
                os.mkdir(self.base_location + '/logs')
            # workspace for dags which are building the simulations
            if not os.path.exists(self.base_location+"/dags"):
                os.mkdir(self.base_location + '/dags')

        def generate_simulation(self, sim_params, **kwargs):
            """In a condor-backed archive, 'generate' = 'register'. The
            generator runs in the submitted job, not at this call site."""
            return self.register_simulation(sim_params, **kwargs)

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

        def get_node_for_dag(self, sim_id_internal=None, **kwargs):
            ile_node = pipeline.CondorDAGNode(self._internal_job)
            ile_node.add_macro("macro_sim_id", sim_id_internal)
            # Pass per-sim params, output path, and metadata path as macros
            # so a worker can find its inputs and write its result.
            sim_params = self.simulations[sim_id_internal][0]
            sim_path = self.simulations[sim_id_internal][1]
            sim_meta = self.simulations[sim_id_internal][2]
            ile_node.add_macro("macro_sim_params", str(sim_params))
            ile_node.add_macro("macro_sim_path", sim_path)
            ile_node.add_macro("macro_sim_meta", sim_meta)
            return ile_node

        def generate_dag_for_all_ready_simulations(self, tag='sim_dag', **kwargs):
            if not has_glue_pipeline:
                logger.error(" glue.pipeline not available, cannot generate DAG ")
                return None
            if self._internal_job is None:
                logger.error(" build_master_job() must be called before generate_dag_for_all_ready_simulations ")
                return None

            dag = pipeline.CondorDAG(log=self.base_location + "/logs")
            ready_names = self.simulations_with_status('ready')
            for sim_id in ready_names:
                node = self.get_node_for_dag(sim_id)
                dag.add_node(node)
                self.set_status('submit_ready', sim_name=sim_id)

            if not ready_names:
                logger.info(" No ready simulations found for DAG ")
                return None

            # glue.pipeline appends ".dag" itself (see e.g.
            # RIFT/misc/dag_utils.py); pass the basename without extension.
            dag_basename = os.path.join(self.base_location, "dags", tag)
            dag.set_dag_file(dag_basename)
            dag.write_concrete_dag()
            dag_path = dag_basename + ".dag"
            logger.info(" Generated DAG with %d simulations at %s ",
                        len(ready_names), dag_path)
            return dag_path

        def submit_dag(self, dag_path):
            """Submit a DAG via condor_submit_dag. Captures the cluster id
            for later schedd queries. Returns True on success."""
            import re, subprocess
            try:
                result = subprocess.run(
                    ['condor_submit_dag', '-f', dag_path],
                    capture_output=True, text=True, check=True)
                logger.info(" DAG submitted successfully: %s ", result.stdout)
                m = re.search(r'submitted to cluster (\d+)', result.stdout)
                if m:
                    self._internal_dag_cluster_id = int(m.group(1))
                # All nodes are now in the schedd's queue.
                for sim_id in self.simulations_with_status('submit_ready'):
                    self.set_status('running', sim_name=sim_id)
                return True
            except subprocess.CalledProcessError as e:
                logger.error(" Failed to submit DAG %s: %s ", dag_path, e.stderr)
                return False
            except FileNotFoundError:
                logger.error(" condor_submit_dag not found on PATH ")
                return False

        def refresh_status_from_condor(self):
            """Use the htcondor python bindings (if available) to inspect the
            schedd. Sims whose nodes are still in the queue are marked
            'running'; sims whose nodes left the queue but whose output is
            missing are marked 'stuck'. Output-on-disk is the authoritative
            'complete' signal — call refresh_status_from_disk() afterwards.

            Works with either the modern `htcondor2` bindings or the legacy
            `htcondor` bindings; the choice is made at module import time."""
            if not has_htcondor:
                logger.info(" refresh_status_from_condor: htcondor bindings not available ")
                return None
            _htcondor = _htcondor_module
            try:
                schedd = _htcondor.Schedd()
                in_queue_sim_ids = set()
                # If we have a cluster id from submit_dag, scope the query.
                constraint = None
                if self._internal_dag_cluster_id is not None:
                    constraint = "DAGManJobId =?= {}".format(self._internal_dag_cluster_id)
                ads = schedd.query(constraint=constraint,
                                   projection=['ClusterId', 'ProcId', 'JobStatus',
                                               'Args', 'Cmd'])
                for ad in ads:
                    args = ad.get('Args', '') or ''
                    for sim_id in self.simulations:
                        # macros for sim_id are passed as $(macro_sim_id)
                        # which is rendered into the Args at submit time.
                        if str(sim_id) in args:
                            in_queue_sim_ids.add(sim_id)
                # update statuses
                for sim_id in self.simulations:
                    current = self.get_status(sim_name=sim_id)
                    if current in (None, 'complete', 'stuck'):
                        continue
                    if sim_id in in_queue_sim_ids:
                        if current != 'running':
                            self.set_status('running', sim_name=sim_id)
                    else:
                        # Job is no longer in the queue. If output is missing,
                        # mark stuck. Otherwise refresh_status_from_disk will
                        # promote to complete.
                        sim_here = self.simulations[sim_id]
                        if not condor_check_complete(
                                params=sim_here[0], sim_path=sim_here[1],
                                sim_meta_path=sim_here[2],
                                sim_annotation=sim_here[3:]):
                            if current == 'running':
                                self.set_status('stuck', sim_name=sim_id)
                return in_queue_sim_ids
            except Exception as exc:
                logger.warning(" refresh_status_from_condor failed: %s ", exc)
                return None
    

if __name__ == "__main__":
    def my_generator(k, **kwargs):
        return k*np.sqrt(2)  # argument

    base = "foo"
    archive = SimulationArchiveOnLocalDiskIntegratedCondorQueue(
        name="test", base_location=base,
        _internal_annotator=bm.append_queue_default)
    archive.generator = my_generator
    archive.build_master_job(tag="me")  # Create condor submit file prototype

    # Register two sims (no inline run; no schedd contacted).
    archive.generate_simulation(0.5)
    archive.generate_simulation(1.5)
    print(" Registered sims:", list(archive.simulations.keys()))
    print(" Statuses:",
          {k: archive.get_status(sim_name=k) for k in archive.simulations})

    # Build a DAG covering all 'ready' sims; status flips to 'submit_ready'.
    dag_path = archive.generate_dag_for_all_ready_simulations(tag="my_dag")
    print(" DAG written to:", dag_path)
    print(" Statuses after build_dag:",
          {k: archive.get_status(sim_name=k) for k in archive.simulations})
