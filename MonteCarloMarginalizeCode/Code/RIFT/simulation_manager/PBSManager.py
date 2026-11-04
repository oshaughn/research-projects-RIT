import sys, os
import subprocess
import re
import logging
import RIFT.simulation_manager.BaseManager as bm

logger = logging.getLogger(__name__)

def default_pbs_build_job(tag='job_tag_default_pbs', exe=None, arg_str=None, sim_path=None, pbs_args=None, array_range=None, **kwargs):
    """
    Builds a PBS submit script.
    pbs_args: dict of PBS options, e.g., {'queue': 'workq', 'walltime': '01:00:00', 'nodes': '1:ppn=1'}
    array_range: PBS array job specification, e.g., '1-10' or '1-10:2' (step size)
    """
    if pbs_args is None:
        pbs_args = {}
    
    # Standard PBS headers
    script_lines = ["#!/bin/bash", "#PBS -N " + tag]
    
    # Add array job directive if specified
    if array_range:
        script_lines.append(f"#PBS -J {array_range}")
    
    # Add custom PBS arguments
    if 'queue' in pbs_args:
        script_lines.append(f"#PBS -q {pbs_args['queue']}")
    if 'walltime' in pbs_args:
        script_lines.append(f"#PBS -l walltime={pbs_args['walltime']}")
    if 'nodes' in pbs_args:
        script_lines.append(f"#PBS -l {pbs_args['nodes']}")
    if 'mem' in pbs_args:
        script_lines.append(f"#PBS -l mem={pbs_args['mem']}")
    
    script_lines.append("")  # newline
    script_lines.append(f"cd $PBS_O_WORKDIR")
    script_lines.append(f"{exe} {arg_str}")
    
    script_content = "\n".join(script_lines)
    return script_content, tag + ".pbs_sub"

class SimulationArchiveOnLocalDiskIntegratedPBSQueue(bm.SimulationArchiveOnLocalDiskExternalQueue):
    """
    Simulation archive whose entries are queued and run via PBS/Larus.
    """
    def __init__(self, **kwargs):
        self._internal_build_submit = default_pbs_build_job
        self._internal_exe = 'echo'
        self._internal_job_id = None 
        self._internal_simulations_have_sub_directories = True
        super().__init__(**kwargs)
        
        if not os.path.exists(self.base_location + "/logs"):
            os.mkdir(self.base_location + "/logs")
        if not os.path.exists(self.base_location + "/pbs_submit_files/"):
            os.mkdir(self.base_location + "/pbs_submit_files/")

    def generate_simulation(self, sim_params, **kwargs):
        return self.register_simulation(sim_params, **kwargs)

    def build_single_job(self, tag=None, **kwargs):
        """Builds and writes a single PBS submit script."""
        build_args = {}
        build_args.update(kwargs)
        
        script_content, script_name = self._internal_build_submit(tag=tag, **build_args)
        
        full_path = os.path.join(self.base_location, "pbs_submit_files", script_name)
        with open(full_path, 'w') as f:
            f.write(script_content)
        
        self._internal_job_id = script_name  # Temporary reference
        return full_path

    def build_array_job(self, tag=None, array_range='1-10', **kwargs):
        """
        Builds and writes a PBS array job submit script.
        array_range: PBS array job specification, e.g., '1-10' or '1-10:2'
        """
        build_args = dict(kwargs)
        build_args['array_range'] = array_range
        
        script_content, script_name = self._internal_build_submit(tag=tag, **build_args)
        
        full_path = os.path.join(self.base_location, "pbs_submit_files", script_name)
        with open(full_path, 'w') as f:
            f.write(script_content)
        
        self._internal_job_id = script_name
        return full_path

    def submit_job(self, script_path):
        """Submits a PBS script via qsub and captures the job ID."""
        try:
            result = subprocess.run(['qsub', script_path], 
                                   capture_output=True, text=True, check=True)
            # qsub usually returns the job ID directly to stdout
            job_id = result.stdout.strip()
            logger.info(" PBS job submitted successfully: %s ", job_id)
            return job_id
        except subprocess.CalledProcessError as e:
            logger.error(" Failed to submit PBS job %s: %s ", script_path, e.stderr)
            return None
        except FileNotFoundError:
            logger.error(" qsub not found on PATH ")
            return None

    def refresh_status_from_pbs(self, job_id):
        """
        Checks the status of a PBS job using qstat.
        Returns 'running' if job is in queue/running, 'complete' if gone (and output exists).
        """
        try:
            # qstat -f provides full details; -n for numeric IDs
            result = subprocess.run(['qstat', '-n', job_id], 
                                   capture_output=True, text=True)
            
            if result.returncode == 0:
                # Job is still in the system
                return 'running'
            else:
                # Job is no longer in qstat; check if output exists via _internal_check_complete
                return 'complete'
        except Exception as exc:
            logger.warning(" refresh_status_from_pbs failed for %s: %s ", job_id, exc)
            return None

if __name__ == "__main__":
    # Simple smoke test
    archive = SimulationArchiveOnLocalDiskIntegratedPBSQueue(name="pbs_test", base_location="pbs_test_dir")
    archive.generator = lambda k: k*2
    archive.generate_simulation(0.5)
    
    # Test build
    path = archive.build_single_job(tag="smoke_test", exe="echo", arg_str="hello world")
    print(f"PBS script built at: {path}")
