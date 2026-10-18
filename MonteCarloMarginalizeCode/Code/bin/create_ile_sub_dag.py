#!/usr/bin/env python

import argparse
import sys
import os
import shutil
import re
from RIFT.misc.pipeline_arguments import parse_submit_arguments, format_submit_arguments
from pathlib import Path
# Backend-neutral pipeline namespace (htcondor/glue/slurm) provided by dag_utils_generic
from RIFT.misc.dag_utils_generic import pipeline
from igwn_ligolw import utils, ligolw, lsctables
import RIFT.lalsimutils as  lsu
import numpy as np
from math import ceil

cwd = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('--sim-xml',type=str,help="the sim-xml which encodes the points this dag will evaluate")
parser.add_argument('--cap-points',type=int,default=None,help="if you want to put a limit on how many points to load")
parser.add_argument('--submit-script',type=str,help="the path to the ile.sub which will be used for these points")
parser.add_argument("--macroiteration",type=int)
parser.add_argument("--target-dir",type=str,default=cwd,help="the directory to write the sub into")
parser.add_argument("--output-suffix",type=str,default="the suffix of the subdag to write to: iteration_{macroiteration}_{opts.suffix}.dag")
opts = parser.parse_args()

n_events= 0
xmldoc = utils.load_filename( opts.sim_xml, contenthandler=lsu.cthdler )

try:
        # Read SimInspiralTable from the xml file, set row bounds
        sim_insp = lsctables.SimInspiralTable.get_table(xmldoc) #table.get_table(xmldoc, lsctables.SimInspiralTable.tableName)
        n_events = len(sim_insp)
        print(" === NUMBER OF EVENTS === ")
        print(n_events)
except ValueError:
        parser.error("No SimInspiral table found in xml file")
if n_events == 0:
    parser.error("Intrinsic grid must contain at least one point")

if opts.cap_points is not None:
    if opts.cap_points < 1:
        parser.error("--cap-points must be positive")
    n_events = min(n_events, opts.cap_points)

dag = pipeline.CondorDAG(log=os.getcwd())

submit_text = Path(opts.submit_script).read_text()
exe_match = re.search(r'^\s*executable\s*=\s*(.*?)\s*$', submit_text, re.I | re.M)
if exe_match is None:
    parser.error("Submit description must specify executable")
exe = exe_match.group(1)
assignments = list(re.finditer(r'^\s*arguments\s*=\s*(.*?)\s*$', submit_text, re.I | re.M))
if len(assignments) > 1:
    parser.error("Submit description has multiple arguments assignments")
raw_args = assignments[0].group(1) if assignments else ""
argv = parse_submit_arguments(raw_args)
counts = []
for index, arg in enumerate(argv):
    if arg == "--n-events-to-analyze":
        if index + 1 == len(argv):
            parser.error("Missing --n-events-to-analyze value")
        counts.append(argv[index + 1])
    elif arg.startswith("--n-events-to-analyze="):
        counts.append(arg.split("=", 1)[1])
try:
    counts = [int(value) for value in counts]
except ValueError:
    parser.error("--n-events-to-analyze must be a positive integer")
if len(set(counts)) > 1 or any(value < 1 for value in counts):
    parser.error("Conflicting or nonpositive --n-events-to-analyze settings")
# The ILE driver's default is one point; older builders omit this flag for one.
n_events_per_job = counts[0] if counts else 1
submit_script = opts.submit_script
if opts.cap_points is not None and n_events % n_events_per_job:
    # Preserve the shared template. A private copy makes the final worker obey
    # an exact cap even when it ends in the middle of a worker's normal batch.
    capped_argv = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == "--n-events-to-analyze":
            skip = True
            continue
        if arg.startswith("--n-events-to-analyze="):
            continue
        capped_argv.append(arg)
    capped_argv += ["--n-events-to-analyze", "$(macrobatchsize)"]
    replacement = "arguments = " + format_submit_arguments(capped_argv)
    if assignments:
        match = assignments[0]
        submit_text = submit_text[:match.start()] + replacement + submit_text[match.end():]
    else:
        submit_text += "\n" + replacement + "\n"
    submit_script = os.path.join(opts.target_dir, f"iteration_{opts.macroiteration}_{opts.output_suffix}_capped.sub")
    Path(submit_script).write_text(submit_text)

print(f"exe is {exe}")
print(f"num events per job is {n_events_per_job}")

num_jobs = ceil(n_events/n_events_per_job)
# Create one node per index
for i in np.arange(num_jobs):        
    ile_blank =  pipeline.CondorDAGJob(universe="vanilla", executable=exe)
    ile_blank.set_sub_file(submit_script)

    ile_node = pipeline.CondorDAGNode(ile_blank)
    ile_node.add_macro("macroevent", n_events_per_job*i)
    ile_node.add_macro("macroiteration",opts.macroiteration)
    if submit_script != opts.submit_script:
        ile_node.add_macro("macrobatchsize", min(n_events_per_job, n_events - n_events_per_job*i))

    ile_node.set_category("ILE")
    dag.add_node(ile_node)


dag_name=os.path.join(opts.target_dir,f"iteration_{opts.macroiteration}_{opts.output_suffix}")
dag.set_dag_file(dag_name)
dag.write_concrete_dag()


