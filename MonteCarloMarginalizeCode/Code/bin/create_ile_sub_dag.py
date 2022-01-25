#!/usr/bin/env python

import argparse
import sys
import os
import shutil
from glue import pipeline # https://github.com/lscsoft/lalsuite-archive/blob/5a47239a877032e93b1ca34445640360d6c3c990/glue/glue/pipeline.py
from ligo.lw import utils, ligolw, lsctables
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
        print("No SimInspiral table found in xml file", file=sys.stderr)

if not (opts.cap_points is None):
    if n_events > opts.cap_points:
        n_event = opts.cap_points

dag = pipeline.CondorDAG(log=os.getcwd())

with open(opts.submit_script,'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'executable = ' in line:
            exe = line.split("=")[-1].strip()
        if 'arguments' in line:
            argsplit = line.split('"')[1].split()
            for i,arg in enumerate(argsplit):
                if arg == "--n-events-to-analyze":
                    n_events_per_job = int(argsplit[i+1])

print(f"exe is {exe}")
print(f"num events per job is {n_events_per_job}")

num_jobs = ceil(n_events/n_events_per_job)
# Create one node per index
for i in np.arange(num_jobs):        
    ile_blank =  pipeline.CondorDAGJob(universe="vanilla", executable=exe)
    ile_blank.set_sub_file(opts.submit_script)

    ile_node = pipeline.CondorDAGNode(ile_blank)
    ile_node.add_macro("macroevent", n_events_per_job*i)
    ile_node.add_macro("macroiteration",opts.macroiteration)

    ile_node.set_category("ILE")
    dag.add_node(ile_node)


dag_name=os.path.join(opts.target_dir,f"iteration_{opts.macroiteration}_{opts.output_suffix}")
dag.set_dag_file(dag_name)
dag.write_concrete_dag()


