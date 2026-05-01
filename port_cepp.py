#!/usr/bin/env python
import re

# Read original script
with open('/Users/rossma/RIFT_ralph/MonteCarloMarginalizeCode/Code/bin/create_event_parameter_pipeline_BasicIteration', 'r') as f:
    content = f.read()

# Replace imports
content = re.sub(r'from RIFT\.misc\.dag_utils_generic import pipeline', '', content)
content = re.sub(r'import RIFT\.misc\.dag_utils_generic as dag_utils', 'from RIFT.misc import dag_utils_htcondor as dag_utils', content)
content = re.sub(r'from RIFT\.misc\.dag_utils_generic import mkdir', 'from RIFT.misc.dag_utils_htcondor import mkdir', content)
content = re.sub(r'from RIFT\.misc\.dag_utils_generic import which', 'from RIFT.misc.dag_utils_htcondor import which', content)

# Replace pipeline.CondorDAG with htcondor.DAG
content = re.sub(r'pipeline\.CondorDAG\(\)', 'htcondor.DAG()', content)
content = re.sub(r'pipeline\.CondorDAGNode', 'htcondor.Node', content)
content = re.sub(r'pipeline\.CondorJob', 'htcondor.Submit', content)

# Replace job/node methods
content = re.sub(r'\.set_retry\(', '.retry = ', content)
content = re.sub(r'\.add_macro\(', '.add_variable(', content)
content = re.sub(r'\.add_parent\(', '.add_parent(', content)
content = re.sub(r'\.add_node\(', '.add_node(', content)

# Replace dag_utils_generic references with dag_utils_htcondor
content = re.sub(r'dag_utils_generic', 'dag_utils_htcondor', content)

# Add htcondor2 import
content = re.sub(r'#! /usr/bin/env python', '#! /usr/bin/env python\nimport htcondor2 as htcondor', content)

# Write ported script
with open('/Users/rossma/RIFT_ralph/MonteCarloMarginalizeCode/Code/bin/cepp_basic_htcondor', 'w') as f:
    f.write(content)

print("Porting complete")
