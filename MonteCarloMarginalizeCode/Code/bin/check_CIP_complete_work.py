#!/usr/bin/env python

# Example Script Provided by:
# https://htcondor.readthedocs.io/en/latest/auto-redirect.html?category=example&tag=bagman-percent-done

import sys
import glob
try:
    import htcondor
except ImportError:
    error("Failed to import htcondor python bindings", 1)

def error(msg: str, code: int):
    """Print error message and exit with specified exit code"""
    print(f"ERROR: {msg}", file=sys.stderr)
    exit(code)


def parse_args():
    """Parse command line arguments"""
    if len(sys.argv) != 3:
        error("Missing argument(s) Job Id and/or Job return code", 1)

    # Parse this nodes Job ID
    try:
        ID = int(sys.argv[1].split(".")[0])
    except ValueError:
        error(f"Failed to convert Job Id ({sys.argv[1]}) to integer", 1)

    # Parse this nodes exit code to preserve node success/failure based on job exit
    try:
        CODE = int(sys.argv[2])
    except ValueError:
        error(f"Failed to convert Job exit code ({sys.argv[2]}) to integer", 1)

    try:
        IT = int(sys.argv[3])
    except ValueError:
        error("No it",1)

    try:
        TARGET = int(sys.argv[4])
    except ValueError:
        error("No target ", 1)

    return (ID, CODE,IT,TARGET)

def get_job_ad(job_id: int, exit_code: int):
    """Query and return the parent DAGMan proper job ad"""
    DAG_ATTRS = ["Iwd"]
    found = False

    schedd = htcondor.Schedd()

    Iwd = None
    # Get workflow Job ID from this a job history ad for this node
    for ad in schedd.history(f"ClusterId=={job_id}", projection=['ProcId', 'ClusterId', 'JobStatus','Iwd'],match=1):
        print(ad)
        if "Iwd" in ad:
            found = True
            Iwd = ad['Iwd']

    if not found:
        error(f"Failed to query job ad for cluster {job_id}", exit_code)


    return Iwd

def main():
    # Threshold to exit if  we reach a threshold of samples
    THRESHOLD = 1000

    job_id, exit_code, it, THRESHOLD = parse_args()

    Iwd = get_job_ad(job_id, exit_code)

    # work size
    n_work = len(glob.glob(Iwd + "/overlap-grid-*.xml.gz"))
    # net n_eff. Using RIFT default n_eff (not n_ESS)
    n_eff_net = 0
    for fname in glob.glob(Iwd+"/overlap-grid*[0-9]+annotation.dat"):
        dat = np.loadtxt(fname) # single line file
        n_eff_net += dat[-1] # last column


    if n_eff_net >= THRESHOLD:
        print(f"CIP jobs have completed {n_eff_net}% effective samples")
        sys.exit(124)

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
