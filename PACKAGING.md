



# Archival notes

## Virtual environments (for dependencies)
The prerequisites for this code may not be available, can take a while to install, and (if used without care) may impact your other workflows via pip user installls.
To simplify safe sharing, make a virtual environment which holds these dependencies.

```
mkdir virtualenvs
virtualenv virtualenvs/rapidpe_gpu_clean
source virtualenvs/rapidpe_gpu_clean/bin/activate
cd ${ILE_DIR}
python setup.py install 
pip install cupy   # must run on GPU-enabled machine
```

## Containers (for OSG)

**Temporary notes**: 

```
# Installed in /cvmfs/ligo-containers.opensciencegrid.org/james-clark/research-projects-rit/rift/latest

singularity shell --writable  /cvmfs/ligo-containers.opensciencegrid.org/james-clark/research-projects-rit/rift/latest


# GW_SURROGATE needs to be non-empty for the code to run at all
export GW_SURROGATE=''


## no longer necessary!
# ILE_DIR should be defined as below ... SHOULD be in 
#export ILE_DIR=/research-projects-RIT/MonteCarloMarginalizeCode/Code
#export PATH=${ILE_DIR}:${PATH}
#export PYTHONPATH=${ILE_DIR}:${PYTHONPATH}


# Installation check
cd ${ILE_DIR}
python test_like_and_samp.py --LikelihoodType_MargTdisc_array  # should run
```
