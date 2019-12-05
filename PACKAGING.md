

# pip  and pypi 

We are creating pip-installable package, posted at [pypi testing](https://test.pypi.org/project/RIFT/)


**Generate a pip-installable package**: Right now the code isn't structured as a standard python package, so we use MANIFEST.in to get all necessary files into.  We also *can't use a wheel*, since that mode  seems to omit our code.

```
# edit setup.py to update the version number correctly, first
rm -rf build dist RIFT.egg-info
python setup.py sdist
```

**Upload the package**: See the [pypi docs](https://packaging.python.org/guides/using-testpypi/):
```
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

**Validate the package installs** As described in [pypi docs](https://packaging.python.org/guides/using-testpypi/), you should confirm the downloaded package is installed.  For a user install, performed by the following command
```
  pip install  --user --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple  RIFT
 ``` 
the necessary packages and source will be in 
```
 .local/bin
.local/lib/python2.7/site-packages/RIFT-<VERSION_NUMBER>-py2.7.egg_info/installed-files.txt # list of files
```

For a completely clean check, use a virtual environment
```
mkdir virtualenvs
virtualenv virtualenvs/RIFT_pip
source virtualenvs/RIFT_pip/bin/activate
pip install  --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple  RIFT
```
and perform the test described in [GETTING_STARTED.md](GETTING_STARTED.md)

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
