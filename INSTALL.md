We have very recently refactored the code to make it properly use setuptools.  
If you as an 

 * **pip install**: Installation with pip is the easiest.   Until we remove a namespace squatter, we are using ``test.pypi.org``.
```
  pip install  --user --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple  RIFT
```
 * **source install with setup.py**: If you retrieve the source code, you can run the setup script directly.  That's very helpful if you need to edit the source
```
git clone https://github.com/oshaughn/research-projects-RIT.git # use for HTTPS
cd research-projects-RIT
python setup.py install --user
export GW_SURROGATE=''
```
   * Alternatively you can use ``pip install --user -e .`` from the source directory
 * **source install, directly** :  Finally, if you don't want to use pip, you can work directly with the source
```bash
git clone https://github.com/oshaughn/research-projects-RIT.git # use for HTTPS
# git clone git@github.com:oshaughn/research-projects-RIT.git # use instead for SSH
cd research-projects-RIT
git checkout temp-RIT-Tides-port_master-GPUIntegration   
export INSTALL_DIR=`pwd`
export GW_SURROGATE=''
```

      * put the following directory in your `PATH` and `PYTHONPATH`
```bash
export ILE_DIR=${INSTALL_DIR}/MonteCarloMarginalizeCode/Code
export PATH=${PATH}:${ILE_DIR}
export PYTHONPATH=${PYTHONPATH}:${ILE_DIR}
export GW_SURROGATE=''
```

    *  make sure you have installed the necessary dependencies.  The following command will at least ensure that these dependencies are up to date 
```bash
python setup.py install --user
```

## lalsuite
The code relies heavily on lalsuite.    Please have it installed and working properly.

The code also requires a working version of glue, supporting `glue.ligolw.ligolw`.  The pip installable version of glue has diverged from the version installed on LDG clusters, and this part of glue has moved into lalsuite.  We'll update accordingly, but be warned that ongoing rapid glue evolution may cause you version-matching headaches (e.g., if the XML format changes). 

## GPU-sensitive installs

The code uses cupy to access GPUs.  If you don't have one, the code will still work.
If you do need one, make sure to install cupy **on a machine that supports GPUs **

```bash
pip install --user cupy
```

## Other environment variables

If you run on an LDG cluster, you need accounting flags

```bash
export LIGO_USER_NAME=albert.einstein
export LIGO_ACCOUNTING=ligo.dev.o3.cbc.pe.lalinferencerapid
```




### Alternative setup (singularity)
If you would rather use a pre-packaged environment *and* you have access to singularity and CVMFS (e.g., on an LDG cluster), you can do the following to use a pre-packaged version :

```bash
singularity shell --writable  /cvmfs/ligo-containers.opensciencegrid.org/james-clark/research-projects-rit/rift/latest
```

You can use this setup for testing and job launching.
