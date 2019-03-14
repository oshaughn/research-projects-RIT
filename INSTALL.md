We are working on refactoring the code to make it a properly-install-able package.

Right now, however, you install the code by downloading the source,
```
git clone https://github.com/oshaughn/research-projects-RIT.git
cd research-projects-RIT
git checkout temp-RIT-Tides-port_master-GPUIntegration   
export INSTALL_DIR=`pwd`
```

putting the following directory in your PATH and PYTHONPATH

```
   export ILE_DIR=${INSTALL_DIR}/MonteCarloMarginalizeCode/Code
   export PATH=${PATH}:${ILE_DIR}
   export PYTHONPATH=${PYTHONPATH}:${ILE_DIR}
```

and making sure you have installed the necessary dependencies.  The following command will at least insure that these dependencies are up to date 
```
  python setup.py install --user
```

## lalsuite
The code relies heavily on lalsuite.    Please have it installed and working properly.

The code also requires a working version of glue, supporting glue.ligolw.ligolw.  The pip installable version of glue has diverged from the version installed on LDG clusters, and this part of glue has moved into lalsuite.  We'll update accordingly, but be warned that ongoing rapid glue evolution may cause you version-matching headaches (e.g., if the XML format changes). 

## GPU-sensitive installs

The code uses cupy to access GPUs.  If you don't have one, the code will still work.
If you do need one, make sure to install cupy:

```
   pip install --user cupy
```

## Other environment variables

If you run on an LDG cluster, you need accounting flags

```
export LIGO_USER_NAME=albert.einstein
export LIGO_ACCOUNTING=
```
