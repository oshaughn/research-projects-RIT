# Instructions

The scripts in this directory are intended for use with population studies. The directory contains the setup and execution workflow for generating GW signal injections from a pre-generated set of parameters to be passed to RIFT for parameter estimation.

- TODO: get PSDs

### Overview
The Makefile in this directory automates three key steps:

1. **Environment Setup**: creates a custom conda environment and clones the RIFT repository. Installs necessary python packages for eccenric waveforms (notably `SEOBNRv5EHM`).

2. **Injection Generation**: creates a working directory and writes mock injection data using specified parameters and waveform approximant.

3. **RIFT Run Directory**: Runs standard script `pp_RIFT_with_ini` to create a run directory using a production style setup with default settings based on the options set in the ini file.

### User set options in the Makefile
There are some options at the top of the Makefile that you should ensure are correct before making the targets. They are:
```
USER=albert.einstein
# note this *MUST* match dirname in your ini file:            
RUNDIR='${PWD}/ecc_injections'
PARAM_FILE='${PWD}/inj_demo.dat'
INI_FILE='${PWD}/pop-example.ini'
```
Note also that all the user desired options *must* be set in the ini file. This includes things like what spin settings you need (none, aligned, precessing), whether or not you want an eccentric analysis, etc. 

🚨 **Caution:** you must ensure the prior ranges in your ini file MATCH what was used to generate your injection parameter file, as those are the priors that will be used for PE. 🚨

### Makefile Targets

#### `make setup-env`
Clones the [research-projects-RIT](https://github.com/oshaughn/research-projects-RIT) repository and sets up a conda environment named `pop-ecc-inj`. This environment is cloned from `igwn-py310` and configured to work with RIFT. This is the environment that will be used

NOTE: this setup is configured to be run on the LDG - make sure you have access to `igwn-py310`.

#### `make injections`
Generates a set of GW signal injections using the script `write_mdc.py`. Injections are written into the run directory specified at the top of the Makefile (`RUNDIR`). The workflow will use the specified parameter file and waveform approximant (`APPROX`).

#### `make rundir`
Runs `pp_RIFT_with_ini` using the given .ini configuration file and the generated injection directory. For more info on this build script, refer to the [documention](https://rift-documentation.readthedocs.io/en/latest/injections.html)

#### `make_submit`
Sources the above environment and submits the dag that runs PE with RIFT.