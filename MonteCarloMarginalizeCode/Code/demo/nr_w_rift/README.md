## Intro

Using RIFT, we can compare gravitational wave data to numerical relativity simulations.  Numerical relativity simulations contain more information than analytical waveform models, and as such, may be more accurate for extracting true parameters from gravitational wave data.  This tutorial assumes the user has  previously accessed LIGO computing resources and has completed the basic RIFT tutorial (i.e. has a proper environment and experience setting appropriate path variables).  

Each numerical relativity simulation corresponds to a particular value for each of seven of the intrinsic  parameters (mass ratio  q  and the three components of each spin vector) but can be scaled to an specified  
value of the total mass. The total mass can be changed and the individual masses will then be calculated  according to  q  - this means each simulation essentially has a single variable parameter for an eight-dimensional  parameter space. For eccentric simulations, this becomes a nine-dimensional parameter space with intrinsic  
parameters  λ, but still only total mass can be changed.

## Setup
We currently store numerical relativity data in directories on CIT, primarily hosted by ROS and accessed  by others via setting environment variables.  

```
1  HOME_ROS=/home/richard.oshaughnessy  
2  export ROS_PROJS=${HOME_ROS}/unixhome/Projects  
3  export NR_BASE=${HOME_ROS}/unixhome/PersonalNRArchive/Archives/  
4  export PATH=${PATH}:${HOME_ROS}/unixhome/PersonalNRArchive/Archives  
5  export PYTHONPATH=${PYTHONPATH}:${HOME_ROS}/unixhome/PersonalNRArchive/Archives  
6  export ROM_SPLINE=/home/oshaughn/unixhome/PersonalNRArchive/  
7  export PYTHONPATH=${PYTHONPATH}:${ROM_SPLINE}  
```
This setup is reasonable for those who wish to quickly and easily access the simulations, and is generally the advised method. However, others have also built specific catalogs in their own directories. There will be instructions for building NR metadata and/or adding individual simulations to the metadata in the future as the infrastructure and documentation get updated.

## Instructions
In general, RIFT uses a waveform approximation/template and a set of intrinsic parameters to perform likelihood calculations on an unstructured grid in its first stage, integrate likelihoods extrinsic (ILE). However, this grid can also be constructed in such a way that each grid point is a numerical relativity simulation. For certain events, grid granularity may pose problems. The pipeline setup is a bit different than standard RIFT runs, but much of the infrastructure exists since something like this was done previously for GW190521.

Because the pipeline writer script (`util_RIFT_pseudo_pipe.py`) is designed to perform setup with a variety of waveform models rather than NR simulations, the setup for RIFT with NR must be done a bit more manually. We can use a Makefile to do most of the heavy lifting.

### Data
The Makefile contains a "data" target, which uses `gracedb` to access the online catalog data for the user-specified event. You must provide both the GID and the SID for the data target to work. Se these at the top of the Makefile. For example, for GW190521
```
    GID=G333631
    SID=S190521g
```

The GID can be found using the online GraceDB portal (https://gracedb.ligo.org/search/) and searching the SID. Note that you must login to see all the information, including the GID. You may need to run `ligo-proxy-init albert.einstein` before you are able to download the files using GraceDB. The data target downloads the files you need, finds the production ini file based on your specified SID, and uses `util_RIFT_pseudo_pipe.py` to create the `local.cache` file needed by RIFT. This will get reused by the NR setup run. To do this, simply run `make data`.

Next, you will have to check the ini file in the data directory to ensure the channel names in your Makefile are correct. For now, you must manually open `data/base.ini`, find the `channels` variable, and copy the strings into the Makefile. For example, in the Makefile for GW190521:
```
    CHANNEL_NAME=DCS-CALIB_STRAIN_CLEAN_C01
    CHANNEL_NAME2=Hrec_hoft_16384Hz
```

These channel names match the channel names in the ini file, where `CHANNEL_NAME` is for H1/L1, and `CHANNEL_NAME2` is for V1. These variables in the Makefile then set the `--channel-name` arguments in the ILE argument string, so it is very important that these are correct. Eventually this will be done automatically, but the user must do this manually for now. This only needs to be performed once per real event.

### Initial Grid
The Makefile creates an initial grid which stores information for an NR simulation at each point that will get passed to the ILE stage of RIFT. It calls the `util_NRExtrudeOverlapGrid.py` script from the RIFT repository. Command line arguments passed to this script are specified by user-set variables in the Makefile, such as the NR group name, mass range, number of grid points, and the eta range. For example, again for GW190521:
```
    NR_GROUP=Sequence-RIT-All
    MTOT_RANGE='[180,220]'
    ETA_RANGE='[0.07,0.260]'
    NPTS=30
```

The grid is then made simply by running `make grid`.

The number of points specifies how many different points per simulation file with mass being the varied quantity. The mass ratio, spins, and eccentricity are fixed by the simulation parameters. Currently, you can specify that eccentricity is included with a `--eccentricity` tag in the arguments in the Makefile, and you can request either `--aligned-only` or `--nospin-only`.

### RIFT Run Directory
Finally, once the data and the grid are present, the RIFTxNR run directory is ready to be built. First, the `EVENT_TIME` variable also needs to be specified in the Makefile. This value can be found on the GraceDB portal for that event. Similar to the channels, this value is passed to the ILE argument string. The user may also choose to change the reference frequency and the number of higher order modes to include. Once the arguments are set, the user runs  `make rundir`  to create the run directory based on the argument strings and a call to RIFT's internal script, `create_event_nr_pipeline_with_cip`. Once the run directory is build, go into the `rundir` and try running a command single to make sure that everything is running as expected. If it looks okay, the job can be submitted. You may need to verify the PSDs are present.

One step worth mentioning that differs from a standard RIFT run is the "refine" stage. Refine calls an internal script named `util_TestSpokesIO.py`, which runs after ILE. This step removes duplicate mass entries in the output `all.net`, and refines the grid to points within a certain fraction of the peak log-likelihood. The refined grid gets passed back to a second (and usually for NR, final) iteration of ILE. CIP only runs a single time at the end to construct the final resultant posterior distribution.
