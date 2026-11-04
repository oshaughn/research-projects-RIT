# Generating Mock Compact Binary Events with GWKokab and Producing Realistic PEs with RIFT

This tutorial describes a **complete, reproducible workflow** for:

1. Generating mock compact-binary injection populations using **GWKokab**, and  
2. Producing realistic **parameter estimates (PEs)** for those injections using **RIFT**.

To avoid dependency and version conflicts, **GWKokab and RIFT must be installed in separate conda environments**.

---

## Overview of the Workflow

1. Install and validate **GWKokab**
2. Generate mock injections (`injections.dat`)
3. Sanity-check GWKokab population inference
4. Configure **RIFT** and the Makefile
5. Prepare injections for RIFT
6. Generate MDC injections
7. Create RIFT run directories
8. Submit PE jobs
9. Produce plots
10. Run diagnostics and checks

## Step 1 — Set Up GWKokab

Follow the official installation guide:

https://gwkokab.readthedocs.io/en/latest/installation.html

Create and activate a dedicated conda environment for gwkokab:

```bash
conda create -n gwkenv python=3.13
conda activate gwkenv
```
Clone and install GWKokab:

```bash
git clone https://github.com/gwkokab/gwkokab.git
cd gwkokab
make install PIP_FLAGS=--upgrade EXTRA=cuda13
```

## Step 2 — Generate Mock Events
Use the GWKokab example workflow to generate mock posterior estimates:
- Documentation:
https://gwkokab.readthedocs.io/en/latest/examples/generating_mock_posterior_estimates.html
- Example repository:
https://github.com/gwkokab/hello-gwkokab/tree/main/generating_mock_posterior_estimates

- Run one of the generators (e.g. genie_n_pls_m_gs).
This will create a directory named data/realization_* containing: `injections.dat` and fake posteriors files of each event.
- Each row corresponds to the intrinsic parameters of one compact-binary system.
The number of rows equals the number of simulated events.

## Step 3 — Sanity Check: GWKokab Population Inference
Before proceeding to RIFT, verify that GWKokab can recover the injected population hyperparameters either with delta error or fake posteriors.
- Documentation:
https://gwkokab.readthedocs.io/en/latest/examples/hbi_discrete_method.html
- Example repository:
https://github.com/gwkokab/hello-gwkokab/tree/main/hbi_discrete_method

Only proceed once this step works correctly.

## Step 4 — Set Up RIFT (Separate Environment)
- RIFT uses a different software stack and must be installed in a separate conda environment.
- The scripts in this repository automate the RIFT workflow for population studies:
    - Environment creation
    - Injection conversion
    - Run-directory setup
    - Job submission

All steps are controlled through a single Makefile.
## Step 5 — Configure the Makefile
Before running any RIFT commands, update the required variables in the Makefile.
- Mainly you can provide desired file names, repo names, env name, absolute paths, and user name.

```bash
ENV_NAME=rift-pop
USER = muhammad.zeeshan

# Must match the run directory specified in the ini file
RUNDIR     = '${PWD}/ecc_injections'
PARAM_FILE = '${PWD}/inj_demo.dat'
INI_FILE   = '${PWD}/pop-example.ini'
```
All the user desired options *must* be set in the ini file. This includes things like what spin settings you need (none, aligned, precessing), whether or not you want an eccentric analysis, etc. All analysis choices (mass ranges, spins, eccentricity, waveform approximant, priors) must be specified in `pop-example.ini` file as follows.

```bash
[pp]     
n_events=204
working_directory=ecc_injections
test_convergence=True
[priors]
mc_min=4.35
mc_max=70.0
m_min=3.0
eta_min=0.05
eta_max=0.24999
... spin, eccentricity, redshift etc ...
```

Important:
All prior ranges in pop-example.ini must exactly match the ranges used to generate injections.dat with GWKokab.
If they differ, the resulting PEs will be invalid. So, always double check `[pp]` and `[priors]` section in `.ini` file.
You also need to update noise frame paths and user name in `.ini` file.

## Step 6 — Create the RIFT Environment
```bash
make setup-env
```
This command:
- Creates the rift-pop environment based on igwn-py310
- Clones the RIFT codebase
- Installs all required dependencies
- Enables eccentric waveform support (e.g. SEOBNRv5EHM)

Note: This setup assumes access to the LDG and igwn-py310.
## Step 7 — Prepare Injections for RIFT
- Copy `injections.dat` from the GWKokab output directory into this RIFT repository.
- Add luminosity distance, which is required by RIFT:
```bash
python lum_distance.py --input ./injections.dat --output ./injections.dat
```
## Step 8 — Generate MDC Injections

This step:
- Converts injections into RIFT-compatible format
- Generates `mdc.xml.gz` and save it into `ecc_injections`
- Writes all outputs into the directory specified by RUNDIR (ecc_injections)
- It uses:
    - PARAM_FILE for parameters
    - The waveform approximant specified in the Makefile

## Step 9 — Create RIFT Run Directories
```bash
make rundir
```
This command:
- Generates signal and combined frames for each event
- Executes pp_RIFT_with_ini
- Creates production-style RIFT run directories

Important: Copy PSDs into Each Run Directory, because ach event requires its own PSD files:

```bash
for d in /home/muhammad.zeeshan/projects/research-projects-RIT/MonteCarloMarginalizeCode/Code/demo/populations/ecc_injections/analy*/rundir; do
  cp /home/muhammad.zeeshan/projects/research-projects-RIT/MonteCarloMarginalizeCode/Code/demo/populations/psds/rundir_psds/*psd.xml.gz "$d"
done
```
## Step 10 — Submit PE Jobs
```bash
make submit
```
it will submit your PE production jobs and will take days to complete. So, you can monitor using the following commands.
```bash
condor_q
condor_q -hold
condor_q -run
condor_rm -all
```
For additional information use `condor -h`
## Step 11 — Plotting Results

RIFT provides automated plotting scripts which you can also use during runs to see the plots.
- plot_iterationsAndSubdag_animation_with.sh
- plotme_anim_ecc.sh

To generate corner plots for all events:
```bash
plot_all.sh ./ecc_injections
```
## Debugging, Sanity Checks, and Common Issues
### Posterior Completion Check
```bash
for i in analysis_event_*; do
  echo $i $(ls $i/rundir/post*.dat $i/rundir/iter*cip/post*.dat 2>/dev/null | wc -l)
done
```
A completed run should contain `posterior_samples-4.dat`
## Step 12 — Gather PE file for Population Inference
- We need the PE files from each rundir in a single folder to do the population inference using GWKokab. You can run the script
```bash
./collect_all.sh
```
It will collect the required file `extrinsic_posterior_samples.dat` for each rundir and save them with their numbers in a folder called `collected_dat_files`.
- Finally, you can run the python script as follows to rename the header accepted by GWKokab and also let you choose random PEs per event.
```bash
python gwk_pop_conversion.py --input-dir ./collected_dat_files --output-dir ./rift_pes --n-rows 2000
```



### Job Progress Tracking
```bash
condor_q -submitter muhammad.zeeshan -long -nobatch | grep Iwd | sort | uniq -c
```
### Likelihood Diagnostics
```bash
wc -l analysis_event_*/run*/con*.composite
```
and 
```bash
for i in an*/run*; do
  echo $i $(sort -rg -k11 $i/all.net | head -n 1 | awk '{print $11,$12,$10}')
done | sort -rg -k2
```
### Common Error: lal_path2cache
Error: FileNotFoundError: No such file or directory
Fix: remove .local folder in your home repo.
```bash
rm -rf ~/.local/bin/lal_path2cache` then rerun `make rundir
```

### Important Notes
- Always use absolute paths in RIFT configuration files.
- local.cache files must not be empty.
- Very loud events may take significantly longer to converge.
- Discard or re-run the events who give log-likelihood less than 5.
- If your corner plot of each event iteration legend shows `F3`, it shows a good complete run.

