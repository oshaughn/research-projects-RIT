import setuptools
import numpy as np

with open("README.md", "r") as fh:
    long_description = fh.read()


with open("requirements.txt",'r') as f:
    lines = f.readlines()
    for indx in np.arange(len(lines)):
        lines[indx]=lines[indx].rstrip()
REQUIREMENTS = {
  "install" : lines #["numpy>=1.14.0","scipy>=1.0.1","h5py", "corner", "numba", "scikit-learn<=0.20"]

 }

#print REQUIREMENTS

import glob
my_scripts = glob.glob("MonteCarloMarginalizeCode/Code/*.py")
my_scripts += glob.glob("MonteCarloMarginalizeCode/Code/create*py")
my_scripts += glob.glob("MonteCarloMarginalizeCode/Code/helper*py")
my_scripts += glob.glob("MonteCarloMarginalizeCode/Code/util*py")
my_scripts = list(set([x for x in my_scripts if not ('test_' in x)]))  # try to remove duplicates, and tests
my_scripts += glob.glob("MonteCarloMarginalizeCode/Code/*.sh")
my_scripts += glob.glob("MonteCarloMarginalizeCode/Code/integrate_likelihood*")
my_scripts += glob.glob("MonteCarloMarginalizeCode/Code/convert*")
my_scripts += glob.glob("MonteCarloMarginalizeCode/Code/switcheroo")
#print my_scripts
# No packages found
#print setuptools.find_packages('MonteCarloMarginalizeCode/Code')

my_extra_source  = glob.glob("MonteCarloMarginalizeCode/Code/cuda*.cu")

# Remove old things
#  - ourio.py, ourparams.py
#  - anything with 'test'
#print " Identified scripts ", my_scripts\
#print setuptools.find_packages('MonteCarloMarginalizeCode/Code')

setuptools.setup(
    name="RIFT",
    version="0.0.7",
    author="Richard O'Shaughnessy",
    author_email="richard.oshaughnessy@ligo.org",
    description="RIFT parameter estimation pipeline. Note branch used is temp-RIT-Tides-port_master-GPUIntegration!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ligo.org/richard-oshaughnessy/research-projects-RIT",
    package_dir = {'':'MonteCarloMarginalizeCode/Code'},
    py_modules = {"mcsampler", "mcsamplerGPU", "mcsamplerEnsemble", "MonteCarloEnsemble", "lalsimutils",'optimized_GPU_tools', 'Q_inner_product', 'SphericalHarmonics_gpu','vectorized_lal_tools','ROMWaveformManager','factored_likelihood','xmlutils', 'priors_utils', 'dag_utils','statutils', 'bounded_kde','multivariate_truncnorm', 'senni',"PrecessingFisherMatrix","EOSManager", "EOBTidalExternalC","BayesianLeastSquares"},
    packages=setuptools.find_packages('MonteCarloMarginalizeCode/Code'),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=my_scripts,
#https://docs.python.org/3/distutils/setupscript.html
   data_files=[('bin',my_extra_source)],
   install_requires=REQUIREMENTS["install"]
)
