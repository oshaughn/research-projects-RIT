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
my_scripts += glob.glob("MonteCarloMarginalizeCode/Code/*.sh")
my_scripts += glob.glob("MonteCarloMarginalizeCode/Code/integrate_likelihood*")
my_scripts += glob.glob("MonteCarloMarginalizeCode/Code/convert*")

# Remove old things
#  - ourio.py, ourparams.py
#  - anything with 'test'

print " Identified scripts ", my_scripts

setuptools.setup(
    name="RIFT",
    version="0.0.3",
    author="Richard O'Shaughnessy",
    author_email="richard.oshaughnessy@ligo.org",
    description="RIFT parameter estimation pipeline. Note branch used is temp-RIT-Tides-port_master-GPUIntegration!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ligo.org/richard-oshaughnessy/research-projects-RIT",
    package_dir = {'':'MonteCarloMarginalizeCode/Code'},
    py_modules = {"mcsampler", "mcsamplerGPU", "mcsamplerEnsemble", "MonteCarloEnsemble", "lalsimutils",'optimized_GPU_tools', 'Q_inner_product', 'SphericalHarmonics_gpu','vectorized_lal_tools','ROMWaveformManager','factored_likelihood','xmlutils', 'priors_utils', 'dag_utils','statutils', 'bounded_kde','multivariate_truncnorm', 'senni',"PrecessingFisherMatrix","EOSManager", "EOBTidalExternalC"},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=my_scripts,
   install_requires=REQUIREMENTS["install"]
)
