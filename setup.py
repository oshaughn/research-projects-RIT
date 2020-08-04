import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


with open("requirements.txt",'r') as f:
    lines = f.readlines()
    for indx in range(len(lines)):
        lines[indx]=lines[indx].rstrip()
REQUIREMENTS = {
  "install" : lines #["numpy>=1.14.0","scipy>=1.0.1","h5py", "corner", "numba", "scikit-learn<=0.20"]

 }

#print REQUIREMENTS

my_library_prefixes=["mcsampler", "mcsamplerGPU", "mcsamplerEnsemble", "MonteCarloEnsemble", "lalsimutils",'optimized_gpu_tools', 'Q_inner_product', 'SphericalHarmonics_gpu','vectorized_lal_tools','vectorized_general_tools','ROMWaveformManager','factored_likelihood','xmlutils', 'priors_utils', 'dag_utils','statutils', 'bounded_kde','multivariate_truncnorm', 'senni',"PrecessingFisherMatrix","EOSManager", "EOBTidalExternal","EOBTidalExternalC","BayesianLeastSquares","samples_utils","effectiveFisher",'gmm','weighted_gmm','gaussian_mixture_model','multivariate_truncnorm', "ModifiedScikitFit",'gp','our_corner','spokes','weight_simulations','MonotonicSpline','bounded_kde','xmlutils']
my_library_total = [("MonteCarloMarginalizeCode/Code/"+x+".py") for x in my_library_prefixes]

import glob
my_scripts = glob.glob("MonteCarloMarginalizeCode/Code/bin/*")
#print my_scripts
# No packages found
#print setuptools.find_packages('MonteCarloMarginalizeCode/Code')

my_extra_source  = glob.glob("MonteCarloMarginalizeCode/Code/RIFT/likelihood/cuda*.cu")

# Remove old things
#  - ourio.py, ourparams.py
#  - anything with 'test'
#print " Identified scripts ", my_scripts\
#print setuptools.find_packages('MonteCarloMarginalizeCode/Code')

setuptools.setup(
    name="RIFT",
    version="0.0.15.3rc3", # do not build on OSX machine, side effects
    author="Richard O'Shaughnessy",
    author_email="richard.oshaughnessy@ligo.org",
    description="RIFT parameter estimation pipeline. Note branch used is temp-RIT-Tides-port_python3_restructure_package (which will become master shortly)!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ligo.org/richard-oshaughnessy/research-projects-RIT",
    package_dir = {'':'MonteCarloMarginalizeCode/Code'},
#    py_modules =set(my_library_prefixes),
    packages=setuptools.find_packages('MonteCarloMarginalizeCode/Code'),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=my_scripts,
#https://docs.python.org/3/distutils/setupscript.html
# https://docs.python.org/2/distutils/setupscript.html
# Would be preferable to be *global* path, not relative to install. Depends on if doing user install or not
# This pathname puts it in the same place as the other files, in site-packages/
   data_files=[('RIFT/likelihood',my_extra_source)],
   install_requires=REQUIREMENTS["install"]
)
