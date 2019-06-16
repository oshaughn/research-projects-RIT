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

setuptools.setup(
    name="RIFT",
    version="0.0.1",
    author="Richard O'Shaughnessy",
    author_email="richard.oshaughnessy@ligo.org",
    description="RIFT parameter estimation pipeline. Note branch used is temp-RIT-Tides-port_master-GPUIntegration!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ligo.org/richard-oshaughnessy/research-projects-RIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
   install_requires=REQUIREMENTS["install"]
)
