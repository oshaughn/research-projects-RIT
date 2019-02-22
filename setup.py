import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


REQUIREMENTS = {
  "install" : ["numpy>=1.14.0","scipy>=1.0.1"]

 }

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
