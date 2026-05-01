============
Installation
============

This documentation describes how to perform parameter estimation of CBC triggers using RIFT. It is designed to be used by LVK members who have access to the LIGO DataGrid, which provides computing clusters and detector data to analyze. You can find more information on the `LDG wiki page <https://wiki.ligo.org/Computing/LDG/WebHome>`_.

Before you begin, login to your cluster of choice (`options here <https://wiki.ligo.org/Computing/LDG/ClusterLogin>`_) as usual using your :code:`albert.einstein` username.


Basic Installation from release
-------------------------------
There are a couple ways to install RIFT from the standard release. Note that if you plan to work on development for RIFT, it is recommended that you follow the setup instructions below using a virtual environment and installation from the source.

.. tabs::

   .. tab:: conda
	    
      To install the latest :code:`RIFT` release from `conda-forge
      <https://anaconda.org/conda-forge/RIFT>`_, run

      .. code-block:: console

         $ conda install -c conda-forge RIFT

      Note, this is the recommended installation process as it ensures all
      dependencies are met.

   .. tab:: pypi

      To install the latest :code:`RIFT` release from `PyPi
      <https://pypi.org/project/RIFT/>`_, run

      .. code-block:: console

         $ pip install --upgrade RIFT

      WARNING: this is not the recommended installation process, some
      dependencies (see below) are only automatically installed by using the
      conda installation method.


Creating an environment
=======================
A python virtual environment an be very helpful to ensure that the correct version of RIFT is being used when you are performing your analyses. Creating an environment where everything can be installed is straightforward.

.. tabs::

   .. tab:: conda

      :code:`conda` is a recommended package manager which allows you to manage
      installation and maintenance of various packages in environments. For
      help getting started, see the `LSCsoft documentation <https://lscsoft.docs.ligo.org/conda/>`_.

      For detailed help on creating and managing environments see `these help pages
      <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.
      Here is an example of creating and activating an environment named RIFT

      .. code-block:: console

         $ conda create -n RIFT python=3.7
         $ conda activate RIFT

   .. tab:: venv

      :code:`venv` is a similar tool to conda. To obtain an environment, run

      .. code-block:: console

         $ python3 -m venv /<choose a path>/
         $ source <your_path>/bin/activate

      You will next either need to :code:`pip install` RIFT or install it as a developer, as described below.

   .. tab:: CVMFS

      To source a :code:`Python 3.9` installation on the LDG using CVMFS, run the
      commands

      .. code-block:: console

         $ source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/bin/activate
         $ conda activate igwn-py39

     Documentation for this conda setup can be found here: https://computing.docs.ligo.org/conda/

Installing RIFT
===============

Once you have a working environment, you can do a basic :code:`RIFT` install with the command

.. code-block:: console

   $ pip install --upgrade RIFT

Install RIFT for development
----------------------------
However, some users may want to install RIFT for development, allowing them to add features and test them. In the
following, we demonstrate how to install a development version of :code:`RIFT` on a LIGO Data Grid (LDG) cluster.

First, clone the repository

.. code-block:: console

   $ git clone git@git.ligo.org:rapidpe-rift/rift.git
   $ cd RIFT/

.. note::
   If you receive an error message:

   .. code-block:: console

      git@git.ligo.org: Permission denied (publickey,gssapi-keyex,gssapi-with-mic).
      fatal: Could not read from remote repository.

   Then this indicates you have not correctly authenticated with your
   git.ligo account. It is recommended to resolve the authentication issue, but
   you can alternatively use the HTTPS URL: replace the first line above with

   .. code-block:: console

      $ git clone https://git.ligo.org/rapidpe-rift/rift.git

Once you have cloned the repository, you need to install the software.

.. code-block:: console

   $ python setup.py install --user

This method is helpful if you need to edit the source. This method also ensures all the necessary dependencies are installed.


Environment Variables
=====================

Once you are logged in, you will need to set environment variables. We recommend you put these into a script you run before commencing an analysis.

.. code-block:: console

    cat > setup_RIFT.sh
    export LIGO_USER_NAME=albert.einstein
    export LIGO_ACCOUNTING=ligo.sim.o4.cbc.pe.rift
    export PATH=${PATH}: # your path to RIFT here
    export CUDA_DIR=/usr/local/cuda  # only needed for GPU code
    export PATH=${PATH}:${CUDA_DIR}/bin  # only needed for GPU code


Dependencies
------------

:code:`RIFT` handles data from the interferometers directly using  :code:`lal` library.

:code:`RIFT` uses several libraries to provide waveforms, including :code:`lalsimulation`.

Additional environment variables are needed if you want to use waveforms through a non-lalsimulation interface. Such waveforms may include the python implementation of surrogate waveforms, NR waveforms, or the C++ implementation of TEOBResumS.

