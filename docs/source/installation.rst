============
Installation
============

Installing RIFT from release
----------------------------------
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

Install RIFT for development
----------------------------------

:code:`RIFT` is developed and tested for Python 3.6, and 3.7. In the
following, we demonstrate how to install a development version of
:code:`RIFT` on a LIGO Data Grid (LDG) cluster.

First off, clone the repository

.. code-block:: console

   $ git clone git@git.ligo.org:lscsoft/RIFT.git
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

      $ git clone https://git.ligo.org/lscsoft/bilby.git

Once you have cloned the repository, you need to install the software. How you
do this will depend on the python installation you intend to use. Below are
several easy-to-use options. Feel free to disregard these should you already
have an alternative.

Python installation
===================

.. tabs::

   .. tab:: conda

      :code:`conda` is a recommended package manager which allows you to manage
      installation and maintenance of various packages in environments. For
      help getting started, see the `LSCsoft documentation <https://lscsoft.docs.ligo.org/conda/>`_.

      For detailed help on creating and managing environments see `these help pages
      <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.
      Here is an example of creating and activating an environment named  bilby

      .. code-block:: console

         $ conda create -n bilby python=3.7
         $ conda activate bilby

   .. tab:: virtualenv

      :code`virtualenv` is a similar tool to conda. To obtain an environment, run

      .. code-block:: console

         $ virtualenv  $HOME/virtualenvs/RIFT
         $ source virtualenvs/RIFT/bin/activate

   .. tab:: CVMFS

      To source a :code:`Python 3.9` installation on the LDG using CVMFS, run the
      commands

      .. code-block:: console

         $ source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
         $ conda activate igwn-py39

     Documentation for this conda setup can be found here: https://computing.docs.ligo.org/conda/

Installing RIFT
=====================

Once you have a working version of :code:`python`, you can install
:code:`RIFT` with the command

.. code-block:: console

   $ pip install --upgrade git+file://${HOME}/PATH/TO/RIFT

Or, alternatively, if you already have a git version

.. code-block:: console

   $ pip install -e .

We recommend the second method, as it ensures the code you edit will be used



Dependencies
------------

:code:`RIFT` handles data from the interferometers directly using  lal library.

:code:`RIFT` uses several libraries to provide waveforms, including lalsimulation.
