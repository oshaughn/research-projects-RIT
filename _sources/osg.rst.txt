Using the OSG
-------------

Computing resources are available through the  International Gravitational-Wave
Observatory Network `(IGWN) <https://computing.docs.ligo.org/guide/>`__
Computing Grid, which is comprised of computing resources provided by member
groups and external partners in partnership with the Open Science Grid
`(OSG) <https://osg-htc.org/>`__. The grid relies on HTCondor for use of the
resources. 

Using the OSG for RIFT jobs is *fast*. It is a great tool, just make sure
the appropriate arguments are supplied both in the workflow and to the
environment so the jobs actually run. There are a few requirements for
submitting RIFT jobs via HTCondor on the OSG which differ from other
machines in the network. To submit a job, you must use
a flag, :code:`--import-env`, which imports all environment variables

.. code-block:: console
		
	condor_submit_dag -import_env master.dag

Ensure that the ini file you used to create your workflow included arguments
indicating that your job uses the OSG. This adds the appropriate arguments
in the .sub files used by your job.

Finally, you need various additional environment variables, ranging from
absolutely essential to extremely helpful:

.. code-block:: console
		
	RIFT_REQUIRE_GPUS=(DeviceName=!="Tesla K10.G1.8GB")&&(DeviceName=!="Tesla K10.G2.8GB")
	RIFT_LOWLATENCY=True
	RIFT_GETENV=LD_LIBRARY_PATH,PATH,PYTHONPATH,*RIFT*,LIBRARY_PATH
	SINGULARITY_RIFT_IMAGE=/cvmfs/singularity.opensciencegrid.org/james-clark/research-projects-rit/rift:production

Additionally, if you are using a waveform model implemented in `gwsignal`,
you must export an extra environment variable:

..code-block:: console

	RIFT_GETENV_OSG=*RIFT*,NUMBA_CACHE_DIR

		
A sample OSG command line might look like this:

.. code-block:: console

	util_RIFT_pseudo_pipe.py --gracedb-id GID_GOES_HERE --use-online-psd --online --choose-data-LI-seglen --use-ini `pwd`/tryme.ini --use-coinc `pwd`/coinc.xml

