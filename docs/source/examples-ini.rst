====================================
RIFT pipeline examples: the ini file
====================================

The ini file provides a way to provide fine-grained control of the many inputs a RIFT analysis can use.
The ini file format corresponds to the `lalinference ini format <https://github.com/lscsoft/lalsuite-archive/blob/master/lalapps/src/inspiral/posterior/lalinference_pipe_example.ini>`__
with a single named section of options, corresponding to arguments of the pipeline.  As a concrete example, the
following ini file block will select a specific approximant and fitting method:


.. code-block:: console

   [rift-pseudo-pipe]
   approx="IMRPhenomPv2"
   cip-fit-method="rf"    # fitting method for CIP
   internal-use-aligned-phase-coordinates=True  
   cip-sampler-method="GMM"   # integration method for CIP
   internal-correlate-parameters-default=True   
   add-extrinsic=True   # run provides extrinsic samples at the end
   l-max=2
   ile-n-eff=10


Note that any command-line arguments *take precedence* over parameters specified in the ini file.
If an ini file is specified, the pipeline will not make certain guesses (e.g., which interferometers to use in the
analysis; the calibration versions or channel names to use).
  

Using an ini file for fake data
----------------------------------

.. code-block:: console
		
  [lalinference]
  fake-cache =   ....  # insert example


Using an ini file for a production analysis
-------------------------------------------
For a production analysis, you will be provided with a pre-existing ini file, with the correct channel names, cache
files, event times, PSD files, 
