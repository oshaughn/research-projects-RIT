util_ParameterPuffball
=======================

The ``util_ParameterPuffball.py`` executable is used to "puff up" a distribution of parameters. It reads an existing set of parameters, assesses their covariance, and generates a new, expanded set of parameters by adding random noise based on that covariance. This is typically used to create a broader sampling distribution for subsequent integration or testing steps.

Overview
--------

In many RIFT workflows, it is necessary to move from a highly constrained set of parameters (e.g., a maximum likelihood point or a narrow grid) to a wider distribution that still respects the underlying correlations of the parameters. ``util_ParameterPuffball.py`` achieves this by calculating the covariance matrix of the input parameters and sampling from a multivariate normal distribution to perturb the points.

Usage
-----

Basic Command Line
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python util_ParameterPuffball.py [options]

Primary Options
~~~~~~~~~~~~~~

**Input and Output**

- ``--inj-file``: Path to the input XML file containing the parameter set.
- ``--inj-file-out``: Path to the output XML file (default: ``output-puffball``).
- ``--puff-factor``: Scaling factor for the covariance matrix. A value of 1.0 maintains the original covariance; values > 1.0 "puff up" the distribution (default: 1.0).
- ``--approx-output``: Waveform approximant to use when writing the output XML (default: ``SEOBNRv2``).

**Parameter Selection**

- ``--parameter``: Parameters to be used for covariance assessment and perturbation. These are the "fitting" parameters.
- ``--no-correlation``: A list of parameter pairs (e.g., ``"['mc','eta']"``) whose mutual correlation should be eliminated by zeroing out the corresponding covariance matrix terms.
- ``--random-parameter``: Parameters to be sampled uniformly at random over a specified range, uncorrelated with others.
- ``--random-parameter-range``: The ranges for the ``--random-parameter`` variables (passed as string evaluations of Python lists).

**Constraints and Filtering**

- ``--downselect-parameter``: Parameters used to filter out points that fall outside a specific range.
- ``--downselect-parameter-range``: The allowed ranges for the downselection parameters.
- ``--mc-range``, ``--eta-range``, ``--mtot-range``: Convenient shortcuts for downselecting based on chirp mass, eta, or total mass.
- ``--reflect-parameter``: Parameters that should be "reflected" back into their allowed range if they exceed the boundaries.
- ``--force-away``: If > 0, uses the inverse covariance matrix to compute a Mahalanobis distance and discards points that are too close to existing points.
- ``--enforce-duration-bound``: Maximum allowed waveform duration. Points producing signals longer than this are discarded.
- ``--fail-if-empty``: Causes the tool to exit with an error if no points survive the filtering process.

Functional Logic
----------------

1. **Coordinate Extraction**: Reads the input XML and extracts the specified ``--parameter`` values. Physical masses are scaled to solar masses.
2. **Transformation**: Performs a natural log transformation on parameters that are strictly positive (e.g., tidal parameters like ``lambda1``, ``lambda2``, ``LambdaTilde``) to ensure perturbed values remain positive.
3. **Covariance Calculation**:
    - Computes the covariance matrix of the extracted parameters.
    - Applies the ``--puff-factor`` scaling.
    - Handles singular matrices by adding a small regularization term to the diagonal of the inverse covariance (pseudo-inverse).
    - Zeroes out correlations for pairs specified via ``--no-correlation``.
4. **Perturbation**:
    - Samples random offsets from a multivariate normal distribution based on the modified covariance matrix.
    - Adds these offsets to the original parameters.
5. **Boundary Handling**:
    - **Reflection**: Reflects parameters specified by ``--reflect-parameter`` back into their defined ranges.
    - **Eta Constraint**: Specifically ensures ``eta`` remains within the physical range [0, 0.25].
6. **Filtering (Downselection)**:
    - Converts parameters to a common coordinate system to check constraints.
    - Discards points that violate ``--downselect-parameter-range`` or the predefined physical bounds for masses and eta.
    - Discards points that violate the ``--enforce-duration-bound`` or the unit spin magnitude constraint ($\chi \le 1$).
    - Optionally discards points too close to the original set using the ``--force-away`` Mahalanobis distance check.
7. **Finalization**: 
    - Undoes the log transformations.
    - Applies random values to parameters specified by ``--random-parameter``.
    - Writes the resulting parameter set to the output XML.

Output Details
---------------

The tool prints the updated covariance matrix and the final count of exported points. The output is a standard `ligolw` XML file containing the "puffed" parameter set.

Opt in to mu coordinates and random transverse discovery
------------------------------------------------------

For an initially aligned precessing-BBH grid, use a covariance puff in
``mu1, mu2, q=m2/m1, s2z`` and an independent **random redraw** of both
transverse spins. Transverse coordinates are never included in the fitted
covariance: an aligned seed grid has no transverse variance. This mode does
not use the legacy singular-covariance fallback. Genuine zero directions in
the four-dimensional covariance remain zero.

.. code-block:: bash

   util_ParameterPuffball.py --inj-file selected-grid.xml.gz \
       --inj-file-out puff-mu --puff-coordinate-system mu-random-transverse \
       --puff-factor 1 --mu-puff-n-proposals 1000 --random-seed 448 \
       --downselect-parameter chi1 --downselect-parameter-range '[0,0.99]' \
       --downselect-parameter chi2 --downselect-parameter-range '[0,0.99]' \
       --fail-if-empty

Omit ``--parameter``: this mode uses a fixed chart. Input seeds should already
be selected using likelihood results; this executable does not rank them.
The default budget makes one attempt per input seed, preserving seed-branch
coverage. An explicit proposal count instead selects parents uniformly.
Each attempt perturbs the joint mass/aligned coordinates by ``--puff-factor``
times their measured covariance factor. An atomic vectorized inverse reconstructs both masses and aligned
spins together. The inverse uses the existing mu coordinate convention
(``tools.fref=200 Hz``), independently of waveform ``--fref``.

For each body the transverse radius is uniform between zero and
``min(mu_puff_transverse_max, sqrt(1-siz**2))``; azimuth is uniform on
``[0,2*pi)``. Both spins are redrawn independently, even when
``--puff-factor 0`` leaves the mass/aligned coordinates fixed. The optional
``--mu-puff-transverse-max`` defaults to 1. This is an exploration distribution,
not an isotropic-spin prior, and not an additive Gaussian transverse puff.

``--mu-puff-n-proposals`` is an attempted budget, defaulting to the input row
count. Invalid inverses and failed native downselections are discarded without
retries or clipping. Other waveform fields are copied from the selected parent.
Both XML and hyperpipeline ASCII grids are supported; ASCII columns are
preserved. A ``<output>_mu_puff.json`` sidecar records the attempted, inverse-
accepted and exported counts, random seed and proposal settings.

The prototype supports native mass/spin downselection, frequency overrides and
duration bounds. It refuses legacy ``--parameter``, ``--reflect-parameter``,
``--no-correlation``, ``--force-away``, random/append overrides, effective
eccentric-phase puffing and hyperbolic force modes instead of ignoring them.
There is no automatic pipeline behavior change without the opt-in below.

Select discovery by CIP batch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   util_RIFT_pseudo_pipe.py [usual precessing BBH arguments] \
       --internal-puff-mu-random-transverse

This BasicIteration opt-in enables the existing ``internal_puff_transverse``
procedure and writes ``args_puff_list.txt`` paired line-for-line with the final
``args_cip_list.txt``. Only the first finite CIP batch uses mu/random-transverse
discovery; every later batch uses the existing reflected transverse covariance
puff. A ``Z`` refinement subdag inherits its own stage's ordinary puff settings,
not the discovery configuration. The first puff follows the first CIP and its
points are scored by the next iteration's ILE puff jobs; this does not alter
the initial grid before iteration-zero ILE.

The first batch's count determines the discovery duration. Existing cadence
and stop settings govern when puff jobs run; the opt-in ensures ``puff-max-it``
covers the first batch. It retains helper bounds, frequency settings, duration
limits and puff factor. Early discovery omits legacy force-away exclusion:
mass/aligned proximity does not make a random transverse orientation redundant.
Later stages retain it. ``args_puff.txt`` remains the ordinary single-stage
configuration. With the opt-in absent, the old argument files and job routing
are unchanged. The initial prototype rejects nonprecessing, matter, eccentric,
hyperbolic, AMR, Alternate/MultiApprox and transverse-tail-bundle combinations.

BasicIteration also accepts a manually supplied ``--puff-args-list`` with the
same count/``G``/``Z`` prefixes and number of lines as ``--cip-args-list``::

   2 --puff-coordinate-system mu-random-transverse --puff-factor 1
   3 --parameter mc --parameter delta_mc --parameter s1z --parameter s2z
   Z --parameter mc --parameter delta_mc --parameter s1z --parameter s2z

Pass it with ``--puff-cadence`` and ``--puff-max-it``. It is mutually exclusive
with the legacy ``--puff-args``. Stage selection uses the builder's actual CIP
batch expansion, including repeated and final-extended stages. A ``Z`` child
receives a single ordinary ``--puff-args`` file for its selected stage.

Qualification
~~~~~~~~~~~~~

The vectorized inverse was checked against forward mu coordinates, including
negative mu1, equal masses, near-Kerr spins and invalid proposals. Earlier
PP005 retrospective support tests motivated the mu chart; those tests used
Gaussian transverse floors and do not establish likelihood yield for this
random-redraw proposal. Re-score candidates with production ILE before using
them in inference. No posterior or evidence weights come from these coordinates.
