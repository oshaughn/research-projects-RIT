Raw distance likelihood contracts
=================================

``RIFT.compression.distance`` evaluates raw distance exports. It is separate
from :doc:`posterior_compression`, whose reconstructed likelihood has an
arbitrary event constant. Existing compression executables, INI options and
pipeline defaults are unchanged. No catalog tail model is enabled by this API.

Normalization
~~~~~~~~~~~~~

``DistanceLikelihood`` requires schema ``rift-raw-distance-v1`` and quantity
``raw_log_likelihood``. Its core predictor must return offset-restored native
log likelihood, not a posterior density, normalized conditional, or the
hybrid that preserves an earlier intrinsic marginal. Numerical overflow offsets
must be restored before fitting. No event or intrinsic-state re-zeroing occurs.

The contract records:

* ``features``: the exported ``FEATURES`` tuple, in that order;
* ``spin_frame``, ``reference_frequency_hz``, ``spin_caps`` and source provenance;
* ``intrinsic_bounds``: lower and upper eight-coordinate support arrays;
* ``distance_support_mpc`` and ``distance_scale_mpc``;
* ``normalization``: native measure, provenance, restored-offset declaration,
  and a noise baseline;
* ``validation_status`` and optional tail overlap/validation provenance.

The native measure records the effective time grid, window, quadrature and
nuisance-prior mass. A constant baseline uses ``kind = constant`` and explicit
``raw_zero_signal_log_likelihood``. A state-dependent baseline uses
``kind = state_evaluator`` and ``evaluator_id``. Supply its predictor and matching
identity to the constructor. A missing baseline is an error, never a default
zero. The ratio is ``raw_log_likelihood - native_zero_signal_log_likelihood``;
the raw quantity can carry an unnormalized time-quadrature volume even when
the noise ratio has a unit zero-signal limit.

``save_contract`` and ``load_contract`` round-trip JSON metadata. Backend model
serialization remains backend-owned; these functions do not save predictors or
deserialize executable objects. Load a matching raw core predictor explicitly.
``load_contract`` rejects experimental metadata unless
``allow_experimental=True``; the constructor does not check the status. The
status field is a producer declaration, not independent scientific
certification.

Conditioning provenance
~~~~~~~~~~~~~~~~~~~~~~~

For physical exports, ``source_provenance`` must bind the native implementation
name, version and source identity. Record conditioning separately for each
detector: the raw PSD table hash and frequency band, data-window and PSD-window
definitions, sample counts, and their evaluated mean-square factors. Record the
formula and numerical factor that convert the raw table into the effective PSD
used by the likelihood. Include runtime overrides, their evidence, and any
unverified assumptions. Equal raw PSD hashes do not establish equal weighting.

For example, ``S_effective = S_raw * mean(w_data**2)`` differs from
``S_effective = S_raw``. A producer that uses the ratio of data-window and
PSD-window powers must record both factors. Window shape parameters alone omit
finite-length and implementation differences. Resolve these quantities from
the event's deployed implementation and runtime; no package-version default is
chosen by this evaluator.

Keep this record in ``source_provenance["conditioning"]`` so contract persistence
and prepared snapshots retain it with the model. The validator requires source
provenance but does not audit this record or certify native-likelihood parity.
Raw/noise normalization controls test accounting separately from conditioning.
A constant time-prior normalization cancels in normalized PE; preserve its
measure when exporting absolute raw likelihoods.

Evaluation and caching
~~~~~~~~~~~~~~~~~~~~~~

The intrinsic chart is natural log detector-frame chirp mass in solar masses,
``delta = (m1-m2)/(m1+m2)``, followed by all six Cartesian component spins.
Component ordering is ``m1 >= m2``. Spins retain the declared frame and reference
frequency. The core's ninth feature is ``distance_scale_mpc / dL_Mpc``.

.. code-block:: python

   from RIFT.compression.distance import DistanceLikelihood, intrinsic_coordinates

   # contract.json and raw_core belong to the same validated export/backend.
   fit = DistanceLikelihood.load_contract("contract.json", raw_core,
                                         allow_experimental=True)
   x = intrinsic_coordinates(m1_detector, m2_detector, cartesian_spins,
                             mass_frame="detector")
   prepared = fit.prepare(x)
   raw = prepared.raw_log_likelihood(distance_nodes_mpc)
   ratio = prepared.log_noise_ratio(distance_nodes_mpc)

Outputs are ``(N,K)`` for ``N`` intrinsic states and a shared ``(K,)`` distance
grid. Per-state ``(N,K)`` grids are also supported. A 1-D input is always a
shared grid, even when ``K = N``; pass one distance per state as ``(N,1)``.
Scalars represent one
distance node. Queries outside declared support are rejected. The separate
``raw_zero_signal_log_likelihood()`` returns the recorded baseline without
extending accepted finite-distance support to infinity.

RF-only evaluation is the default. To enable a candidate tail, provide a
predictor returning ``(N,degree)`` coefficient arrays and pass
``enable_tail=True``. A prepared batch predicts coefficients once per unique
intrinsic state, including nonadjacent duplicates, and reuses them across
distance calls. Preparation snapshots metadata and captures its core predictor.
Do not refit predictors in place while prepared batches remain in use; create
new predictor objects and prepare new batches for a changed fit.
The tail is the recorded baseline plus a zero-intercept
polynomial in inverse distance. Its required ``tail.distance_scale_mpc`` binds
the coefficient basis independently of the core's distance feature scale.
``tail.degree`` declares the coefficient count, ordered from power one upward.
Coefficients fitted in ``anchor_mpc / dL`` use that anchor as their tail scale;
they must not be interpreted in a different inverse-distance basis.
Its overlap must be explicit and inside
accepted support; smoothstep blends raw log likelihood over that interval.
No pilot overlap or cosmology is a framework default.

Source masses and prior integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``intrinsic_coordinates(..., mass_frame="source", redshift=z)`` converts source
masses using explicit redshifts. Obtain these from the analysis cosmology,
including an inherited compression ``CosmologyContext`` where appropriate.
Scalar likelihood transformations add no density Jacobian. Integration weights
must include the population/reference measure and the appropriate Jacobian.

A prepared state fixes detector-frame intrinsic coordinates. At fixed source
masses, varying distance changes detector-frame masses: prepare those changed
states instead of reusing one detector-frame cache. No hidden-spin reduction
occurs. A reduced population likelihood must integrate the declared conditional
nuisance prior and its proposal quotient; that operation is separate from
distance evaluation.

For reference distance prior ``pi_ref``, a conditional density is proportional
to ``exp(raw) * pi_ref`` with its distance integral divided out. Its density
measure and bounds must be explicit. Combining that conditional with an older
intrinsic shape preserves the older amplitude convention; neither quantity is
the raw likelihood or its noise ratio.

Validation scope
~~~~~~~~~~~~~~~~

Known-polynomial tests cover baseline subtraction, the smoothstep overlap,
coefficient reuse, metadata round-trip, source-mass conversion, and the
metadata, support and shape guards. They do not validate finite-distance strain
likelihoods or a learned tail.
Grouped held-out intrinsic states, blind likelihood checks, overlap accuracy,
backend round-trip and converged population integrals remain required before
adopting a new fitted representation. Keep the RF fallback available.
