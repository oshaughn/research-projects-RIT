---------------------------------------------
``util_RIFT_hyperpipe.py``: Configuration-driven pipeline
---------------------------------------------

``util_RIFT_hyperpipe.py`` is the top-level driver for the hyperpipeline.
It consumes a Hydra/OmegaConf configuration file and drives the
same ``create_eos_posterior_pipeline.py`` machinery that the legacy
Makefile path uses — but without the manual args-file authoring.

Usage::

    util_RIFT_hyperpipe.py --config hyperpipe_conf.yaml

Hydra overrides work as expected::

    util_RIFT_hyperpipe.py --config hyperpipe_conf.yaml \
        arch.n-iterations=10 general.use-osg=true

A bare run (no ``--config``) loads the installed default
``hyperpipe_conf.yaml`` next to the script. The demo configuration in
``MonteCarloMarginalizeCode/Code/demo/hyperpipe/hyperpipe_conf.yaml`` is
the reference template.


Configuration schema
^^^^^^^^^^^^^^^^^^^^

The configuration is organized into six top-level sections.

``arch``
    Controls iteration count, chunk size, and parallelization.

``post``
    Configures the posterior-construction stage
    (``util_ConstructEOSPosterior.py``): which parameters to fit,
    integration ranges, coordinate module, and MC sampler settings.

``marg-list``
    One entry per (likelihood driver, event) pair. Each entry names an
    executable, its argument string, the event file, and batch size.
    Multiple entries enable heterogeneous multi-event inference.

``puff``
    Configures the puffball / parsimonious-placement stage
    (``util_HyperparameterPuffball.py`` or a tracer updater):
    puff factor, force-away distance, and (for tracers) sampler
    hyperparameters.

``test``
    Convergence-test settings (which stage runs
    ``convergence_test_samples``).

``init``
    Sourcing for the initial hyperparameter grid: either a path to an
    existing grid file, or a ``generation`` block that calls
    ``util_HyperparameterGrid.py`` automatically.

``general``
    Working directory, Condor resource requests, OSG/singularity flags,
    and retry behaviour.
