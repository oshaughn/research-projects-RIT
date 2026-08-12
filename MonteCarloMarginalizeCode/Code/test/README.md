


# PP plots
See pp


# Monte carlo integration

* ``test_mcsampler_foridiots.py``: easy-to-read test code, not that stringent.

* ``test_mcsamplerEnsemble_extended.py`` : best single-contact test.  3d gaussian integration, with plot of recovered CDF.

* ``test_mcsampler_rosenbrock``: Simple 2d test

* ``expensive_before_merging/integrators``: **posterior shape-recovery merge gate** — REQUIRED before merging any integrator change into a production line; much stronger than the integral tests above (catches integral-invisible shape failures and silent n_eff collapse). See RIFT/integrators/TESTING.md.
