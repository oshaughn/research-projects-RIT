


# PP plots
See pp


# Monte carlo integration

* ``demo_mcsampler_foridiots.py``: easy-to-read demo code, not that stringent.
  Run it by hand; it is not a pytest test, and it writes a figure into the
  current directory.

* ``test_mcsamplerEnsemble_extended.py`` : best single-contact test.  3d gaussian integration, with plot of recovered CDF.
  It is a script, not a pytest target: ``.travis/test-integrate.sh`` runs it with
  ``--as-test``, and ``conftest.py`` keeps pytest from importing it.

* ``demo_mcsampler_rosenbrock.py``: simple 2d Rosenbrock demo.  Run it by hand;
  it is not a pytest test, and it writes ``fairdraw_rosenbrock_*.dat`` and
  ``cdf_rosenbrock.png`` into the current directory.

* ``expensive_before_merging/integrators``: **posterior shape-recovery merge gate** — REQUIRED before merging any integrator change into a production line; much stronger than the integral tests above (catches integral-invisible shape failures and silent n_eff collapse). See RIFT/integrators/TESTING.md.
