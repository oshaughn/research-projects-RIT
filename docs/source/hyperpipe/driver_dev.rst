Marg Driver Development
=========================

A *marg driver* is the executable that evaluates the (marginalized)
likelihood for each hyperparameter point. The hyperpipeline expects
drivers to accept a specific CLI contract; a base class and a Gaussian
toy driver are provided in ``RIFT.hyperpipe.drivers``.

Driver Contract
----------------

Every marg driver must accept the following CLI contract to integrate with the pipeline:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Flag
     - Description
   * - ``--using-eos PATH``
     - Input hyperparameter grid. The ``file:`` prefix is tolerated. Format: ``# lnL sigma_lnL <params...>``
   * - ``--eos_start_index INT``
     - First row to evaluate (inclusive).
   * - ``--eos_end_index INT``
     - Last row to evaluate (exclusive).
   * - ``--fname-output-integral PATH``
     - Output file path. Driver must write back same rows with evaluated ``lnL`` and ``sigma_lnL``.
   * - ``--outdir PATH``
     - Output directory; created if absent.
   * - ``--conforming-output-name``
     - If set, append ``+annotation.dat`` to the output file name.
   * - ``--fname``
     - Legacy passthrough; may be None or ignored.

Built-in Driver: ``util_HyperMargGaussian.py``
--------------------------------------------

The simplest way to run the Gaussian demo without authoring an executable
is to use the structured driver installed as a command-line tool::

    util_HyperMargGaussian.py \
        --using-eos file:my_grid.dat \
        --eos_start_index 0 --eos_end_index 1000 \
        --outdir out \
        --fname-output-integral lnL.txt \
        --conforming-output-name

The driver is a thin shim around
``RIFT.hyperpipe.drivers.gaussian.GaussianMargDriver`` and supports:

``--x-offset FLOAT``
    Position of the two modes along x (default: 4.0).

``--sigma2 FLOAT``
    Diagonal of the covariance matrix (default: 2.0).

``--unimodal``
    Drop the second mode at ``-x_offset`` for a single-mode Gaussian.

``--params X,Y,Z``
    Comma-separated names of the three grid columns (default: ``x,y,z``).

DIY Driver with ``MargDriverBase``
---------------------------------

For custom physics, subclass ``MargDriverBase``::

    from RIFT.hyperpipe.drivers.base import MargDriverBase

    class MyDriver(MargDriverBase):
        description = "My custom marg driver."

        def log_likelihood(self, row_values, column_names, opts):
            # row_values: strings from columns 2.. of the grid
            # column_names: list of parameter names from the header
            params = dict(zip(column_names, map(float, row_values)))
            # your physics here
            lnL = -0.5 * (params["x"]**2 + params["y"]**2)
            return lnL, 1e-3

    if __name__ == "__main__":
        MyDriver().run()

``MargDriverBase`` handles all CLI parsing, grid I/O, and output
formatting. Implement ``log_likelihood`` and (optionally)
``add_arguments`` for driver-specific flags.
