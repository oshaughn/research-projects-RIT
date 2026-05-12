"""
RIFT.hyperpipe.drivers
======================

Toolkit for building "marg drivers" --- the per-event likelihood
evaluators the hyperpipeline orchestrates --- with the minimum of
ceremony.

Every driver in the hyperpipeline must obey the contract:

  Inputs (CLI, as seen by ``create_eos_posterior_pipeline``):
    --using-eos file:<grid>      # path to the current hyperparameter grid
    --using-eos-index <i>        # OR
    --eos_start_index <a>        #   index range [a, b) to evaluate
    --eos_end_index   <b>
    --fname-output-integral <X>  # name for output annotated file
    --fname-output-samples  <Y>  # ignored by most drivers (legacy)
    --outdir <D>                 # write outputs into D/
    --conforming-output-name     # if set, append '+annotation.dat'
    --fname <F>                  # dummy/passthrough; sometimes a real path

  Output:
    <outdir>/<fname-output-integral>[+annotation.dat]
        whitespace-separated, header:
            # lnL  sigma_lnL  <original grid column names>
        Rows are the rows of the input grid in [a, b), with the first
        two columns replaced by ln L and an estimate of its uncertainty.

The :class:`MargDriverBase` class encapsulates the boilerplate (arg
parsing, header sniffing, slicing, output formatting) so a concrete
driver only has to implement ``log_likelihood``.

See :mod:`.gaussian` for the canonical example.
"""

from .base import MargDriverBase, parse_marg_driver_args, write_marg_output

__all__ = ["MargDriverBase", "parse_marg_driver_args", "write_marg_output"]
