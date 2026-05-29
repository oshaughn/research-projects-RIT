"""
RIFT.hyperpipe
==============

Pipeline-construction utilities for the RIFT "hyperpipeline" --- the iterative
marginalize + fit + puff loop used to infer hyperparameters (e.g. EOS,
population-model, or generic high-level) from one or more underlying
likelihood evaluators ("marg drivers"). Analogous in spirit to the
``util_RIFT_pseudo_pipe.py`` driver for the single-event GW PE pipeline,
but with:

  * ini-based / Hydra-based configuration (see ``config.py``)
  * flexible multi-event input (``marg-list`` semantics, see ``marg_list.py``)
  * a coordinate-transformation framework that mirrors the CIP
    ``--supplementary-coordinate-code`` / ``--parameter`` / ``--integration-parameter-range``
    conventions already used by ``util_ConstructEOSPosterior.py``
    (see ``coords.py``)
  * driver toolkits to spare downstream users from re-deriving the
    fragile argument strings, output-file names, and contract points
    each marg driver must respect (see ``drivers``)

The top-level CLI driver remains ``bin/util_RIFT_hyperpipe.py``; this
package is what it delegates to.
"""

from . import coords
from . import config
from . import marg_list

__all__ = ['coords', 'config', 'marg_list', 'drivers']
