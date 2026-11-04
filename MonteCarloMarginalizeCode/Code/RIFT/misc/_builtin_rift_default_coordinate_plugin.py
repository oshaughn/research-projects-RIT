"""
Built-in coordinate-convert plugin used when the user passes
``--supplementary-coordinate-code rift_default``.

This is a thin adapter around the legacy hardcoded converter
``RIFT.lalsimutils.convert_waveform_coordinates`` plus the default RIFT
prior dictionary in ``RIFT.likelihood.rift_priors``.  It is *not* meant to
be imported directly by users -- it exists so that ``rift_default`` flows
through the same loader as third-party plugins and we don't need a
special-case branch in the driver script.

If you want to copy this as a starting point for your own coordinate
plugin, prefer ``demo/hyperpipe/linear_coordinate_convert.py`` -- it has
the same shape and is a lot simpler.
"""

from __future__ import annotations

import numpy as np

NAME = "rift_default"


def convert_coordinates(x_in, coord_names, low_level_coord_names, **kwargs):
    # Lazy import: ``lalsimutils`` is heavy, and not every plugin user has
    # the gravitational-wave stack installed.  Only pay the cost when
    # ``rift_default`` is actually selected.
    from RIFT import lalsimutils

    forwarded = {
        k: v
        for k, v in kwargs.items()
        if k in ("enforce_kerr", "source_redshift")
    }
    return lalsimutils.convert_waveform_coordinates(
        x_in,
        coord_names=list(coord_names),
        low_level_coord_names=list(low_level_coord_names),
        **forwarded,
    )


def register_priors(prior_map, prior_range_map=None, coord_names=None,
                    low_level_coord_names=None, **kwargs):
    # Install the canonical RIFT priors keyed on the standard parameter
    # names.  Callers seed ``prior_map`` with a uniform default beforehand,
    # so the ``update`` call here only overrides names the RIFT defaults
    # actually know about.
    try:
        from RIFT.likelihood import rift_priors
    except ImportError:  # pragma: no cover - import path may vary by install
        from RIFT.misc.likelihood import rift_priors  # type: ignore
    prior_map.update(rift_priors.prior_map)
