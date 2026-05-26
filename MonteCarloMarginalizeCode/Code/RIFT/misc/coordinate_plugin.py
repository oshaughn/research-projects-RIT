"""
RIFT.misc.coordinate_plugin
===========================

Loader and interface contract for user-supplied coordinate-conversion code,
used by ``util_ConstructEOSPosterior.py`` (and intended for reuse by any other
RIFT executable that today hard-codes a coordinate transform).

Why this exists
---------------
``util_ConstructIntrinsicPosterior_GenericCoordinates.py`` uses a hardcoded
converter, ``RIFT.lalsimutils.convert_waveform_coordinates``, to translate
between the columns of a ``grid-N.dat`` file (``dat_orig_names`` /
``low_level_coord_names``) and the fitting / sampling coordinates
(``coord_names``).  ``util_ConstructEOSPosterior.py`` historically could only
fit using the literal columns of its input file.  The
``--supplementary-coordinate-code`` option already existed as a stub, but only
accepted the magic value ``'rift_default'``.

This module turns that stub into a real plugin mechanism: the user names a
piece of Python (by importable dotted-name OR by file path), the loader
imports it, validates that it conforms to the API below, optionally hands it
an ini file for configuration, and returns a callable with the same signature
RIFT already uses for ``convert_waveform_coordinates``.


How the user points at their code
---------------------------------
The driver script exposes three CLI flags (mirroring the
``--supplementary-likelihood-factor-*`` family that already lives in the same
file):

  --supplementary-coordinate-code      SPEC
      Discovery target.  One of:
        * the literal string ``rift_default`` -- use
          ``RIFT.lalsimutils.convert_waveform_coordinates`` plus the priors
          in ``RIFT.likelihood.rift_priors.prior_map``;
        * a filesystem path ending in ``.py`` (or otherwise existing on
          disk) -- loaded with ``importlib.util.spec_from_file_location``;
        * any other string -- treated as a dotted importable module name
          and resolved via ``importlib.import_module`` against the active
          ``PYTHONPATH``.

  --supplementary-coordinate-function  NAME      (optional, default ``convert_coordinates``)
      Name of the entry-point callable inside the loaded module.

  --supplementary-coordinate-ini       PATH      (optional)
      Path to an ini file parsed with ``configparser`` and handed to the
      plugin's ``prepare(...)`` hook so it can read its own configuration
      block(s).


Plugin contract
---------------
A coordinate-convert plugin is a Python module that exposes at minimum a
single callable, by default named ``convert_coordinates``::

    def convert_coordinates(x_in, coord_names, low_level_coord_names, **kwargs):
        '''
        Parameters
        ----------
        x_in : numpy.ndarray of shape (N, len(low_level_coord_names))
            Rows are samples.  Columns are ordered to match
            ``low_level_coord_names``.
        coord_names : list[str]
            Desired output columns, in order.
        low_level_coord_names : list[str]
            Input column names, in the order they appear in ``x_in``.
        **kwargs
            Forward-compatibility hook.  Existing callers will pass nothing
            extra; future RIFT versions may inject e.g. ``opts=`` or
            ``source_redshift=``.  Plugins should accept ``**kwargs`` and
            ignore the rest.

        Returns
        -------
        numpy.ndarray of shape (N, len(coord_names))
        '''

Optional module-level attributes & hooks (all introspected by the loader):

  ``NAME : str``                     -- human-readable identifier for logs.
  ``INPUT_PARAMETERS : list[str]``   -- declarative list of names this
                                        plugin understands as inputs; if
                                        present the loader will warn when
                                        ``low_level_coord_names`` contains
                                        anything not in this set.
  ``OUTPUT_PARAMETERS : list[str]``  -- same idea for outputs vs
                                        ``coord_names``.

  ``CHARTS : dict[str, ChartSpec]``  -- declarative atlas (see below).
  ``DEFAULT_CHART : str``            -- name of the chart to use when the
                                        caller doesn't pass ``chart=``.

  ``prepare(config=None, coord_names=None, low_level_coord_names=None,
            chart=None, opts=None, **kwargs)``
      One-shot setup hook.  Called once at load time, after the ini file
      (if any) has been parsed.  Use it to read configuration, cache
      transformation matrices, and raise on misconfiguration -- much
      cheaper than discovering problems halfway through a Monte Carlo run.
      ``chart`` is the chart name the loader resolved (see below).

  ``register_priors(prior_map, prior_range_map=None, coord_names=None,
                    low_level_coord_names=None, chart=None, **kwargs)``
      Hook for the plugin to install priors keyed on the parameter names it
      produces.  Mutates ``prior_map`` (and optionally ``prior_range_map``)
      in place.  Normally you'd use ``CHARTS`` for this instead, and skip
      ``register_priors``; the hook is kept for plugins that want full
      programmatic control.

  ``jacobian(x_in, coord_names, low_level_coord_names, chart=None, **kwargs)``
      Optional.  Returns per-row ``log|det J|`` for change-of-variables
      corrections.  Not consumed by the prototype loader -- documented here
      so plugin authors and downstream RIFT code converge on one signature.


Charts and priors
-----------------
RIFT currently uses *separable* priors -- ``prior_map`` is a flat dict from
parameter name to a one-variable callable.  That representation is fine
inside a single chart, but the same parameter name in two different charts
is two different things: the implicit prior on ``y`` in the chart
``(x, y, z)`` is not the same as the implicit prior on ``y`` in the chart
``(r, y, z)``, because the Jacobian relative to whatever reference measure
you started from depends on the other coordinates that share the chart.
So a plugin that wants to expose more than one chart cannot just publish a
single ``{name: callable}`` mapping -- it has to expose *one such mapping
per chart*, and the caller must pick which chart it is sampling in.

We encode that by letting a plugin define::

    CHARTS = {
        "uvw_rotated": {
            "parameters": ["u", "v", "w"],
            "priors":     {"u": prior_u, "v": prior_v, "w": prior_w},
            "ranges":     {"u": (-7., 7.), ...},          # optional
            "description": "45 degree rotation of (x,y,z)", # optional
        },
        "ryz_cylindrical": {
            "parameters": ["r", "y", "z"],
            "priors":     {"r": prior_r, "y": prior_y_in_cyl, "z": prior_z},
            ...
        },
    }

The chart used for a given run is resolved (in priority order):
  1. explicit ``chart=`` kwarg to ``load_coordinate_converter``;
  2. ``DEFAULT_CHART`` module attribute, if defined;
  3. the only key in ``CHARTS`` when there's exactly one;
  4. otherwise a ``ValueError`` -- the loader will not pick arbitrarily.

The resolved chart name is then forwarded to ``prepare(chart=...)`` and
to every call of the converter as ``chart=...`` so the plugin can route
internally.  The loader installs ``CHARTS[chart]["priors"]`` into
``prior_map`` and ``CHARTS[chart]["ranges"]`` (if present) into
``prior_range_map`` whenever those names are not already keyed by the
driver.

Plugins that don't need this -- single basis, single prior set -- can omit
``CHARTS`` entirely and use the flat ``register_priors`` hook as before.

The loader fails closed: missing entry-point function, unimportable module,
unreadable ini, or a ``prepare`` that raises will all cause
``load_coordinate_converter`` to raise, never silently fall back.
"""

from __future__ import annotations

import configparser
import importlib
import importlib.util
import os
import sys
import warnings
from typing import Callable, Optional, Sequence, Tuple


_DEFAULT_FUNCTION_NAME = "convert_coordinates"


def _load_module_from_spec(spec: str):
    """Resolve ``spec`` to an imported module.

    Order of attempts:
      1. ``spec == 'rift_default'``  -> built-in shim (see below).
      2. spec looks like a file path -> ``importlib.util.spec_from_file_location``.
      3. otherwise                    -> ``importlib.import_module``.
    """
    if spec == "rift_default":
        return importlib.import_module(
            "RIFT.misc._builtin_rift_default_coordinate_plugin"
        )

    # File path branch.  We treat anything that exists on disk, or anything
    # that ends in '.py', as a file-path spec.  Using both heuristics lets
    # the user write either an absolute path or a relative one, while still
    # leaving dotted-name imports unambiguous.
    looks_like_path = spec.endswith(".py") or os.path.sep in spec or os.path.isfile(spec)
    if looks_like_path:
        if not os.path.isfile(spec):
            raise FileNotFoundError(
                f"--supplementary-coordinate-code: file not found: {spec!r}"
            )
        module_name = "_rift_coord_plugin_" + os.path.splitext(os.path.basename(spec))[0]
        loader_spec = importlib.util.spec_from_file_location(module_name, spec)
        if loader_spec is None or loader_spec.loader is None:
            raise ImportError(f"Could not build import spec for {spec!r}")
        module = importlib.util.module_from_spec(loader_spec)
        sys.modules[module_name] = module
        loader_spec.loader.exec_module(module)
        return module

    # Dotted-name branch.
    return importlib.import_module(spec)


def _parse_ini(ini_path: Optional[str]) -> Optional[configparser.ConfigParser]:
    """Parse the optional ini file.  Preserve key case (matches the existing
    ``--supplementary-likelihood-factor-ini`` behaviour)."""
    if not ini_path:
        return None
    if not os.path.isfile(ini_path):
        raise FileNotFoundError(
            f"--supplementary-coordinate-ini: file not found: {ini_path!r}"
        )
    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # preserve case
    cfg.read(ini_path)
    return cfg


def _warn_unknown_names(
    declared: Optional[Sequence[str]],
    requested: Optional[Sequence[str]],
    kind: str,
    attr_name: str,
    plugin_name: str,
) -> None:
    # ``declared`` may legitimately be empty if the plugin populates its
    # INPUT_PARAMETERS / OUTPUT_PARAMETERS during prepare().  Empty -> no
    # opinion; only warn when the plugin *did* declare something and the
    # driver asked for a name outside that set.
    if not declared or not requested:
        return
    unknown = [name for name in requested if name not in declared]
    if unknown:
        warnings.warn(
            f"coordinate plugin {plugin_name!r}: requested {kind} {unknown!r} "
            f"not in plugin's declared {attr_name}={list(declared)!r}",
            RuntimeWarning,
            stacklevel=3,
        )


def _resolve_chart(module, requested: Optional[str]) -> Optional[str]:
    """Decide which CHARTS entry to use for this load.

    Priority: explicit ``requested`` > ``DEFAULT_CHART`` > sole entry > None.
    Raises if the plugin has multiple charts and the caller didn't pick one,
    or if the requested chart name is not in ``CHARTS``.
    """
    charts = getattr(module, "CHARTS", None)
    if not charts:
        if requested:
            warnings.warn(
                f"coordinate plugin {getattr(module, 'NAME', module.__name__)!r}: "
                f"--supplementary-coordinate-chart={requested!r} supplied but the "
                "plugin does not define CHARTS; ignoring.",
                RuntimeWarning, stacklevel=3,
            )
        return None

    if requested:
        if requested not in charts:
            raise ValueError(
                f"coordinate plugin {getattr(module, 'NAME', module.__name__)!r}: "
                f"requested chart {requested!r} not in CHARTS "
                f"(available: {sorted(charts)!r})"
            )
        return requested

    default = getattr(module, "DEFAULT_CHART", None)
    if default is not None:
        if default not in charts:
            raise ValueError(
                f"coordinate plugin {getattr(module, 'NAME', module.__name__)!r}: "
                f"DEFAULT_CHART={default!r} not in CHARTS={sorted(charts)!r}"
            )
        return default

    if len(charts) == 1:
        return next(iter(charts))

    raise ValueError(
        f"coordinate plugin {getattr(module, 'NAME', module.__name__)!r}: defines "
        f"multiple charts ({sorted(charts)!r}); pass --supplementary-coordinate-chart "
        f"or set DEFAULT_CHART in the plugin."
    )


def _install_chart_priors(
    module,
    chart_name: Optional[str],
    prior_map: Optional[dict],
    prior_range_map: Optional[dict],
) -> None:
    """Install CHARTS[chart_name]['priors'] and ['ranges'] into the driver's
    maps.  Skips any name already present (so explicit CLI overrides win).
    """
    if not chart_name or prior_map is None:
        return
    chart = getattr(module, "CHARTS", {}).get(chart_name)
    if not chart:
        return
    priors = chart.get("priors", {}) or {}
    for name, fn in priors.items():
        # Only install if the driver hasn't already keyed this name with
        # something stronger than the bare uniform fallback.  We can't tell
        # that for sure, so the contract is: chart priors override the
        # uniform default but yield to the rift_default prior_map updates
        # (which run first when 'rift_default' is the plugin).  In
        # practice the driver seeds prior_map with uniform_prior for every
        # low_level_coord_name -- those uniform fallbacks get replaced.
        if not callable(fn):
            raise TypeError(
                f"chart {chart_name!r} prior for {name!r} is not callable"
            )
        prior_map[name] = fn

    if prior_range_map is None:
        return
    ranges = chart.get("ranges", {}) or {}
    for name, lo_hi in ranges.items():
        if name in prior_range_map:
            # Driver already set this from --integration-parameter-range or
            # from the data file; don't clobber an explicit user choice.
            continue
        prior_range_map[name] = lo_hi


def load_coordinate_converter(
    spec: str,
    function_name: Optional[str] = None,
    ini_path: Optional[str] = None,
    coord_names: Optional[Sequence[str]] = None,
    low_level_coord_names: Optional[Sequence[str]] = None,
    chart: Optional[str] = None,
    opts=None,
    prior_map: Optional[dict] = None,
    prior_range_map: Optional[dict] = None,
) -> Tuple[Callable, object]:
    """Load a user-supplied coordinate-convert plugin.

    Returns
    -------
    (converter, module)
        ``converter`` is a callable with the canonical RIFT signature
        ``converter(x_in, coord_names=..., low_level_coord_names=..., **kwargs)``
        suitable for direct assignment to ``supplemental_coordinate_convert``.
        ``module`` is the loaded plugin module object (handy for tests, and
        so the caller can inspect e.g. ``module.NAME``).

    Side effects
    ------------
    * Calls ``module.prepare(...)`` if defined, passing the parsed ini, the
      coordinate name lists, and the driver's ``opts`` namespace.
    * Calls ``module.register_priors(...)`` if defined, mutating
      ``prior_map`` (and optionally ``prior_range_map``) in place.

    The wrapper this function returns intentionally drops any extra kwargs
    callers may pass (matching ``convert_waveform_coordinates``, which
    accepts but ignores its non-relevant kwargs).  This keeps plugin
    authors honest: their function must accept ``**kwargs``.
    """
    if not spec:
        raise ValueError("load_coordinate_converter requires a non-empty spec")

    module = _load_module_from_spec(spec)
    plugin_name = getattr(module, "NAME", getattr(module, "__name__", spec))

    fn_name = function_name or _DEFAULT_FUNCTION_NAME
    if not hasattr(module, fn_name):
        raise AttributeError(
            f"coordinate plugin {plugin_name!r} (from {spec!r}) does not expose "
            f"{fn_name!r}; either define it or pass "
            f"--supplementary-coordinate-function to point at the right name."
        )
    convert_fn = getattr(module, fn_name)
    if not callable(convert_fn):
        raise TypeError(
            f"coordinate plugin {plugin_name!r}: attribute {fn_name!r} is not callable"
        )

    cfg = _parse_ini(ini_path)

    # Resolve the working chart *before* prepare() so the plugin can use it.
    resolved_chart = _resolve_chart(module, chart)

    prepare_fn = getattr(module, "prepare", None)
    if callable(prepare_fn):
        prepare_fn(
            config=cfg,
            coord_names=list(coord_names) if coord_names is not None else None,
            low_level_coord_names=list(low_level_coord_names)
            if low_level_coord_names is not None
            else None,
            chart=resolved_chart,
            opts=opts,
        )

    # If the plugin uses CHARTS, the chart's parameter list authoritatively
    # describes the output basis -- treat it as OUTPUT_PARAMETERS for the
    # declarative check below.  Same for inputs if the chart declares them.
    chart_spec = (
        getattr(module, "CHARTS", {}).get(resolved_chart)
        if resolved_chart
        else None
    )
    declared_outputs = (
        chart_spec.get("parameters")
        if chart_spec
        else getattr(module, "OUTPUT_PARAMETERS", None)
    )
    declared_inputs = (
        (chart_spec.get("input_parameters") if chart_spec else None)
        or getattr(module, "INPUT_PARAMETERS", None)
    )

    # Declarative sanity checks (warnings only).  Run *after* prepare so
    # plugins that populate INPUT_PARAMETERS / OUTPUT_PARAMETERS from their
    # ini file inside prepare() get a fair shake.
    _warn_unknown_names(
        declared_inputs,
        low_level_coord_names,
        "inputs",
        "INPUT_PARAMETERS",
        plugin_name,
    )
    _warn_unknown_names(
        declared_outputs,
        coord_names,
        "outputs",
        "OUTPUT_PARAMETERS",
        plugin_name,
    )

    # Chart-driven prior installation (preferred) then optional explicit
    # register_priors hook for plugins that want full control.
    _install_chart_priors(module, resolved_chart, prior_map, prior_range_map)

    register_priors = getattr(module, "register_priors", None)
    if callable(register_priors) and prior_map is not None:
        register_priors(
            prior_map=prior_map,
            prior_range_map=prior_range_map,
            coord_names=list(coord_names) if coord_names is not None else None,
            low_level_coord_names=list(low_level_coord_names)
            if low_level_coord_names is not None
            else None,
            chart=resolved_chart,
        )

    print(
        " COORDINATE PLUGIN loaded: {} from {} (entry={}, chart={})".format(
            plugin_name, spec, fn_name, resolved_chart
        )
    )

    def _wrapper(x_in, coord_names=coord_names, low_level_coord_names=low_level_coord_names, **kwargs):
        # Match the canonical signature used by
        # RIFT.lalsimutils.convert_waveform_coordinates so callers in the
        # driver script don't have to change.  We additionally inject the
        # resolved chart name so plugins can route per chart without
        # plumbing it through the driver each time.
        kwargs.setdefault("chart", resolved_chart)
        return convert_fn(
            x_in,
            coord_names=coord_names,
            low_level_coord_names=low_level_coord_names,
            **kwargs,
        )

    return _wrapper, module
