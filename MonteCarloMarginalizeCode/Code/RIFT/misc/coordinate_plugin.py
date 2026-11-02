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

  ``inverse_convert_coordinates(y_in, coord_names, low_level_coord_names,
                                 chart=None, **kwargs)``
      Optional but STRONGLY recommended for invertible transforms.  Maps
      plugin-basis points back to the file basis: ``y_in`` has columns
      named by ``coord_names`` (plugin-basis names) and the return value
      has columns named by ``low_level_coord_names`` (file-basis names),
      shape ``(N, len(low_level_coord_names))``.  Consumed by
      ``util_HyperparameterPuffball.py`` / ``util_HyperparameterTracerUpdate.py``
      (round-tripping the puff displacement) and by
      ``util_ConstructEOSPosterior.py`` (writing the final posterior
      samples in fiducial coordinates when the MC sampled in the plugin
      basis).  Without it the EOS-posterior driver cannot express
      plugin-basis samples in the output file, and warns.

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


Two plugin API shapes: ``rift`` and ``inplace``
-----------------------------------------------
Everything above describes the ``rift`` API, which every existing plugin
uses.  The loader also accepts a second, smaller shape (the one proposed in
PR #434), called ``inplace``::

    PLUGIN_API = "inplace"                     # recommended; see selection below

    def rotate(X, coord_names, **kwargs):      # name set by --supplementary-coordinate-function
        '''X has columns labelled by coord_names.  Return an array of the
        SAME shape whose columns hold the transformed values under the SAME
        names (e.g. x0, x1 -> rotated x0, x1).  Columns the plugin does not
        know are returned unchanged.'''

    def inverse_rotate(X, coord_names, **kwargs):  # always "inverse_" + function name
        '''Same shape and labels; undoes rotate.'''

    def get_bounds(coord_names, ranges, **kwargs):   # optional
        '''ranges: {name: [lo, hi]} in the FILE basis.  Return {name: [lo, hi]}
        in the transformed basis, under the original names, for every name
        in coord_names.'''

The loader wraps an ``inplace`` plugin so drivers see the ``rift`` call
signature: the wrapper copies its input before the call (the caller's array
is never mutated), calls ``fn(x, low_level_coord_names, **kw)``, and then
selects the requested ``coord_names`` by name (a requested name that is not
an input name raises).  The inverse is wrapped the same way.  Values given
with ``--external-range-args key=value`` (parsed as python literals) are
passed to the forward and inverse calls in this mode only; ``rift`` plugins
receive only ``chart=``, as before.

Selection, in order:
  1. explicit: the module attribute ``PLUGIN_API = "rift" | "inplace"`` or
     the CLI flag ``--supplementary-coordinate-api {auto,rift,inplace}``
     (default ``auto``).  If both are given they must agree.
  2. ``auto`` reads only the module's own namespace and ``inspect.signature``;
     it never calls the plugin and never runs a module-level ``__getattr__``
     (the loader also reads the entry point statically until the API is
     decided; only after that do hooks go through ordinary attribute access).
     It never selects ``inplace``:
       * ``rift`` if the entry point's signature explicitly names
         ``low_level_coord_names`` (every production plugin does);
       * otherwise, if the module defines ``inverse_<function>`` (or a
         module ``__getattr__`` that could supply it), REFUSE and ask for
         ``PLUGIN_API`` or the CLI flag.  An unreadable or uninformative
         signature (``*args``/``**kwargs`` only, a decorator without
         ``functools.wraps``, a builtin) counts as "does not name".  Helpers
         named ``prepare``, ``CHARTS``, ``jacobian`` and so on do not change
         this: calling an inplace plugin with the rift contract would write
         transformed values as file values, with exit 0;
       * otherwise (forward-only) the current contract, ``rift``, as before
         this adapter, with a RuntimeWarning that says what that call means;
         a forward-only signature that cannot take the rift call is refused.
     A ``rift`` plugin whose entry point takes ``low_level_coord_names`` only
     through ``**kwargs`` and also defines an inverse must now set
     ``PLUGIN_API = "rift"``.

Only ``util_ConstructEOSPosterior.py`` and ``util_HyperparameterPuffball.py``
accept ``inplace`` plugins today; other drivers refuse them.  In those two
drivers an ``inplace`` plugin's sampled (or puffed) names are all in the
transformed basis, ``--integration-parameter-range`` / ``--parameter-range``
values for them are FILE-basis ranges: they are inputs to ``get_bounds``
(and are refused without ``--get-range-from-external``, since their frame
would be ambiguous), and they are ENFORCED on the output -- the transformed
box from ``get_bounds`` only bounds the sampling or puff, and PUFF then drops
rows, and CEP gives zero weight to samples, whose inverse lies outside the
file-basis box (in CEP that box is every range handed to ``get_bounds``,
including data-derived ones).  Every sampled column is passed through the
inverse before it is written, and CEP checks forward -> sampled columns ->
inverse on a few data rows before the fit, so a sampled set the inverse
cannot invert fails at once.  The check uses only rows where the forward map
is finite (rows outside the plugin's domain are skipped, as the fit drops
them), needs at least one, and fails if the inverse is non-finite where the
forward map is finite.  It allows an error of
``--supplementary-coordinate-roundtrip-tol`` (else the plugin's
``ROUNDTRIP_TOL``, else 1e-5) times each column's scale: its spread in the
data file, floored at 1e-3 of its largest magnitude (so a constant column at
value c allows about 1e-8 c), so a tabulated or iterative inverse passes.  PUFF puffs in the transformed basis
(PR #434 puffed in the file basis); for a linear map the two agree.  PUFF
refuses ``--reflect-parameter`` with an ``inplace`` plugin (reflection in the
transformed box would not respect the file-basis box), and exits nonzero when
the file-basis downselect leaves no rows.

Known limitations of the ``inplace`` path
-----------------------------------------
* CEP samples the transformed bounding box and gives zero weight (log weight
  -inf) outside the file-basis box.  Efficiency is the ratio of the two
  volumes: 0.5 for a square rotated by 45 degrees, much less for a thin box,
  so n_eff falls at fixed n_max.
* With those -inf log weights the GMM sampler's EM loop can stop after two
  iterations (``abs(-inf - -inf)`` is NaN; this predates the adapter).
  No n_eff loss has been measured.
* ``util_RIFT_hyperpipe`` does not route an ``inplace`` plugin to PUFF: its
  sampled names are file-column names, so ``puff_basis('auto')`` puffs in the
  file basis without the plugin, and ``'plugin'`` mode refuses.  "PUFF puffs
  in the transformed basis" holds only when PUFF is called directly.
* ``util_RIFT_hyperpipe`` forwards neither ``--supplementary-coordinate-api``
  nor ``--supplementary-coordinate-roundtrip-tol`` to its stages: an
  ``inplace`` plugin used through hyperpipe must set ``PLUGIN_API`` (and, if
  needed, ``ROUNDTRIP_TOL``) in the module.
* A column whose spread is small next to its value, sampled through an
  iterative inverse, can fail the round-trip check at the default tolerance:
  for a BNS chirp mass (1.2, spread 5e-4) the allowed error is about 1e-8,
  below ``scipy.optimize.newton``'s default 1.48e-8.  Set ``ROUNDTRIP_TOL``
  or ``--supplementary-coordinate-roundtrip-tol``.
"""

from __future__ import annotations

import configparser
import importlib
import importlib.util
import inspect
import os
import sys
import warnings
from typing import Callable, Optional, Sequence, Tuple


_DEFAULT_FUNCTION_NAME = "convert_coordinates"

PLUGIN_APIS = ("rift", "inplace")
_MISSING = object()


def _static_attr(module, name, default=None):
    """Module attribute without running a PEP 562 module ``__getattr__``."""
    return vars(module).get(name, default)


def _signature(obj):
    try:
        return inspect.signature(obj)
    except (TypeError, ValueError):
        return None


def _names_low_level_coord_names(sig) -> bool:
    """True only when the signature explicitly names ``low_level_coord_names``.
    An unreadable signature, ``(*args, **kwargs)`` (e.g. a decorator without
    ``functools.wraps``), or a ``**kwargs`` that might receive it do not count."""
    if sig is None:
        return False
    p = sig.parameters.get("low_level_coord_names")
    return p is not None and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)


def _ambiguous(plugin_name, why):
    return ValueError(
        f"coordinate plugin {plugin_name!r}: cannot tell which plugin API it implements ({why}). "
        "Set PLUGIN_API = 'rift' or 'inplace' in the plugin, or pass "
        "--supplementary-coordinate-api rift|inplace.  See RIFT.misc.coordinate_plugin."
    )


def classify_plugin_api(module, function_name: Optional[str] = None, api: Optional[str] = None) -> str:
    """Return ``'rift'`` or ``'inplace'`` for a loaded plugin module.

    ``api`` is the CLI choice (``None``/``'auto'`` = not given).  An explicit
    choice (CLI or the module's ``PLUGIN_API``) wins; both given must agree.
    Otherwise (``auto``) classify from the module's own namespace and
    ``inspect.signature`` alone: the plugin is never called, and a module-level
    ``__getattr__`` is never run.  ``auto`` never returns ``'inplace'``; see the
    module docstring for the rules.
    """
    plugin_name = _static_attr(module, "NAME", _static_attr(module, "__name__", "?"))
    declared = _static_attr(module, "PLUGIN_API", None)
    requested = None if api in (None, "auto") else api
    for value, source in ((declared, "PLUGIN_API"), (requested, "--supplementary-coordinate-api")):
        if value is not None and value not in PLUGIN_APIS:
            raise ValueError(f"coordinate plugin {plugin_name!r}: {source}={value!r} "
                             f"is not one of {PLUGIN_APIS!r}")
    if declared is not None and requested is not None and declared != requested:
        raise ValueError(f"coordinate plugin {plugin_name!r}: PLUGIN_API={declared!r} in the plugin "
                         f"but --supplementary-coordinate-api {requested}")
    if requested is not None or declared is not None:
        return requested or declared

    fn_name = function_name or _DEFAULT_FUNCTION_NAME
    fn = _static_attr(module, fn_name, _MISSING)
    sig = None if fn is _MISSING else _signature(fn)
    if _names_low_level_coord_names(sig):
        return "rift"

    # The entry point does not name low_level_coord_names (or its signature cannot be
    # read: *args/**kwargs, a decorator without functools.wraps, a builtin).  An
    # inverse_<fn> is how an inplace (#434) plugin pairs its maps, and calling such a
    # plugin with the rift contract writes transformed values as file values with
    # exit 0.  So refuse rather than guess.
    inv_name = "inverse_" + fn_name
    if inv_name in vars(module):
        raise _ambiguous(plugin_name, f"{fn_name} does not name low_level_coord_names "
                                      f"(signature {sig if sig is not None else 'unreadable'}) "
                                      f"and the module defines {inv_name}")
    if "__getattr__" in vars(module):
        raise _ambiguous(plugin_name, f"{fn_name} does not name low_level_coord_names and the "
                                      f"module defines __getattr__, so whether {inv_name} exists "
                                      "cannot be checked without running plugin code")
    if fn is _MISSING:
        raise _ambiguous(plugin_name, f"{fn_name} is not defined in the module namespace")

    # rift-only hooks (prepare, CHARTS, jacobian, ...) are deliberately NOT evidence: a
    # helper with one of those names must not decide the API.
    if sig is not None:
        try:
            sig.bind(None, coord_names=None, low_level_coord_names=None, chart=None)
        except TypeError:
            raise _ambiguous(plugin_name, f"{fn_name}{sig} cannot be called with the rift contract "
                                          "(x_in, coord_names, low_level_coord_names, **kw) and the "
                                          f"module defines no {inv_name}")
    # Forward-only, no low_level_coord_names in the signature: the current contract,
    # as before this adapter existed.  Say plainly what that means for a #434 plugin.
    warnings.warn(
        f"coordinate plugin {plugin_name!r}: {fn_name}"
        f"{sig if sig is not None else '(<unreadable signature>)'} does not name "
        "low_level_coord_names and there is no " + inv_name + ", so it is called with the rift "
        "contract: coord_names = the requested OUTPUT names, low_level_coord_names (in **kwargs) "
        "= the data-file columns.  If it was written for the inplace API (#434: transform X "
        "under its input names), its results are WRONG under this contract: add " + inv_name +
        " and PLUGIN_API = 'inplace'.  If it is a rift plugin, name low_level_coord_names in "
        "its signature.",
        RuntimeWarning, stacklevel=3)
    return "rift"


def _select_columns(Y, have, want, what, plugin_name):
    import numpy as np
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] != len(have):
        raise ValueError(f"coordinate plugin {plugin_name!r}: {what} returned shape {Y.shape}; "
                         f"the inplace API returns the input's shape (N, {len(have)})")
    missing = [n for n in want if n not in have]
    if missing:
        raise ValueError(f"coordinate plugin {plugin_name!r} (inplace API): {what} cannot produce "
                         f"{missing!r}; an inplace plugin keeps its input names, so every requested "
                         f"name must be one of {list(have)!r}")
    return Y[:, [list(have).index(n) for n in want]]


def _inplace_wrappers(convert_fn, inverse_fn, extra_kwargs, coord_names, low_level_coord_names, plugin_name):
    import numpy as np

    def _forward(x_in, coord_names=coord_names, low_level_coord_names=low_level_coord_names, **kwargs):
        kwargs.pop("chart", None)
        names_in = list(low_level_coord_names)
        x = np.array(np.atleast_2d(x_in), dtype=float, copy=True)   # never hand over the caller's array
        y = convert_fn(x, list(names_in), **{**extra_kwargs, **kwargs})
        return _select_columns(y, names_in, list(coord_names), "forward", plugin_name)

    def _inverse(y_in, coord_names=coord_names, low_level_coord_names=low_level_coord_names, **kwargs):
        kwargs.pop("chart", None)
        names_in = list(coord_names)
        y = np.array(np.atleast_2d(y_in), dtype=float, copy=True)
        x = inverse_fn(y, list(names_in), **{**extra_kwargs, **kwargs})
        return _select_columns(x, names_in, list(low_level_coord_names), "inverse", plugin_name)

    return _forward, _inverse


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
    api: Optional[str] = None,
    allowed_apis: Sequence[str] = ("rift",),
    external_range_args=None,
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

    Plugin API (see the module docstring): ``api`` is the CLI choice
    (``auto``/``rift``/``inplace``; ``None`` = ``auto``).  ``allowed_apis``
    lists the shapes the calling driver supports; the default, ``("rift",)``,
    keeps every existing caller on the current contract and makes it refuse an
    ``inplace`` plugin.  The returned converter carries two attributes:
    ``converter.plugin_api`` (``'rift'`` or ``'inplace'``) and
    ``converter.inverse`` (a callable with the rift inverse signature, or
    None).  For a ``rift`` plugin ``converter.inverse`` is the module's own
    ``inverse_convert_coordinates``, unwrapped.  ``external_range_args``
    (``key=value`` strings) reach the forward and inverse calls only for an
    ``inplace`` plugin.
    """
    if not spec:
        raise ValueError("load_coordinate_converter requires a non-empty spec")

    module = _load_module_from_spec(spec)
    plugin_name = _static_attr(module, "NAME", _static_attr(module, "__name__", spec))

    fn_name = function_name or _DEFAULT_FUNCTION_NAME
    # Read the entry point from the module namespace only: a module-level __getattr__
    # must not run before classify_plugin_api has decided (it refuses, in auto mode, a
    # module whose entry point or inverse could come only from __getattr__).
    if fn_name not in vars(module) and "__getattr__" not in vars(module):
        raise AttributeError(
            f"coordinate plugin {plugin_name!r} (from {spec!r}) does not expose "
            f"{fn_name!r}; either define it or pass "
            f"--supplementary-coordinate-function to point at the right name."
        )

    plugin_api = classify_plugin_api(module, fn_name, api)

    # Classified (an explicit API, or a statically defined entry point): ordinary
    # attribute access from here on.
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
    if plugin_api not in tuple(allowed_apis):
        raise ValueError(
            f"coordinate plugin {plugin_name!r} uses the {plugin_api!r} plugin API, which this "
            f"driver does not support (supported: {list(allowed_apis)!r})."
        )
    if plugin_api == "inplace":
        inverse_fn = getattr(module, "inverse_" + fn_name, None)
        if not callable(inverse_fn):
            raise AttributeError(
                f"coordinate plugin {plugin_name!r} (inplace API) does not define "
                f"inverse_{fn_name}; the inplace API needs it to write samples in the file basis."
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
    #
    # Direction matters and differs between inputs and outputs.  For
    # INPUTS the failure mode is the plugin needing a column the driver
    # cannot feed it -- extra available columns (e.g. non-sampled nuisance
    # parameters carried in the data file) are harmless and must NOT warn.
    if declared_inputs and low_level_coord_names:
        _missing_inputs = [n for n in declared_inputs if n not in low_level_coord_names]
        if _missing_inputs:
            warnings.warn(
                f"coordinate plugin {plugin_name!r}: declares input(s) "
                f"{_missing_inputs!r} that the driver does not provide "
                f"(available: {list(low_level_coord_names)!r})",
                RuntimeWarning,
                stacklevel=2,
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
        " COORDINATE PLUGIN loaded: {} from {} (entry={}, chart={}, api={})".format(
            plugin_name, spec, fn_name, resolved_chart, plugin_api
        )
    )

    if plugin_api == "inplace":
        _forward, _inverse = _inplace_wrappers(
            convert_fn, inverse_fn, parse_external_range_args(external_range_args),
            coord_names, low_level_coord_names, plugin_name)
        _forward.plugin_api = "inplace"
        _forward.inverse = _inverse
        return _forward, module

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

    _wrapper.plugin_api = "rift"
    _wrapper.inverse = getattr(module, "inverse_convert_coordinates", None)
    return _wrapper, module


def resolve_input_parameters(module, chart=None, cli_override=None):
    """Return the FILE-basis column names the plugin maps from/to.

    These are the names of the columns ``convert_coordinates`` consumes and
    ``inverse_convert_coordinates`` produces.  Resolution priority:

      1. ``cli_override`` (e.g. --supplementary-coordinate-input-parameter);
      2. ``CHARTS[chart]['input_parameters']`` for the resolved chart (when
         ``chart`` is None and the plugin has exactly one chart, that chart
         is used);
      3. the module-level ``INPUT_PARAMETERS`` attribute (often populated by
         ``prepare()`` from the plugin's ini file).

    Returns a list (possibly empty when the plugin declares nothing).  This
    consolidates the resolution logic previously inlined in
    ``util_HyperparameterPuffball.py`` so every driver resolves the same way.
    """
    if cli_override:
        return list(cli_override)
    charts = getattr(module, "CHARTS", None) or {}
    chart_spec = None
    if chart:
        chart_spec = charts.get(chart)
    elif len(charts) == 1:
        chart_spec = next(iter(charts.values()))
    elif charts:
        default = getattr(module, "DEFAULT_CHART", None)
        if default:
            chart_spec = charts.get(default)
    if chart_spec and chart_spec.get("input_parameters"):
        return list(chart_spec["input_parameters"])
    return list(getattr(module, "INPUT_PARAMETERS", []) or [])


def parse_external_range_args(items):
    """Parse ``key=value`` strings into a dict; values become python literals when they parse."""
    import ast
    out = {}
    for item in (items or []):
        if "=" not in item:
            raise ValueError(f"--external-range-args expects key=value; got {item!r}")
        key, raw = item.split("=", 1)
        try:
            out[key.strip()] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key.strip()] = raw
    return out


def call_get_bounds(module, coord_names, ranges, external_range_args=None):
    """Call the plugin's ``get_bounds(coord_names, ranges, **kwargs)``.

    ``ranges`` is a copy of the ranges known so far.  The hook returns
    ``{name: [lo, hi]}`` in the basis named by ``coord_names`` (the sampling
    or puff basis, i.e. the plugin's output names).  Errors propagate.

    For an ``inplace`` plugin ``ranges`` are in the FILE basis and the result
    is in the transformed basis under the same names; use
    ``inplace_sampling_ranges``, which also requires full coverage.
    """
    fn = getattr(module, "get_bounds", None)
    if not callable(fn):
        raise AttributeError(
            "--get-range-from-external: the coordinate plugin does not define get_bounds"
        )
    out = fn(list(coord_names), {k: list(v) for k, v in dict(ranges).items()},
             **parse_external_range_args(external_range_args))
    result = {}
    for name, rng in dict(out).items():
        lo, hi = (float(v) for v in rng)
        if not lo < hi:
            raise ValueError(f"get_bounds returned a non-increasing range for {name!r}: {rng!r}")
        result[name] = [lo, hi]
    return result


def inplace_sampling_ranges(module, names, file_ranges, cli_range_names, use_get_bounds,
                            external_range_args=None):
    """Sampling-basis ranges for an ``inplace`` plugin's sampled ``names``.

    An inplace plugin reuses the file's names for its transformed coordinates,
    so a range given on the command line for one of them has no stated frame.
    It is read as a FILE-basis range and handed to ``get_bounds``, which must
    then return a range for every name in ``names``.  Without
    ``use_get_bounds`` a command-line range on a sampled name is refused, and
    the return value is ``None``: the caller derives ranges itself (CEP: from
    the forward-transformed input grid).
    """
    names = list(names)
    if use_get_bounds:
        out = call_get_bounds(module, names, file_ranges, external_range_args)
        missing = [n for n in names if n not in out]
        if missing:
            raise ValueError(f"get_bounds (inplace API) returned no range for {missing!r}; with an "
                             "inplace plugin it must cover every sampled name")
        return {n: out[n] for n in names}
    clash = [n for n in names if n in set(cli_range_names)]
    if clash:
        raise ValueError(
            f"range(s) given on the command line for {clash!r}, which an inplace coordinate plugin "
            "transforms under the same name(s): the frame of those ranges is ambiguous.  Pass "
            "--get-range-from-external so the plugin's get_bounds maps these file-basis ranges, "
            "or drop them.")
    return None


def in_file_box(X_file, names, box):
    """Boolean mask of the rows of ``X_file`` (columns labelled by ``names``) that lie
    inside ``box`` = ``{name: [lo, hi]}`` (closed); names absent from ``box`` are not
    constrained, and a NaN in a constrained column is outside.  Used to enforce an
    ``inplace`` plugin's FILE-basis ranges after the inverse."""
    import numpy as np
    X_file = np.atleast_2d(np.asarray(X_file, dtype=float))
    ok = np.ones(X_file.shape[0], dtype=bool)
    for j, name in enumerate(names):
        if name in box:
            lo, hi = (float(v) for v in box[name])
            ok &= (X_file[:, j] >= lo) & (X_file[:, j] <= hi)
    return ok


DEFAULT_ROUNDTRIP_TOL = 1e-5


def roundtrip_tolerance(module, cli_value=None):
    """Tolerance for the ``inplace`` round-trip preflight, as ``(tol, source)``.

    ``tol`` bounds ``|inverse(forward(x)) - x|`` divided by the column's scale
    (:func:`roundtrip_column_scale`).  Priority: ``cli_value`` (``--supplementary-coordinate-roundtrip-tol``),
    then the plugin's ``ROUNDTRIP_TOL`` (read without running a module ``__getattr__``),
    then ``DEFAULT_ROUNDTRIP_TOL`` (1e-5: passes a tabulated or iterative inverse,
    refuses a wrong one).  A value that is not a positive finite number raises."""
    import math
    if cli_value is not None:
        tol, src = cli_value, "--supplementary-coordinate-roundtrip-tol"
    elif module is not None and _static_attr(module, "ROUNDTRIP_TOL", None) is not None:
        tol, src = _static_attr(module, "ROUNDTRIP_TOL"), "plugin ROUNDTRIP_TOL"
    else:
        tol, src = DEFAULT_ROUNDTRIP_TOL, "default"
    try:
        tol = float(tol)
    except (TypeError, ValueError):
        raise ValueError(f"round-trip tolerance from {src} is not a number: {tol!r}")
    if not (math.isfinite(tol) and tol > 0):
        raise ValueError(f"round-trip tolerance from {src} must be a positive finite number, got {tol!r}")
    return tol, src


# Floor on the round-trip scale, as a fraction of the column's magnitude.  It sets the
# scale of a constant column (an absolute tolerance of tol * 1e-3 * |x|, about 1e-8 of
# the value at the default tol) and keeps a column whose spread is tiny next to its value
# (a BNS chirp mass) from demanding a round trip below what an iterative inverse delivers.
ROUNDTRIP_MAGNITUDE_FLOOR = 1e-3


def roundtrip_column_scale(values):
    """Scale that divides a round-trip error in one column: ``max(spread,
    ROUNDTRIP_MAGNITUDE_FLOOR * max|x|)`` over the finite values, or 1 if that is zero."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    s = max(float(np.ptp(v)), ROUNDTRIP_MAGNITUDE_FLOOR * float(np.max(np.abs(v))))
    return s if s > 0 else 1.0


def inplace_roundtrip_preflight(convert, inverse, X_file, file_names, sampled_names, tol, tol_src, n_check=8):
    """Round-trip check of an ``inplace`` plugin on data rows, run before the fit.

    Forward on every data row (all file columns), keep the sampled columns, invert up to
    ``n_check`` rows spread through the rows inside the forward map's domain (forward
    output and file value finite in every sampled column), and compare with the file
    values.  Rows outside the domain are skipped: the fit drops them anyway.  Raises
    ValueError with one of four causes: the inverse raised (it may need columns that are
    not sampled), no row is in the domain, the inverse is non-finite on a row whose
    forward map is finite, or the inverse does not reproduce the rows within ``tol``
    times :func:`roundtrip_column_scale`.  Returns ``(n_checked, n_outside, max_err)``."""
    import numpy as np
    file_names, sampled_names = list(file_names), list(sampled_names)
    X_file = np.atleast_2d(np.asarray(X_file, dtype=float))
    cols = [file_names.index(p) for p in sampled_names]
    head = " inplace coordinate plugin: round-trip check failed before the fit: "
    fwd = np.asarray(convert(X_file, coord_names=sampled_names, low_level_coord_names=file_names), dtype=float)
    if fwd.shape != (X_file.shape[0], len(sampled_names)):
        raise ValueError(head + "the forward map returned shape {} for {} rows and sampled columns {!r}".format(
            fwd.shape, X_file.shape[0], sampled_names))
    want_all = X_file[:, cols]
    in_domain = np.all(np.isfinite(fwd), axis=1) & np.all(np.isfinite(want_all), axis=1)
    idx = np.flatnonzero(in_domain)
    if idx.size == 0:
        raise ValueError(head + "no data row is in the plugin's domain: the forward map is non-finite in the "
                         "sampled columns {!r} on all {} rows".format(sampled_names, X_file.shape[0]))
    pick = np.unique(idx[np.linspace(0, idx.size - 1, min(idx.size, n_check)).astype(int)])
    want = want_all[pick]
    try:
        back = np.asarray(inverse(fwd[pick], coord_names=sampled_names, low_level_coord_names=sampled_names),
                          dtype=float)
    except Exception as err:
        raise ValueError(head + "the inverse, given the sampled columns {!r} only, raised {}: {}.  The inverse may "
                         "need a column the forward map mixes into the sampled ones (file columns {!r}): add it "
                         "with --parameter or --parameter-nofit.".format(
                             sampled_names, type(err).__name__, err, file_names)) from err
    if back.shape != want.shape:
        raise ValueError(head + "the inverse returned shape {} for {} rows and sampled columns {!r}".format(
            back.shape, want.shape[0], sampled_names))
    bad_inv = ~np.isfinite(back)
    if np.any(bad_inv):
        raise ValueError(head + "the inverse is non-finite on rows whose forward map is finite (rows per "
                         "column: {!r}).  The inverse must cover the forward map's image.".format(
                             {n: int(c) for n, c in zip(sampled_names, bad_inv.sum(axis=0)) if c}))
    scale = np.array([roundtrip_column_scale(want_all[idx, k]) for k in range(len(cols))])
    maxerr = np.max(np.abs(back - want) / scale, axis=0)
    if np.any(maxerr > tol):
        raise ValueError(head + "inverse mismatch: inverse(forward(x)) restricted to the sampled columns {!r} "
                         "does not reproduce the data rows (max error / column scale: {!r}; tolerance {:g} from "
                         "{}).  Either the inverse needs columns that are not sampled (add them with --parameter "
                         "or --parameter-nofit), or the forward and inverse maps (or the --external-range-args "
                         "they both receive) disagree.  An inverse that is accurate but approximate (tabulated, "
                         "iterative) can raise the tolerance with --supplementary-coordinate-roundtrip-tol or "
                         "ROUNDTRIP_TOL in the plugin.".format(
                             sampled_names, {n: float(e) for n, e in zip(sampled_names, maxerr) if e > tol},
                             tol, tol_src))
    return int(pick.size), int(X_file.shape[0] - idx.size), float(np.max(maxerr))
