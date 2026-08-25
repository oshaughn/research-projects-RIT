"""
hyperpipeline_io
================

A small, dependency-light I/O helper for the *hyperpipeline* ASCII format used
as an alternative to the legacy XML / positional-ASCII intermediate files
exchanged between ILE and CIP.

The format is a plain-text whitespace-separated table with a leading
``#``-prefixed header naming each column.  Column 0 is always ``lnL`` and
column 1 is always ``sigma_lnL``; the remaining columns are the (possibly
extensible) physical parameters describing each intrinsic sample.  This is
deliberately compatible with ``numpy.genfromtxt(..., names=True)``,
``numpy.loadtxt`` (when ``# `` headers are stripped as comments), and
``pandas.read_csv(..., sep=r'\\s+', comment='#')``.

Activation
----------

The new format is opt-in.  Both producers (``integrate_likelihood_extrinsic_batchmode``)
and consumers (``util_ConstructIntrinsicPosterior_GenericCoordinates.py``)
check the environment variable :envvar:`RIFT_HYPERPIPELINE_FORMAT`; when its
value is truthy (``1``, ``true``, ``yes``, case-insensitive), the new code
paths are taken.  When the variable is unset or falsy, the executables behave
exactly as before -- existing pipelines are unaffected.

Default columns
---------------

By default the format describes a quasi-circular precessing binary using
Cartesian component spins::

    lnL sigma_lnL m1 m2 a1x a1y a1z a2x a2y a2z

Optional column groups (auto-detected from the header):

* ``eccentricity`` (and optionally ``meanPerAno``)
* ``lambda1 lambda2`` (and optionally ``eos_table_index``)
* ``distance``  (in Mpc)
* ``ecliptic_longitude ecliptic_latitude`` (sky position treated as intrinsic)

The reader returns a structured ``numpy.ndarray`` so callers can address
columns by name regardless of order, and a small adaptor function is provided
to convert that structured array into the legacy positional ``(N, K)`` matrix
the existing CIP pipeline still expects.
"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

#: Environment-variable flag.  When truthy, ILE writes / CIP reads in the
#: hyperpipeline format.
ENV_FLAG = "RIFT_HYPERPIPELINE_FORMAT"

#: Magic token placed on the first line of every hyperpipeline file so
#: that downstream tools can sniff the format unambiguously.  Both writers
#: prepend ``# `` so the on-disk first line is ``# RIFT_HYPERPIPELINE_V1``.
#: Sniff/read code strips any leading ``#`` and whitespace before matching.
MAGIC = "RIFT_HYPERPIPELINE_V1"

#: Default base columns -- always present, in this order, for every file.
DEFAULT_BASE_COLUMNS = (
    "lnL", "sigma_lnL",
    "m1", "m2",
    "a1x", "a1y", "a1z",
    "a2x", "a2y", "a2z",
)

#: Columns that may optionally be appended after the base set.  The reader
#: accepts any subset; the writer emits only those that are active.
OPTIONAL_COLUMNS = (
    "eccentricity",
    "meanPerAno",
    "lambda1",
    "lambda2",
    "eos_table_index",
    "a6c",
    "E0",
    "p_phi0",
    "distance",
    "ecliptic_longitude",
    "ecliptic_latitude",
)


def is_active(env=None):
    """Return ``True`` if the hyperpipeline format is enabled in *env*.

    Parameters
    ----------
    env : Mapping or None
        Environment mapping.  Defaults to ``os.environ``.
    """
    if env is None:
        env = os.environ
    val = str(env.get(ENV_FLAG, "")).strip().lower()
    return val in ("1", "true", "yes", "on")


def build_column_list(use_eccentricity=False, use_meanPerAno=False,
                      use_tides=False, use_eos_index=False,
                      use_eob_parameters=False, use_hyperbolic=False,
                      use_distance=False, use_sky=False):
    """Compose the column-name tuple for a given physics configuration.

    The resulting tuple always begins with :data:`DEFAULT_BASE_COLUMNS` and
    appends the requested optional groups in a fixed canonical order::

        (eccentricity, [meanPerAno,] [lambda1, lambda2,]
         [eos_table_index,] [a6c,] [E0, p_phi0,] [distance,]
         [ecliptic_longitude, ecliptic_latitude])
    """
    cols = list(DEFAULT_BASE_COLUMNS)
    if use_eccentricity:
        cols.append("eccentricity")
        if use_meanPerAno:
            cols.append("meanPerAno")
    if use_tides:
        cols.extend(["lambda1", "lambda2"])
        if use_eos_index:
            cols.append("eos_table_index")
    if use_eob_parameters:
        cols.append("a6c")
    if use_hyperbolic:
        cols.extend(["E0", "p_phi0"])
    if use_distance:
        cols.append("distance")
    if use_sky:
        cols.extend(["ecliptic_longitude", "ecliptic_latitude"])
    return tuple(cols)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_row(fname, columns, values, append=False):
    """Write a single row in the hyperpipeline format.

    Parameters
    ----------
    fname : str
        Output filename.
    columns : sequence of str
        Names of the columns (length K).
    values : sequence of float
        Values, in the same order as *columns* (length K).
    append : bool
        If ``True``, append to *fname* without rewriting the header.  Used
        for accumulating multi-event ILE shards into a single file.

    Notes
    -----
    The function writes (in this order) a magic line, a column-name header
    line, and the data row.  When *append* is ``True`` the magic and header
    lines are skipped on the assumption that the file already begins with
    them.
    """
    columns = tuple(columns)
    values = np.asarray(values, dtype=float).reshape(1, -1)
    if values.shape[1] != len(columns):
        raise ValueError(
            "hyperpipeline_io.write_row: got {} values for {} columns".format(
                values.shape[1], len(columns)))

    mode = "a" if append else "w"
    with open(fname, mode) as fp:
        if not append:
            fp.write("# " + MAGIC + "\n")
            fp.write("# " + " ".join(columns) + "\n")
        # %.18e mirrors numpy.savetxt's default precision so this round-trips.
        fp.write(" ".join("{:.18e}".format(v) for v in values[0]) + "\n")


def write_table(fname, columns, data):
    """Write a multi-row table.  *data* must have shape ``(N, len(columns))``."""
    columns = tuple(columns)
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(columns):
        raise ValueError(
            "hyperpipeline_io.write_table: data has {} cols, header has {}"
            .format(arr.shape[1], len(columns)))
    # numpy.savetxt prepends `# ` to every line of `header`, so we ask it
    # to produce two `#`-prefixed comment lines: the magic marker and the
    # column-name line.
    header = MAGIC + "\n" + " ".join(columns)
    np.savetxt(fname, arr, header=header)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def _strip_comment(line):
    """Strip a leading ``#`` (or repeated ``#``) and surrounding whitespace."""
    return line.lstrip("#").strip()


def sniff(fname):
    """Return ``True`` if *fname* looks like a hyperpipeline file.

    The check is cheap: read the first non-empty line and look for the
    magic marker, or fall back to a header-only sniff (a ``#`` line listing
    the canonical first column names).

    Robust to non-ASCII inputs: the pipeline routinely passes gzipped XML
    grids (``overlap-grid-N.xml.gz``).  Those are binary, so a text-mode read
    raises ``UnicodeDecodeError`` on the gzip magic (``1f 8b``); we peek the
    first bytes and bail early for gzip / XML, and also treat any decode error
    as "not a hyperpipeline ASCII file" rather than letting it propagate.
    """
    try:
        # Binary peek: gzip magic (1f 8b) or a leading '<' (XML) is never a
        # hyperpipeline ASCII grid -- reject before any text decode.
        with open(fname, "rb") as fb:
            head = fb.read(2)
        if head[:2] == b"\x1f\x8b" or head[:1] == b"<":
            return False
    except (OSError, IOError):
        return False
    try:
        with open(fname, "r") as fp:
            for raw in fp:
                line = raw.strip()
                if not line:
                    continue
                if not line.startswith("#"):
                    return False
                payload = _strip_comment(line)
                if payload.startswith(MAGIC):
                    return True
                # Fallback: a header line listing the canonical first columns.
                toks = payload.split()
                if len(toks) >= 2 and toks[0] == "lnL" and toks[1] == "sigma_lnL":
                    return True
                return False
    except (OSError, IOError, UnicodeDecodeError):
        return False
    return False


def read_header(fname):
    """Return the tuple of column names declared in *fname*'s header.

    Skips the optional magic line; expects the next ``#``-prefixed line to
    be the column-name header.
    """
    with open(fname, "r") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            if not line.startswith("#"):
                break  # Hit data without finding a header.
            payload = _strip_comment(line)
            if payload.startswith(MAGIC):
                continue
            return tuple(payload.split())
    raise ValueError(
        "hyperpipeline_io.read_header: no column header found in {}".format(fname))


def read_table(fname):
    """Read *fname* and return ``(structured_array, column_names_tuple)``.

    Uses :func:`numpy.genfromtxt` with ``names=True`` so columns are
    addressable by name.  Multiple ``#``-prefixed lines (e.g. the magic
    marker plus repeated headers from a concatenated multi-shard file) are
    all treated as comments by genfromtxt; the column names come from the
    first such line, which we determine explicitly via :func:`read_header`.
    """
    columns = read_header(fname)
    # genfromtxt's auto name detection tries to infer from the *last*
    # comment block immediately preceding the data, which is brittle for
    # concatenated multi-shard files.  Pass names explicitly to be robust.
    arr = np.genfromtxt(fname, names=columns, comments="#",
                        dtype=float, invalid_raise=False)
    if arr.ndim == 0:
        # Single-row file -> genfromtxt returns 0-d structured scalar;
        # promote to length-1 array for consistent downstream handling.
        arr = np.atleast_1d(arr)
    return arr, columns


# ---------------------------------------------------------------------------
# Adaptor: structured array -> legacy positional matrix expected by CIP
# ---------------------------------------------------------------------------

def to_legacy_dat(arr, use_eccentricity=False, use_meanPerAno=False,
                  use_tides=False, use_eos_index=False, use_distance=False,
                  use_eob_parameters=False, use_hyperbolic=False,
                  use_sky=False):
    """Reshape a hyperpipeline structured array into the legacy CIP layout.

    The legacy CIP reader expects a plain ``(N, K)`` ``ndarray`` whose
    columns are::

        [event_id, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,
         (distance?), (lambda1, lambda2, (eos_index)?)?, (a6c?)?,
         (E0, p_phi0)?,
         (eccentricity, (meanPerAno)?)?,
         (ecliptic_longitude, ecliptic_latitude)?,
         lnL, sigma_lnL]

    The ordering of optional groups before ``lnL`` mirrors what the legacy
    ``col_lnL`` increment chain in
    ``util_ConstructIntrinsicPosterior_GenericCoordinates.py`` produces.
    The synthetic ``event_id`` column is filled with ``-1`` because the
    hyperpipeline format does not carry that book-keeping field.
    """
    n = len(arr)

    cols = ["event_id", "m1", "m2", "a1x", "a1y", "a1z", "a2x", "a2y", "a2z"]
    if use_distance:
        cols.append("distance")
    if use_tides:
        cols.extend(["lambda1", "lambda2"])
        if use_eos_index:
            cols.append("eos_table_index")
    if use_eob_parameters:
        cols.append("a6c")
    if use_hyperbolic:
        cols.extend(["E0", "p_phi0"])
    if use_eccentricity:
        cols.append("eccentricity")
        if use_meanPerAno:
            cols.append("meanPerAno")
    if use_sky:
        cols.extend(["ecliptic_longitude", "ecliptic_latitude"])
    cols.extend(["lnL", "sigma_lnL"])

    out = np.zeros((n, len(cols)), dtype=float)
    for j, name in enumerate(cols):
        if name == "event_id":
            out[:, j] = -1.0
        else:
            if name not in arr.dtype.names:
                raise KeyError(
                    "hyperpipeline_io.to_legacy_dat: requested column '{}' "
                    "not present in file (have {})".format(name, arr.dtype.names))
            out[:, j] = arr[name]
    return out


def read_many(filenames):
    """Read and vertically stack a list of hyperpipeline files.

    All files must share the same column header.  Empty / unreadable / wrong
    files are skipped with a warning printed to stderr (this mirrors
    util_CleanILE.py's tolerance for malformed shards).  Returns
    ``(structured_array, column_names_tuple)``.
    """
    import sys
    chunks = []
    columns = None
    for fname in filenames:
        try:
            if not os.path.exists(fname):
                continue
            if os.stat(fname).st_size == 0:
                continue
            arr, hdr = read_table(fname)
        except Exception as exc:
            sys.stderr.write(
                "hyperpipeline_io.read_many: skipping {}: {}\n".format(fname, exc))
            continue
        if columns is None:
            columns = hdr
        elif hdr != columns:
            sys.stderr.write(
                "hyperpipeline_io.read_many: column mismatch in {} "
                "(have {}, expected {}); skipping\n".format(fname, hdr, columns))
            continue
        if arr.ndim == 0:
            arr = np.atleast_1d(arr)
        chunks.append(arr)
    if not chunks:
        raise ValueError(
            "hyperpipeline_io.read_many: no readable input files among "
            "{} candidate(s)".format(len(filenames)))
    out = np.concatenate(chunks)
    return out, columns


def consolidate(arr, columns, sigma_cut=0.9, digits=5):
    """Consolidate duplicate intrinsic-parameter rows by weighted averaging.

    Mirrors the per-key averaging in ``util_CleanILE.py``: rows are grouped
    by their intrinsic-parameter values (every column except ``lnL`` and
    ``sigma_lnL``, rounded to *digits* decimals), rows with
    ``sigma_lnL > sigma_cut`` are discarded, and within each group ``lnL`` is
    weighted-averaged using the same ``1/sigma^2`` (after Lmax factoring)
    weights as ``RIFT.misc.weight_simulations.AverageSimulationWeights``::

        wts = (sigma_k)^-2 / sum_k (sigma_k)^-2
        <I> = sum_k wts_k * I_k
        sigma_<I> = sqrt(1 / sum_k 1/sigma_k^2) / <I>

    Returns ``(consolidated_array, columns)`` where the array is sorted by
    ``lnL`` in descending order.
    """
    columns = tuple(columns)
    if "lnL" not in columns or "sigma_lnL" not in columns:
        raise ValueError(
            "hyperpipeline_io.consolidate: input is missing 'lnL' or 'sigma_lnL'"
            " columns: {}".format(columns))
    intrinsic_cols = tuple(c for c in columns if c not in ("lnL", "sigma_lnL"))

    # Drop poorly-resolved samples (mirrors util_CleanILE behaviour).
    keep = arr["sigma_lnL"] <= sigma_cut
    arr = arr[keep]

    if len(arr) == 0:
        return np.empty(0, dtype=arr.dtype), columns

    # Group by rounded intrinsic-parameter tuple.
    groups = {}
    for row in arr:
        key = tuple(round(float(row[c]), digits) for c in intrinsic_cols)
        groups.setdefault(key, []).append(row)

    out = np.empty(len(groups), dtype=arr.dtype)
    for i, (key, rows) in enumerate(groups.items()):
        lnL = np.array([r["lnL"] for r in rows], dtype=float)
        sigOverL = np.array([r["sigma_lnL"] for r in rows], dtype=float)
        sigOverL = np.maximum(sigOverL, 1e-7)
        lnLmax = np.max(lnL)
        # sigma_k absolute (factor out Lmax for numerical stability).
        sig = sigOverL * np.exp(lnL - lnLmax)
        wts = 1.0 / (sig * sig)
        wts = wts / np.sum(wts)
        lnL_mean_minus_max = np.log(np.sum(np.exp(lnL - lnLmax) * wts))
        sigma_net_overL = np.sqrt(1.0 / np.sum(1.0 / (sig * sig))) / np.exp(lnL_mean_minus_max)
        # Build the consolidated row: intrinsic values from the key, lnL/sigma
        # from the weighted average.
        for j, name in enumerate(intrinsic_cols):
            out[i][name] = key[j]
        out[i]["lnL"] = lnL_mean_minus_max + lnLmax
        out[i]["sigma_lnL"] = sigma_net_overL

    # Sort by lnL descending so the composite has the highest-likelihood row
    # first -- preserves the convention of the legacy `sort -rg -kN` step.
    order = np.argsort(out["lnL"])[::-1]
    return out[order], columns


#: Bidirectional alias map between hyperpipeline on-disk column names and
#: the corresponding ``ChooseWaveformParams`` attribute names.  The on-disk
#: convention follows the LALInference / posterior-export naming
#: (``a1x``, ``a1y``, ...) while ``ChooseWaveformParams`` stores spins as
#: ``s1x``, ``s1y``, etc.  The codebase already defines this mapping in
#: several places (e.g. util_FitAndEvaluate_GenericCoordinates.py and
#: util_ConstructIntrinsicPosterior_GenericCoordinates.py); we centralise it
#: here so the hyperpipeline I/O layer is the single bridge.
COLUMN_ALIAS_DISK_TO_ATTR = {
    "a1x": "s1x", "a1y": "s1y", "a1z": "s1z",
    "a2x": "s2x", "a2y": "s2y", "a2z": "s2z",
    "ecliptic_longitude": "phi",
    "ecliptic_latitude": "theta",
}
COLUMN_ALIAS_ATTR_TO_DISK = {v: k for k, v in COLUMN_ALIAS_DISK_TO_ATTR.items()}


def disk_to_attr(name):
    """Map an on-disk column name to its ChooseWaveformParams attribute."""
    return COLUMN_ALIAS_DISK_TO_ATTR.get(name, name)


def attr_to_disk(name):
    """Map a ChooseWaveformParams attribute to its on-disk column name."""
    return COLUMN_ALIAS_ATTR_TO_DISK.get(name, name)


#: On-disk units for parameters that have a non-trivial scale.  m1 / m2 are
#: stored in solar masses for human readability (matches the legacy ILE ASCII
#: convention) but ``ChooseWaveformParams.m1`` is internally in kg, so any
#: writer / reader that round-trips through a P object MUST apply this
#: scaling.  Distance is stored in Mpc; ``ChooseWaveformParams.dist`` is in
#: metres.  Other parameters are dimensionless / pre-scaled and pass through.
PARAM_DISK_TO_SI = {
    "m1": "MSUN_SI",
    "m2": "MSUN_SI",
    "dist": "PC_SI_MPC",  # special: metres = value(Mpc) * 1e6 * lal.PC_SI
    "distance": "PC_SI_MPC",
}


def _disk_to_si(name, val, lal_module):
    """Convert *val* (on-disk units) to the SI convention used by P."""
    scale = PARAM_DISK_TO_SI.get(name)
    if scale is None:
        return float(val)
    if scale == "MSUN_SI":
        return float(val) * lal_module.MSUN_SI
    if scale == "PC_SI_MPC":
        return float(val) * 1.0e6 * lal_module.PC_SI
    return float(val)


def _si_to_disk(name, val, lal_module):
    """Convert *val* (P's SI units) to the on-disk convention."""
    scale = PARAM_DISK_TO_SI.get(name)
    if scale is None:
        return float(val)
    if scale == "MSUN_SI":
        return float(val) / lal_module.MSUN_SI
    if scale == "PC_SI_MPC":
        return float(val) / (1.0e6 * lal_module.PC_SI)
    return float(val)


#: Default filename suffix for hyperpipeline grid files emitted by
#: :func:`write_grid_from_P_list` when the caller passes a basename
#: without one.  Mirrors the auto-append behaviour of
#: ``lalsimutils.ChooseWaveformParams_array_to_xml`` (which appends
#: ``.xml.gz`` when it isn't already present), so writer call sites can
#: pass the same basename to either helper.
DEFAULT_GRID_SUFFIX = ".dat"


def with_grid_suffix(fname):
    """Append :data:`DEFAULT_GRID_SUFFIX` if *fname* lacks a recognised one."""
    if fname.endswith(".dat") or fname.endswith(".txt"):
        return fname
    return fname + DEFAULT_GRID_SUFFIX


def write_grid_from_P_list(fname, P_list, columns,
                           lal_module=None, lalsimutils_module=None,
                           lnL_values=None, sigma_lnL_values=None):
    """Write a hyperpipeline grid file from a list of ChooseWaveformParams.

    Used by CIP / puffball / fetch when emitting the next-iteration
    intrinsic-parameter grid in hyperpipeline format instead of XML.

    *columns* must include ``lnL`` and ``sigma_lnL`` (filled with 0 by
    default if *lnL_values* / *sigma_lnL_values* are not provided -- these
    columns carry no information for posterior draws but are kept to
    preserve the universal column-0/1 invariant).  Mass and distance values
    are converted from P's internal SI units to the on-disk convention
    (solar masses, Mpc) via :data:`PARAM_DISK_TO_SI`.

    Parameters
    ----------
    fname : str
        Output filename.
    P_list : sequence
        ``ChooseWaveformParams`` instances.
    columns : sequence of str
        Output column names (must contain ``lnL`` and ``sigma_lnL``).
    lal_module, lalsimutils_module : modules
        Pass ``lal`` and ``RIFT.lalsimutils`` explicitly.  Avoids importing
        them at module-load time so :mod:`hyperpipeline_io` stays
        dependency-light for tests that only exercise pure-numpy code paths.
    lnL_values, sigma_lnL_values : array-like or None
        Optional per-row likelihood values.  Filled with zeros if omitted.
    """
    columns = tuple(columns)
    if "lnL" not in columns or "sigma_lnL" not in columns:
        raise ValueError(
            "hyperpipeline_io.write_grid_from_P_list: 'lnL' and 'sigma_lnL' "
            "must appear in columns; got {}".format(columns))
    fname = with_grid_suffix(fname)
    n = len(P_list)
    mat = np.zeros((n, len(columns)), dtype=float)
    if lnL_values is not None:
        mat[:, columns.index("lnL")] = np.asarray(lnL_values, dtype=float)
    if sigma_lnL_values is not None:
        mat[:, columns.index("sigma_lnL")] = np.asarray(sigma_lnL_values, dtype=float)
    for i, P in enumerate(P_list):
        for j, name in enumerate(columns):
            if name in ("lnL", "sigma_lnL"):
                continue
            # Resolve disk-name -> attr-name via the alias map (a1x -> s1x).
            attr_name = disk_to_attr(name)
            # First try direct attribute, then fall through to extract_param
            # (handles derived coordinates: mc, eta, q, chi1, ...)
            if hasattr(P, attr_name):
                raw = getattr(P, attr_name)
            elif lalsimutils_module is not None and hasattr(P, "extract_param"):
                raw = P.extract_param(attr_name)
            else:
                raise AttributeError(
                    "hyperpipeline_io.write_grid_from_P_list: P has no '{}' "
                    "(or alias '{}') attribute and no lalsimutils provided "
                    "for extract_param".format(name, attr_name))
            if lal_module is not None:
                mat[i, j] = _si_to_disk(name, raw, lal_module)
            else:
                mat[i, j] = float(raw)
    write_table(fname, columns, mat)


def read_grid_to_P_list(fname, P_factory, lal_module=None,
                        valid_params=None):
    """Read a hyperpipeline grid file and return ``(P_list, columns)``.

    For each row, instantiates a ``ChooseWaveformParams`` via *P_factory()*
    and assigns the named columns.  Mass and distance values are converted
    from on-disk units (solar masses, Mpc) to P's internal SI units via
    :data:`PARAM_DISK_TO_SI`.

    Parameters
    ----------
    fname : str
        Input filename.
    P_factory : callable
        Zero-arg callable that returns a fresh ``ChooseWaveformParams``.
        Typically ``lalsimutils.ChooseWaveformParams``.
    lal_module : module
        Pass ``lal`` for unit conversions.  If None, no conversion is done
        (units stay on-disk -- only safe when downstream code knows that).
    valid_params : iterable of str
        Restrict assignment to these names.  Defaults to the full column
        set.  Pass ``lalsimutils.valid_params`` to mirror ILE's --sim-grid
        intersection behaviour.
    """
    arr, columns = read_table(fname)
    if valid_params is not None:
        valid_params = set(valid_params)
        # A column is "active" if its disk-name OR its alias-resolved
        # attr-name is in valid_params.  This lets the caller pass
        # lalsimutils.valid_params (which knows about s1x, not a1x) and
        # still get spin columns assigned correctly.
        active = [c for c in columns
                  if c in valid_params or disk_to_attr(c) in valid_params]
    else:
        active = [c for c in columns if c not in ("lnL", "sigma_lnL")]
    P_list = []
    for row in arr:
        P = P_factory()
        for name in active:
            raw = float(row[name])
            if lal_module is not None:
                val = _disk_to_si(name, raw, lal_module)
            else:
                val = raw
            attr_name = disk_to_attr(name)
            if hasattr(P, attr_name):
                setattr(P, attr_name, val)
            elif hasattr(P, "assign_param"):
                P.assign_param(attr_name, val)
        P_list.append(P)
    return P_list, columns


def legacy_column_indices(use_eccentricity=False, use_meanPerAno=False,
                          use_tides=False, use_eos_index=False,
                          use_distance=False, use_eob_parameters=False,
                          use_hyperbolic=False, use_sky=False):
    """Return the positional column indices the legacy CIP loop expects.

    Mirrors the layout produced by :func:`to_legacy_dat` so callers can
    reset the ``col_lnL`` / ``col_lambda1`` / ``col_distance`` /
    ``col_eccentricity`` / ``col_meanPerAno`` integers used by the existing
    CIP indexing logic without having to recompute them.

    Returns a dict keyed by ``'lnL'``, ``'sigma_lnL'``, ``'distance'``,
    ``'lambda1'``, ``'a6c'``, ``'E0'``, ``'p_phi0'``,
    ``'eccentricity'``, ``'meanPerAno'``,
    ``'ecliptic_longitude'``, ``'ecliptic_latitude'``.  Any column not
    present in the configuration maps to ``None``.
    """
    # event_id m1 m2 a1x a1y a1z a2x a2y a2z = 9 leading columns.
    idx = 9
    out = {"lnL": None, "sigma_lnL": None, "distance": None,
           "lambda1": None, "a6c": None, "E0": None, "p_phi0": None,
           "eccentricity": None, "meanPerAno": None,
           "ecliptic_longitude": None, "ecliptic_latitude": None}
    if use_distance:
        out["distance"] = idx
        idx += 1
    if use_tides:
        out["lambda1"] = idx
        idx += 2  # lambda1, lambda2
        if use_eos_index:
            idx += 1  # eos_table_index (no positional alias used by CIP)
    if use_eob_parameters:
        out["a6c"] = idx
        idx += 1
    if use_hyperbolic:
        out["E0"] = idx
        out["p_phi0"] = idx + 1
        idx += 2
    if use_eccentricity:
        out["eccentricity"] = idx
        idx += 1
        if use_meanPerAno:
            out["meanPerAno"] = idx
            idx += 1
    if use_sky:
        out["ecliptic_longitude"] = idx
        out["ecliptic_latitude"] = idx + 1
        idx += 2
    out["lnL"] = idx
    out["sigma_lnL"] = idx + 1
    return out
