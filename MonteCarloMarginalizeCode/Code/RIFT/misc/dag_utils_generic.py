# Copyright (C) 2013  Evan Ochsner
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Backend-neutral workflow / DAG utilities for RIFT.

This module is a re-implementation of ``RIFT.misc.dag_utils`` whose original
form was tied directly to ``glue.pipeline`` (a piece of LIGO/glue infrastructure
that is fragile to install and not always available -- particularly on macOS).

Architecture
============

The module separates three concerns:

1. **A backend-neutral workflow data model** -- :class:`_GenericJob`,
   :class:`_GenericNode`, :class:`_GenericDAG`, :class:`_GenericManJob`.  These
   are *plain data containers*: they record the executable, arguments,
   resource requests, environment, files, queue count, dependencies, etc.
   They expose the historical ``glue.pipeline.CondorDAGJob`` API
   (``add_opt``, ``add_arg``, ``add_condor_cmd``, ``set_sub_file``,
   ``write_sub_file``, ...) as a *facade* on top of that data model, so
   existing RIFT code keeps working.  The underlying state is fully generic;
   nothing about it presupposes Condor.

2. **A pluggable backend layer** -- :class:`WorkflowBackend` is an abstract
   base class.  Each backend implements ``emit_job(job, path)`` (turn a
   :class:`_GenericJob` into a per-system submit description) and
   ``emit_dag(dag, path)`` (turn a :class:`_GenericDAG` into a per-system
   workflow driver).  Three backends ship with this module:

   - :class:`HTCondorBackend` -- modern HTCondor python bindings
     (``htcondor.Submit``).  Emits ``.sub`` files and a HTCondor DAGMan
     ``.dag`` file.
   - :class:`GluePipelineBackend` -- legacy ``glue.pipeline``-based fallback,
     useful where the htcondor python bindings are not available (e.g. some
     macOS installs).  Translates the generic job spec into ``glue.pipeline``
     calls at emit time.
   - :class:`SlurmBackend` -- emits ``.sbatch`` scripts plus a shell driver
     script that submits the jobs in dependency order (using
     ``sbatch --dependency=afterok:JOBID``).

3. **A backend registry** -- :func:`register_backend`, :func:`set_backend`,
   :func:`get_backend`.  Backends are looked up by name.  New backends can be
   registered by user code without touching this module.

Backend selection
=================

At import time this module performs an *auto-detection*: it tries the
HTCondor python bindings first, then ``glue.pipeline``.  Slurm is never
auto-selected because no reliable purely-pythonic test for "we are on a Slurm
cluster" exists; pick it explicitly.

The active backend can be controlled in three ways, in order of precedence:

1. Calling :func:`set_backend("name")` from python.
2. The ``RIFT_DAG_BACKEND`` environment variable.  Recognised values:
   ``htcondor``, ``glue``, ``slurm``, ``auto`` (default).
3. Auto-detection (above).

Custom backends
===============

To add a new execution-system backend (e.g. PBS/Torque, LSF, Kubernetes, ...)::

    from RIFT.misc.dag_utils_generic import (
        WorkflowBackend, register_backend, set_backend
    )

    class PBSBackend(WorkflowBackend):
        name = "pbs"

        def emit_job(self, job, path):
            ...

        def emit_dag(self, dag, path):
            ...

    register_backend(PBSBackend())
    set_backend("pbs")

Public API
==========

For consumers of this module, the public surface is:

- The factories ``CondorDAGJob``, ``CondorDAG``, ``CondorDAGNode``,
  ``CondorDAGManJob`` (named for historical reasons; they return the
  backend-neutral wrappers).  A ``pipeline`` namespace exposes the same four
  factories so code that previously did ``from glue import pipeline`` can do
  ``from RIFT.misc.dag_utils_generic import pipeline`` instead.
- All the ``write_*_sub`` helpers (``write_CIP_sub``, ``write_ILE_sub_simple``,
  ``write_consolidate_sub_simple``, ...) ported verbatim from the original
  ``dag_utils.py``.  Because they are written against the facade API, they
  work under any backend.
- The utility helpers ``which``, ``mkdir``, ``quote_arguments``,
  ``safely_quote_arg_str``, ``bilby_ish_string_to_dict``,
  ``build_resolved_env``.
"""

import os
import re
import sys
import abc
import shlex
from time import time
from hashlib import md5

import numpy as np
import configparser

__author__ = (
    "Evan Ochsner <evano@gravity.phys.uwm.edu>, "
    "Chris Pankow <pankow@gravity.phys.uwm.edu>"
)


# ===========================================================================
# Utility helpers
# ===========================================================================

# getenv=True deprecated, will need workaround to explicitly pull extra environment variables
default_getenv_value = 'True'
default_getenv_osg_value = 'True'
if 'RIFT_GETENV' in os.environ:
    default_getenv_value = os.environ['RIFT_GETENV']
if 'RIFT_GETENV_OSG' in os.environ:
    default_getenv_osg_value = os.environ['RIFT_GETENV_OSG']


def is_exe(fpath):
    return os.path.exists(fpath) and os.access(fpath, os.X_OK)


def which(program):
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def mkdir(dir_name):
    try:
        os.mkdir(dir_name)
    except OSError:
        pass


def generate_job_id():
    """Generate a unique md5 hash for use as a job ID."""
    t = str(int(time() * 1000))
    r = str(int(np.random.random() * 100000000000000000))
    return md5((t + r).encode("utf-8")).hexdigest()


def _double_up_quotes(instr):
    return instr.replace("'", "''").replace('"', '""')


def quote_arguments(args):
    """Quote a string or list of strings using the Condor submit-file 'new' argument-quoting rules."""
    if isinstance(args, str):
        args_list = [args]
    else:
        args_list = args
    quoted_args = []
    for a in args_list:
        qa = _double_up_quotes(a)
        if " " in qa or "'" in qa:
            qa = "'" + qa + "'"
        quoted_args.append(qa)
    return " ".join(quoted_args)


def safely_quote_arg_str(arg_str):
    if not ('"' in arg_str):
        return quote_arguments
    quote_breaks = arg_str.split('"')
    if len(quote_breaks) != 3:
        raise Exception(" Arg parsing: multiple quoted argument strings provided, not ready to handle ")
    args0 = quote_arguments(quote_breaks[0].split())
    args2 = quote_arguments(quote_breaks[2].split())
    args1 = quote_arguments('"{}"'.format(quote_breaks[1]))
    return "{} {} {}".format(args0, args1, args2)


def bilby_ish_string_to_dict(my_str):
    items = my_str.replace('{', '').replace('}', '').strip().split(',')
    items = [x for x in items if len(x) > 0]
    pseudo_dict = {}
    for item in items:
        key, val = item.split(':')
        pseudo_dict[key] = val
    return pseudo_dict


def match_expr(my_list, my_expr):
    list_out = []
    p = re.compile(my_expr.replace('*', '.*'))
    for name in my_list:
        if p.match(name):
            list_out.append(name)
    print("  RESOLVE: for {} found ".format(my_expr), list_out)
    return list_out


def build_resolved_env(my_str):
    env_dict = os.environ
    str_out = ""
    for pat in my_str.split(','):
        print("  RESOLVE: building env for ", pat)
        if ('*' in pat):
            list_out = match_expr(list(env_dict.keys()), pat)
        else:
            list_out = [pat] if pat in env_dict else []
        for name in list_out:
            str_out += (" {}={} ".format(name, env_dict[name]))
    return str_out


default_resolved_env = None
default_resolved_osg_env = None
if 'RIFT_GETENV_RESOLVE' in os.environ:
    default_resolved_env = build_resolved_env(default_getenv_value)
    default_resolved_osg_env = build_resolved_env(default_getenv_osg_value)


# ===========================================================================
# Backend-neutral workflow data model
# ===========================================================================
#
# These classes are plain data containers.  They expose the historical
# ``glue.pipeline.CondorDAGJob`` API as a facade so that existing RIFT code
# continues to work, but the underlying state is fully generic and contains
# no Condor-specific encoding.  At ``write_sub_file()`` / ``write_concrete_dag()``
# time, the active :class:`WorkflowBackend` is asked to translate this state
# into the appropriate per-system artefact (Condor submit file, sbatch script,
# ...).
# ===========================================================================


# Set of HTCondor "request_*" / submit-file commands that map to semantic
# resource concepts shared by most batch systems.  When ``add_condor_cmd`` is
# called with one of these, we *also* record it as a structured resource so
# non-Condor backends can use it.
_RESOURCE_KEYS = {
    "request_memory": "memory",
    "request_disk": "disk",
    "request_cpus": "cpus",
    "request_gpus": "gpus",
    "request_GPUs": "gpus",
    "+MaxRunTimeMinutes": "runtime_minutes",
    "MY.MaxRunTimeMinutes": "runtime_minutes",
    "max_runtime_minutes": "runtime_minutes",
}


class _GenericJob(object):
    """Backend-neutral description of a single job/task.

    Exposes the legacy ``glue.pipeline.CondorDAGJob`` API as a facade.  All
    ``add_*`` / ``set_*`` calls just record state; the actual on-disk artefact
    is produced by the active :class:`WorkflowBackend` when
    :meth:`write_sub_file` is called.
    """

    def __init__(self, universe="vanilla", executable=None):
        # Generic, backend-neutral state
        self.universe = universe         # "vanilla", "local", ... -- backends interpret per-system
        self.executable = executable
        self.opts = []                   # list of (name, value-or-None) for --opts
        self.short_opts = []             # list of (name, value)        for -opts
        self.file_opts = []              # list of (name, value)        for file--opts
        self.var_opts = []               # variable opts (filled per-node from macros)
        self.arguments = []              # positional arguments
        self.environment = {}            # explicit env: dict of name -> value
        self.inherit_environment = False # condor's "getenv = True" / slurm's "--export=ALL"
        self.condor_cmds = []            # raw Condor submit-file commands (escape hatch / passthrough)
        self.resources = {}              # structured: memory, disk, cpus, gpus, runtime_minutes
        # Files
        self.sub_file = None
        self.log_file = None
        self.stdout_file = None
        self.stderr_file = None
        # Replicas (Condor "queue N", Slurm "--array")
        self.queue_count = 1

    # ------------------------------------------------------------------
    # Compatibility shims: legacy "private" attribute access
    # ------------------------------------------------------------------
    @property
    def _CondorJob__queue(self):
        return self.queue_count

    @_CondorJob__queue.setter
    def _CondorJob__queue(self, value):
        self.queue_count = int(value)

    @property
    def _CondorJob__arguments(self):
        return self.arguments

    @_CondorJob__arguments.setter
    def _CondorJob__arguments(self, value):
        self.arguments = list(value)

    # ------------------------------------------------------------------
    # Facade API (legacy CondorDAGJob method names)
    # ------------------------------------------------------------------
    def set_universe(self, universe):
        self.universe = universe

    def set_executable(self, executable):
        self.executable = executable

    def set_sub_file(self, fname):
        self.sub_file = fname

    def get_sub_file(self):
        return self.sub_file

    def set_log_file(self, fname):
        self.log_file = fname

    def set_stdout_file(self, fname):
        self.stdout_file = fname

    def set_stderr_file(self, fname):
        self.stderr_file = fname

    def add_opt(self, name, value=None):
        self.opts.append((name, value))

    def add_short_opt(self, name, value):
        self.short_opts.append((name, value))

    def add_var_opt(self, name):
        self.var_opts.append(name)

    def add_file_opt(self, name, value):
        self.file_opts.append((name, value))

    def add_arg(self, arg):
        self.arguments.append(arg)

    def add_condor_cmd(self, key, value):
        """Record a HTCondor submit-file command.

        For commands that map to a *semantic* resource concept
        (``request_memory``, ``request_disk``, ``request_cpus``,
        ``request_gpus``, runtime limits) we also store the value as a
        structured :attr:`resources` entry so non-Condor backends can use it.
        Commands that have no portable equivalent (``MY.flock_local``,
        ``+SingularityImage``, ...) are stored verbatim and only Condor-style
        backends will emit them; other backends may emit them as comments.
        """
        # Capture a few well-known semantic commands
        if key == "getenv":
            self.inherit_environment = (str(value).strip().lower() in ("true", "1", "yes"))
        elif key == "environment":
            # Condor environment string: a flat "K=V K=V" sequence (possibly quoted).
            self._merge_environment_string(value)
        else:
            mapped = _RESOURCE_KEYS.get(key)
            if mapped is not None:
                self.resources[mapped] = value
        self.condor_cmds.append((key, value))

    def _merge_environment_string(self, value):
        """Best-effort parse of a Condor 'environment = ...' string."""
        s = str(value).strip()
        # Strip surrounding quotes if present
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1]
        try:
            tokens = shlex.split(s)
        except ValueError:
            tokens = s.split()
        for tok in tokens:
            if "=" in tok:
                k, v = tok.split("=", 1)
                self.environment[k] = v

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------
    def write_sub_file(self):
        """Hand the job to the active backend, which writes the submit artefact."""
        if self.sub_file is None:
            raise RuntimeError("write_sub_file: no sub-file path set")
        get_backend().emit_job(self, self.sub_file)

    # Useful for tests / debugging
    def to_dict(self):
        return {
            "universe": self.universe,
            "executable": self.executable,
            "opts": list(self.opts),
            "short_opts": list(self.short_opts),
            "file_opts": list(self.file_opts),
            "var_opts": list(self.var_opts),
            "arguments": list(self.arguments),
            "environment": dict(self.environment),
            "inherit_environment": self.inherit_environment,
            "condor_cmds": list(self.condor_cmds),
            "resources": dict(self.resources),
            "sub_file": self.sub_file,
            "log_file": self.log_file,
            "stdout_file": self.stdout_file,
            "stderr_file": self.stderr_file,
            "queue_count": self.queue_count,
        }


def _make_unique_node_name(prefix):
    """Return a globally-unique node name of the form ``<prefix>-<md5>``.

    Mirrors ``glue.pipeline.CondorDAGNode``: the md5 hash is generated from
    ``time() * 1000`` plus a random integer, so two workflows constructed in
    different processes (or the same process) cannot collide accidentally
    when their ``.dag`` files are merged.  ``generate_job_id()`` is exposed
    as the helper that produces the hash.
    """
    safe_prefix = re.sub(r"\W+", "_", prefix or "job") or "job"
    return "{}-{}".format(safe_prefix, generate_job_id())


class _GenericNode(object):
    """A node in a workflow DAG.  Bound to a :class:`_GenericJob`.

    Each node carries a globally-unique ``name`` (and equivalent
    ``_CondorDAGNode__md5name``) so multiple workflows can be combined
    without identifier collisions.  The legacy ``_CondorDAGNode__md5name``
    private attribute is preserved so external code that reaches for that
    name (e.g. when wiring up ``SCRIPT POST`` items by hand) keeps working.
    """

    def __init__(self, job):
        self.job = job
        self.macros = {}
        self.category = None
        self.retry = 0
        self.parents = []
        prefix = os.path.basename(getattr(job, "executable", None) or "") or "job"
        self.name = _make_unique_node_name(prefix)
        # glue.pipeline.CondorDAGNode stores its globally-unique id under
        # ``self.__md5name`` (which Python name-mangles to
        # ``_CondorDAGNode__md5name``).  Mirror that exact attribute name so
        # consumer code that hard-codes it keeps working.
        self._CondorDAGNode__md5name = self.name

    # Compatibility names from glue.pipeline
    def get_name(self):
        return self.name

    def set_name(self, name):
        """Override the node name (e.g. for legibility).  Keeps the legacy
        ``_CondorDAGNode__md5name`` attribute in sync."""
        self.name = name
        self._CondorDAGNode__md5name = name

    def add_macro(self, key, value):
        self.macros[str(key)] = value

    def set_category(self, category):
        self.category = category

    def set_retry(self, n):
        self.retry = int(n)

    def add_parent(self, parent_node):
        self.parents.append(parent_node)


class _GenericSubdagNode(_GenericNode):
    """A node that represents an external sub-DAG (Condor: ``SUBDAG EXTERNAL``)."""

    def __init__(self, subdag_file):
        # Don't go through _GenericNode.__init__ -- there's no underlying job.
        self.job = None
        self.macros = {}
        self.category = None
        self.retry = 0
        self.parents = []
        self.subdag_file = subdag_file
        self.name = _make_unique_node_name("subdag")
        self._CondorDAGNode__md5name = self.name


class _GenericManJob(object):
    """Wrapper for an external sub-DAG, the moral equivalent of
    ``glue.pipeline.CondorDAGManJob``."""

    def __init__(self, dag_file):
        self.dag_file = dag_file

    def create_node(self):
        return _GenericSubdagNode(self.dag_file)


class _GenericDAG(object):
    """Backend-neutral workflow / DAG container.

    In addition to the basic node graph, the DAG carries the *control-logic*
    overlays that HTCondor's DAGMan supports natively:

      * **per-node pre/post scripts** -- analogous to ``SCRIPT PRE`` /
        ``SCRIPT POST`` directives.  These run on the submit node before /
        after the corresponding job.
      * **abort-on hooks** -- analogous to ``ABORT-DAG-ON``: if the named
        node returns the given exit code, the DAG aborts (optionally with
        a specific success/failure return value).
      * **dot visualisation file** -- ``DOT`` directive.
      * **escape-hatch raw directives** -- arbitrary backend-native lines
        appended verbatim by the active backend.

    Each backend is responsible for translating these into its native form:
    HTCondor / glue.pipeline emit DAGMan directives unchanged; the Slurm
    backend translates pre/post into pre-sbatch shell stanzas and
    afterany-dependency wrapper jobs respectively, and emits the rest as
    documented comments where there's no direct equivalent.

    The control-logic accessors can be called either *before*
    :meth:`write_concrete_dag` (in which case they're emitted alongside
    the rest of the DAG) or *after* (in which case the active backend is
    asked to append them to the artefact it already produced -- this
    matches the legacy pattern of ``open(dag_file, "a").write(...)`` that
    older RIFT bin/ scripts use).
    """

    def __init__(self, log=None):
        self.log = log
        self.dag_file = None
        self.nodes = []
        # Control-logic overlays
        self.script_pre = []        # list of (node_or_name, exe, args_str)
        self.script_post = []       # list of (node_or_name, exe, args_str)
        self.abort_on = []          # list of (node_or_name, exit_code, return_value)
        self.dot_file = None
        self.extra_directives = []  # list of raw lines (escape hatch)
        self._already_written = False

    def set_dag_file(self, name):
        self.dag_file = name

    def add_node(self, node):
        self.nodes.append(node)

    @staticmethod
    def _node_name(node_or_name):
        """Accept either a node instance or its bare name string."""
        if isinstance(node_or_name, str):
            return node_or_name
        # Prefer the legacy ``_CondorDAGNode__md5name`` attribute so we
        # match exactly what HTCondor / glue.pipeline already use.
        return (
            getattr(node_or_name, "_CondorDAGNode__md5name", None)
            or getattr(node_or_name, "name", None)
            or str(node_or_name)
        )

    def add_script_pre(self, node, executable, *args):
        """Add a SCRIPT PRE hook for *node* (executable + args)."""
        entry = (self._node_name(node), str(executable),
                 " ".join(str(a) for a in args))
        self.script_pre.append(entry)
        if self._already_written:
            get_backend().append_script_pre(self, *entry)

    def add_script_post(self, node, executable, *args):
        """Add a SCRIPT POST hook for *node* (executable + args)."""
        entry = (self._node_name(node), str(executable),
                 " ".join(str(a) for a in args))
        self.script_post.append(entry)
        if self._already_written:
            get_backend().append_script_post(self, *entry)

    def add_abort_on(self, node, exit_code, return_value=0):
        """Add an ABORT-DAG-ON hook: if *node* returns *exit_code*, abort
        the DAG with overall exit *return_value*."""
        entry = (self._node_name(node), int(exit_code), int(return_value))
        self.abort_on.append(entry)
        if self._already_written:
            get_backend().append_abort_on(self, *entry)

    def set_dot_file(self, path):
        """Request a DAG visualisation file (DOT directive)."""
        self.dot_file = path
        if self._already_written:
            get_backend().append_dot_file(self, path)

    def add_extra_directive(self, line):
        """Append a raw backend-native directive line.  Escape hatch for
        anything the structured API doesn't cover."""
        self.extra_directives.append(line)
        if self._already_written:
            get_backend().append_extra_directive(self, line)

    def write_concrete_dag(self):
        """Hand the DAG to the active backend, which writes the workflow artefact."""
        if self.dag_file is None:
            raise RuntimeError("write_concrete_dag: no dag file set")
        get_backend().emit_dag(self, self.dag_file)
        self._already_written = True

    @property
    def output_path(self):
        """Path the active backend actually wrote to.

        Useful when external code wants to append further directives.  For
        HTCondor / glue this is ``<dag_file>.dag``; for Slurm it's the
        ``_dag.sh`` driver script.
        """
        if self.dag_file is None:
            return None
        return get_backend().output_path_for_dag(self.dag_file)


# ===========================================================================
# Backend abstraction
# ===========================================================================


class WorkflowBackend(abc.ABC):
    """Abstract base class for execution-system backends.

    A backend takes the backend-neutral :class:`_GenericJob` /
    :class:`_GenericDAG` containers and produces whatever artefacts the
    target execution system expects (submit files, sbatch scripts, shell
    drivers, ...).
    """

    #: Short identifier used to register the backend.
    name = None

    @classmethod
    def is_available(cls):
        """Return ``True`` if the backend's runtime dependencies are installed.

        Used during auto-detection.  Backends that have no python-importable
        dependency (e.g. Slurm, which is detected by the presence of
        ``sbatch`` on the host running the workflow) should return ``False``
        from auto-detect-time and be selected explicitly.
        """
        return True

    @abc.abstractmethod
    def emit_job(self, job, path):
        """Write the submit artefact for *job* to *path*."""

    @abc.abstractmethod
    def emit_dag(self, dag, path):
        """Write the workflow driver artefact for *dag* to *path*."""

    # ------------------------------------------------------------------
    # DAG-level control-logic overlays.
    #
    # The default behaviour of every ``append_*`` method is to write a
    # single backend-native line into ``output_path_for_dag(dag.dag_file)``.
    # Subclasses override the formatters via ``format_*``; only Slurm
    # needs to fully override ``append_script_post`` etc. because its
    # control-logic equivalent isn't a single text line.
    # ------------------------------------------------------------------

    def output_path_for_dag(self, dag_file):
        """Return the absolute path the backend wrote (or will write)
        for *dag_file*.  Default: append ``.dag`` if missing."""
        if dag_file.endswith(".dag"):
            return dag_file
        return dag_file + ".dag"

    def format_script_pre(self, name, exe, args_str):
        return "SCRIPT PRE {} {} {}\n".format(name, exe, args_str).rstrip(" \n") + "\n"

    def format_script_post(self, name, exe, args_str):
        return "SCRIPT POST {} {} {}\n".format(name, exe, args_str).rstrip(" \n") + "\n"

    def format_abort_on(self, name, exit_code, return_value):
        return "ABORT-DAG-ON {} {} RETURN {}\n".format(name, exit_code, return_value)

    def format_dot_file(self, path):
        return "DOT {}\n".format(path)

    def _append_to_output(self, dag, text):
        path = self.output_path_for_dag(dag.dag_file)
        with open(path, "a") as fh:
            fh.write(text)

    def append_script_pre(self, dag, name, exe, args_str):
        self._append_to_output(dag, self.format_script_pre(name, exe, args_str))

    def append_script_post(self, dag, name, exe, args_str):
        self._append_to_output(dag, self.format_script_post(name, exe, args_str))

    def append_abort_on(self, dag, name, exit_code, return_value):
        self._append_to_output(dag, self.format_abort_on(name, exit_code, return_value))

    def append_dot_file(self, dag, path):
        self._append_to_output(dag, self.format_dot_file(path))

    def append_extra_directive(self, dag, line):
        if not line.endswith("\n"):
            line = line + "\n"
        self._append_to_output(dag, line)

    def _emit_dag_control_overlays(self, dag, fh):
        """Helper for emit_dag implementations that produce a single
        text artefact: write all the captured control-logic overlays
        appended to the workflow body."""
        for name, exe, args_str in dag.script_pre:
            fh.write(self.format_script_pre(name, exe, args_str))
        for name, exe, args_str in dag.script_post:
            fh.write(self.format_script_post(name, exe, args_str))
        for name, exit_code, return_value in dag.abort_on:
            fh.write(self.format_abort_on(name, exit_code, return_value))
        if dag.dot_file is not None:
            fh.write(self.format_dot_file(dag.dot_file))
        for line in dag.extra_directives:
            if not line.endswith("\n"):
                line = line + "\n"
            fh.write(line)

    # Defaults useful for sub-classes
    @staticmethod
    def _build_argument_string(job, var_ref):
        """Render a job's argument list as a single command-line string.

        *var_ref* is a callable mapping a variable-opt name to whatever string
        should be substituted at submit time (e.g. ``"$(macroevent)"`` for
        Condor, ``"${event}"`` for Slurm).
        """
        parts = []
        for name, value in job.opts:
            if value is None or value == "":
                parts.append("--{}".format(name))
            else:
                parts.append("--{}={}".format(name, value))
        for name, value in job.short_opts:
            parts.append("-{} {}".format(name, value))
        for name, value in job.file_opts:
            parts.append("--{}={}".format(name, value))
        for name in job.var_opts:
            parts.append("--{}={}".format(name, var_ref(name)))
        for arg in job.arguments:
            parts.append(str(arg))
        return " ".join(parts)


# ---------------------------------------------------------------------------
# HTCondor backend (modern python bindings)
# ---------------------------------------------------------------------------

class HTCondorBackend(WorkflowBackend):
    """HTCondor backend using the modern ``htcondor`` python bindings.

    Submit descriptions are constructed via :class:`htcondor.Submit` (which
    validates the result) and rendered to text.  DAG files are emitted as
    plain text in HTCondor DAGMan format.
    """

    name = "htcondor"

    @classmethod
    def is_available(cls):
        try:
            import htcondor  # noqa: F401
            return True
        except Exception:
            return False

    def __init__(self):
        try:
            import htcondor
            self._htcondor = htcondor
        except Exception:
            self._htcondor = None

    @staticmethod
    def _var_ref(name):
        return "$(macro{})".format(name.replace("-", "_"))

    def _build_submit_dict(self, job):
        sub = {}
        if job.universe is not None:
            sub["universe"] = job.universe
        if job.executable is not None:
            sub["executable"] = str(job.executable)
        args = self._build_argument_string(job, self._var_ref)
        if args:
            sub["arguments"] = args
        if job.log_file:
            sub["log"] = job.log_file
        if job.stdout_file:
            sub["output"] = job.stdout_file
        if job.stderr_file:
            sub["error"] = job.stderr_file
        # condor_cmds includes both well-known and custom commands; pass them all through.
        for key, value in job.condor_cmds:
            sub[key] = "" if value is None else str(value)
        return sub

    def emit_job(self, job, path):
        sub_dict = self._build_submit_dict(job)
        text = None
        if self._htcondor is not None:
            try:
                submit = self._htcondor.Submit(sub_dict)
                text = str(submit)
                if not text.endswith("\n"):
                    text += "\n"
                text += "queue {}\n".format(job.queue_count)
            except Exception:
                text = None
        if text is None:
            text = self._render_submit_text(sub_dict, job.queue_count)
        with open(path, "w") as fh:
            fh.write(text)

    def _render_submit_text(self, sub_dict, queue_count):
        ordered = ["universe", "executable", "arguments", "log", "output", "error"]
        lines, seen = [], set()
        for k in ordered:
            if k in sub_dict:
                lines.append("{} = {}".format(k, sub_dict[k]))
                seen.add(k)
        for k, v in sub_dict.items():
            if k in seen:
                continue
            lines.append("{} = {}".format(k, v))
        lines.append("queue {}".format(queue_count))
        return "\n".join(lines) + "\n"

    def emit_dag(self, dag, path):
        if not path.endswith(".dag"):
            path = path + ".dag"
        with open(path, "w") as fh:
            for node in dag.nodes:
                if isinstance(node, _GenericSubdagNode):
                    fh.write("SUBDAG EXTERNAL {} {}\n".format(node.name, node.subdag_file))
                else:
                    sub = node.job.get_sub_file()
                    if sub is None:
                        raise RuntimeError(
                            "node {} references a job with no sub-file".format(node.name)
                        )
                    fh.write("JOB {} {}\n".format(node.name, sub))
                if node.macros:
                    items = " ".join('{}="{}"'.format(k, v) for k, v in node.macros.items())
                    fh.write("VARS {} {}\n".format(node.name, items))
                if node.category:
                    fh.write("CATEGORY {} {}\n".format(node.name, node.category))
                if node.retry:
                    fh.write("RETRY {} {}\n".format(node.name, node.retry))
            for node in dag.nodes:
                for parent in node.parents:
                    fh.write("PARENT {} CHILD {}\n".format(parent.name, node.name))
            # Control-logic overlays (SCRIPT POST, ABORT-DAG-ON, DOT, ...)
            self._emit_dag_control_overlays(dag, fh)


# ---------------------------------------------------------------------------
# glue.pipeline fallback backend
# ---------------------------------------------------------------------------

class GluePipelineBackend(WorkflowBackend):
    """Legacy fallback that drives ``glue.pipeline`` to produce submit / DAG
    files.

    The generic :class:`_GenericJob` / :class:`_GenericDAG` state is replayed
    onto a freshly-constructed ``glue.pipeline.CondorDAGJob`` /
    ``glue.pipeline.CondorDAG`` at emit time.  This means consumers see the
    same backend-neutral surface as under HTCondor, but the actual file I/O
    is handled by ``glue``.
    """

    name = "glue"

    @classmethod
    def is_available(cls):
        try:
            from glue import pipeline  # noqa: F401
            return True
        except Exception:
            return False

    def __init__(self):
        from glue import pipeline as _p
        self._pipeline = _p
        # Translation cache so a node in emit_dag can find the glue object
        # that was created to back its underlying _GenericJob, and so an
        # already-submitted glue node isn't rebuilt.
        self._glue_jobs_by_id = {}
        self._glue_nodes_by_id = {}

    def _make_glue_job(self, job):
        gid = id(job)
        if gid in self._glue_jobs_by_id:
            return self._glue_jobs_by_id[gid]
        gjob = self._pipeline.CondorDAGJob(universe=job.universe, executable=job.executable)
        if job.sub_file is not None:
            gjob.set_sub_file(job.sub_file)
        # glue.pipeline insists that log/stdout/stderr files all be set,
        # raising CondorSubmitError otherwise; htcondor and slurm tolerate
        # their absence.  Smooth over the difference by deriving defaults
        # from the sub-file whenever the caller didn't supply one.
        if job.sub_file is not None:
            base, _ = os.path.splitext(job.sub_file)
        else:
            base = None
        log_file = job.log_file
        if log_file is None and base is not None:
            log_file = base + ".log"
        if log_file is not None:
            gjob.set_log_file(log_file)
        stdout_file = job.stdout_file
        if stdout_file is None and base is not None:
            stdout_file = base + ".out"
        if stdout_file is not None:
            gjob.set_stdout_file(stdout_file)
        stderr_file = job.stderr_file
        if stderr_file is None and base is not None:
            stderr_file = base + ".err"
        if stderr_file is not None:
            gjob.set_stderr_file(stderr_file)
        for n, v in job.opts:
            gjob.add_opt(n, v)
        for n, v in job.short_opts:
            gjob.add_short_opt(n, v)
        for n, v in job.file_opts:
            gjob.add_file_opt(n, v)
        for n in job.var_opts:
            gjob.add_var_opt(n)
        for arg in job.arguments:
            gjob.add_arg(arg)
        for k, v in job.condor_cmds:
            gjob.add_condor_cmd(k, v)
        try:
            # glue.pipeline stores the queue count in this private attribute.
            gjob._CondorJob__queue = job.queue_count
        except Exception:
            pass
        self._glue_jobs_by_id[gid] = gjob
        return gjob

    def emit_job(self, job, path):
        gjob = self._make_glue_job(job)
        # Make sure sub-file path matches what the caller asked for
        gjob.set_sub_file(path)
        gjob.write_sub_file()

    def emit_dag(self, dag, path):
        gdag = self._pipeline.CondorDAG(log=(dag.log or os.getcwd()))
        # Build glue nodes
        glue_nodes = {}
        for node in dag.nodes:
            if isinstance(node, _GenericSubdagNode):
                # glue's CondorDAGManJob/create_node()
                man = self._pipeline.CondorDAGManJob(node.subdag_file)
                gnode = man.create_node()
            else:
                gjob = self._make_glue_job(node.job)
                gnode = self._pipeline.CondorDAGNode(gjob)
            for k, v in node.macros.items():
                gnode.add_macro(k, v)
            if node.category:
                gnode.set_category(node.category)
            if node.retry:
                gnode.set_retry(node.retry)
            glue_nodes[id(node)] = gnode
        # Wire up parents
        for node in dag.nodes:
            gnode = glue_nodes[id(node)]
            for parent in node.parents:
                if id(parent) in glue_nodes:
                    gnode.add_parent(glue_nodes[id(parent)])
        for node in dag.nodes:
            gdag.add_node(glue_nodes[id(node)])
        # glue.pipeline.CondorDAG.write_concrete_dag() always appends a
        # ".dag" suffix.  If the caller already passed a path ending in
        # ".dag" we'd get "wf.dag.dag" otherwise; strip the suffix so the
        # final file lands at the path we were asked for.
        if path.endswith(".dag"):
            glue_path = path[:-4]
        else:
            glue_path = path
        gdag.set_dag_file(glue_path)
        gdag.write_concrete_dag()
        # Append the control-logic overlays directly to the file glue
        # produced (glue.pipeline does not expose its own SCRIPT POST /
        # ABORT-DAG-ON / DOT API).
        final_path = self.output_path_for_dag(path)
        with open(final_path, "a") as fh:
            self._emit_dag_control_overlays(dag, fh)

    def output_path_for_dag(self, dag_file):
        # glue.pipeline always lands the dag at ``<basename>.dag``.
        return dag_file if dag_file.endswith(".dag") else dag_file + ".dag"


# ---------------------------------------------------------------------------
# Slurm backend
# ---------------------------------------------------------------------------

class SlurmBackend(WorkflowBackend):
    """Slurm backend.

    Each :class:`_GenericJob` is emitted as an ``sbatch`` script.  Resource
    requests, queue counts, environments, etc. are translated into
    ``#SBATCH`` directives where a portable mapping exists.  A workflow
    :class:`_GenericDAG` is emitted as a shell driver script that submits
    its jobs in topological order with ``sbatch --dependency=afterok:...``
    chains.

    Things that don't have a portable Slurm equivalent (e.g.
    ``MY.flock_local``, ``+SingularityImage``, condor universe ``local``)
    are emitted as ``# HTCondor: ...`` comments in the sbatch script so the
    information is preserved but inert under Slurm.
    """

    name = "slurm"

    @classmethod
    def is_available(cls):
        # We can run on a login node without sbatch installed (we'll just
        # write the scripts), so don't gate on `sbatch`.  However, we don't
        # want auto-detection to pick Slurm by default since that would
        # silently override Condor on misconfigured machines.
        return False

    def emit_job(self, job, path):
        lines = ["#!/bin/bash", ""]

        # Map structured resources / common semantics to #SBATCH directives.
        directives = []
        if job.resources.get("memory"):
            directives.append("--mem={}".format(self._coerce_mem(job.resources["memory"])))
        if job.resources.get("cpus"):
            directives.append("--cpus-per-task={}".format(job.resources["cpus"]))
        if job.resources.get("gpus"):
            directives.append("--gres=gpu:{}".format(job.resources["gpus"]))
        if job.resources.get("runtime_minutes"):
            directives.append("--time={}".format(self._coerce_time(job.resources["runtime_minutes"])))
        if job.resources.get("disk"):
            directives.append("--tmp={}".format(self._coerce_mem(job.resources["disk"])))
        if job.queue_count and int(job.queue_count) > 1:
            directives.append("--array=0-{}".format(int(job.queue_count) - 1))
        if job.stdout_file:
            directives.append("--output={}".format(job.stdout_file))
        if job.stderr_file:
            directives.append("--error={}".format(job.stderr_file))
        # Job name from the executable, useful in squeue
        if job.executable:
            directives.append("--job-name={}".format(os.path.basename(str(job.executable))))
        # Map condor's "accounting_group" / "accounting_group_user" if present
        for key, value in job.condor_cmds:
            if key == "accounting_group":
                directives.append("--account={}".format(value))
            elif key == "+SlurmPartition" or key == "MY.SlurmPartition":
                directives.append("--partition={}".format(self._unquote(value)))
            elif key == "+SlurmQOS" or key == "MY.SlurmQOS":
                directives.append("--qos={}".format(self._unquote(value)))

        for d in directives:
            lines.append("#SBATCH {}".format(d))

        # Environment
        if job.inherit_environment:
            lines.append("#SBATCH --export=ALL")
        elif job.environment:
            kvs = ",".join("{}={}".format(k, v) for k, v in job.environment.items())
            lines.append("#SBATCH --export={}".format(kvs))

        # Preserve all condor commands as comments so the information isn't lost
        if job.condor_cmds:
            lines.append("")
            lines.append("# Original HTCondor submit-file commands (preserved for reference):")
            for k, v in job.condor_cmds:
                lines.append("#   {} = {}".format(k, v))

        # When queue_count > 1, the per-task variable opt substitution should
        # use SLURM_ARRAY_TASK_ID; when there's only one task we still pass
        # macros via per-job environment variables (the dag driver script
        # will set them via --export when submitting).
        def slurm_var(name):
            ev = "SLURM_VAR_" + re.sub(r"\W+", "_", name).upper()
            return "${" + ev + "}"

        args = self._build_argument_string(job, slurm_var)
        lines.append("")
        if not job.executable:
            raise RuntimeError("SlurmBackend.emit_job: job has no executable")
        if args:
            lines.append("exec {} {}".format(job.executable, args))
        else:
            lines.append("exec {}".format(job.executable))
        text = "\n".join(lines) + "\n"
        with open(path, "w") as fh:
            fh.write(text)
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass

    def output_path_for_dag(self, dag_file):
        """For Slurm, the workflow lands in a shell driver, not a .dag.

        Mapping:
          ``foo.dag``  → ``foo_dag.sh``
          ``foo.sh``   → ``foo.sh``
          otherwise    → ``foo.sh``
        """
        if dag_file.endswith(".sh"):
            return dag_file
        if dag_file.endswith(".dag"):
            return dag_file[:-4] + "_dag.sh"
        return dag_file + ".sh"

    def emit_dag(self, dag, path):
        """Emit a shell driver that submits the DAG via sbatch dependency chains.

        High-level DAGMan control directives are translated as follows:

        * ``RETRY`` per node          → ``--requeue`` on the corresponding sbatch
        * ``VARS k="v"`` per node     → ``--export=ALL,SLURM_VAR_K=v``
        * ``SCRIPT PRE`` per node     → bash command run before sbatch
        * ``SCRIPT POST`` per node    → ``sbatch --dependency=afterany:<jobid> --wrap=...``
                                        scheduled to run regardless of success/failure
        * ``ABORT-DAG-ON`` per node   → bash check that inspects the named
                                        job's exit code, scancels future
                                        dependents, and exits with the
                                        requested return value
        * ``DOT`` directive           → comment (Slurm has no equivalent)
        * Extra raw directives        → comment (escape hatch); preserved verbatim
        """
        path = self.output_path_for_dag(path)

        # Topological sort so we can emit submissions in dependency order
        order = self._topological_sort(dag.nodes)
        var_for_node_id = {}
        var_for_node_name = {}
        for i, node in enumerate(order):
            var = "JOBID_{}".format(i)
            var_for_node_id[id(node)] = var
            var_for_node_name[node.name] = var
        # Index pre/post hooks per node-name for quick lookup.
        pre_by_name = {}
        post_by_name = {}
        for name, exe, args in dag.script_pre:
            pre_by_name.setdefault(name, []).append((exe, args))
        for name, exe, args in dag.script_post:
            post_by_name.setdefault(name, []).append((exe, args))

        lines = [
            "#!/bin/bash",
            "# Slurm driver script for workflow",
            "# Submits each job via sbatch and chains them with --dependency=afterok",
            "set -euo pipefail",
            "",
        ]
        # Track post-script jobs we submitted so subsequent abort-on checks
        # can wait for them too.
        for node in order:
            var = var_for_node_id[id(node)]
            # SCRIPT PRE -- run synchronously before sbatch.
            for exe, args in pre_by_name.get(node.name, []):
                lines.append("# SCRIPT PRE for {}: {} {}".format(node.name, exe, args))
                lines.append("{} {}".format(exe, args).rstrip())

            if isinstance(node, _GenericSubdagNode):
                lines.append("# Sub-DAG: {}".format(node.subdag_file))
                lines.append('echo "Slurm backend: SUBDAG nodes are not supported natively; '
                             'driving sub-script {} via bash" >&2'.format(node.subdag_file))
                deps = self._dep_clause(node, var_for_node_id)
                lines.append("{}=$(sbatch{} --wrap=\"bash {}\" | awk '{{print $4}}')".format(
                    var, deps, node.subdag_file))
            else:
                sub = node.job.get_sub_file()
                if sub is None:
                    raise RuntimeError(
                        "Slurm: node {} has no sbatch script".format(node.name))
                # Per-node variable opts get exported via --export=ALL,VAR=val
                export_args = ""
                if node.macros:
                    kvs = ",".join("SLURM_VAR_{}={}".format(
                        re.sub(r"\W+", "_", k).upper(), v)
                        for k, v in node.macros.items())
                    export_args = " --export=ALL,{}".format(kvs)
                deps = self._dep_clause(node, var_for_node_id)
                retry_clause = ""
                if node.retry:
                    retry_clause = " --requeue"
                lines.append("{}=$(sbatch{}{}{} {} | awk '{{print $4}}')".format(
                    var, deps, export_args, retry_clause, sub))
                lines.append('echo "Submitted {} as ${{{}}}"'.format(node.name, var))

            # SCRIPT POST -- a separate sbatch with afterany dependency so
            # the post-script runs regardless of success / failure.
            for j, (exe, args) in enumerate(post_by_name.get(node.name, [])):
                post_var = "{}_post_{}".format(var, j)
                lines.append("# SCRIPT POST for {}: {} {}".format(node.name, exe, args))
                lines.append(
                    "{}=$(sbatch --dependency=afterany:${{{}}} --wrap=\"{} {}\" "
                    "| awk '{{print $4}}')".format(post_var, var, exe, args.rstrip())
                )
                lines.append('echo "Post-script for {} as ${{{}}}"'.format(node.name, post_var))

        # ABORT-DAG-ON -- block waiting for the named job, then if its
        # exit code matches, scancel everything else and exit with the
        # requested return value.  Scheduled at the end of the driver
        # (after all sbatch calls have been issued) so we have all jobids.
        if dag.abort_on:
            lines.append("")
            lines.append("# ABORT-DAG-ON checks: monitor the named jobs and abort the workflow")
            lines.append("# if any of them returns the matching exit code.")
            for name, exit_code, return_value in dag.abort_on:
                if name not in var_for_node_name:
                    lines.append("# (skipped abort-on for unknown node {})".format(name))
                    continue
                target_var = var_for_node_name[name]
                lines.append("(")
                lines.append("    sleep_until_finished_var=${{{}}}".format(target_var))
                lines.append('    while squeue -j "${sleep_until_finished_var}" -h '
                             '-o "%i" 2>/dev/null | grep -q .; do sleep 30; done')
                lines.append('    code=$(sacct -j "${sleep_until_finished_var}" -X '
                             '--format=ExitCode --noheader | head -1 | awk -F: '
                             "'{print $1}')")
                lines.append('    if [[ "${{code}}" -eq {} ]]; then'.format(exit_code))
                lines.append('        echo "ABORT-DAG-ON: node {} returned exit ${{code}};'
                             ' scancelling remaining jobs and exiting {}" >&2'
                             .format(name, return_value))
                lines.append('        scancel ' + ' '.join(
                    '${{{}}}'.format(v) for v in var_for_node_id.values() if v != target_var
                ))
                lines.append('        exit {}'.format(return_value))
                lines.append('    fi')
                lines.append(") &")
            lines.append("wait")

        # DOT directive -- best-effort: emit a graphviz file describing the
        # dependency graph, since Slurm has no native equivalent.
        if dag.dot_file is not None:
            lines.append("")
            lines.append("# DOT visualisation requested by DAG; written to {}".format(dag.dot_file))
            lines.append("cat <<'__DOT_EOF__' > '{}'".format(dag.dot_file))
            lines.append("digraph workflow {")
            for node in order:
                lines.append('  "{}";'.format(node.name))
            for node in order:
                for parent in node.parents:
                    lines.append('  "{}" -> "{}";'.format(parent.name, node.name))
            lines.append("}")
            lines.append("__DOT_EOF__")

        # Escape-hatch raw directives -- emit each as a comment so the
        # information is preserved (these are usually condor-specific and
        # don't have a Slurm equivalent).
        if dag.extra_directives:
            lines.append("")
            lines.append("# Original DAG-language directives preserved as comments:")
            for line in dag.extra_directives:
                lines.append("#   " + line.rstrip("\n"))

        text = "\n".join(lines) + "\n"
        with open(path, "w") as fh:
            fh.write(text)
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Post-emission append hooks.
    #
    # For backends that emit a single DAG file (HTCondor / glue) the
    # default ``WorkflowBackend._append_to_output`` is fine -- it just
    # appends a ``SCRIPT POST ...\n`` line to the .dag and DAGMan picks
    # it up.  For Slurm the .sh driver is a bash program; we can't just
    # append a foreign DAG-language line to it.  So if the caller is
    # adding control logic *after* ``write_concrete_dag()`` we re-emit
    # the whole driver, picking up the new state captured on ``dag``.
    # ------------------------------------------------------------------
    def _reemit(self, dag):
        # Re-run emit_dag on the same target.  We can do this safely
        # because ``dag`` already holds the updated state.
        self.emit_dag(dag, dag.dag_file)

    def append_script_pre(self, dag, name, exe, args_str):
        self._reemit(dag)

    def append_script_post(self, dag, name, exe, args_str):
        self._reemit(dag)

    def append_abort_on(self, dag, name, exit_code, return_value):
        self._reemit(dag)

    def append_dot_file(self, dag, path):
        self._reemit(dag)

    def append_extra_directive(self, dag, line):
        self._reemit(dag)

    @staticmethod
    def _dep_clause(node, var_for_node):
        if not node.parents:
            return ""
        ids = ":".join("${" + var_for_node[id(p)] + "}" for p in node.parents if id(p) in var_for_node)
        if not ids:
            return ""
        return " --dependency=afterok:" + ids

    @staticmethod
    def _topological_sort(nodes):
        # Kahn's algorithm
        in_edges = {id(n): set(id(p) for p in n.parents) for n in nodes}
        node_by_id = {id(n): n for n in nodes}
        out = []
        ready = [n for n in nodes if not in_edges[id(n)]]
        while ready:
            n = ready.pop(0)
            out.append(n)
            for m in nodes:
                if id(n) in in_edges[id(m)]:
                    in_edges[id(m)].remove(id(n))
                    if not in_edges[id(m)]:
                        ready.append(m)
        if len(out) != len(nodes):
            # Cycle, just emit in input order
            return list(nodes)
        return out

    @staticmethod
    def _coerce_mem(value):
        # Accept "2048M", "2G", "1024", 1024
        s = str(value).strip()
        # Condor sometimes uses just numbers (interpreted as MB); Slurm uses M/G suffix.
        if re.fullmatch(r"\d+", s):
            return s + "M"
        return s

    @staticmethod
    def _coerce_time(minutes):
        try:
            m = int(minutes)
        except (TypeError, ValueError):
            return str(minutes)
        h, mm = divmod(m, 60)
        return "{}:{:02d}:00".format(h, mm)

    @staticmethod
    def _unquote(s):
        s = str(s)
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
            s = s[1:-1]
        return s


# ===========================================================================
# Backend registry & selection
# ===========================================================================

# Registry of backend instances, keyed by name.
_BACKENDS = {}
_ACTIVE_BACKEND_NAME = None


def register_backend(backend):
    """Register a :class:`WorkflowBackend` instance.

    Subsequent calls to :func:`set_backend` (and the auto-detection path) can
    reference *backend* by its ``name`` attribute.
    """
    if backend.name is None:
        raise ValueError("backend instance must have a non-None .name attribute")
    _BACKENDS[backend.name] = backend


def set_backend(name):
    """Select the active backend by name.  Raises ``KeyError`` if unknown."""
    if name not in _BACKENDS:
        raise KeyError(
            "Unknown workflow backend: {!r}.  Registered backends: {}".format(
                name, sorted(_BACKENDS.keys())
            )
        )
    global _ACTIVE_BACKEND_NAME
    _ACTIVE_BACKEND_NAME = name


def get_backend():
    """Return the active :class:`WorkflowBackend` instance.

    Raises :class:`RuntimeError` if no backend has been selected and none is
    available for auto-selection.
    """
    if _ACTIVE_BACKEND_NAME is None:
        raise RuntimeError(
            "No workflow backend is selected.  Install one of "
            "`htcondor` or `lscsoft-glue`, or call set_backend('slurm') / "
            "set_backend('your_backend_name')."
        )
    return _BACKENDS[_ACTIVE_BACKEND_NAME]


def current_backend_name():
    """Return the name of the active backend, or ``None`` if not selected."""
    return _ACTIVE_BACKEND_NAME


def _auto_select_backend():
    # Honor explicit override via environment variable
    env_choice = os.environ.get("RIFT_DAG_BACKEND", "auto").lower()
    if env_choice not in ("auto", "", "none"):
        if env_choice in _BACKENDS:
            set_backend(env_choice)
            return
        else:
            print(
                "[dag_utils_generic] RIFT_DAG_BACKEND={!r} not registered; "
                "falling back to auto-detection".format(env_choice),
                file=sys.stderr,
            )
    # Auto: prefer HTCondor python bindings, then glue.pipeline.
    for name in ("htcondor", "glue"):
        if name in _BACKENDS:
            try:
                set_backend(name)
                return
            except Exception:
                pass


# Register the bundled backends.
#
# We *always* register HTCondorBackend and SlurmBackend because both have a
# pure-python emission path with no library dependency (HTCondor falls back
# to a built-in submit-file text renderer when ``htcondor.Submit`` isn't
# available; Slurm only emits shell text).  GluePipelineBackend is the
# exception: its emit_* paths construct real glue.pipeline objects, so we
# can only register it when glue is actually importable.
try:
    register_backend(HTCondorBackend())
except Exception as _exc:
    print(
        "[dag_utils_generic] WARNING: could not register HTCondorBackend: {}"
        .format(_exc),
        file=sys.stderr,
    )
if GluePipelineBackend.is_available():
    try:
        register_backend(GluePipelineBackend())
    except Exception:
        pass
register_backend(SlurmBackend())

_auto_select_backend()


# ===========================================================================
# Public factories (preserve historical names) + pipeline namespace shim
# ===========================================================================

def CondorDAGJob(universe="vanilla", executable=None):
    """Return a backend-neutral job container.

    Despite the legacy name, the returned object is *not* Condor-specific; it
    just exposes the historical ``glue.pipeline.CondorDAGJob`` API.  The
    actual submit artefact emitted by :meth:`_GenericJob.write_sub_file` is
    determined by the active :class:`WorkflowBackend`.
    """
    return _GenericJob(universe=universe, executable=executable)


def CondorDAG(log=None):
    return _GenericDAG(log=log)


def CondorDAGNode(job):
    return _GenericNode(job)


def CondorDAGManJob(dag_file):
    return _GenericManJob(dag_file)


# Expose a ``pipeline``-flavored namespace so consumers that previously did
# ``from glue import pipeline`` can switch to
# ``from RIFT.misc.dag_utils_generic import pipeline`` with no other changes.
class _PipelineNamespace(object):
    CondorDAGJob = staticmethod(CondorDAGJob)
    CondorDAG = staticmethod(CondorDAG)
    CondorDAGNode = staticmethod(CondorDAGNode)
    CondorDAGManJob = staticmethod(CondorDAGManJob)


pipeline = _PipelineNamespace()


# Legacy module-level constant.  Kept for any external code that might check
# it (e.g. tests).  Set lazily so it reflects the active backend rather than
# a static "auto-detected at import time" value.
def _legacy_BACKEND():
    return _ACTIVE_BACKEND_NAME


_BACKEND = _ACTIVE_BACKEND_NAME  # snapshot for backwards-compat



# ---------------------------------------------------------------------------
# Ported write_*_sub helpers
# ---------------------------------------------------------------------------
#
# Everything below is the original logic from RIFT.misc.dag_utils, copied
# verbatim except that ``pipeline.CondorDAGJob`` / ``pipeline.CondorDAG`` /
# ``pipeline.CondorDAGNode`` / ``pipeline.CondorDAGManJob`` references have
# been rewritten to use the backend-neutral classes defined above.  This means
# every helper here works under both the htcondor python bindings backend and
# the glue.pipeline fallback backend.
# ---------------------------------------------------------------------------

def write_integrate_likelihood_extrinsic_grid_sub(tag='integrate', exe=None, log_dir=None, ncopies=1, **kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over
    extrinsic parameters.
    Like the other case (below), but modified to use the sim_xml
    and loop over 'event'

    Inputs:
        - 'tag' is a string to specify the base name of output files. The output
          submit file will be named tag.sub, and the jobs will write their
          output to tag-ID.out, tag-ID.err, tag.log, where 'ID' is a unique
          identifier for each instance of a job run from the sub file.
        - 'cache' is the path to a cache file which gives the location of the
          data to be analyzed.
        - 'sim' is the path to the XML file with the grid
        - 'channelH1/L1/V1' is the channel name to be read for each of the
          H1, L1 and V1 detectors.
        - 'psdH1/L1/V1' is the path to an XML file specifying the PSD of
          each of the H1, L1, V1 detectors.
        - 'ncopies' is the number of runs with identical input parameters to
          submit per condor 'cluster'

    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """

    assert len(kwargs["psd_file"]) == len(kwargs["channel_name"])

    exe = exe or which("integrate_likelihood_extrinsic")
    ile_job = CondorDAGJob(universe="vanilla", executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    if "output_file" in kwargs and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if "save_samples" in kwargs and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]

    #
    # Add normal arguments
    # FIXME: Get valid options from a module
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    #
    # Macro based options
    #
    ile_job.add_var_opt("event")

    if default_resolved_env:
        ile_job.add_condor_cmd('environment', default_resolved_env)
    else:
        ile_job.add_condor_cmd('getenv', default_getenv_value)
    ile_job.add_condor_cmd('request_memory', '2048M')

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")
        
    

    ###
    ### SUGGESTION FROM STUART (for later)
    # request_memory = ifthenelse( (LastHoldReasonCode=!=34 && LastHoldReasonCode=!=26), InitialRequestMemory, int(1.5 * NumJobStarts * MemoryUsage) )
    # periodic_release = ((HoldReasonCode =?= 34) || (HoldReasonCode =?= 26))
    # This will automatically release a job that is put on hold for using too much memory with a 50% increased memory request each tim.e


    return ile_job, ile_sub_name


# FIXME: Keep in sync with arguments of integrate_likelihood_extrinsic
def write_integrate_likelihood_extrinsic_sub(tag='integrate', exe=None, log_dir=None, ncopies=1, **kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over
    extrinsic parameters.

    Inputs:
        - 'tag' is a string to specify the base name of output files. The output
          submit file will be named tag.sub, and the jobs will write their
          output to tag-ID.out, tag-ID.err, tag.log, where 'ID' is a unique
          identifier for each instance of a job run from the sub file.
        - 'cache' is the path to a cache file which gives the location of the
          data to be analyzed.
        - 'coinc' is the path to a coincident XML file, from which masses and
          times will be drawn FIXME: remove this once it's no longer needed.
        - 'channelH1/L1/V1' is the channel name to be read for each of the
          H1, L1 and V1 detectors.
        - 'psdH1/L1/V1' is the path to an XML file specifying the PSD of
          each of the H1, L1, V1 detectors.
        - 'ncopies' is the number of runs with identical input parameters to
          submit per condor 'cluster'

    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """

    assert len(kwargs["psd_file"]) == len(kwargs["channel_name"])

    exe = exe or which("integrate_likelihood_extrinsic")
    ile_job = CondorDAGJob(universe="vanilla", executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    if "output_file" in kwargs and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if "save_samples" in kwargs and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]

    #
    # Add normal arguments
    # FIXME: Get valid options from a module
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    #
    # Macro based options
    #
    ile_job.add_var_opt("mass1")
    ile_job.add_var_opt("mass2")

    if default_resolved_env:
        ile_job.add_condor_cmd('environment', default_resolved_env)
    else:
        ile_job.add_condor_cmd('getenv', default_getenv_value)
    ile_job.add_condor_cmd('request_memory', '2048M')
    
    return ile_job, ile_sub_name

def write_result_coalescence_sub(tag='coalesce', exe=None, log_dir=None, output_dir="./", use_default_cache=True):
    """
    Write a submit file for launching jobs to coalesce ILE output
    """

    exe = exe or which("ligolw_sqlite")
    sql_job = CondorDAGJob(universe="vanilla", executable=exe)

    sql_sub_name = tag + '.sub'
    sql_job.set_sub_file(sql_sub_name)

    #
    # Logging options
    #
    uniq_str = "$(cluster)-$(process)"
    sql_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    sql_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    sql_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    if use_default_cache:
        sql_job.add_opt("input-cache", "ILE_$(macromassid).cache")
    else:
        sql_job.add_arg("$(macrofiles)")
    #sql_job.add_arg("*$(macromassid)*.xml.gz")
    sql_job.add_opt("database", "ILE_$(macromassid).sqlite")
    #if os.environ.has_key("TMPDIR"):
        #tmpdir = os.environ["TMPDIR"]
    #else:
        #print >>sys.stderr, "WARNING, TMPDIR environment variable not set. Will default to /tmp/, but this could be dangerous."
        #tmpdir = "/tmp/"
    tmpdir = "/dev/shm/"
    sql_job.add_opt("tmp-space", tmpdir)
    sql_job.add_opt("verbose", None)

    if default_resolved_env:
        sql_job.add_condor_cmd('environment', default_resolved_env)
    else:
        sql_job.add_condor_cmd('getenv', default_getenv_value)
    sql_job.add_condor_cmd('request_memory', '1024')
    
    return sql_job, sql_sub_name

def write_posterior_plot_sub(tag='plot_post', exe=None, log_dir=None, output_dir="./"):
    """
    Write a submit file for launching jobs to coalesce ILE output
    """

    exe = exe or which("plot_like_contours")
    plot_job = CondorDAGJob(universe="vanilla", executable=exe)

    plot_sub_name = tag + '.sub'
    plot_job.set_sub_file(plot_sub_name)

    #
    # Logging options
    #
    uniq_str = "$(cluster)-$(process)"
    plot_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    plot_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    plot_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    plot_job.add_opt("show-points", None)
    plot_job.add_opt("dimension1", "mchirp")
    plot_job.add_opt("dimension2", "eta")
    plot_job.add_opt("input-cache", "ILE_all.cache")
    plot_job.add_opt("log-evidence", None)

    plot_job.add_condor_cmd('getenv', default_getenv_value)
    plot_job.add_condor_cmd('request_memory', '1024')
    
    return plot_job, plot_sub_name

def write_tri_plot_sub(tag='plot_tri', injection_file=None, exe=None, log_dir=None, output_dir="./"):
    """
    Write a submit file for launching jobs to coalesce ILE output
    """

    exe = exe or which("make_triplot")
    plot_job = CondorDAGJob(universe="vanilla", executable=exe)

    plot_sub_name = tag + '.sub'
    plot_job.set_sub_file(plot_sub_name)

    #
    # Logging options
    #
    uniq_str = "$(cluster)-$(process)"
    plot_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    plot_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    plot_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    plot_job.add_opt("output", "ILE_triplot_$(macromassid).png")
    if injection_file is not None:
        plot_job.add_opt("injection", injection_file)
    plot_job.add_arg("ILE_$(macromassid).sqlite")

    plot_job.add_condor_cmd('getenv', default_getenv_value)
    #plot_job.add_condor_cmd('request_memory', '2048M')
    
    return plot_job, plot_sub_name

def write_1dpos_plot_sub(tag='1d_post_plot', exe=None, log_dir=None, output_dir="./"):
    """
    Write a submit file for launching jobs to coalesce ILE output
    """

    exe = exe or which("postprocess_1d_cumulative")
    plot_job = CondorDAGJob(universe="vanilla", executable=exe)

    plot_sub_name = tag + '.sub'
    plot_job.set_sub_file(plot_sub_name)

    #
    # Logging options
    #
    uniq_str = "$(cluster)-$(process)"
    plot_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    plot_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    plot_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    plot_job.add_opt("save-sampler-file", "ILE_$(macromassid).sqlite")
    plot_job.add_opt("disable-triplot", None)
    plot_job.add_opt("disable-1d-density", None)

    plot_job.add_condor_cmd('getenv', default_getenv_value)
    plot_job.add_condor_cmd('request_memory', '2048M')
    
    return plot_job, plot_sub_name


def write_CIP_single_iteration_subdag(cip_worker_job,it,unique_postfix,subdag_dir,n_retries=3,n_explode=1):
    # Assume subdag_dir exists, we will append onto filename
    dag = CondorDAG(log=os.getcwd())
    for indx in range(n_explode):
        worker_node =CondorDAGNode(cip_worker_job)
        worker_node.add_macro("macroiteration", it)
        worker_node.add_macro("macroiterationnext", it+1)
        worker_node.set_category("CIP_worker")
        worker_node.set_retry(n_retries)
        dag.add_node(worker_node)
    dag_name=subdag_dir+"/subdag_CIP_{}".format(unique_postfix)
    dag.set_dag_file(dag_name)
    dag.write_concrete_dag()
    return dag_name + ".dag"



def write_CIP_sub(tag='integrate', exe=None, input_net='all.net',output='output-ILE-samples',universe="vanilla",out_dir=None,log_dir=None, use_eos=False,ncopies=1,arg_str=None,request_memory=8192,request_memory_flex=False, arg_vals=None, no_grid=False,request_disk=False, transfer_files=None,transfer_output_files=None,use_singularity=False,use_osg=False,use_oauth_files=False,use_simple_osg_requirements=False,singularity_image=None,max_runtime_minutes=None,condor_commands=None,**kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over intrinsic parameters.

    Inputs:
    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """

    if use_singularity and (singularity_image == None)  :
        print(" FAIL : Need to specify singularity_image to use singularity ")
        sys.exit(0)
    if use_singularity and (transfer_files == None)  :
        print(" FAIL : Need to specify transfer_files to use singularity at present!  (we will append the prescript; you should transfer any PSDs as well as the grid file ")
        sys.exit(0)

    singularity_image_used = "{}".format(singularity_image) # make copy
    extra_files = []
    if singularity_image:
        if 'osdf:' in singularity_image:
            singularity_image_used  = "./{}".format(singularity_image.split('/')[-1])
            extra_files += [singularity_image]

    exe = exe or which("util_ConstructIntrinsicPosterior_GenericCoordinates.py")
    if use_singularity:
        exe_base = os.path.basename(exe)
#        path_split = exe.split("/")
#        print((" Executable: name breakdown ", path_split, " from ", exe))
        singularity_base_exe_path = "/usr/bin/"  # should not hardcode this ...!
        if 'SINGULARITY_BASE_EXE_DIR_HYPERPIPE' in list(os.environ.keys()) : # allow a DIFFERENT exe to be used here for hyperpipe : CIP used for remote
            singularity_base_exe_path = os.environ['SINGULARITY_BASE_EXE_DIR_HYPERPIPE']
        elif 'SINGULARITY_BASE_EXE_DIR' in list(os.environ.keys()) :
            singularity_base_exe_path = os.environ['SINGULARITY_BASE_EXE_DIR']
        exe=singularity_base_exe_path + exe_base
        if exe_base == 'true':  # special universal path for /bin/true, don't override it!
            exe = "/usr/bin/true"
    ile_job = CondorDAGJob(universe=universe, executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies


    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')

    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    #
    # Add options en mass, by brute force
    #
    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
    arg_str = arg_str.lstrip('-')
    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

    ile_job.add_opt("fname", input_net)
    ile_job.add_opt("fname-output-samples", out_dir+"/"+output)
    ile_job.add_opt("fname-output-integral", out_dir+"/"+output)

    #
    # Macro based options.
    #     - select EOS from list (done via macro)
    #     - pass spectral parameters
    #
#    ile_job.add_var_opt("event")
    if use_eos:
        ile_job.add_var_opt("using-eos")


    #
    # Logging options
    #
    uniq_str = "$(macroevent)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    if "fname_output_samples" in kwargs and kwargs["fname_output_samples"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["fname_output_samples"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
    if "fname_output_integral" in kwargs and kwargs["fname_output_integral"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["fname_output_integral"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))

    #
    # Add normal arguments
    # FIXME: Get valid options from a module
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    if not use_osg:
        if default_resolved_env:
            ile_job.add_condor_cmd('environment', default_resolved_env)
        else:
            ile_job.add_condor_cmd('getenv', default_getenv_value)
    if not(request_memory_flex):
        ile_job.add_condor_cmd('request_memory', str(request_memory)+"M") 
    if request_memory_flex:
        ile_job.add_condor_cmd("MY.InitialRequestMemory",str(request_memory))
        ile_job.add_condor_cmd('periodic_release', "HoldReasonCode =?= 34")
        ile_job.add_condor_cmd('request_memory',  'ifthenelse( LastHoldReasonCode=!=34, InitialRequestMemory, int(1.5 * MemoryUsage) )')
    if not(request_disk is False):
        ile_job.add_condor_cmd('request_disk', str(request_disk)) 
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    requirements = []
    if use_singularity:
        # Compare to https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
        ile_job.add_condor_cmd('request_CPUs', str(1))
        ile_job.add_condor_cmd('transfer_executable', 'False')
        ile_job.add_condor_cmd("MY.SingularityBindCVMFS", 'True')
        ile_job.add_condor_cmd("MY.SingularityImage", '"' + singularity_image_used + '"')
        requirements.append("HAS_SINGULARITY=?=TRUE")

    if use_oauth_files:
        # we are using some authentication to retrieve files from the file transfer list, for example, from distributed hosts, not just submit. eg urls provided
            ile_job.add_condor_cmd('use_oauth_services',use_oauth_files)
    if use_osg:
           # avoid black-holing jobs to specific machines that consistently fail. Uses history attribute for ad
           #    https://htcondor.readthedocs.io/en/latest/classad-attributes/job-classad-attributes.html
           ile_job.add_condor_cmd('periodic_release','((HoldReasonCode == 45) && (HoldReasonSubCode == 0)) || (HoldReasonCode == 13)')
           ile_job.add_condor_cmd('job_machine_attrs','Machine')
           ile_job.add_condor_cmd('job_machine_attrs_history_length','4')
#           for indx in [1,2,3,4]:
#               requirements.append("TARGET.GLIDEIN_ResourceName=!=MY.MachineAttrGLIDEIN_ResourceName{}".format(indx))
           if "OSG_DESIRED_SITES" in os.environ:
               ile_job.add_condor_cmd('+DESIRED_SITES',os.environ["OSG_DESIRED_SITES"])
           if "OSG_UNDESIRED_SITES" in os.environ:
               ile_job.add_condor_cmd('+UNDESIRED_SITES',os.environ["OSG_UNDESIRED_SITES"])
           # Some options to automate restarts, acts on top of RETRY in dag
    if use_singularity or use_osg:
            # Set up file transfer options
           ile_job.add_condor_cmd("when_to_transfer_output",'ON_EXIT')

           # Stream log info
           if not ('RIFT_NOSTREAM_LOG' in os.environ):
               ile_job.add_condor_cmd("stream_error",'True')
               ile_job.add_condor_cmd("stream_output",'True')

    if use_osg and ( 'RIFT_BOOLEAN_LIST' in os.environ):
        extra_requirements = [ "{} =?= TRUE".format(x) for x in os.environ['RIFT_BOOLEAN_LIST'].split(',')]
        requirements += extra_requirements

    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    # Stream log info: always stream CIP error, it is a critical bottleneck
    if True: # not ('RIFT_NOSTREAM_LOG' in os.environ):
        ile_job.add_condor_cmd("stream_error",'True')
        ile_job.add_condor_cmd("stream_output",'True')

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")
        
    if not transfer_files is None:
        if not isinstance(transfer_files, list):
            fname_str=transfer_files + ' '.join(extra_files)
        else:
            fname_str = ','.join(transfer_files + extra_files)
        fname_str=fname_str.strip()
        ile_job.add_condor_cmd('transfer_input_files', fname_str)
        ile_job.add_condor_cmd('should_transfer_files','YES')

    # Periodic remove: kill jobs running longer than max runtime
    # https://stackoverflow.com/questions/5900400/maximum-run-time-in-condor
    if not(max_runtime_minutes is None):
        remove_str = 'JobStatus =?= 2 && (CurrentTime - JobStartDate) > ( {})'.format(60*max_runtime_minutes)
        ile_job.add_condor_cmd('periodic_remove', remove_str)


    ###
    ### SUGGESTION FROM STUART (for later)
    # request_memory = ifthenelse( (LastHoldReasonCode=!=34 && LastHoldReasonCode=!=26), InitialRequestMemory, int(1.5 * NumJobStarts * MemoryUsage) )
    # periodic_release = ((HoldReasonCode =?= 34) || (HoldReasonCode =?= 26))
    # This will automatically release a job that is put on hold for using too much memory with a 50% increased memory request each tim.e
    if condor_commands is not None:
        for cmd, value in condor_commands.items():
            ile_job.add_condor_cmd(cmd, value)


    return ile_job, ile_sub_name


def write_puff_sub(tag='puffball', exe=None, base=None,input_net='output-ILE-samples',output='puffball',universe="vanilla",out_dir=None,log_dir=None, use_eos=False,ncopies=1,arg_str=None,request_memory=1024,arg_vals=None, no_grid=False,extra_text='',**kwargs):
    """
    Perform puffball calculation 
    Inputs:
    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """

    exe = exe or which("util_ParameterPuffball.py")
    # Create executable if needed  (using extra_text as flag for now)
    base_str = ''
    if len(extra_text) > 0:
        if not (base is None):
            base_str = ' ' + base +"/"

        cmdname = "puff_sub.sh"
        
        with open(cmdname,'w') as f:        
            f.write("#! /usr/bin/env bash\n")
            f.write(extra_text+"\n")
            extra_args = ''
            f.write( exe +  " $@  \n")

        st = os.stat(cmdname)
        import stat
        os.chmod(cmdname, st.st_mode | stat.S_IEXEC)

        exe = base_str + "puff_sub.sh"


    
    ile_job = CondorDAGJob(universe=universe, executable=exe)
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    #
    # Add options en mass, by brute force
    #
    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
    arg_str = arg_str.lstrip('-')
    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

    if not(input_net is None):
        ile_job.add_opt("inj-file", input_net)   # using this double-duty for FETCH, other use cases
    if not(output is None):
        ile_job.add_opt("inj-file-out", output)


    #
    # Logging options
    #
    uniq_str = "$(macroevent)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    #
    # Add normal arguments
    # FIXME: Get valid options from a module
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    if default_resolved_env:
        ile_job.add_condor_cmd('environment', default_resolved_env)
    else:
        ile_job.add_condor_cmd('getenv', default_getenv_value)
    ile_job.add_condor_cmd('request_memory', str(request_memory)+"M") 
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")
        
    
    return ile_job, ile_sub_name


def write_ILE_sub_simple(tag='integrate', exe=None, log_dir=None, use_eos=False,simple_unique=False,ncopies=1,arg_str=None,request_memory=4096,request_gpu=False,request_cross_platform=False,request_disk=False,arg_vals=None, transfer_files=None,transfer_output_files=None,use_singularity=False,use_osg=False,use_simple_osg_requirements=False,singularity_image=None,use_cvmfs_frames=False,use_oauth_files=False,frames_dir=None,cache_file=None,fragile_hold=False,max_runtime_minutes=None,condor_commands=None,**kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over intrinsic parameters.

    Inputs:
    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """
    if use_singularity and (singularity_image == None)  :
        print(" FAIL : Need to specify singularity_image to use singularity ")
        sys.exit(0)
    if use_singularity and (frames_dir == None)  and (cache_file == None) :
        print(" FAIL : Need to specify frames_dir or cache_file to use singularity (at present) ")
        sys.exit(0)
    if use_singularity and (transfer_files == None)  :
        print(" FAIL : Need to specify transfer_files to use singularity at present!  (we will append the prescript; you should transfer any PSDs as well as the grid file ")
        sys.exit(0)

    singularity_image_used = "{}".format(singularity_image) # make copy
    extra_files = []
    if singularity_image:
        if 'osdf:' in singularity_image:
            singularity_image_used  = "./{}".format(singularity_image.split('/')[-1])
            extra_files += [singularity_image]

        
    exe = exe or which("integrate_likelihood_extrinsic")
    frames_local = None
    if use_singularity:
        exe_base = os.path.basename(exe)
#        print((" Executable: name breakdown ", path_split, " from ", exe))
        singularity_base_exe_path = "/opt/lscsoft/rift/MonteCarloMarginalizeCode/Code/"  # should not hardcode this ...!
        if 'SINGULARITY_BASE_EXE_DIR' in list(os.environ.keys()) :
            singularity_base_exe_path = os.environ['SINGULARITY_BASE_EXE_DIR']
        else:
#            singularity_base_exe_path = "/opt/lscsoft/rift/MonteCarloMarginalizeCode/Code/"  # should not hardcode this ...!
            singularity_base_exe_path = "/usr/bin/"  # should not hardcode this ...!
        exe=singularity_base_exe_path + exe_base
        if not(frames_dir is None):
            frames_local = frames_dir.split("/")[-1]
    elif use_osg:  # NOT using singularity!
        if not(frames_dir is None):
            frames_local = frames_dir.split("/")[-1]
        exe = os.path.basename(exe)
        exe_here = 'my_wrapper.sh'
        if transfer_files is None:
            transfer_files = []
        transfer_files += ['../my_wrapper.sh']
        with open(exe_here,'w') as f:
            f.write("#! /bin/bash  \n")
            f.write(r"""
#!/bin/bash
# Modules and scripts run directly from repository
# Note the repo and branch are self-referential ! Not a robust solution long-term
# Exit on failure:
# set -e
export INSTALL_DIR=research-projects-RIT
export ILE_DIR=${INSTALL_DIR}/MonteCarloMarginalizeCode/Code
export PATH=${PATH}:${ILE_DIR}
export PYTHONPATH=${PYTHONPATH}:${ILE_DIR}
export GW_SURROGATE=gwsurrogate
git clone https://git.ligo.org/richard-oshaughnessy/research-projects-RIT.git
pushd ${INSTALL_DIR} 
git checkout temp-RIT-Tides-port_master-GPUIntegration 
popd

ls 
cat local.cache
echo Starting ...
./research-projects-RIT/MonteCarloMarginalizeCode/Code/""" + exe + " $@ \n")
            os.system("chmod a+x "+exe_here)
            exe = exe_here  # update executable


    ile_job = CondorDAGJob(universe="vanilla", executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    #
    # Add options en mass, by brute force
    #
    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
    arg_str = arg_str.lstrip('-')
    if '"' in arg_str:
        arg_str = safely_quote_arg_str(arg_str)
        #arg_str = arg_str.replace('"','""') # double quote for condor - weird but true
    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

    #
    # Macro based options.
    #     - select EOS from list (done via macro)
    #     - pass spectral parameters
    #
#    ile_job.add_var_opt("event")
    if use_eos:
        ile_job.add_var_opt("using-eos")


    requirements =[]
    #
    # Logging options
    #
    uniq_str = "$(macroevent)-$(cluster)-$(process)"
    if simple_unique:
        uniq_str = "$(macroevent)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    # Add lame initial argument

    if "output_file" in kwargs and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if "save_samples" in kwargs and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]


    #
    # Add normal arguments
    # FIXME: Get valid options from a module
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    if cache_file:
        ile_job.add_opt("cache-file",cache_file)

    ile_job.add_var_opt("event")

    if not use_osg:
        ile_job.add_condor_cmd('getenv', default_getenv_value)
    else:
        env_statement="*RIFT*"
        if 'RIFT_GETENV_OSG' in os.environ:
            env_statement = os.environ['RIFT_GETENV_OSG']  # for example use NUMBA_CACHE_DIR=/tmp; see https://git.ligo.org/computing/helpdesk/-/issues/4616
        # special-purpose  environment variable to help tweak remote execution/driver issues
        if 'CUDA_LAUNCH_BLOCKING' in os.environ:
            env_statement+= ",CUDA_LAUNCH_BLOCKING"
        if default_resolved_env:
            new_resolved_env = build_resolved_env(env_statement)
            ile_job.add_condor_cmd('environment', new_resolved_env)
        else:
            ile_job.add_condor_cmd('getenv', env_statement)  # retrieve any RIFT commands -- specifically RIFT_LOWLATENCY
    ile_job.add_condor_cmd('request_memory', str(request_memory)+"M") 
    if not(request_disk is False):
        ile_job.add_condor_cmd('request_disk', str(request_disk)) 
    nGPUs =0
    requirements = []
    if request_gpu:
        nGPUs=1
        if request_cross_platform:
            # recipe from https://opensciencegrid.atlassian.net/browse/HTCONDOR-2200
            nGPUs = 'countMatches(RequireGPUs, AvailableGPUs) >= 1 ? 1 : 0'
            ile_job.add_condor_cmd('rank', 'RequestGPUs')
        ile_job.add_condor_cmd('request_GPUs', str(nGPUs)) 
# Claim we don't need to make this request anymore to avoid out-of-memory errors. Also, no longer in 'requirements'
#        requirements.append("CUDAGlobalMemoryMb >= 2048")  
    if use_singularity:
        # Compare to https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
        ile_job.add_condor_cmd('request_CPUs', str(1))
        ile_job.add_condor_cmd('transfer_executable', 'False')
        ile_job.add_condor_cmd("MY.SingularityBindCVMFS", 'True')
        ile_job.add_condor_cmd("MY.SingularityImage", '"' + singularity_image_used + '"')
        ile_job.add_condor_cmd("MY.flock_local",'true')  # jobs can match to local pool !
        requirements.append("HAS_SINGULARITY=?=TRUE")
#               if not(use_simple_osg_requirements):
#                requirements.append("HAS_CVMFS_LIGO_CONTAINERS=?=TRUE")
            #ile_job.add_condor_cmd("requirements", ' (IS_GLIDEIN=?=True) && (HAS_LIGO_FRAMES=?=True) && (HAS_SINGULARITY=?=TRUE) && (HAS_CVMFS_LIGO_CONTAINERS=?=TRUE)')

    if use_oauth_files:
        # we are using some authentication to retrieve files from the file transfer list, for example, from distributed hosts, not just submit. eg urls provided
            ile_job.add_condor_cmd('use_oauth_services',use_oauth_files)
    if use_cvmfs_frames:
        requirements.append("HAS_LIGO_FRAMES=?=TRUE")
        if 'LIGO_OATH_SCOPE' in os.environ:
            ile_job.add_condor_cmd('use_oauth_services','igwn')
            ile_job.add_condor_cmd('igwn_oauth_permissions',os.environ['LIGO_OATH_SCOPE'])
        else:
            ile_job.add_condor_cmd('use_x509userproxy','True')
            if 'X509_USER_PROXY' in list(os.environ.keys()):
                print(" Storing copy of X509 user proxy -- beware expiration! ")
                cwd = os.getcwd()
                fname_proxy = cwd +"/my_proxy"  # this can get overwritten, that's fine - just renews, feature not bug
                os.system("cp ${X509_USER_PROXY} "  + fname_proxy)
            #            ile_job.add_condor_cmd('x509userproxy',os.environ['X509_USER_PROXY'])
                ile_job.add_condor_cmd('x509userproxy',fname_proxy)

    if use_osg:
#           if not(use_simple_osg_requirements):
#               requirements.append("IS_GLIDEIN=?=TRUE")
           # avoid black-holing jobs to specific machines that consistently fail. Uses history attribute for ad
           ile_job.add_condor_cmd('periodic_release','((HoldReasonCode == 45) && (HoldReasonSubCode == 0)) || (HoldReasonCode == 13)')
           ile_job.add_condor_cmd('job_machine_attrs','Machine')
           ile_job.add_condor_cmd('job_machine_attrs_history_length','4')
#           for indx in [1,2,3,4]:
#               requirements.append("TARGET.GLIDEIN_ResourceName=!=MY.MachineAttrGLIDEIN_ResourceName{}".format(indx))
           if "OSG_DESIRED_SITES" in os.environ:
               ile_job.add_condor_cmd('+DESIRED_SITES',os.environ["OSG_DESIRED_SITES"])
           if "OSG_UNDESIRED_SITES" in os.environ:
               ile_job.add_condor_cmd('+UNDESIRED_SITES',os.environ["OSG_UNDESIRED_SITES"])
           # Some options to automate restarts, acts on top of RETRY in dag
           if fragile_hold:
               ile_job.add_condor_cmd("periodic_release","(NumJobStarts < 5) && ((CurrentTime - EnteredCurrentStatus) > 600)")
               ile_job.add_condor_cmd("on_exit_hold","(ExitBySignal == True) || (ExitCode != 0)")
    if use_singularity or use_osg:
            # Set up file transfer options
           ile_job.add_condor_cmd("when_to_transfer_output",'ON_EXIT')

           # Stream log info
           if not ('RIFT_NOSTREAM_LOG' in os.environ):
               ile_job.add_condor_cmd("stream_error",'True')
               ile_job.add_condor_cmd("stream_output",'True')

    if use_osg and ( 'RIFT_BOOLEAN_LIST' in os.environ):
        extra_requirements = [ "{} =?= TRUE".format(x) for x in os.environ['RIFT_BOOLEAN_LIST'].split(',')]
        requirements += extra_requirements

    # Create prescript command to set up local.cache, only if frames are needed
    # if we have CVMFS frames, we should be copying local.cache over directly, with it already populated !
    if not(frames_local is None) and not(use_cvmfs_frames):   # should be required for singularity or osg
        try:
            lalapps_path2cache=os.environ['LALAPPS_PATH2CACHE']
        except KeyError:
            print("Variable LALAPPS_PATH2CACHE is unset, assume default lal_path2cache is appropriate")
            lalapps_path2cache="lal_path2cache"
        cmdname = 'ile_pre.sh'
        if transfer_files is None:
            transfer_files = []
        transfer_files += [frames_dir]
        # Test if we *need* ile_pre.sh : are path names already relative? DOES NOT WORK
        pre_needed = True
        if False: # try:
          with open('local.cache', 'r') as f:
            lines = f.readlines()
            fnames = [x.split()[-1] for x in lines]
            fnames_no_prefix = [x.replace('file:/','').replace('osdf:/','') for x in fnames]
            for name in fnames_no_prefix:
                if name[0] == '/':
                    pre_needed =True
        else: # except:
            print(" WARNING: local.cache file not present, reverting to ile_pre.sh ")
        if not(pre_needed):
            transfer_files += ['../local.cache']
        else:            
          transfer_files += ["../ile_pre.sh"]  # assuming default working directory setup
          with open(cmdname,'w') as f:
            f.write("#! /bin/bash -xe \n")
            f.write( "ls "+frames_local+" | {lalapps_path2cache} 1> local.cache \n".format(lalapps_path2cache=lalapps_path2cache))  # Danger: need user to correctly specify local.cache directory
            # Rewrite cache file to use relative paths, not a file:// operation
            f.write(" cat local.cache | awk '{print $1, $2, $3, $4}' > local_stripped.cache \n")
            f.write("for i in `ls " + frames_local + "`; do echo "+ frames_local + "/$i; done  > base_paths.dat \n")
            f.write("paste local_stripped.cache base_paths.dat > local_relative.cache \n")
            f.write("cp local_relative.cache local.cache \n")
            f.write('{exe}  "$@" '.format(exe=exe))
            os.system("chmod a+x ile_pre.sh")
            ile_job.set_executable("ile_pre.sh")  # transferred, used as executable
#          ile_job.add_condor_cmd('+PreCmd', '"ile_pre.sh"')


#    if use_osg:
#        ile_job.add_condor_cmd("MY.OpenScienceGrid",'True')
#    if use_cvmfs_frames:
#        transfer_files += ["../local.cache"]
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    # Avoid undesirable hosts in RIFT_AVOID_HOSTS
    if 'RIFT_AVOID_HOSTS' in os.environ:
        line = os.environ['RIFT_AVOID_HOSTS']
        line = line.rstrip()
        if line:
            name_list = line.split(',')
            for name in name_list:
                requirements.append('TARGET.Machine =!= "{}" '.format(name))

    # Write requirements
    # From https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    if not transfer_files is None:
        if not isinstance(transfer_files, list):
            fname_str=transfer_files + ' '.join(extra_files)
        else:
            fname_str = ','.join(transfer_files+extra_files)
        fname_str=fname_str.strip()
        ile_job.add_condor_cmd('transfer_input_files', fname_str)
        ile_job.add_condor_cmd('should_transfer_files','YES')

    if not transfer_output_files is None:
        if not isinstance(transfer_output_files, list):
            fname_str=transfer_output_files
        else:
            fname_str = ','.join(transfer_output_files)
        fname_str=fname_str.strip()
        ile_job.add_condor_cmd('transfer_output_files', fname_str)
 
    # Periodic remove: kill jobs running longer than max runtime
    # https://stackoverflow.com/questions/5900400/maximum-run-time-in-condor
    if not(max_runtime_minutes is None):
        remove_str = 'JobStatus =?= 2 && (CurrentTime - JobStartDate) > ( {})'.format(60*max_runtime_minutes)
        ile_job.add_condor_cmd('periodic_remove', remove_str)

    if 'RIFT_REQUIRE_GPUS' in os.environ:  # new convention 'require_gpus = ' to specify conditions on GPU properties
        ile_job.add_condor_cmd('require_gpus',os.environ['RIFT_REQUIRE_GPUS'])
    

    ###
    ### SUGGESTION FROM STUART (for later)
    # request_memory = ifthenelse( (LastHoldReasonCode=!=34 && LastHoldReasonCode=!=26), InitialRequestMemory, int(1.5 * NumJobStarts * MemoryUsage) )
    # periodic_release = ((HoldReasonCode =?= 34) || (HoldReasonCode =?= 26))
    # This will automatically release a job that is put on hold for using too much memory with a 50% increased memory request each tim.e
    if condor_commands is not None:
        for cmd, value in condor_commands.items():
            ile_job.add_condor_cmd(cmd, value)

    return ile_job, ile_sub_name



def write_consolidate_sub_simple(tag='consolidate', exe=None, base=None,target=None,universe="vanilla",arg_str=None,log_dir=None, use_eos=False,ncopies=1,no_grid=False, max_runtime_minutes=120,extra_text='',**kwargs):
    """
    Write a submit file for launching a consolidation job
       util_ILEdagPostprocess.sh   # suitable for ILE consolidation.  
       arg_str   # add argument (used for NR postprocessing, to identify group)


    """

    exe = exe or which("util_ILEdagPostprocess.sh")

    # Create executable if needed  (using extra_text as flag for now)
    # Note 'base' refers to the working diretory here, so we need to back up
    base_str = ''
    if len(extra_text) > 0:
        if not (base is None):
            base_0 = base[0]
            remove_last_path = '/'.join(base.split('/')[:-1])
            base_str = ' ' + remove_last_path +"/"

        cmdname = "con_sub.sh"
        
        with open(cmdname,'w') as f:        
            f.write("#! /usr/bin/env bash\n")
            f.write(extra_text+"\n")
            extra_args = ''
            f.write( exe +  " $@  \n")

        st = os.stat(cmdname)
        import stat
        os.chmod(cmdname, st.st_mode | stat.S_IEXEC)

        exe = base_str + "con_sub.sh"

    
    ile_job = CondorDAGJob(universe=universe, executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")


    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    # Add manual options for input, output
    ile_job.add_arg(base) # what directory to load
    ile_job.add_arg(target) # where to put the output (label), in CWD
    ile_job.add_arg(arg_str)
    #
    # NO OPTIONS
    #
#    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
#    arg_str = arg_str.lstrip('-')
#    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line


    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))



    #
    # Add normal arguments
    # FIXME: Get valid options from a module
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")
        
    

    ###
    ### SUGGESTION FROM STUART (for later)
    # request_memory = ifthenelse( (LastHoldReasonCode=!=34 && LastHoldReasonCode=!=26), InitialRequestMemory, int(1.5 * NumJobStarts * MemoryUsage) )
    # periodic_release = ((HoldReasonCode =?= 34) || (HoldReasonCode =?= 26))
    # This will automatically release a job that is put on hold for using too much memory with a 50% increased memory request each tim.e

    # Periodic remove: kill jobs running longer than max runtime
    # https://stackoverflow.com/questions/5900400/maximum-run-time-in-condor
    if not(max_runtime_minutes is None):
        remove_str = 'JobStatus =?= 2 && (CurrentTime - JobStartDate) > ( {})'.format(60*max_runtime_minutes)
        ile_job.add_condor_cmd('periodic_remove', remove_str)


    return ile_job, ile_sub_name



def write_unify_sub_simple(tag='unify', exe=None, base=None,target=None,universe="vanilla",arg_str=None,log_dir=None, use_eos=False,ncopies=1,no_grid=False, max_runtime_minutes=60,extra_text='',**kwargs):
    """
    Write a submit file for launching a consolidation job
       util_ILEdagPostprocess.sh   # suitable for ILE consolidation.  
       arg_str   # add argument (used for NR postprocessing, to identify group)


    """

    exe = exe or which("util_CleanILE.py")  # like cat, but properly accounts for *independent* duplicates. (Danger if identical). Also strips large errors

    # Write unify.sh
    #    - problem of globbing inside condor commands
    #    - problem that *.composite files from intermediate results will generally NOT be present 
    cmdname ='unify.sh'
    base_str = ''
    if not (base is None):
        base_str = ' ' + base +"/"
    with open(cmdname,'w') as f:        
        f.write("#! /usr/bin/env bash\n")
        if len(extra_text) > 0:
            f.write(extra_text+"\n")
        f.write( "ls " + base_str+"*.composite  1>&2 \n")  # write filenames being concatenated to stderr
        # Sometimes we need to pass --eccentricity or --tabular-eos-file etc to util_CleanILE.py
        extra_args = ''
        if arg_str:
            extra_args = arg_str
        f.write( exe + extra_args+ base_str+ "*.composite \n")
        # Backstop code for untify.sh
        f.write("""ret_value=$?
if [ $ret_value -eq 0 ]; then
  exit 0
else
  cat {}
fi
""".format(base_str+"*.composite"))
    st = os.stat(cmdname)
    import stat
    os.chmod(cmdname, st.st_mode | stat.S_IEXEC)


    ile_job = CondorDAGJob(universe=universe, executable=base_str+cmdname) # force full prefix
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    # Add manual options for input, output
#    ile_job.add_arg('*.composite') # what to do

    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file(target)

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    # Periodic remove: kill jobs running longer than max runtime
    # https://stackoverflow.com/questions/5900400/maximum-run-time-in-condor
    if not(max_runtime_minutes is None):
        remove_str = 'JobStatus =?= 2 && (CurrentTime - JobStartDate) > ( {})'.format(60*max_runtime_minutes)
        ile_job.add_condor_cmd('periodic_remove', remove_str)

    return ile_job, ile_sub_name

def write_convert_sub(tag='convert', exe=None, file_input=None,file_output=None,universe="vanilla",arg_str='',log_dir=None, use_eos=False,ncopies=1, no_grid=False,max_runtime_minutes=120,**kwargs):
    """
    Write a submit file for launching a 'convert' job
       convert_output_format_ile2inference

    """

    exe = exe or which("convert_output_format_ile2inference")  # like cat, but properly accounts for *independent* duplicates. (Danger if identical). Also strips large errors

    ile_job = CondorDAGJob(universe=universe, executable=exe)
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    if not(arg_str is None or len(arg_str)<2):
        arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
        arg_str = arg_str.lstrip('-')
        ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#        ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
    ile_job.add_arg(file_input)
    
    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file(file_output)

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    # no grid
    if no_grid:  # very aggressively enforce staying on the current filesystem!
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"none"')
        ile_job.add_condor_cmd("MY.flock_local",'true')
        try:
            os.system("condor_config_val UID_DOMAIN > uid_domain.txt")
            with open("uid_domain.txt", 'r') as f:
                uid_domain = f.readline().strip()
                requirements.append(' UidDomain =?= "{}"'.format(uid_domain))
        except:
            True

    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))
        
    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    # Periodic remove: kill jobs running longer than max runtime
    # https://stackoverflow.com/questions/5900400/maximum-run-time-in-condor
    if not(max_runtime_minutes is None):
        remove_str = 'JobStatus =?= 2 && (CurrentTime - JobStartDate) > ( {})'.format(60*max_runtime_minutes)
        ile_job.add_condor_cmd('periodic_remove', remove_str)

    return ile_job, ile_sub_name


def write_test_sub(tag='converge', exe=None,samples_files=None, base=None,target=None,universe="target",arg_str=None,log_dir=None, use_eos=False,ncopies=1, no_grid=False,**kwargs):
    """
    Write a submit file for launching a convergence test job

    """

    exe = exe or which("convergence_test_samples.py") 

    ile_job = CondorDAGJob(universe=universe, executable=exe)
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
    arg_str = arg_str.lstrip('-')
    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

    # Add options for two parameter files
    for name in samples_files:
#        ile_job.add_opt("samples",name)  # do not add in usual fashion, because otherwise the key's value is overwritten
        ile_job.add_opt("samples " + name,'')  

    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file(target)

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')
        try:
            os.system("condor_config_val UID_DOMAIN > uid_domain.txt")
            with open("uid_domain.txt", 'r') as f:
                uid_domain = f.readline().strip()
                requirements.append(' UidDomain =?= "{}"'.format(uid_domain))
        except:
            True


    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name

def write_refine_sub(tag='refine', exe=None, input_net=None,input_grid=None,output=None,universe="vanilla",out_dir=None,log_dir=None, use_eos=False,ncopies=1,arg_str=None,request_memory=1024,arg_vals=None, target=None,no_grid=False,**kwargs):
    """
    Write a submit file for creating a refined CIP grid for NR-based runs.
    """
    
    exe = exe or which("util_TestSpokesIO.py")

    ile_job = CondorDAGJob(universe="vanilla", executable=exe)

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs                                                                                                                     
    arg_str = arg_str.lstrip('-')
    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
    #    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line                                                    

    ile_job.add_opt("fname-dat", input_net)
    ile_job.add_opt("fname", input_grid)
    ile_job.add_opt("save-refinement-fname", output)

    # Add normal arguments
    # FIXME: Get valid options from a module
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option                                                                                                                   
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file(target)

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    # To change interactively:
    #   condor_qedit
    # for example:
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all
    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to refine.sub !")

    return ile_job, ile_sub_name

def write_plot_sub(tag='converge', exe=None,samples_files=None, base=None,target=None,arg_str=None,log_dir=None, use_eos=False,ncopies=1, **kwargs):
    """
    Write a submit file for launching a final plot.  Note the user can in principle specify several samples (e.g., several iterations, if we want to diagnose them)

    """

    exe = exe or which("plot_posterior_corner.py") 

    ile_job = CondorDAGJob(universe="vanilla", executable=exe)

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
    arg_str = arg_str.lstrip('-')
    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

    # Add options for two parameter files
    for name in samples_files:
#        ile_job.add_opt("samples",name)  # do not add in usual fashion, because otherwise the key's value is overwritten
        ile_job.add_opt("posterior-file " + name,'')  

    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file(target)

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name




def write_init_sub(tag='gridinit', exe=None,arg_str=None,log_dir=None, use_eos=False,ncopies=1, **kwargs):
    """
    Write a submit file for launching a grid initialization job.
    Note this routine MUST create whatever files are needed by the ILE iteration

    """

    exe = exe or which("util_ManualOverlapGrid.py") 

    ile_job = CondorDAGJob(universe="vanilla", executable=exe)

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
    arg_str = arg_str.lstrip('-')
    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 


    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name



def write_psd_sub_BW_monoblock(tag='PSD_BW_mono', exe=None, log_dir=None, ncopies=1,arg_str=None,request_memory=4096,arg_vals=None, transfer_files=None,transfer_output_files=None,use_singularity=False,use_osg=False,singularity_image=None,frames_dir=None,cache_file=None,psd_length=4,srate=4096,data_start_time=None,event_time=None,universe='local',no_grid=False,**kwargs):
    """
    Write a submit file for constructing the PSD using BW
    Modern argument syntax for BW
    Note that *all ifo-specific results must be set outside this loop*, to work sensibly, and passed as an argument

    Inputs:
      - channel_dict['H1']  = [channel_name, flow_ifo]
    Outputs:
        - An instance of the CondorDAGJob that was generated for BW
    """
    exe = exe or which("BayesWave")
    if exe is None:
        print(" BayesWave not available, hard fail ")
        sys.exit(0)
    frames_local = None

    ile_job = CondorDAGJob(universe=universe, executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')



    requirements =[]
    #
    # Logging options
    #
    uniq_str = "$(macroevent)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))


    #
    # Loop over IFOs
    # You should only have one, in the workflow for which this is intended
    # Problem: 
    ile_job.add_arg("$(macroargument0)")


    #
    # Add mandatory options
    ile_job.add_opt('Niter', '1000100')
    ile_job.add_opt('Nchain', '20')
    ile_job.add_opt('Dmax', '200')  # limit number of dimensions in model
    ile_job.add_opt('resume', '')
    ile_job.add_opt('progress', '')
    ile_job.add_opt('checkpoint', '')
    ile_job.add_opt('bayesLine', '')
    ile_job.add_opt('cleanOnly', '')
    ile_job.add_opt('updateGeocenterPSD', '')
    ile_job.add_opt('dataseed', '1234')  # make reproducible

    ile_job.add_opt('trigtime', str(event_time))
    ile_job.add_opt('psdstart', str(event_time-(psd_length-2)))
    ile_job.add_opt('segment-start', str(event_time-(psd_length-2)))
    ile_job.add_opt('seglen', str(psd_length))
    ile_job.add_opt('psdlength', str(psd_length))
    ile_job.add_opt('srate', str(srate))
    ile_job.add_opt('outputDir', 'output_$(ifo)')





    # Add lame initial argument
    if "output_file" in kwargs and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if "save_samples" in kwargs and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]


    #
    # Add normal arguments
    # FIXME: Get valid options from a module
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    ile_job.add_condor_cmd('request_memory', str(request_memory)+"M") 

    # Write requirements
    # From https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")


    return ile_job, ile_sub_name


def write_psd_sub_BW_step1(tag='PSD_BW_post', exe=None, log_dir=None, ncopies=1,arg_str=None,request_memory=4096,arg_vals=None, transfer_files=None,transfer_output_files=None,use_singularity=False,use_osg=False,singularity_image=None,frames_dir=None,cache_file=None,channel_dict=None,psd_length=4,srate=4096,data_start_time=None,event_time=None,**kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over intrinsic parameters.

    Inputs:
      - channel_dict['H1']  = [channel_name, flow_ifo]
    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """
    exe = exe or which("BayesWavePost")
    if exe is None:
        print(" BayesWavePost not available, hard fail ")
        import sys
        sys.exit(0)
    frames_local = None

    ile_job = CondorDAGJob(universe="vanilla", executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)


    requirements =[]
    #
    # Logging options
    #
    uniq_str = "$(macroevent)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    #
    # Add mandatory options
    ile_job.add_opt('checkpoint', '')
    ile_job.add_opt('bayesLine', '')
    ile_job.add_opt('cleanOnly', '')
    ile_job.add_opt('updateGeocenterPSD', '')
    ile_job.add_opt('Nchain', '20')
    ile_job.add_opt('Niter', '4000000')
    ile_job.add_opt('Nbayesline', '2000')
    ile_job.add_opt('dataseed', '1234')  # make reproducible

    ile_job.add_opt('trigtime', str(event_time))
    ile_job.add_opt('psdstart', str(event_time-(psd_length-2)))
    ile_job.add_opt('segment-start', str(event_time-(psd_length-2)))
    ile_job.add_opt('seglen', str(psd_length))
    ile_job.add_opt('srate', str(srate))



    #
    # Loop over IFOs
    # Not needed, can do one job per PSD
#    ile_job.add_opt("ifo","$(ifo)")
#    ile_job.add_opt("$(ifo)-cache",cache_file)
    for ifo in channel_dict:
        channel_name, channel_flow = channel_dict[ifo]
        ile_job.add_arg("--ifo "+ ifo)  # need to prevent overwriting!
        ile_job.add_opt(ifo+"-channel", ifo+":"+channel_name)
        ile_job.add_opt(ifo+"-cache", cache_file)
        ile_job.add_opt(ifo+"-flow", str(channel_flow))

    # Add lame initial argument
    if "output_file" in kwargs and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if "save_samples" in kwargs and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]


    #
    # Add normal arguments
    # FIXME: Get valid options from a module
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    ile_job.add_condor_cmd('request_memory', str(request_memory)+"M") 

    # Write requirements
    # From https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")


    return ile_job, ile_sub_name


def write_psd_sub_BW_step0(tag='PSD_BW', exe=None, log_dir=None, ncopies=1,arg_str=None,request_memory=4096,arg_vals=None, transfer_files=None,transfer_output_files=None,use_singularity=False,use_osg=False,singularity_image=None,frames_dir=None,cache_file=None,channel_dict=None,psd_length=4,srate=4096,data_start_time=None,event_time=None,**kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over intrinsic parameters.

    Inputs:
      - channel_dict['H1']  = [channel_name, flow_ifo]
    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """
    exe = exe or which("BayesWave")
    if exe is None:
        print(" BayesWave not available, hard fail ")
        sys.exit(0)
    frames_local = None

    ile_job = CondorDAGJob(universe="vanilla", executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)


    requirements =[]
    #
    # Logging options
    #
    uniq_str = "$(macroevent)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    #
    # Add mandatory options
    ile_job.add_opt('checkpoint', '')
    ile_job.add_opt('bayesLine', '')
    ile_job.add_opt('cleanOnly', '')
    ile_job.add_opt('updateGeocenterPSD', '')
    ile_job.add_opt('Nchain', '20')
    ile_job.add_opt('Niter', '4000000')
    ile_job.add_opt('Nbayesline', '2000')
    ile_job.add_opt('dataseed', '1234')  # make reproducible

    ile_job.add_opt('trigtime', str(event_time))
    ile_job.add_opt('psdstart', str(event_time-(psd_length-2)))
    ile_job.add_opt('segment-start', str(event_time-(psd_length-2)))
    ile_job.add_opt('seglen', str(psd_length))
    ile_job.add_opt('srate', str(srate))



    #
    # Loop over IFOs
    for ifo in channel_dict:
        channel_name, channel_flow = channel_dict[ifo]
        ile_job.add_arg("--ifo " + ifo)
        ile_job.add_opt(ifo+"-channel", ifo+":"+channel_name)
        ile_job.add_opt(ifo+"-cache", cache_file)
        ile_job.add_opt(ifo+"-flow", str(channel_flow))
        ile_job.add_opt(ifo+"-timeslide", str(0.0))


    # Add lame initial argument
    if "output_file" in kwargs and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if "save_samples" in kwargs and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]


    #
    # Add normal arguments
    # FIXME: Get valid options from a module
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    ile_job.add_condor_cmd('request_memory', str(request_memory)+"M") 

    # Write requirements
    # From https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")


    return ile_job, ile_sub_name


def write_resample_sub(tag='resample', exe=None, file_input=None,file_output=None,universe="vanilla",arg_str='',log_dir=None, use_eos=False,ncopies=1, no_grid=False,**kwargs):
    """
    Write a submit file for launching a 'resample' job
       util_ResampleILEOutputWithExtrinsic.py

    """

    exe = exe or which("util_ResampleILEOutputWithExtrinsic.py")  # like cat, but properly accounts for *independent* duplicates. (Danger if identical). Also strips large errors

    ile_job = CondorDAGJob(universe=universe, executable=exe)
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    if not(arg_str is None or len(arg_str)<2):
        arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
        arg_str = arg_str.lstrip('-')
        ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#        ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
    ile_job.add_opt('fname',file_input)
    ile_job.add_opt('fname-out',file_output)
    
    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file(file_output)

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')


    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name



def write_cat_sub(tag='cat', exe=None, file_prefix=None,file_postfix=None,file_output=None,universe="vanilla",arg_str='',log_dir=None, use_eos=False,ncopies=1, no_grid=False,**kwargs):
    """
    Write a submit file for launching a 'resample' job
       util_ResampleILEOutputWithExtrinsic.py

    """

    exe = exe or which("find")  # like cat, but properly accounts for *independent* duplicates. (Danger if identical). Also strips large errors
    exe_switch = which("switcheroo")  # tool for patterend search-replace, to fix first line of output file

    cmdname = 'catjob.sh'
    with open(cmdname,'w') as f:
        f.write("#! /bin/bash\n")
        f.write(exe+"  . -name '"+file_prefix+"*"+file_postfix+r"' -exec cat {} \; | sort -r | uniq > "+file_output+";\n")
        f.write(exe_switch + " 'm1 ' '# m1 ' "+file_output)  # add standard prefix
        os.system("chmod a+x "+cmdname)

    ile_job = CondorDAGJob(universe=universe, executable='catjob.sh')
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")
    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')
        try:
            os.system("condor_config_val UID_DOMAIN > uid_domain.txt")
            with open("uid_domain.txt", 'r') as f:
                uid_domain = f.readline().strip()
                requirements.append(' UidDomain =?= "{}"'.format(uid_domain))
        except:
            True


    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)


#    ile_job.add_arg(" . -name '" + file_prefix + "*" +file_postfix+"' -exec cat {} \; ")
    
    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")


    return ile_job, ile_sub_name



def write_convertpsd_sub(tag='convert_psd', exe=None, ifo=None,file_input=None,target_dir=None,arg_str='',log_dir=None,  universe='local',**kwargs):
    """
    Write script to convert PSD from one format to another.  Needs to be called once per PSD file being used.
    """

    exe = exe or which("convert_psd_ascii2xml")  # like cat, but properly accounts for *independent* duplicates. (Danger if identical). Also strips large errors
    ile_job = CondorDAGJob(universe=universe, executable=exe)

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    ile_job.add_opt("fname-psd-ascii",file_input)
    ile_job.add_opt("ifo",ifo)
    ile_job.add_arg("--conventional-postfix")
    
    #
    # Logging options
    #
    uniq_str = "$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    if not (target_dir is None):
        # Copy output PSD into place
        ile_job.add_condor_cmd("MY.PostCmd", '" cp '+ifo+'-psd.xml.gz ' + target_dir +'"')

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name


def write_joingrids_sub(tag='join_grids', exe=None, universe='vanilla', input_pattern=None,target_dir=None,output_base=None,log_dir=None,n_explode=1, gzip="/usr/bin/gzip", old_add=False, old_style_add=False,no_grid=False,extra_text='', **kwargs):
    """
    Write script to merge CIP 'overlap-grid-(iteration)-*.xml.gz  results.  Issue is that
    """
    default_add = "util_RandomizeOverlapOrder.py"
    if old_style_add:
        default_add = "igwn_ligolw_add"
    
    exe = exe or which(default_add)  
    if not(exe):
        exe = "igwn_ligolw_add"   # go back to fallback if there is a weird disaster -- eg we are using an old-style install before this was updated

    working_dir = log_dir.replace("/logs", '') # assumption about workflow/naming! Danger!

    fname_out =target_dir + "/" +output_base + ".xml.gz"
    if n_explode ==1:   # we are really doing a glob match
        fname_out = fname_out.replace('$(macroiteration)','$1')
        fname_out = fname_out.replace('$(macroiterationnext)','$2')
        alt_work_dir = working_dir.replace('$(macroiteration)','$1')
        alt_out = output_base.replace('$(macroiterationnext)','$2')
        extra_arg = ''
        if old_add:
            extra_arg = " --ilwdchar-compat "  # should never be used anymore
        with open("join_grids.sh",'w') as f:
            f.write("#! /bin/bash  \n")
            f.write(r"""
# merge using glob command called from shell
{}
{}  {} --output {}  {}/{}*.xml.gz 
""".format(extra_text,exe,extra_arg,fname_out,alt_work_dir,alt_out))
        os.system("chmod a+x join_grids.sh")
        exe = target_dir + "/join_grids.sh"

    ile_job = CondorDAGJob(universe=universe, executable=exe)

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')
        try:
            os.system("condor_config_val UID_DOMAIN > uid_domain.txt")
            with open("uid_domain.txt", 'r') as f:
                uid_domain = f.readline().strip()
                requirements.append(' UidDomain =?= "{}"'.format(uid_domain))
        except:
            True

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    if n_explode > 1:
        ile_job.add_arg("--output="+fname_out)


    #
    # Logging options
    #
    uniq_str = "$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))
#    ile_job.set_stdout_file(fname_out)

#    ile_job.add_condor_cmd("MY.PostCmd",  ' "' + gzip + ' ' +fname_out + '"')

    explode_str = ""
    explode_str += " {}/{}.xml.gz ".format(working_dir,output_base)  # base result from fitting job
    if n_explode >1:
     for indx in np.arange(n_explode):
        explode_str+= " {}/{}-{}.xml.gz ".format(working_dir,output_base,indx)
        ile_job.add_arg(explode_str)
    else:
        ile_job.add_arg(" $(macroiteration) $(macroiterationnext) ")
#        explode_str += " {}/{}-*.xml.gz ".format(working_dir,output_base)  # if n_explode is 1 or 0, use a matching pattern 
#    ile_job.add_arg("overlap-grid*.xml.gz")  # working in our current directory
    
    if old_add and n_explode > 1:
        ile_job.add_opt("ilwdchar-compat",'')  # needed?

    ile_job.add_condor_cmd('getenv', default_getenv_value)

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name





def write_subdagILE_sub(tag='subdag_ile', full_path_name=True, exe=None, universe='vanilla', submit_file=None,input_pattern=None,target_dir=None,output_suffix=None,log_dir=None,sim_xml=None, **kwargs):

    """
    Write script to convert PSD from one format to another.  Needs to be called once per PSD file being used.
    """
    exe = exe or which("create_ile_sub_dag.py") 
    subfile = submit_file or 'ILE.sub'
    if full_path_name and target_dir:
        if subfile[0]!= '/': # if not already a full path
            subfile = target_dir + "/"+subfile

    ile_job = CondorDAGJob(universe=universe, executable=exe)

    ile_sub_name = tag + '.sub'
#    if full_path_name and target_dir:
#        ile_sub_name = target_dir +"/" + ile_sub_name
    ile_job.set_sub_file(ile_sub_name)

    ile_job.add_arg("--target-dir "+target_dir)
    ile_job.add_arg("--output-suffix "+output_suffix)
    ile_job.add_arg("--submit-script "+subfile)
    ile_job.add_arg("--macroiteration $(macroiteration)")
    ile_job.add_arg("--sim-xml "+sim_xml)

    working_dir = log_dir.replace("/logs", '') # assumption about workflow/naming! Danger!

    #
    # Logging options
    #
    uniq_str = "$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))
#    ile_job.set_stdout_file(fname_out)

#    ile_job.add_condor_cmd("MY.PostCmd",  ' "' + gzip + ' ' +fname_out + '"')

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name


def write_calibration_uncertainty_reweighting_sub(tag='Calib_reweight', exe=None, log_dir=None, ncopies=1,request_memory=8192,time_marg=True,pickle_file=None,posterior_file=None,universe='vanilla',no_grid=False,ile_args=None,n_cal=100,use_osg=False,use_oauth_files=False,use_singularity=False,singularity_image=None,transfer_files=None,**kwargs):
    """
    Write a submit file for launching jobs to reweight final posterior samples due to calibration uncertainty 

    Inputs:
     - posterior samples, event pickle file (generated by Bilby)
    Outputs:
     - reweighted samples due to calibration uncertainty and corresponding weights
    """
    if use_singularity and (singularity_image == None)  :
        print(" FAIL : Need to specify singularity_image to use singularity ")
        sys.exit(0)
    if use_singularity and (transfer_files == None)  :
        print(" FAIL : Need to specify transfer_files to use singularity at present!  (we will append the prescript; you should transfer any PSDs as well as the grid file ")
        sys.exit(0)

    singularity_image_used = "{}".format(singularity_image) # make copy
    extra_files = []
    if singularity_image:
            if 'osdf:' in singularity_image:
                singularity_image_used  = "./{}".format(singularity_image.split('/')[-1])
                extra_files += [singularity_image]


    
    exe = exe or which("calibration_reweighting.py")
    if exe is None:
        print(" Calibration Reweighting code not available. ")
        sys.exit(0)
    if use_singularity:
        exe_base = os.path.basename(exe)
#        print((" Executable: name breakdown ", path_split, " from ", exe))
        singularity_base_exe_path = "/opt/lscsoft/rift/MonteCarloMarginalizeCode/Code/"  # should not hardcode this ...!
        if 'SINGULARITY_BASE_EXE_DIR' in list(os.environ.keys()) :
            singularity_base_exe_path = os.environ['SINGULARITY_BASE_EXE_DIR']
        else:
#            singularity_base_exe_path = "/opt/lscsoft/rift/MonteCarloMarginalizeCode/Code/"  # should not hardcode this ...!
            singularity_base_exe_path = "/usr/bin/"  # should not hardcode this ...!
        exe=singularity_base_exe_path + exe_base

    ile_job = CondorDAGJob(universe="vanilla", executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)
#    if not(request_disk is False):
#        ile_job.add_condor_cmd('request_disk', str(request_disk))


    requirements =[]
    # Containerization basics
    if use_singularity:
        # Compare to https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
        ile_job.add_condor_cmd('request_CPUs', str(1))
        ile_job.add_condor_cmd('transfer_executable', 'False')
        ile_job.add_condor_cmd("MY.SingularityBindCVMFS", 'True')
        ile_job.add_condor_cmd("MY.SingularityImage", '"' + singularity_image_used + '"')
        ile_job.add_condor_cmd("transfer_output_files", "weight_files")
        requirements.append("HAS_SINGULARITY=?=TRUE")
        print(" WARNING: cal reweighting requires bilby. Directories are moved to cal_evelopes")
#        os.system("condor_config_val UID_DOMAIN > uid_domain.txt")
#       with open("uid_domain.txt", 'r') as f:
#            uid_domain = f.readline().strip()
#            requirements.append(' UidDomain =?= "{}"'.format(uid_domain))
    if use_oauth_files:
        # we are using some authentication to retrieve files from the file transfer list, for example, from distributed hosts, not just submit. eg urls provided
            ile_job.add_condor_cmd('use_oauth_services',use_oauth_files)
    if use_singularity or use_osg:
            # Set up file transfer options
           ile_job.add_condor_cmd("when_to_transfer_output",'ON_EXIT')

    #
    # Logging options
    #
    uniq_str = "$(macroevent)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    #
    # Add mandatory options
    pickle_file_arg = str(pickle_file)
    post_file_arg = str(posterior_file)
    if use_osg:
        transfer_files += [pickle_file_arg , post_file_arg]
        pickle_file_arg = os.path.basename(pickle_file_arg)
        post_file_arg = os.path.basename(post_file_arg)
        if os.path.exists('cal_envelopes'):
            transfer_files += ['./cal_envelopes'] # note initial dir configured so this will work
            ile_job.add_arg(" --use_local_cal_files ")
    ile_job.add_opt('data_dump_file', str(pickle_file_arg))
    ile_job.add_opt('posterior_sample_file', str(post_file_arg))
    ile_job.add_opt('number_of_calibration_curves', str(n_cal))
    ile_job.add_opt('reevaluate_likelihood', 'True')
    ile_job.add_opt('use_rift_samples', 'True')
    if time_marg:
        # problem with this argument: 'False' is often parsed as 'True' by argparsing (weird). Default is 'false'
        ile_job.add_opt('time_marginalization', str(time_marg))

    lmax=None
    approx=None
    if ile_args:
        ile_args_split = ile_args.split('--')
        fmin_list = []
        fmin_template = None
        for line in ile_args_split:
            line_split = line.split()
            if len(line_split)>1:
                if line_split[0] == 'fmin-ifo':
                    fmin_list += [line_split[1]]
                if line_split[0] == 'fmin-template':
                    fmin_template = line_split[1]
                elif line_split[0] == 'l-max':
                    lmax = int(line_split[1])
                elif line_split[0] == 'approx':
                    approx = line_split[1]
#        fmin = np.min(fmin_list)
        if fmin_template:
            ile_job.add_arg(" --fmin {} ".format(fmin_template))  # code will fail without this, and it is always written anyways, but 
        if lmax:
            ile_job.add_arg(" --l-max {} ".format(lmax))
        if approx:
            ile_job.add_arg(" --waveform_approximant {} ".format(approx))
    #
    # Add normal arguments
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    getenv_calmarg = 'PATH,PYTHONPATH,LIBRARY_PATH,LD_LIBRARY_PATH,*RIFT*' # local !
    if not(default_getenv_value == 'True'):
        ile_job.add_condor_cmd('getenv', default_getenv_value)
    else:
        ile_job.add_condor_cmd('getenv', getenv_calmarg)
    # use a smaller request initially, then increase. Should improve throughput
    ile_job.add_condor_cmd('request_memory', str(request_memory/2)+"M")
    ile_job.add_condor_cmd('retry_request_memory', str(request_memory)+"M")

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')

    # Write requirements
    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    # Write transfer file list.  Will handle any surrogates + pickle/container files.
    if not transfer_files is None:
        if not isinstance(transfer_files, list):
            fname_str=transfer_files  + ' '.join(extra_files)
        else:
            fname_str = ','.join(transfer_files+extra_files)
        fname_str=fname_str.strip()
        ile_job.add_condor_cmd('transfer_input_files', fname_str)
        ile_job.add_condor_cmd('should_transfer_files','YES')

    # Stream log info
    if not ('RIFT_NOSTREAM_LOG' in os.environ):
        ile_job.add_condor_cmd("stream_error",'True')
        ile_job.add_condor_cmd("stream_output",'True')
    
    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
         print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")


    return ile_job, ile_sub_name

def bilby_prior_dict_string_from_mc_q(mc_range,dmax_Mpc):
    out_str = """chirp-mass: bilby.gw.prior.UniformInComponentsChirpMass(minimum={}, maximum={}, name='chirp_mass', boundary=None), mass-ratio: bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.05, maximum=1.0, name='mass_ratio', latex_label='$q$', unit=None, boundary=None), mass-1: Constraint(minimum=1, maximum=1000, name='mass_1', latex_label='$m_1$', unit=None), mass-2: Constraint(minimum=1, maximum=1000, name='mass_2', latex_label='$m_2$', unit=None), a-1: Uniform(minimum=0, maximum=0.99, name='a_1', latex_label='$a_1$', unit=None, boundary=None), a-2: Uniform(minimum=0, maximum=0.99, name='a_2', latex_label='$a_2$', unit=None, boundary=None), tilt-1: Sine(minimum=0, maximum=3.141592653589793, name='tilt_1'), tilt-2: Sine(minimum=0, maximum=3.141592653589793, name='tilt_2'), phi-12: Uniform(minimum=0, maximum=6.283185307179586, name='phi_12', boundary='periodic'), phi-jl: Uniform(minimum=0, maximum=6.283185307179586, name='phi_jl', boundary='periodic'), luminosity-distance: PowerLaw(alpha=2, minimum=10, maximum={}, name='luminosity_distance', latex_label='$d_L$', unit='Mpc', boundary=None), theta-jn: Sine(minimum=0, maximum=3.141592653589793, name='theta_jn'), psi: Uniform(minimum=0, maximum=3.141592653589793, name='psi', boundary='periodic'), phase: Uniform(minimum=0, maximum=6.283185307179586, name='phase', boundary='periodic'), dec: Cosine(name='dec'), ra: Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')
""".format(mc_range[0],mc_range[1],dmax_Mpc)
    out_str = "{" + out_str.rstrip() + "}"
    return out_str

def write_bilby_pickle_sub(tag='Bilby_pickle', exe=None, universe='local', log_dir=None, ncopies=1,request_memory=4096,bilby_ini_file=None,no_grid=False,frames_dir=None,cache_file=None,ile_args=None,**kwargs):
    """
    Write a submit file for launching a job to generate a pickle file based off a bilby ini file; needed for  reweight final posterior samples due to calibration uncertainty
    
    Inputs:
     - bilby ini file
    Outputs:
     - pickle file of event settings; needed as input for calibration reweighting

     Notes:
       - local universe is generally safer: we need access to frame files in a standard location (typically datafind returns cvmfs, etc). That may not be available on remote nodes.
    """
    exe = exe or which("bilby_pipe_generation")
    if exe is None:
        print(" Pickle generation code unavailable. ")
        sys.exit(0)
    ile_job = CondorDAGJob(universe=universe, executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")


    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    #
    #Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))


    # Add manual options for input, output.  Hopefully this all happens in order as needed, if not we will just concatenate
    # 
    ile_job.add_arg(str(bilby_ini_file)) # needs to be a bilby ini file for the particular event being analyzed
    ile_job.add_arg(' --data-dump-file calmarg/data/calmarg_data_dump.pickle')

    # Problem: bilby ini file may not have 'data-dict', in which case we need to backstop it with data from 'frames_dir' or 'cache_file'
    # Problem: bilby ini file does not have sections.
    # Workaround: https://stackoverflow.com/questions/2885190/using-configparser-to-read-a-file-without-section-name
    config = configparser.ConfigParser()
    config.optionxform=str # force preserve case! Important for --choose-data-LI-seglen
    with open(bilby_ini_file) as stream:
        config.read_string("[top]\n" + stream.read())
        bilby_items = dict(config["top"])
        # Backstop horrible parsing situations where it returns a string and not dict
        if not(isinstance(bilby_items['channel-dict'], dict)):
            # Safer string parsing - in case no comma at end, etc
            bilby_items['channel-dict'] = bilby_ish_string_to_dict(bilby_items['channel-dict'])
            # base_list=bilby_items['channel-dict'][1:-1].split(',')[:-1]
            # base_dict = {}
            # for item in base_list:
            #     if item:
            #        key,value =item.split(':')
            #        key = key.lstrip()
            #        base_dict[key] = value
            # bilby_items['channel-dict'] = base_dict
        ifo_list = list(bilby_items['channel-dict'])  # PSDs must be listed, implicitly provides all ifos
    # remove entries with the None keyword, as misleading
    dict_names = list(bilby_items)
    for name in dict_names:
        if bilby_items[name] == 'None':
            del bilby_items[name]
    if not('data-dict' in bilby_items):
        bilby_data_dict = {}
        if cache_file:
            print(" calmarg: bilby ini file does not have data_dict, attempting to identify data from (host) cache file: {} ".format(cache_file))
            cache_lines = np.loadtxt(cache_file,dtype=str)
            if len(ifo_list)==1 and len(cache_lines.shape)==1:
                ifo = cache_lines[0] + '1'
                bilby_data_dict[ifo] = cache_lines[-1].replace('file://localhost','')
            else:
                if len(cache_lines) <= len(ifo_list):
                    for indx in np.arange(len(cache_lines)):
                        ifo = cache_lines[indx][0]+"1"
                        bilby_data_dict[ifo] = cache_lines[indx][-1].replace('file://localhost','')
                else:
                    import glob
                    print(" WARNING: cache file ideallly contain one line per IFO to identify files in this approach")
                    if  not(frames_dir) or not os.path.exists('./frames_dir'):
                        print(" WARNING: Backstop method being applied - regenerating frames into frames_dir")
                        shutil.copyfile(cache_file, 'local.cache')
                        os.system("util_ForOSG_MakeTruncatedLocalFramesDir.sh .")
                    fnames_gwf = list(glob.glob(frames_dir+"/*.gwf")  )
                    # get dictionary matching files
                    for name in fnames_gwf:
                        this_frame_ifo = None
                        for ifo in ifo_list:
                            if name.startswith(frames_dir+"/{}-".format(ifo)):
                                this_frame_ifo=ifo
                        bilby_data_dict[ifo] = this_frame_ifo
        elif frames_dir:  # Danger : this directory might be EMPTY and generated at runtile
            import glob
            print(" calmarg: bilby ini file does not have data_dict, attempting to identify data from directory: {} ".format(frames_dir))
            fnames_gwf = list(glob.glob(frames_dir+"/*.gwf")  )
            # get dictionary matching files
            for name in fnames_gwf:
                this_frame_ifo = None
                for ifo in ifo_list:
                    if name.startswith(frames_dir+"/{}-".format(ifo)):
                        this_frame_ifo=ifo
                bilby_data_dict[ifo] = this_frame_ifo
            if len(list(bilby_data_dict)) ==0 :
                print("  Failed to find files in frames_dir, warning! ")
        else:
            print(" ==== WARNING FALLTHROUGH : calmarg attempting to identify correct frame files to use but falling back to 'magic' options from bilby ===")
        # add to command-line arguments, IF NONEMPTY.  Otherwise we're stuck, and we have to hope magic works
        if len(list(bilby_data_dict))>0:
            data_argstr = '{}'.format(bilby_data_dict)
            data_argstr = '  --data-dict ""{}""  '.format(data_argstr.replace(' ',''))  # double "" because we are in a condor submit script!  Annoying but seemt to be correct
            ile_job.add_arg(data_argstr)
        else:
            print(" ==== WARNING FALLTHROUGH : calmarg failed to pull out options  ===",bilby_data_dict,bilby_items)

    # make LOCAL COPIES OF CAL ENVELOPES with STANDARD NAMES - facilitate remote/OSG use
    if 'spline-calibration-envelope-dict' in bilby_items:
        spline_dict = bilby_ish_string_to_dict(bilby_items['spline-calibration-envelope-dict'])
        if not os.path.exists('cal_envelopes'):
            os.mkdir('cal_envelopes')
        for ifo in spline_dict:
            shutil.copyfile(spline_dict[ifo], 'cal_envelopes/{}.txt'.format(ifo))

    # Other required settings from ILE
    # approximant: if ile_args present, ALWAYS parse it and set it that way, so we are consistent with our own analysis
    if ile_args:
        approx = bilby_items['waveform-approximant']
        ile_args_split = ile_args.split('--')
        start_time =None
        end_time=None
        trigger_time =None
        rift_window_shape=None    # remember this is a dimensionless number, not a time
        rift_srate =None
        fmin_list = []
        channel_list=[]
        fmax=None
        for line in ile_args_split:
            line_split = line.split()
            if len(line_split)>1:
                if line_split[0]=='approx':
                    approx = line_split[1]
                elif line_split[0] == 'event-time':
                    event_time = float(line_split[1])
                elif line_split[0] == 'data-start-time':
                    start_time = float(line_split[1])
                elif line_split[0] == 'data-end-time':
                    end_time = float(line_split[1])
                elif line_split[0] == 'window-shape':
                    rift_window_shape = float(line_split[1])
                elif line_split[0] == 'srate':
                    rift_srate = int(float(line_split[1]))  # safety
                elif line_split[0] == 'fmin-ifo':
                    fmin_list += [line_split[1]]
                elif line_split[0] == 'fmax':
                    fmax = int(float(line_split[1]))  # safety
                elif line_split[0] == 'channel-name':
                    channel_list += [line_split[1]]
        ile_job.add_arg(" --waveform-approximant {} ".format(approx))
        if rift_srate:
            ile_job.add_arg(" --sampling-frequency {} ".format(rift_srate))
        if event_time:
            ile_job.add_arg(" --trigger-time {} ".format(event_time))
        # t_tukey
        t_tukey = (end_time-start_time)*rift_window_shape/2   # basically the fraction of time not in the window; see formula in helper
        ile_job.add_arg(" --tukey-roll-off {} ".format(t_tukey))
        # channel list
        channel_dict ={}
        for channel_id in channel_list:
            if '=' in channel_id:
                ifo, channel_name = channel_id.split('=')
                channel_dict[ifo] = channel_name
        channel_argstr = '{}'.format(channel_dict)
        channel_argstr = '  --channel-dict ""{}""  '.format(channel_argstr.replace(' ',''))
        ile_job.add_arg(channel_argstr)
        # fmin
        if len(fmin_list)>0:
            fmin_dict = {}
            for fmin_id in fmin_list:
                if '=' in fmin_id:
                    ifo, fmin = fmin_id.split('=')
                    fmin_dict[ifo] = float(fmin)
            fmin_argstr = '{}'.format(fmin_dict)
            fmin_argstr = '  --minimum-frequency ""{}""  '.format(fmin_argstr.replace(' ',''))  # inside condor
            ile_job.add_arg(fmin_argstr)
            # fmax.  Use previous to get ifo list
            if fmax:
                fmax_dict = {}
                for ifo in fmin_dict:
                    fmax_dict[ifo] =fmax
                fmax_argstr = '{}'.format(fmax_dict)
                fmax_argstr = '  --maximum-frequency ""{}""  '.format(fmax_argstr.replace(' ',''))
                ile_job.add_arg(fmax_argstr)

    # Add outdir, label so we can control filename for output
    ile_job.add_arg(" --outdir calmarg ")
    ile_job.add_arg(" --label calmarg ")

    #
    # Add normal arguments
    # Note these need to appear *after* the bilby ini file
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))


    # getenv: use default, BUT want to check for datafind so it is passed!
    #
    pickle_getenv_value = str(default_getenv_value) # force re-create
    if not(pickle_getenv_value == 'True') and not('DATAFIND' in pickle_getenv_value):
        names_datafind = []
        for name in os.environ:
            if 'DATAFIND' in name:
                names_datafind.append(name)
        if len(names_datafind)>0:
            pickle_getenv_value= default_getenv_value +',' + ",".join(names_datafind)
    ile_job.add_condor_cmd('getenv', pickle_getenv_value)
    ile_job.add_condor_cmd('request_memory', str(request_memory)+"M")

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')
        try:
            os.system("condor_config_val UID_DOMAIN > uid_domain.txt")
            with open("uid_domain.txt", 'r') as f:
                uid_domain = f.readline().strip()
                requirements.append(' UidDomain =?= "{}"'.format(uid_domain))
        except:
            True

    # Write requirements
    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
         print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")


    return ile_job, ile_sub_name

def write_comov_distance_reweighting_sub(tag='Comov_dist', comov_distance_reweighting_exe=None, reweight_location=None, universe='vanilla', log_dir=None, ncopies=1,request_memory=4096,posterior_file=None,no_grid=False,**kwargs):
    """
    Write a submit file for launching a job to generate reweight posterior samples to reflect a comoving distance prior
    
    Inputs:
     - posterior samples in h5 format
    Outputs:
     - reweighted samples in h5 format
    """
    exe = comov_distance_reweighting_exe or which("make_uni_comov_skymap.py")
    if exe is None:
        print(" Comoving distance reweighting code unavailable. ")
        sys.exit(0)
    ile_job = CondorDAGJob(universe=universe, executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")


    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    # Add manual options for input, output
    ile_job.add_opt('resampled-file', str(reweight_location))
    ile_job.add_arg(str(posterior_file)) # needs to be a bilby ini file for the particular event being analyzed

    #
    #Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))



    #
    # Add normal arguments
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    ile_job.add_condor_cmd('request_memory', str(request_memory)+"M")

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')


    # Write requirements
    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
         print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")


    return ile_job, ile_sub_name

def write_convert_ascii_to_h5_sub(tag='Convert_ascii2h5', convert_ascii_to_h5_exe=None,output_file=None, universe='vanilla', log_dir=None, ncopies=1,request_memory=4096,posterior_file=None,no_grid=False,**kwargs):
    """
    Converts posterior samples file from ascii to h5 format
    
    Inputs:
     - posterior samples in ascii format
    Outputs:
     - posterior samples in h5 format
    """
    exe = convert_ascii_to_h5_exe or which("convert_output_format_ascii2h5.py")
    if exe is None:
        print(" Converting code unavailable. ")
        sys.exit(0)
    ile_job = CondorDAGJob(universe=universe, executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")


    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    # Add manual options for input, output
    ile_job.add_opt('output-file', str(output_file))
    ile_job.add_opt('posterior-file', str(posterior_file))
#    ile_job.add_arg(str(posterior_file)) # needs to be a bilby ini file for the particular event being analyzed

    #
    #Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))



    #
    # Add normal arguments
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    ile_job.add_condor_cmd('getenv', default_getenv_value)
    ile_job.add_condor_cmd('request_memory', str(request_memory)+"M")

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')
        try:
            os.system("condor_config_val UID_DOMAIN > uid_domain.txt")
            with open("uid_domain.txt", 'r') as f:
                uid_domain = f.readline().strip()
                requirements.append(' UidDomain =?= "{}"'.format(uid_domain))
        except:
            True

    # Write requirements
    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
         print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")


    return ile_job, ile_sub_name


def write_hyperpost_sub(tag='HYPER', exe=None, input_net='all.marg_net',output='output-samples',universe="vanilla",out_dir=None,log_dir=None, ncopies=1,arg_str=None,request_memory=8192,arg_vals=None, no_grid=False,request_disk=False, transfer_files=None,transfer_output_files=None,use_singularity=False,use_osg=False,use_simple_osg_requirements=False,singularity_image=None,max_runtime_minutes=None,condor_commands=None,**kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over hyperparameters.
    Almost identical to CIP 

    Inputs:
    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """

    if use_singularity and (singularity_image == None)  :
        print(" FAIL : Need to specify singularity_image to use singularity ")
        sys.exit(0)
    if use_singularity and (transfer_files == None)  :
        print(" FAIL : Need to specify transfer_files to use singularity at present!  (we will append the prescript; you should transfer any PSDs as well as the grid file ")
        sys.exit(0)


    exe = exe or which("util_ConstructEOSPosterior.py")
    if use_singularity:
        path_split = exe.split("/")
        print((" Executable: name breakdown ", path_split, " from ", exe))
        singularity_base_exe_path = "/usr/bin/"  # should not hardcode this ...!
        if 'SINGULARITY_BASE_EXE_DIR' in list(os.environ.keys()) :
            singularity_base_exe_path = os.environ['SINGULARITY_BASE_EXE_DIR']
        exe=singularity_base_exe_path + path_split[-1]
        if path_split[-1] == 'true':  # special universal path for /bin/true, don't override it!
            exe = "/usr/bin/true"
    ile_job = CondorDAGJob(universe=universe, executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies


    # no grid
    if no_grid:
        ile_job.add_condor_cmd("MY.DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("MY.flock_local",'true')
        try:
            os.system("condor_config_val UID_DOMAIN > uid_domain.txt")
            with open("uid_domain.txt", 'r') as f:
                uid_domain = f.readline().strip()
                requirements.append(' UidDomain =?= "{}"'.format(uid_domain))
        except:
            True

    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    #
    # Add options en mass, by brute force
    #
    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
    arg_str = arg_str.lstrip('-')
    ile_job.add_opt(arg_str,'')  

    ile_job.add_opt("fname", input_net)
    ile_job.add_opt("fname-output-samples", out_dir+"/"+output)
    ile_job.add_opt("fname-output-integral", out_dir+"/"+output)

    #
    # Macro based options.
    #     - select EOS from list (done via macro)
    #     - pass spectral parameters
    #

    #
    # Logging options
    #
    uniq_str = "$(macroevent)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    if "fname_output_samples" in kwargs and kwargs["fname_output_samples"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["fname_output_samples"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("fname-output-samples", "%s-%s.%s" % (ofname, uniq_str, ext))

    #
    # Add normal arguments
    # FIXME: Get valid options from a module
    #
    for opt, param in list(kwargs.items()):
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    if not use_osg:
        ile_job.add_condor_cmd('getenv', default_getenv_value)
    ile_job.add_condor_cmd('request_memory', str(request_memory)+"M") 
    if not(request_disk is False):
        ile_job.add_condor_cmd('request_disk', str(request_disk)) 
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    requirements = []
    if use_singularity:
        # Compare to https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
        ile_job.add_condor_cmd('request_CPUs', str(1))
        ile_job.add_condor_cmd('transfer_executable', 'False')
        ile_job.add_condor_cmd("MY.SingularityBindCVMFS", 'True')
        ile_job.add_condor_cmd("MY.SingularityImage", '"' + singularity_image + '"')
        requirements.append("HAS_SINGULARITY=?=TRUE")

    if use_osg:
           # avoid black-holing jobs to specific machines that consistently fail. Uses history attribute for ad
           ile_job.add_condor_cmd('periodic_release','(HoldReasonCode == 45) && (HoldReasonSubCode == 0)')
           ile_job.add_condor_cmd('job_machine_attrs','Machine')
           ile_job.add_condor_cmd('job_machine_attrs_history_length','4')
#           for indx in [1,2,3,4]:
#               requirements.append("TARGET.GLIDEIN_ResourceName=!=MY.MachineAttrGLIDEIN_ResourceName{}".format(indx))
           if "OSG_DESIRED_SITES" in os.environ:
               ile_job.add_condor_cmd('+DESIRED_SITES',os.environ["OSG_DESIRED_SITES"])
           if "OSG_UNDESIRED_SITES" in os.environ:
               ile_job.add_condor_cmd('+UNDESIRED_SITES',os.environ["OSG_UNDESIRED_SITES"])
           # Some options to automate restarts, acts on top of RETRY in dag
    if use_singularity or use_osg:
            # Set up file transfer options
           ile_job.add_condor_cmd("when_to_transfer_output",'ON_EXIT')

           # Stream log info
           if not ('RIFT_NOSTREAM_LOG' in os.environ):
               ile_job.add_condor_cmd("stream_error",'True')
               ile_job.add_condor_cmd("stream_output",'True')


    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    # Stream log info: always stream CIP error, it is a critical bottleneck
    if True: # not ('RIFT_NOSTREAM_LOG' in os.environ):
        ile_job.add_condor_cmd("stream_error",'True')
        ile_job.add_condor_cmd("stream_output",'True')

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")
        
    
    if not transfer_files is None:
        if not isinstance(transfer_files, list):
            fname_str=transfer_files
        else:
            fname_str = ','.join(transfer_files)
        fname_str=fname_str.strip()
        ile_job.add_condor_cmd('transfer_input_files', fname_str)
        ile_job.add_condor_cmd('should_transfer_files','YES')

    # Periodic remove: kill jobs running longer than max runtime
    # https://stackoverflow.com/questions/5900400/maximum-run-time-in-condor
    if not(max_runtime_minutes is None):
        remove_str = 'JobStatus =?= 2 && (CurrentTime - JobStartDate) > ( {})'.format(60*max_runtime_minutes)
        ile_job.add_condor_cmd('periodic_remove', remove_str)


    ###
    ### SUGGESTION FROM STUART (for later)
    # request_memory = ifthenelse( (LastHoldReasonCode=!=34 && LastHoldReasonCode=!=26), InitialRequestMemory, int(1.5 * NumJobStarts * MemoryUsage) )
    # periodic_release = ((HoldReasonCode =?= 34) || (HoldReasonCode =?= 26))
    # This will automatically release a job that is put on hold for using too much memory with a 50% increased memory request each tim.e
    if condor_commands is not None:
        for cmd, value in condor_commands.items():
            ile_job.add_condor_cmd(cmd, value)


    return ile_job, ile_sub_name

