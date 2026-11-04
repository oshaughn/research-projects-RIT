#!/usr/bin/env python

from __future__ import print_function

import os.path
import pkgutil
import re
import sys
from importlib import import_module

import pytest

# Set environment variables as needed
import os
if not('GW_SURROGATE' in os.environ.keys()):
   os.environ["GW_SURROGATE"]=''
# EOBTidalExternal - no one should ever use this code
if not('EOB_BASE' in os.environ.keys()):
   os.environ["EOB_BASE"]=''
if not('EOB_ARCHIVE' in os.environ.keys()):
   os.environ["EOB_ARCHIVE"]='~/'
if not('MATLAB_BASE' in os.environ.keys()):
   os.environ["MATLAB_BASE"]=''
# EOBTidalExternalC - probably ditto
if not('EOB_C_BASE' in os.environ.keys()):
   os.environ["EOB_C_BASE"]=''
if not('EOB_C_ARCHIVE' in os.environ.keys()):
   os.environ["EOB_C_ARCHIVE"]='~/'
if not('EOB_C_ARCHIVE_NMAX' in os.environ.keys()):
   os.environ["EOB_C_ARCHIVE_NMAX"]='1'
# EOSManager
if not('EOS_TABLES' in os.environ.keys()):
   os.environ["EOS_TABLES"]='~/'
# lalsuite
if not('LALSIMULATION_DATADIR' in os.environ.keys()):
   os.environ["LALSIMULATION_DATADIR"]='~/'



pkgname = "RIFT"
package = import_module(pkgname)

# files to ignore
EXCLUDE = re.compile("({})".format("|".join([
    r"\Atests\Z",
    r"\Atest_",
    r"\Aconftest\Z",
    r"\A_",
    r"\Aasimov\Z",
])))

# ignorable failures
IGNORE = re.compile("({})".format("|".join([
    r"\ANo module named torch\Z",
    r"\ANo module named 'torch'\Z",
    r"\ANo module named gpytorch\Z",
    r"\ANo module named 'gpytorch'\Z",
    r"\ANo module named cupy\Z",
    r"\ANo module named 'cupy'\Z",
    r"\ANo module named asimov\Z",
    r"\ANo module named 'asimov'\Z",
    r"\ANo module named jax\Z",
    r"\ANo module named 'jax'\Z",
    r"\ANo module named optax\Z",
    r"\ANo module named 'optax'\Z",
    r"\ANo module named equinox\Z",
    r"\ANo module named 'equinox'\Z",
    r"\ANo module named tinygp\Z",
    r"\ANo module named 'tinygp'\Z",
    r"\ANo module named numpyro\Z",
    r"\ANo module named 'numpyro'\Z",
    r"\ANo module named flowMC\Z",
    r"\ANo module named 'flowMC'\Z",
    r"\ANo module named htcondor2\Z",
    r"\ANo module named 'htcondor2'\Z",
])))


# These modules deliberately wrap dependency errors with actionable diagnostics.
# Skip only an absent optional ROOT; a broken installed API must still fail.
OPTIONAL_MODULE_DEPENDENCIES = {"RIFT.likelihood.td_gs_precompute": "tdanalysis"}


def _missing_optional_module(modname, exc):
    dependency = OPTIONAL_MODULE_DEPENDENCIES.get(modname)
    seen = set()
    while dependency and exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ModuleNotFoundError) and exc.name == dependency:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def iter_all_modules(path, exclude=EXCLUDE):
    name = os.path.basename(path)
    for _, modname, ispkg in pkgutil.iter_modules(path=[path]):
        if exclude and exclude.search(modname):
            continue
        yield "{}.{}".format(name, modname)
        if ispkg:
            for mod2 in iter_all_modules(os.path.join(path, modname), exclude=exclude):
                yield "{}.{}".format(name, mod2)


@pytest.mark.parametrize("modname", iter_all_modules(package.__path__[0]))
def test_import(modname):
    try:
        import_module(modname)
    except Exception as exc:
        if IGNORE.search(str(exc)) or _missing_optional_module(modname, exc):
            pytest.skip(str(exc))
        raise


if __name__ == "__main__":
    if "-v" not in " ".join(sys.argv[1:]):  # default to verbose
        sys.argv.append("-v")
    sys.argv.append("-rs")
    sys.exit(pytest.main(args=[__file__] + sys.argv[1:]))
