#!/usr/bin/env python

from __future__ import print_function

import os.path
import pkgutil
import re
import sys
from importlib import import_module

import pytest

pkgname = "RIFT"
package = import_module(pkgname)

EXCLUDE = re.compile(
    "("
    r"\Atests\Z|"
    r"\Atest_|"
    r"\Aconftest\Z|"
    r"\A_"
    ")"
)


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
    import_module(modname)


if __name__ == "__main__":
    if "-v" not in " ".join(sys.argv[1:]):  # default to verbose
        sys.argv.append("-v")
    sys.argv.append("-rs")
    sys.exit(pytest.main(args=[__file__] + sys.argv[1:]))
