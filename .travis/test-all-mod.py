#! /usr/bin/env python

import os.path
import pkgutil
import re
import sys
from importlib import import_module

pkgname = sys.argv[1]
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


for mod in iter_all_modules(package.__path__[0]):
    print(mod)
    import_module(mod)
