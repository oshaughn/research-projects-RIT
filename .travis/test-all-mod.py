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
    for mod in pkgutil.iter_modules(path=[path]):
        if exclude and exclude.search(mod.name):
            continue
        yield "{}.{}".format(name, mod.name)
        if mod.ispkg:
            for mod2 in iter_all_modules(os.path.join(path, mod.name), exclude=exclude):
                yield "{}.{}".format(name, mod2)


for mod in iter_all_modules(package.__path__[0]):
    print(mod)
    import_module(mod)
