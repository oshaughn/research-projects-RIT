#! /usr/bin/env python

import pkgutil
import RIFT

from importlib import import_module

package= RIFT
for importer, modname,ispkg in pkgutil.walk_packages(path=package.__path__, prefix=package.__name__+".",onerror=lambda x: None):
    if ispkg:
        print(modname)
        import_module(modname)
