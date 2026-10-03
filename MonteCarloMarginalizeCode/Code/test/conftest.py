"""Keep two CI gate SCRIPTS out of pytest's collection.

THE DEFECT.  test_mcsamplerEnsemble_extended.py and test_vector_coordinates.py are named
test_* but are not pytest targets.  Each is a script whose work happens at module level,
starting with an optparse parse_args().  When pytest imports one during collection it
parses pytest's OWN argv, fails on an option like -q, and calls sys.exit.  That raises out
of the collection loop as an INTERNALERROR, which aborts the entire run -- so every file
sorting after it is silently never collected.  `--continue-on-collection-errors` does not
help, because an INTERNALERROR is not a collection error.  Measured on Code/test/: 2278 items
collected before this conftest, 3015 after.

WHY NOT RENAME THEM, which is the fix used for the demos in integrators/ (c43f2e7f8,
b6ef0e66a, #344).  Those were demos that nothing ran.  These two are live gates, invoked BY
NAME with --as-test:

    .travis/test-coord.sh:3       test_vector_coordinates.py
    .travis/test-integrate.sh:167 test_mcsamplerEnsemble_extended.py (and :169 --use-lnL)

A rename has to move with those scripts, and .travis/test-ci-roster.py keys reachability off
the file name, so the census would stop seeing them as covered.  Calling a gate demo_* would
also misdescribe it.  They keep their names and pytest is told to skip them.

THE RISK OF AN IGNORE LIST is that it goes stale without saying so: if one of these later
grows a real test function, pytest would skip it forever and nothing would report that.
_check_still_scripts() below turns that into a loud UsageError at collection time.  It also
fires if an entry names a file that no longer exists, so a delete cannot leave a dead entry
behind.  Keep it cheap -- this conftest loads for every pytest run under Code/test/,
including the core-unit gate.
"""

import gc
import os
import re
import sys

import pytest

# Relative to this directory, which is what pytest expects of collect_ignore.
collect_ignore = [
    "test_mcsamplerEnsemble_extended.py",
    "test_vector_coordinates.py",
]

# Matches how pytest itself decides a module has tests: a test_ function or a Test class at
# any indentation.  Deliberately textual -- this runs before the module is importable.
_HAS_TESTS = re.compile(r"^[ \t]*(?:def test_|class Test)", re.M)


def _check_still_scripts():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in collect_ignore:
        path = os.path.join(here, name)
        if not os.path.exists(path):
            raise pytest.UsageError(
                "conftest.py ignores %s, which does not exist. A collect_ignore entry for a "
                "deleted file is a silent no-op; drop the line." % name)
        if _HAS_TESTS.search(open(path, errors="replace").read()):
            raise pytest.UsageError(
                "conftest.py ignores %s because it is a script with no test functions, but it "
                "now defines one. pytest is skipping that test and will go on skipping it.\n"
                "    Either move the test to a collected file, or give the script an "
                "`if __name__ == \"__main__\":` guard so it is safe to import and drop it "
                "from collect_ignore." % name)


_check_still_scripts()


_JAX_TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jax") + os.sep


@pytest.fixture(autouse=True, scope="module")
def _release_jax_executables(request):
    """Free compiled XLA executables after each test/jax module.

    Every executable stays loaded, and each holds memory mappings.  A full test-jax.sh shard
    reached the per-process limit (vm.max_map_count, 65530 on CIT) and the next compile
    aborted or raised "Failed to materialize symbols".  Clearing at module boundaries bounds
    growth across modules, not within one.  This lives here rather than in test/jax/conftest.py
    because a second module named conftest shadows gpu_precompute's `from conftest import`.
    """
    yield
    if not str(request.fspath).startswith(_JAX_TEST_DIR):
        return
    jax = sys.modules.get("jax")
    if jax is not None and hasattr(jax, "clear_caches"):
        jax.clear_caches()
    gc.collect()
