"""Release compiled XLA executables between test modules.

Every CPU executable compiled in a pytest process stays loaded, and each one holds
memory mappings.  A full .travis/test-jax.sh shard reaches the kernel's per-process
limit (vm.max_map_count, 65530 on CIT), and the next compile or persistent-cache read
aborts or raises "Failed to materialize symbols".  Clearing the caches at each module
boundary costs only the recompiles shared across files.
"""

import gc

import pytest


@pytest.fixture(autouse=True, scope="module")
def _release_compiled_executables():
    yield
    import jax
    jax.clear_caches()
    gc.collect()
