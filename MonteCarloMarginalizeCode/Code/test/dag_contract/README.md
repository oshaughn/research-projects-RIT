# Pipeline DAG build contract

`rift_pipeline_contract.py` statically audits the DAG emitted by
`util_RIFT_pseudo_pipe.py`. It does not submit jobs and does not need HTCondor
Python bindings.

The contract checks:

- DAG syntax references, duplicate nodes, and cycles;
- presence of intrinsic ILE/CIP and the final extrinsic, conversion, Bilby
  pickle, calibration-reweighting, and calibration-combine roles;
- reachability of every extrinsic job through conversion and every calibration
  batch through the unique combine product;
- accidental terminal sinks in any pre-combine final-stage role;
- top-level `ABORT-DAG-ON` convergence nodes, which must be upstream of the
  terminal product and use `--always-succeed` in the emitted submit file;
- adaptive external sub-DAG aborts, which are allowed only after a grid
  conversion when the parent immediately fetches that grid and remains upstream
  of the terminal product.

Sub-DAGs nest. A boundary is evaluated in whichever DAG declares it. The fetch
child is read from that containing graph. The connection to the terminal product
is inherited from the containing SUBDAG node, which cannot complete until every
node inside it has.

The distinction between top-level and adaptive aborts is intentional. A
successful top-level convergence abort can silently skip extrinsic/calibration
work. An adaptive sub-DAG is allowed to converge early because its already-built
grid is the handoff product.

CI coverage has two layers. `test_dag_build_contract.py` mutation-tests the
checker itself, while `.travis/test-build.sh` applies it to an untouched
calibration-marginalization DAG produced by the real pipeline builder.

Run the fast checker tests with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  MonteCarloMarginalizeCode/Code/test/test_dag_build_contract.py
```

Audit an existing build with:

```bash
python MonteCarloMarginalizeCode/Code/test/dag_contract/rift_pipeline_contract.py \
  /path/to/rift/run
```
