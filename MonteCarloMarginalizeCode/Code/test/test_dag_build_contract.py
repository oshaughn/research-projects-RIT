"""Mutation-oriented tests for the build-only RIFT DAG contract."""

from pathlib import Path

from dag_contract.dag_structure import external_dags, parse_dag, validate_dag
from dag_contract.rift_pipeline_contract import DAG_NAME, audit_pipeline


GOOD_DAG = """JOB intrinsic ILE.sub
JOB cip CIP.sub
JOB convergence test.sub
JOB extrinsic_a ILE_extr.sub
JOB extrinsic_b ILE_extr.sub
JOB convert convert_extr.sub
JOB pickle Bilby_pickle.sub
JOB calibration_a Calib_reweight.sub
JOB calibration_b Calib_reweight.sub
JOB combine CAL_REWEIGHT_COMBINE.sub
PARENT intrinsic CHILD cip
PARENT cip CHILD convergence
PARENT convergence CHILD extrinsic_a extrinsic_b
PARENT extrinsic_a extrinsic_b CHILD convert
PARENT convert CHILD pickle
PARENT pickle CHILD calibration_a calibration_b
PARENT calibration_a calibration_b CHILD combine
ABORT-DAG-ON convergence 1 RETURN 0
RETRY convergence 3
"""


EXTERNAL_DAG = """JOB intrinsic ILE.sub
JOB cip CIP.sub
JOB convergence test.sub
SUBDAG EXTERNAL adaptive nested.dag
JOB fetch FETCH_4_subdag.sub
JOB extrinsic_a ILE_extr.sub
JOB extrinsic_b ILE_extr.sub
JOB convert convert_extr.sub
JOB pickle Bilby_pickle.sub
JOB calibration_a Calib_reweight.sub
JOB calibration_b Calib_reweight.sub
JOB combine CAL_REWEIGHT_COMBINE.sub
PARENT intrinsic CHILD cip
PARENT cip CHILD convergence
PARENT convergence CHILD adaptive
PARENT adaptive CHILD fetch
PARENT fetch CHILD extrinsic_a extrinsic_b
PARENT extrinsic_a extrinsic_b CHILD convert
PARENT convert CHILD pickle
PARENT pickle CHILD calibration_a calibration_b
PARENT calibration_a calibration_b CHILD combine
ABORT-DAG-ON convergence 1 RETURN 0
"""


NESTED_DAG = """JOB grid convert.sub
JOB inner_convergence nested_test.sub
PARENT grid CHILD inner_convergence
ABORT-DAG-ON inner_convergence 1 RETURN 0
"""


def _parse(tmp_path, text):
    path = tmp_path / "input.dag"
    path.write_text(text, encoding="utf-8")
    return parse_dag(path)


def _make_tree(tmp_path, dag_text=GOOD_DAG, safe_abort=True):
    (tmp_path / DAG_NAME).write_text(dag_text, encoding="utf-8")
    for name in (
        "ILE.sub",
        "CIP.sub",
        "ILE_extr.sub",
        "convert_extr.sub",
        "Bilby_pickle.sub",
        "Calib_reweight.sub",
        "CAL_REWEIGHT_COMBINE.sub",
    ):
        (tmp_path / name).write_text("queue 1\n", encoding="utf-8")
    safe_argument = " --always-succeed" if safe_abort else ""
    (tmp_path / "test.sub").write_text(
        'arguments = "{}"\n'.format(safe_argument), encoding="utf-8"
    )
    return tmp_path


def _make_external_tree(tmp_path, nested=NESTED_DAG, top=EXTERNAL_DAG):
    _make_tree(tmp_path, top)
    (tmp_path / "nested.dag").write_text(nested, encoding="utf-8")
    (tmp_path / "nested_test.sub").write_text(
        'arguments = "active early convergence"\n', encoding="utf-8"
    )
    (tmp_path / "convert.sub").write_text('arguments = "produce grid"\n', encoding="utf-8")
    (tmp_path / "FETCH_4_subdag.sub").write_text(
        'arguments = "fetch grid"\n', encoding="utf-8"
    )
    return tmp_path


# ``nested.dag`` may itself declare an external sub-DAG.  The abort then lives
# two levels down and its boundary must be read in ``nested.dag``, not in the
# top-level DAG.
NESTED_WITH_EXTERNAL = """JOB grid convert.sub
SUBDAG EXTERNAL inner_adaptive nested2.dag
JOB inner_fetch FETCH_5_subdag.sub
PARENT grid CHILD inner_adaptive
PARENT inner_adaptive CHILD inner_fetch
"""


SECOND_LEVEL_DAG = """JOB inner_grid convert.sub
JOB inner_convergence nested_test.sub
PARENT inner_grid CHILD inner_convergence
ABORT-DAG-ON inner_convergence 1 RETURN 0
"""


# One container file may be instantiated by two SUBDAG nodes.  The detached
# instance is declared first, so a traversal keyed on the container file alone
# evaluates the connected instance and silently drops the detached one.
SHARED_EXTERNAL_DAG = EXTERNAL_DAG.replace(
    "SUBDAG EXTERNAL adaptive nested.dag\n",
    "SUBDAG EXTERNAL adaptive_detached nested.dag\nSUBDAG EXTERNAL adaptive nested.dag\n",
).replace(
    "PARENT convergence CHILD adaptive\n",
    "PARENT convergence CHILD adaptive adaptive_detached\n",
)


SELF_REFERENTIAL_NESTED_DAG = """JOB grid convert.sub
SUBDAG EXTERNAL repeat nested.dag
JOB inner_convergence nested_test.sub
PARENT grid CHILD repeat
PARENT repeat CHILD inner_convergence
ABORT-DAG-ON inner_convergence 1 RETURN 0
"""


def _make_double_external_tree(
    tmp_path,
    nested=NESTED_WITH_EXTERNAL,
    second=SECOND_LEVEL_DAG,
    top=EXTERNAL_DAG,
):
    _make_external_tree(tmp_path, nested=nested, top=top)
    (tmp_path / "nested2.dag").write_text(second, encoding="utf-8")
    (tmp_path / "FETCH_5_subdag.sub").write_text(
        'arguments = "fetch inner grid"\n', encoding="utf-8"
    )
    return tmp_path


def test_parser_models_control_flow_and_subdag(tmp_path):
    dag = _parse(
        tmp_path,
        """JOB start start.sub
SUBDAG EXTERNAL nested child.dag
JOB finish finish.sub
PARENT start CHILD nested
PARENT nested CHILD finish
RETRY start 3 UNLESS-EXIT 17
ABORT-DAG-ON start 1 RETURN 0
SCRIPT DEFER 2 30 POST start check.sh
""",
    )
    assert validate_dag(dag) == []
    assert dag.ancestors("finish") == {"start", "nested"}
    assert dag.descendants("start") == {"nested", "finish"}
    assert dag.abort["start"]["exit"] == "1"
    assert dag.scripts["start"][0]["kind"] == "POST"
    assert external_dags(dag)[0][1].name == "child.dag"


def test_parser_rejects_undefined_references(tmp_path):
    dag = _parse(
        tmp_path,
        """JOB a a.sub
PARENT missing CHILD a
ABORT-DAG-ON absent 1 RETURN 0
""",
    )
    errors = validate_dag(dag)
    assert any("undefined parent missing" in item for item in errors)
    assert any("undefined node absent" in item for item in errors)


def test_parser_rejects_cycle(tmp_path):
    dag = _parse(
        tmp_path,
        """JOB a a.sub
JOB b b.sub
PARENT a CHILD b
PARENT b CHILD a
""",
    )
    assert any("cycle detected" in item for item in validate_dag(dag))


def test_parser_rejects_duplicate_node(tmp_path):
    dag = _parse(tmp_path, "JOB a a.sub\nJOB a other.sub\n")
    assert any("duplicate node" in item for item in validate_dag(dag))


def test_complete_handoff_passes(tmp_path):
    report = audit_pipeline(_make_tree(tmp_path))
    assert report["pass"], report["failures"]


def test_detached_extrinsic_is_caught(tmp_path):
    broken = GOOD_DAG.replace(
        "PARENT extrinsic_a extrinsic_b CHILD convert",
        "PARENT extrinsic_b CHILD convert",
    )
    report = audit_pipeline(_make_tree(tmp_path, broken))
    assert not report["pass"]
    assert any("extrinsic jobs do not feed" in item for item in report["failures"])
    assert report["checks"]["critical_sinks"]


def test_detached_calibration_batch_is_caught(tmp_path):
    broken = GOOD_DAG.replace(
        "PARENT calibration_a calibration_b CHILD combine",
        "PARENT calibration_a CHILD combine",
    )
    report = audit_pipeline(_make_tree(tmp_path, broken))
    assert not report["pass"]
    assert any("calibration combine misses" in item for item in report["failures"])


def test_abort_without_actual_submit_protection_is_caught(tmp_path):
    report = audit_pipeline(_make_tree(tmp_path, safe_abort=False))
    assert not report["pass"]
    assert any("can terminate before handoff" in item for item in report["failures"])


def test_abort_not_upstream_of_product_is_caught(tmp_path):
    broken = GOOD_DAG.replace(
        "PARENT convergence CHILD extrinsic_a extrinsic_b",
        "PARENT cip CHILD extrinsic_a extrinsic_b",
    )
    report = audit_pipeline(_make_tree(tmp_path, broken))
    assert not report["pass"]
    assert any("lack terminal-product descendants" in item for item in report["failures"])


def test_active_nested_abort_after_grid_and_before_fetch_is_safe(tmp_path):
    report = audit_pipeline(_make_external_tree(tmp_path))
    assert report["pass"], report["failures"]
    boundary = report["checks"]["external_dags"][0]
    assert boundary["active_abort_nodes"] == 1
    assert boundary["has_fetch_child"]
    assert boundary["feeds_terminal_product"]


def test_nested_abort_without_prior_grid_is_caught(tmp_path):
    nested = NESTED_DAG.replace("PARENT grid CHILD inner_convergence\n", "")
    report = audit_pipeline(_make_external_tree(tmp_path, nested=nested))
    assert not report["pass"]
    assert any("aborts before a grid conversion" in item for item in report["failures"])


def test_nested_abort_without_fetch_boundary_is_caught(tmp_path):
    top = EXTERNAL_DAG.replace("PARENT adaptive CHILD fetch", "PARENT convergence CHILD fetch")
    report = audit_pipeline(_make_external_tree(tmp_path, top=top))
    assert not report["pass"]
    assert any("lacks immediate FETCH child" in item for item in report["failures"])


def test_second_level_active_abort_is_read_in_its_own_container(tmp_path):
    report = audit_pipeline(_make_double_external_tree(tmp_path))
    assert report["pass"], report["failures"]
    boundaries = {item["node"]: item for item in report["checks"]["external_dags"]}
    inner = boundaries["inner_adaptive"]
    assert inner["container"].endswith("nested.dag")
    assert inner["active_abort_nodes"] == 1
    assert inner["has_fetch_child"]
    assert inner["feeds_terminal_product"]


def test_second_level_abort_without_fetch_boundary_is_caught(tmp_path):
    nested = NESTED_WITH_EXTERNAL.replace(
        "PARENT inner_adaptive CHILD inner_fetch", "PARENT grid CHILD inner_fetch"
    )
    report = audit_pipeline(_make_double_external_tree(tmp_path, nested=nested))
    assert not report["pass"]
    assert any(
        "lacks immediate FETCH child: inner_adaptive" in item
        for item in report["failures"]
    )


def test_second_level_abort_detached_from_product_is_caught(tmp_path):
    top = EXTERNAL_DAG.replace("PARENT adaptive CHILD fetch", "PARENT convergence CHILD fetch")
    report = audit_pipeline(_make_double_external_tree(tmp_path, top=top))
    assert not report["pass"]
    assert any(
        "does not feed terminal product: inner_adaptive" in item
        for item in report["failures"]
    )


def test_repeated_external_invocations_are_evaluated_separately(tmp_path):
    report = audit_pipeline(
        _make_double_external_tree(tmp_path, top=SHARED_EXTERNAL_DAG)
    )
    assert not report["pass"]
    assert any(
        "does not feed terminal product: inner_adaptive" in item
        for item in report["failures"]
    )
    inner = [
        item
        for item in report["checks"]["external_dags"]
        if item["node"] == "inner_adaptive"
    ]
    assert len(inner) == 2
    assert {item["feeds_terminal_product"] for item in inner} == {True, False}
    assert {tuple(item["invocation"]) for item in inner} == {
        ("adaptive", "inner_adaptive"),
        ("adaptive_detached", "inner_adaptive"),
    }


def test_self_referential_external_dag_is_caught(tmp_path):
    report = audit_pipeline(
        _make_external_tree(tmp_path, nested=SELF_REFERENTIAL_NESTED_DAG)
    )
    assert not report["pass"]
    assert any("re-enters its own container" in item for item in report["failures"])


def test_second_level_abort_before_grid_is_caught(tmp_path):
    second = SECOND_LEVEL_DAG.replace("PARENT inner_grid CHILD inner_convergence\n", "")
    report = audit_pipeline(_make_double_external_tree(tmp_path, second=second))
    assert not report["pass"]
    assert any("aborts before a grid conversion" in item for item in report["failures"])
