"""Guard waveform-option parity at the classic compound-driver boundary."""

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


DRIVER = Path(__file__).resolve().parents[2] / "bin" / "integrate_likelihood_extrinsic_batchmode"


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return prefix + "." + node.attr if prefix else node.attr
    return ""


def _analyze_event_tree():
    tree = ast.parse(DRIVER.read_text(), filename=str(DRIVER))
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "analyze_event")


def test_classic_compound_reuses_baseline_waveform_generation_kwargs():
    function = _analyze_event_tree()
    assignments = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name)
                and target.id == "waveform_generation_kwargs"
                for target in node.targets)
    ]
    assert len(assignments) == 1

    option_fields = {
        "use_gwsignal": "use_gwsignal",
        "use_gwsignal_approx": "approximant",
        "use_external_EOB": "use_external_EOB",
        "nr_lookup": "nr_lookup",
        "nr_lookup_valid_groups": "nr_lookup_group",
        "perturbative_extraction": "nr_perturbative_extraction",
        "perturbative_extraction_full": "nr_perturbative_extraction_full",
        "use_provided_strain": "nr_use_provided_strain",
        "hybrid_use": "nr_hybrid_use",
        "hybrid_method": "nr_hybrid_method",
        "ROM_group": "rom_group",
        "ROM_param": "rom_param",
        "ROM_use_basis": "rom_use_basis",
        "ROM_limit_basis_size": "rom_limit_basis_size_to",
        "no_memory": "no_memory",
        "force_22_mode": "force_hyperbolic_22",
    }
    sentinels = {field: object() for field in option_fields.values()}
    opts = SimpleNamespace(**sentinels)
    nr_group, nr_param, nested = object(), object(), object()
    expression = ast.Expression(assignments[0].value)
    ast.fix_missing_locations(expression)
    forwarded = eval(
        compile(expression, str(DRIVER), "eval"),
        dict(opts=opts, NR_template_group=nr_group,
             NR_template_param=nr_param, extra_waveform_kwargs=nested),
    )

    assert forwarded["NR_group"] is nr_group
    assert forwarded["NR_param"] is nr_param
    assert forwarded["extra_waveform_kwargs"] is nested
    for keyword, field in option_fields.items():
        assert forwarded[keyword] is sentinels[field]

    expanded_calls = []
    for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
        if any(keyword.arg is None
               and isinstance(keyword.value, ast.Name)
               and keyword.value.id == "waveform_generation_kwargs"
               for keyword in call.keywords):
            expanded_calls.append(_dotted_name(call.func))

    assert Counter(expanded_calls) == Counter({
        "factored_likelihood.PrecomputeLikelihoodTerms": 1,
        "PrecomputeLikelihoodTermsRotatingFreqResponseGPU": 1,
        "factored_likelihood_rotating_freqresponse.PrecomputeLikelihoodTermsRotatingFreqResponse": 1,
    })
