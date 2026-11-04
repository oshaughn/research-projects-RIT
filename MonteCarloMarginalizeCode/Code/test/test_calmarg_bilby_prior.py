"""Independent Bilby prior comparison; run by calmarg-check, which installs Bilby."""
import pytest
from test_calmarg_calibration import _envelope


@pytest.mark.parametrize("convention", ["data", "template"])
def test_envelope_prior_matches_bilby(tmp_path, convention):
    """Compare actual prior nodes with the independent Bilby implementation."""
    import numpy as np
    from bilby.gw.prior import CalibrationPriorDict
    from RIFT.calmarg.generate_realizations import node_prior
    path = tmp_path / "envelope.txt"
    _envelope(path, varying=True)
    ours = node_prior(path, 21, 120, 6, correction_type=convention)
    reference = CalibrationPriorDict.from_envelope_file(
        str(path), 21, 120, 6, "X1", correction_type=convention)
    mean = [reference[f"recalib_X1_amplitude_{i}"].mu + 1 for i in range(6)]
    mean += [reference[f"recalib_X1_phase_{i}"].mu for i in range(6)]
    sigma = [reference[f"recalib_X1_amplitude_{i}"].sigma for i in range(6)]
    sigma += [reference[f"recalib_X1_phase_{i}"].sigma for i in range(6)]
    np.testing.assert_allclose(ours["mean"], mean, atol=1e-14)
    np.testing.assert_allclose(ours["sigma"], sigma, atol=1e-14)



@pytest.mark.parametrize("setting", [None, {"H1":"template", "V1":"data"}])
def test_reweighting_forwards_instrument_metadata(tmp_path, setting):
    """Execute the production prior-construction site against Bilby."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    import bilby
    from RIFT.calmarg.calibration import correction_type_for_ifo
    path = Path(__file__).resolve().parents[1] / "bin/calibration_reweighting.py"
    tree = ast.parse(path.read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "ifo_calibration_priors" for t in n.targets))
    envelope = tmp_path / "envelope.txt"
    _envelope(envelope)
    for ifo in ["H1", "V1"]:
        ns = dict(bilby=bilby, ifo=SimpleNamespace(name=ifo, minimum_frequency=20, maximum_frequency=120),
                  calibration_file_path=str(envelope), calibration_correction_type=setting,
                  correction_type_for_ifo=correction_type_for_ifo,
                  bilby_pipe=SimpleNamespace(utils=SimpleNamespace(convert_string_to_dict=lambda x:x)))
        exec(compile(ast.Module(body=[call], type_ignores=[]), str(path), "exec"), ns)
        expected = 1/1.1 if correction_type_for_ifo(setting, ifo) == "data" else 1.1
        assert ns["ifo_calibration_priors"][f"recalib_{ifo}_amplitude_0"].mu + 1 == pytest.approx(expected)
