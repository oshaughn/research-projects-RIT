"""Regression tests for the JAX driver's intrinsic-template finalization.

The executable is intentionally not imported: it parses command-line options at
module import.  Extracting the real function by AST exercises its implementation
without constructing a fake copy of the logic under test.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys

import lal
import lalsimulation as lalsim
import numpy as np

from RIFT import lalsimutils


DRIVER = (Path(__file__).resolve().parents[1] /
          "bin" / "integrate_likelihood_extrinsic_jax")


def _load_templates_function():
    source = DRIVER.read_text()
    tree = ast.parse(source, filename=str(DRIVER))
    fn = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == "load_templates")
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "np": np, "sys": sys, "lalsimutils": lalsimutils, "lalsim": lalsim,
        "MSUN": lal.MSUN_SI, "PC": lal.PC_SI,
    }
    exec(compile(module, str(DRIVER), "exec"), namespace)
    return namespace["load_templates"]


def _opts(**updates):
    values = dict(
        reference_freq=100.0, fmin_template=20.0, approximant="TaylorF2",
        sim_xml=None, sim_grid=None, random_event=False, event=0,
        n_events_to_analyze=1, mass1=None, mass2=None,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def _assert_finalized(P, m1, m2, s1z, lambda1, epoch, delta_f, delta_t):
    assert P.phiref == 0.0
    assert P.psi == 0.0
    assert P.incl == 0.0
    np.testing.assert_allclose([P.m1, P.m2], [m1, m2], rtol=2e-7)
    np.testing.assert_allclose(P.s1z, s1z, rtol=0, atol=1e-12)
    np.testing.assert_allclose(P.lambda1, lambda1, rtol=2e-7)
    assert float(P.tref) == float(epoch)
    assert P.deltaF == delta_f and P.deltaT == delta_t
    assert P.dist == 1000.0e6 * lal.PC_SI


def test_xml_template_zeroes_extrinsics_and_preserves_intrinsics(tmp_path):
    original = lalsimutils.ChooseWaveformParams(
        m1=1.45 * lal.MSUN_SI, m2=1.22 * lal.MSUN_SI,
        s1z=0.031, s2z=-0.014, lambda1=527.0, lambda2=811.0,
        phiref=0.73, psi=1.17, incl=2.02)
    base = tmp_path / "nonzero-extrinsics"
    lalsimutils.ChooseWaveformParams_array_to_xml([original], str(base))
    xml = str(base) + ".xml.gz"
    # Compare to serialized values: XML spin fields have finite precision.
    serialized = lalsimutils.xml_to_ChooseWaveformParams_array(xml)[0]
    assert serialized.phiref != 0 and serialized.psi != 0 and serialized.incl != 0

    load_templates = _load_templates_function()
    epoch, delta_f, delta_t = 1000000000.25, 0.25, 1.0/4096
    got = load_templates(_opts(sim_xml=xml), epoch, delta_f, delta_t)[0]
    _assert_finalized(got, serialized.m1, serialized.m2, serialized.s1z,
                      serialized.lambda1, epoch, delta_f, delta_t)


def test_grid_template_zeroes_extrinsics_and_preserves_intrinsics(tmp_path):
    grid = tmp_path / "nonzero-extrinsics-grid.dat"
    grid.write_text(
        "m1 m2 s1z s2z lambda1 lambda2 phiref psi incl\n"
        "1.47 1.19 0.027 -0.011 493 902 0.61 1.03 2.21\n")

    load_templates = _load_templates_function()
    epoch, delta_f, delta_t = 1000000001.5, 0.125, 1.0/8192
    got = load_templates(_opts(sim_grid=str(grid)), epoch, delta_f, delta_t)[0]
    _assert_finalized(got, 1.47*lal.MSUN_SI, 1.19*lal.MSUN_SI,
                      0.027, 493.0, epoch, delta_f, delta_t)
