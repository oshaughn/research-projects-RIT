"""
mcsamplerGPU.MCSampler.add_parameter(..., cdf_inv=None) with a vectorized pdf.

cdf_inverse() builds the CDF by integrating the pdf with scipy's odeint, whose
callback receives a python FLOAT.  Every pdf helper in mcsamplerGPU is vectorized
(ret_uniform_samp_vector_alt returns ones(len(x))/(b-a) since 2022-04), so
add_parameter without cdf_inv raised TypeError on len(float) before any likelihood
call.  Scipy-independent: reproduced on 1.10.1 and 1.13.1.

The fix probes the pdf once with a float and falls back to a length-1 backend
array, so scalar-style pdfs (uniform_samp, numpy.vectorize, the withfloor helper)
keep the call they had before, and 0-d or length-1 returns on either backend reduce
to a float.

In the ILE drivers the only add_parameter call without an analytic cdf_inv was
t_ref, reached with --time-marginalization off under the default
--sampler-method adaptive_cartesian_gpu.  Those calls now pass the analytic
inverse they already built; the driver test at the bottom pins the wiring.

Test ids carry the backend mcsamplerGPU picked (numpy on CI and ldas-grid, cupy on
the CIT GPU head nodes), so a log says which backend a pass is evidence for.
"""
import ast
import functools
from pathlib import Path

import numpy as np
import pytest

import RIFT.integrators.mcsamplerGPU as mcsamplerGPU

BACKEND = mcsamplerGPU.xpy_default.__name__
assert BACKEND in ("numpy", "cupy"), BACKEND


def _zero_d_uniform(lo, hi):
    """pdf that returns a 0-d numpy array (works pre-fix; must keep working)."""
    return lambda x: np.asarray(1.0 / (hi - lo))


# (label, pdf, lo, hi, x at cdf=0.25, x at cdf=0.5)
CASES = [
    ("uniform_alt", mcsamplerGPU.ret_uniform_samp_vector_alt(-0.002, 0.002),
     -0.002, 0.002, -0.001, 0.0),
    ("uniform_phase", mcsamplerGPU.uniform_samp_phase,
     0.0, 2 * np.pi, np.pi / 2, np.pi),
    ("uniform_psi", mcsamplerGPU.uniform_samp_psi,
     0.0, np.pi, np.pi / 4, np.pi / 2),
    # pdf sin(x)/2 on [0,pi]: cdf = (1-cos x)/2, so cdf=0.25 at pi/3, 0.5 at pi/2
    ("theta", mcsamplerGPU.uniform_samp_theta,
     0.0, np.pi, np.pi / 3, np.pi / 2),
    # scalar-style pdfs: these took odeint's float before the fix and still must
    ("uniform_scalar", functools.partial(mcsamplerGPU.uniform_samp, -1.0, 3.0),
     -1.0, 3.0, 0.0, 1.0),
    ("np_vectorize", np.vectorize(functools.partial(mcsamplerGPU.uniform_samp, -1.0, 3.0)),
     -1.0, 3.0, 0.0, 1.0),
    ("zero_d_return", _zero_d_uniform(-1.0, 3.0),
     -1.0, 3.0, 0.0, 1.0),
    # 0.75 on [0,1), 0.25 on [1,2): cdf(1)=0.75, so cdf=0.25 at 1/3 and 0.5 at 2/3
    ("withfloor", functools.partial(mcsamplerGPU.uniform_samp_withfloor_vector, 2.0, 1.0, 0.5),
     0.0, 2.0, 1.0 / 3.0, 2.0 / 3.0),
]
IDS = ["%s-%s" % (c[0], BACKEND) for c in CASES]


@pytest.mark.parametrize("label,pdf,lo,hi,q25,q50", CASES, ids=IDS)
def test_add_parameter_without_cdf_inv(label, pdf, lo, hi, q25, q50):
    s = mcsamplerGPU.MCSampler()
    s.add_parameter(label, pdf=pdf, cdf_inv=None, left_limit=lo, right_limit=hi,
                    prior_pdf=pdf)
    inv = s.cdf_inv[label]
    x = inv(np.array([0.0, 0.25, 0.5, 1.0]))
    # 1000-point grid: one cell is 1e-3*(hi-lo); measured error is below 2e-5, so
    # a one-cell-shifted inverse fails this
    tol = 1e-4 * (hi - lo)
    assert x[0] == pytest.approx(lo, abs=tol)
    assert x[1] == pytest.approx(q25, abs=tol)
    assert x[2] == pytest.approx(q50, abs=tol)
    assert x[3] == pytest.approx(hi, abs=tol)
    # the draw path hands cdf_inv a vector of uniforms
    draws = inv(np.random.default_rng(0).uniform(size=2000))
    assert draws.min() >= lo and draws.max() <= hi


DRIVERS = ["integrate_likelihood_extrinsic_batchmode",
           "integrate_likelihood_extrinsic_batchmode_lisa",
           "integrate_likelihood_extrinsic"]


@pytest.mark.parametrize("driver", DRIVERS)
def test_ile_tref_passes_analytic_cdf_inv(driver):
    """The t_ref add_parameter call must pass the analytic inverse, not None.

    None routes through cdf_inverse -> odeint -> interp1d; on a cupy host interp1d
    then rejects the device array draw_simplified hands it, so the odeint fix alone
    would not make the default sampler run there.
    """
    ile = Path(__file__).parents[1] / "bin" / driver
    tree = ast.parse(ile.read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "add_parameter"
        and node.args and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "t_ref"
    ]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(kw["cdf_inv"], ast.Name)
    assert kw["cdf_inv"].id == "tref_sampler_cdf_inv"
    assigned = {
        t.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
        for t in node.targets if isinstance(t, ast.Name)
    }
    assert "tref_sampler_cdf_inv" in assigned
    # and that inverse is exact on the window ends, in the units the sampler draws in
    inv = functools.partial(mcsamplerGPU.uniform_samp_cdf_inv_vector, -0.002, 0.002)
    assert np.allclose(inv(np.array([0.0, 0.5, 1.0])), [-0.002, 0.0, 0.002])
