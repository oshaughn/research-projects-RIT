"""Portfolio density and explicit CPU-backend regressions on O4c."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from RIFT.integrators import mcsamplerAdaptiveVolume as AV
from RIFT.integrators import mcsamplerPortfolio as PF


@pytest.mark.parametrize('width,fraction', [(2., 1.), (0.3, 1.), (2., 0.5)])
def test_av_draw_returns_normalized_physical_density(width, fraction):
    s = AV.MCSampler(n_chunk=1000)
    s.add_parameter('x', pdf=None, left_limit=0., right_limit=width,
                    prior_pdf=lambda x: np.ones(len(x))/width)
    s.setup()
    # Keep the first bin when the retained fractional volume shrinks.
    s.V = fraction
    s.dx = s.dx0 * fraction
    proposal, prior, rv = s.draw_simplified(1000, 'x')
    assert np.all(rv >= 0)
    assert np.all(rv <= width*fraction)
    np.testing.assert_allclose(AV.identity_convert(proposal), 1./(width*fraction))
    np.testing.assert_allclose(AV.identity_convert(prior)/AV.identity_convert(proposal), fraction)


def test_portfolio_constant_likelihood_has_unit_evidence_on_nonunit_box():
    s = PF.MCSampler(portfolio_freeze_wt=1., portfolio=[AV.MCSampler(n_chunk=1000),
                               AV.MCSampler(n_chunk=1000)])
    for name, width in [('x', 2.), ('y', 3.)]:
        s.add_parameter(name, pdf=None, left_limit=0., right_limit=width,
                        prior_pdf=lambda x, w=width: np.ones(len(x))/w)
    s.setup()
    lnz, _, _, _ = s.integrate_log(lambda x, y: np.zeros(len(x)), 'x', 'y',
                                   n=1000, nmax=1000, neff=10,
                                   save_intg=True, save_no_samples=True)
    assert float(lnz) == pytest.approx(0., abs=1e-12)


def test_explicit_numpy_portfolio_configures_members_and_container():
    # Execute the actual selection branch without loading detector data.
    path = Path(__file__).resolve().parents[1]/'bin/integrate_likelihood_extrinsic_batchmode'
    tree = ast.parse(path.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and 'opts.sampler_method == \'portfolio\'' in ast.unparse(n.test)
                  and any(isinstance(a, ast.Assign) and
                          any(isinstance(t, ast.Name) and t.id == 'sampler_list' for t in a.targets)
                          for a in n.body))
    member = SimpleNamespace(xpy=None)
    container = SimpleNamespace(xpy=None)
    unavailable = lambda x: pytest.fail('device conversion used for explicit numpy')
    opts = SimpleNamespace(sampler_method='portfolio', sampler_portfolio=['AV'],
                           sampler_xpy='numpy', n_chunk=1000)
    scope = dict(opts=opts, mcsampler_Portfolio_ok=True, numpy=np,
                 xpy_default=object(), identity_convert=unavailable,
                 identity_convert_togpu=unavailable,
                 # The core module has no set_xpy_to_numpy method.
                 mcsampler=SimpleNamespace(),
                 mcsamplerAdaptiveVolume=SimpleNamespace(MCSampler=lambda **kw: member),
                 mcsamplerPortfolio=SimpleNamespace(MCSampler=lambda **kw: container))
    exec(compile(ast.Module(body=branch.body, type_ignores=[]), str(path), 'exec'), scope)
    for sampler in [member, container]:
        assert sampler.xpy is np
        values = np.arange(3.)
        assert sampler.identity_convert(values) is values
        assert sampler.identity_convert_togpu(values) is values
