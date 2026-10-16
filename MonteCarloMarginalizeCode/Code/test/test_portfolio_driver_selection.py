"""Exercise real driver selection validation without frame or optional-plugin imports."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

DRIVER = Path(__file__).resolve().parents[1]/'bin/integrate_likelihood_extrinsic_batchmode'


def validator(**flags):
    tree = ast.parse(DRIVER.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == '_validate_sampler_selection')
    scope = dict(mcsampler_gpu_ok=True, mcsampler_AV_ok=True, mcsampler_gmm_ok=True,
                 mcsampler_Portfolio_ok=True,
                 mcsamplerPortfolio=SimpleNamespace(known_pipelines={'custom': object()}))
    scope.update(flags)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(DRIVER), 'exec'), scope)
    return scope['_validate_sampler_selection']


def options(members=None, method='portfolio', args=None):
    return SimpleNamespace(sampler_method=method, sampler_portfolio=members,
                           sampler_portfolio_args=args)


@pytest.mark.parametrize('members', [['AV,GMM'], ['AV', 'GMM'], [' AV, ', 'GMM'], ['custom,AC']])
def test_documented_member_spellings_are_normalized(members):
    opts = options(members)
    validator()(opts)
    assert opts.sampler_portfolio == (['custom','AC'] if members == ['custom,AC'] else ['AV','GMM'])


@pytest.mark.parametrize('members', [None, [], ['', ' , '], ['AV,BOGUS']])
def test_empty_or_unknown_portfolios_are_rejected(members):
    with pytest.raises(ValueError, match='requires at least one|unknown member'):
        validator()(options(members))


@pytest.mark.parametrize('name,flag', [('AV','mcsampler_AV_ok'), ('GMM','mcsampler_gmm_ok'),
                                      ('AC','mcsampler_gpu_ok')])
def test_missing_member_import_has_actionable_error(name, flag):
    with pytest.raises(ValueError, match='member.*unavailable'):
        validator(**{flag:False})(options([name]))


def test_unavailable_portfolio_does_not_dereference_module():
    with pytest.raises(ValueError, match='not available'):
        validator(mcsampler_Portfolio_ok=False, mcsamplerPortfolio=None)(options(['AV']))


@pytest.mark.parametrize('method', ['BOGUS', 'custom'])
def test_missing_plugin_cannot_silently_use_original_sampler(method):
    with pytest.raises(ValueError, match='unknown or unavailable'):
        validator(mcsampler_Portfolio_ok=False, mcsamplerPortfolio=None)(options(method=method))


def test_explicit_original_sampler_remains_supported():
    validator(mcsampler_Portfolio_ok=False, mcsamplerPortfolio=None)(options(method='adaptive_cartesian'))


def test_member_arguments_cannot_be_silently_discarded():
    with pytest.raises(ValueError, match='one dictionary per portfolio member'):
        validator()(options(['AV,GMM'], args=['{}']))
    validator()(options(['AV,GMM'], args=['{}','{}']))


@pytest.mark.parametrize('args,message', [
    (['--sampler-method','portfolio'], 'requires at least one'),
    (['--sampler-method','portfolio','--sampler-portfolio','AV,BOGUS'], 'unknown member'),
    (['--sampler-method','BOGUS'], 'unknown or unavailable'),
    (['--sampler-method','portfolio','--sampler-portfolio','AV,GMM',
      '--sampler-portfolio-args','{}'], 'one dictionary per portfolio member'),
])
def test_cli_rejects_selection_before_requesting_event_or_data(args, message):
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH=str(DRIVER.parents[1]))
    p = subprocess.run([sys.executable,str(DRIVER),*args], env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 2, p.stdout+p.stderr
    assert message in p.stderr
    assert 'Fetching data' not in p.stdout
