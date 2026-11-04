"""Production CLI option definitions and assembly through the real likelihood route.

The standalone ILE requires detector data at import time. Extract only its
actual option definitions and waveform-option assembly; then call the installed
factored_likelihood and GWSignal functions, intercepting the generator boundary.
Full ILE execution against production container dependencies is a separate gate.
"""
import ast
from optparse import OptionParser
from pathlib import Path
import shlex

import pytest

lal = pytest.importorskip('lal')
pytest.importorskip('astropy')
pytest.importorskip('gwpy')
pytest.importorskip('lalsimulation.gwsignal')
from RIFT import lalsimutils as lsu
from RIFT.likelihood import factored_likelihood as fl
from RIFT.physics import GWSignal as rgws

ILE = Path(__file__).resolve().parents[1] / 'bin' / 'integrate_likelihood_extrinsic_batchmode'
FLAGS = {'--gwsignal-mode-conditioning', '--use-gwsignal', '--approximant', '--internal-waveform-extra-kwargs',
         '--internal-waveform-fd-L-frame', '--internal-waveform-fd-no-condition',
         '--use-gwsignal-lmax-nyquist', '--internal-waveform-extra-lalsuite-args',
         '--rom-group', '--e-freq'}


def assembled_options(argv):
    tree = ast.parse(ILE.read_text())
    parser = OptionParser()
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != 'add_option':
            continue
        if any(isinstance(arg, ast.Constant) and arg.value in FLAGS for arg in call.args):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(ILE), 'exec'), {'optp': parser})
    opts, _ = parser.parse_args(argv)
    scope = {'opts': opts, 'optp': parser}
    guard = next(node for node in tree.body if isinstance(node, ast.If)
                 and 'internal_waveform_extra_lalsuite_args' in ast.unparse(node.test)
                 and 'use_gwsignal' in ast.unparse(node.test))
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(ILE), 'exec'), scope)
    for guard in tree.body:
        if isinstance(guard, ast.If) and 'gwsignal_mode_conditioning' in ast.unparse(guard.test):
            exec(compile(ast.Module(body=[guard], type_ignores=[]), str(ILE), 'exec'), scope)
    # Optional wrappers not requested by the production command.
    for key, default in [('rom_group', None), ('e_freq', 0),
                         ('internal_waveform_extra_lalsuite_args', None)]:
        if not hasattr(opts, key):
            setattr(opts, key, default)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'analyze_event')
    start = next(i for i, node in enumerate(fn.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'extra_waveform_kwargs' for t in node.targets))
    end = next(i for i in range(start, len(fn.body)) if isinstance(fn.body[i], ast.If)
               and isinstance(fn.body[i].test, ast.Attribute)
               and fn.body[i].test.attr == 'internal_waveform_extra_kwargs')
    scope = {'opts': opts}
    exec(compile(ast.Module(body=fn.body[start:end+1], type_ignores=[]), str(ILE), 'exec'), scope)
    return opts, scope['extra_waveform_kwargs']


@pytest.mark.parametrize('double_quoted', [False, True])
@pytest.mark.parametrize('direct_lmax', [None, 0, 5])
def test_production_cli_reaches_real_likelihood_generator(monkeypatch, double_quoted, direct_lmax):
    dictionary = "{'antisymmetric_modes_hm': True, 'enable_antisymmetric_modes': True, 'lmax_nyquist': 1}"
    value = repr(dictionary) if double_quoted else dictionary
    argv = shlex.split('--use-gwsignal --approx SEOBNRv5PHM --internal-waveform-fd-L-frame')
    argv += ['--internal-waveform-extra-kwargs', value]
    if direct_lmax is not None:
        argv += ['--use-gwsignal-lmax-nyquist', str(direct_lmax)]
    opts, extras = assembled_options(argv)
    assert extras['fd_L_frame'] is True
    if direct_lmax is not None:
        assert extras['lmax_nyquist'] == direct_lmax
    captured = {}
    class AtGenerator(Exception):
        pass
    def capture(params, generator):
        captured.update(params)
        raise AtGenerator
    monkeypatch.setattr(rgws.wfm, 'GenerateTDModes', capture)
    monkeypatch.setattr(rgws.gws.models, 'gwsignal_get_waveform_generator', lambda name: name)
    monkeypatch.setattr(fl, 'has_GWS', True)
    monkeypatch.setattr(fl, 'rgws', rgws, raising=False)
    p = lsu.ChooseWaveformParams()
    p.m1, p.m2 = 60 * lal.MSUN_SI, 40 * lal.MSUN_SI
    p.fmin = p.fref = 30
    p.deltaT, p.deltaF = 1 / 2048, 1 / 4
    p.dist = 500e6 * lal.PC_SI
    with pytest.raises(AtGenerator):
        fl.internal_hlm_generator(p, 4, use_gwsignal=opts.use_gwsignal,
                                  use_gwsignal_approx=opts.approximant,
                                  extra_waveform_kwargs=extras, quiet=True)
    assert captured['antisymmetric_modes_hm'] is True
    assert captured['enable_antisymmetric_modes'] is True
    assert captured['lmax_nyquist'] == 1  # explicit dictionary wins
    assert not {'fd_L_frame', 'e_freq', 'fd_alignment_postevent_time'} & captured.keys()


def test_non_gwsignal_dictionary_remains_legacy_kwargs():
    opts, extras = assembled_options(['--internal-waveform-extra-kwargs', "{'fd_L_frame': False, 'lmax_nyquist': 0}"])
    assert not opts.use_gwsignal
    assert extras['fd_L_frame'] is False
    assert extras['lmax_nyquist'] == 0
    assert 'gwsignal_extra_kwargs' not in extras


@pytest.mark.parametrize('value', ['[]', '1', "'[]'"])
def test_cli_non_dictionary_rejected(value):
    with pytest.raises(TypeError, match="must be a dictionary"):
        assembled_options(['--use-gwsignal', '--internal-waveform-extra-kwargs', value])


@pytest.mark.parametrize('requested', [None, {
    'future_nested_option': {'modes': [(2, 2)], 'switch': False},
    'future_false_option': False, 'future_zero_option': 0,
    'dA_dict': {'2,2': 0.1, '3,3': 0.0},
    'fd_L_frame': 'backend meaning', 'internal_phase_shift': 'backend meaning'}])
def test_cli_zero_and_future_namespace_options_reach_generator(monkeypatch, requested):
    argv = ['--use-gwsignal', '--approx', 'SEOBNRv5PHM',
            '--use-gwsignal-lmax-nyquist', '0', '--internal-waveform-fd-L-frame']
    if requested is not None:
        argv += ['--internal-waveform-extra-kwargs', repr(requested)]
    opts, extras = assembled_options(argv)
    class AtGenerator(Exception):
        pass
    def capture(params, generator):
        assert params['lmax_nyquist'] == 0
        if requested is not None:
            for key, value in requested.items():
                assert params[key] == value
        else:
            assert 'fd_L_frame' not in params
        raise AtGenerator
    monkeypatch.setattr(rgws.wfm, 'GenerateTDModes', capture)
    monkeypatch.setattr(rgws.gws.models, 'gwsignal_get_waveform_generator', lambda name: name)
    monkeypatch.setattr(fl, 'has_GWS', True)
    monkeypatch.setattr(fl, 'rgws', rgws, raising=False)
    p = lsu.ChooseWaveformParams()
    p.m1, p.m2 = 60 * lal.MSUN_SI, 40 * lal.MSUN_SI
    p.fmin = p.fref = 30
    p.deltaT, p.deltaF = 1 / 2048, 1 / 4
    p.dist = 500e6 * lal.PC_SI
    with pytest.raises(AtGenerator):
        fl.internal_hlm_generator(p, 4, use_gwsignal=opts.use_gwsignal,
                                  use_gwsignal_approx=opts.approximant,
                                  extra_waveform_kwargs=extras, quiet=True)


def test_non_gwsignal_lalsuite_options_reach_lal_dict_builder(monkeypatch):
    _, extras = assembled_options([
        '--internal-waveform-extra-lalsuite-args', "{'PhenomXPrecVersion': 310}",
        '--internal-waveform-extra-kwargs', "{'fd_L_frame': False}"])
    captured = {}
    class AtLALDictionary(Exception):
        pass
    def capture_lalsuite(*args):
        lal_dict = args[-2]
        captured['PhenomXPrecVersion'] = lsu.lalsim.SimInspiralWaveformParamsLookupPhenomXPrecVersion(lal_dict)
        raise AtLALDictionary
    monkeypatch.setattr(lsu.lalsim, 'SimInspiralChooseFDModes', capture_lalsuite)
    p = lsu.ChooseWaveformParams()
    p.approx = lsu.lalsim.IMRPhenomXPHM
    p.m1, p.m2 = 60 * lal.MSUN_SI, 40 * lal.MSUN_SI
    p.fmin = p.fref = 30
    p.deltaT, p.deltaF = 1 / 2048, 1 / 4
    p.dist = 500e6 * lal.PC_SI
    with pytest.raises(AtLALDictionary):
        fl.internal_hlm_generator(p, 4, use_gwsignal=False,
                                  extra_waveform_kwargs=extras, quiet=True)
    assert captured == {'PhenomXPrecVersion': 310}


@pytest.mark.parametrize('payload', ["{'PhenomXPrecVersion': 310}", '{}'])
def test_cli_incompatible_lalsuite_options_fail_before_data_loading(payload, capsys):
    with pytest.raises(SystemExit) as exc:
        assembled_options(['--use-gwsignal', '--internal-waveform-extra-lalsuite-args', payload,
                           '--internal-waveform-extra-kwargs', "{'enable_antisymmetric_modes': True}"])
    assert exc.value.code == 2
    assert 'cannot be used with --use-gwsignal' in capsys.readouterr().err


def test_missing_gwsignal_does_not_fall_back_or_mutate_parameters(monkeypatch):
    monkeypatch.setattr(fl, 'has_GWS', False)
    monkeypatch.setattr(lsu, 'std_and_conj_hlmoff', lambda *a, **kw: pytest.fail('wrong backend fallback'))
    p = lsu.ChooseWaveformParams()
    p.taper = 0
    with pytest.raises(RuntimeError, match='refusing to fall back'):
        fl.internal_hlm_generator(p, 4, use_gwsignal=True, quiet=True)
    assert p.taper == 0


@pytest.mark.parametrize('policy', ['legacy', 'native'])
def test_mode_conditioning_selector_reaches_wrapper_not_backend(monkeypatch, policy):
    opts, extras = assembled_options(['--use-gwsignal', '--approx', 'SEOBNRv5PHM',
                                     '--gwsignal-mode-conditioning', policy])
    assert extras['gwsignal_mode_conditioning'] == policy
    p = lsu.ChooseWaveformParams()
    p.m1, p.m2 = 60 * lal.MSUN_SI, 30 * lal.MSUN_SI
    p.fmin = p.fref = 20.
    p.taper = rgws.lalsim.SIM_INSPIRAL_TAPER_START
    captured = {}
    class AtGenerator(Exception):
        pass
    def capture(params, generator):
        captured.update(params)
        raise AtGenerator
    monkeypatch.setattr(rgws.wfm, 'GenerateTDModes', capture)
    monkeypatch.setattr(rgws.gws.models, 'gwsignal_get_waveform_generator', lambda _: object())
    with pytest.raises(AtGenerator):
        rgws.hlmoft(p, approx_string=opts.approximant, **extras)
    assert 'gwsignal_mode_conditioning' not in captured
    import astropy.units as u
    start = captured['f22_start'].to_value(u.Hz)
    assert (start < p.fmin) if policy == 'native' else (start == p.fmin)


def test_mode_conditioning_defaults_and_invalid_route():
    _, extras = assembled_options(['--use-gwsignal'])
    assert extras['gwsignal_mode_conditioning'] == 'legacy'
    with pytest.raises(SystemExit):
        assembled_options(['--gwsignal-mode-conditioning', 'native'])
    with pytest.raises(SystemExit):
        assembled_options(['--use-gwsignal', '--gwsignal-mode-conditioning', 'other'])

    with pytest.raises(SystemExit):
        assembled_options(['--use-gwsignal', '--gwsignal-mode-conditioning', 'native',
                           '--internal-waveform-fd-no-condition'])
