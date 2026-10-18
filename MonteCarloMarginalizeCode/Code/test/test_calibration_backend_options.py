"""Pseudo-pipeline CLI bridge → calibration CLI assembly → real calmarg source.

The executable scripts need external detector data on import, so execute their
actual argument parsing/assembly AST blocks, then the real source function and
GWSignal wrapper. Only backend generation is intercepted.
"""
import argparse
import ast
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

pytest.importorskip('lal')
pytest.importorskip('lalsimulation.gwsignal')
pytest.importorskip('gwpy')
import numpy as np
from RIFT.calmarg import rift_source
from RIFT.physics import GWSignal

BIN = Path(__file__).resolve().parents[1] / 'bin'


def calibration_cli_from_manual(manual, use_gwsignal=True):
    tree = ast.parse((BIN / 'util_RIFT_pseudo_pipe.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.If)
                and isinstance(n.test, ast.Attribute) and n.test.attr == 'calibration_reweighting'
                and any(isinstance(c, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'my_extra_string'
                                                        for t in c.targets) for c in n.body))
    opts = SimpleNamespace(calibration_reweighting=True, use_gwsignal=use_gwsignal,
                           assume_eccentric=False, manual_extra_ile_args=manual, use_ini=False,
                           internal_mitigate_fd_J_frame='L_frame', calibration_reweighting_initial_extra_args=None)
    scope = {'opts': opts, 'cmd': '', 'argparse': argparse, 'shlex': shlex, 'ast': ast}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(BIN / 'util_RIFT_pseudo_pipe.py'), 'exec'), scope)
    token, = shlex.split(scope['cmd'])
    flag, value = token.split('=', 1)
    assert flag == '--calibration-reweighting-initial-extra-args'
    return shlex.split(value)


def calibration_assembled_options(argv):
    path = BIN / 'calibration_reweighting.py'
    tree = ast.parse(path.read_text())
    parser = argparse.ArgumentParser()
    parser.add = parser.add_argument
    wanted = {'--use-gwsignal', '--use-gwsignal-lmax-nyquist', '--extra-waveform-kwargs',
              '--internal-waveform-fd-L-frame', '--internal-waveform-fd-no-condition'}
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == 'add' and any(
                    isinstance(arg, ast.Constant) and arg.value in wanted for arg in call.args):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), {'parser': parser})
    args = parser.parse_args(argv)
    start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'extra_waveform_kwargs' for t in node.targets))
    end = next(i for i in range(start, len(tree.body)) if isinstance(tree.body[i], ast.Assign)
               and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                       and t.value.id == 'waveform_arguments' for t in tree.body[i].targets))
    scope = {'args': args, 'ast': ast, 'waveform_arguments': {}}
    exec(compile(ast.Module(body=tree.body[start:end+1], type_ignores=[]), str(path), 'exec'), scope)
    return args, scope['extra_waveform_kwargs']


@pytest.mark.parametrize('double_quoted', [False, True])
def test_full_calibration_bridge_preserves_generic_backend_options(monkeypatch, double_quoted):
    requested = {'antisymmetric_modes_hm': True, 'enable_antisymmetric_modes': True,
                 'lmax_nyquist': 0, 'future_false': False,
                 'future_nested': {'label': 'a -- b', 'quote': '"hello"', 'apostrophe': "it's fine"},
                 'fd_L_frame': 'future backend meaning', 'dA_dict': {'2,2': 0.1}}
    payload = repr(repr(requested)) if double_quoted else repr(requested)
    manual = shlex.join(['--internal-waveform-extra-kwargs', payload, '--unrelated-option', 'ignored'])
    argv = calibration_cli_from_manual(manual)
    args, extras = calibration_assembled_options(argv)
    assert args.use_gwsignal
    assert extras['fd_L_frame'] is True
    assert extras['gwsignal_extra_kwargs'] == requested

    class AtGenerator(Exception):
        pass

    def capture(parameters, generator):
        for key, value in requested.items():
            assert parameters[key] == value
        assert 'fd_alignment_postevent_time' not in parameters
        assert 'gwsignal_extra_kwargs' not in parameters
        raise AtGenerator

    monkeypatch.setattr(GWSignal.gws.models, 'gwsignal_get_waveform_generator', lambda name: name)
    monkeypatch.setattr(GWSignal.wfm, 'GenerateTDModes', capture)
    with pytest.raises(AtGenerator):
        rift_source.RIFT_lal_binary_black_hole(
            np.arange(1025) * 0.25, mass_1=60, mass_2=40, luminosity_distance=500,
            spin_1x=0.4, spin_1y=0.2, spin_1z=0.3,
            spin_2x=-0.2, spin_2y=0.1, spin_2z=-0.1, iota=0.4, phase=0,
            waveform_approximant='SEOBNRv5PHM', minimum_frequency=30,
            reference_frequency=30, sampling_frequency=2048,
            h_method='gws_hlmoft', extra_waveform_kwargs=extras)


def test_native_lal_bridge_keeps_lal_options_separate():
    manual = shlex.join(['--internal-waveform-extra-lalsuite-args', "{'PhenomXPrecVersion': 310}",
                        '--internal-waveform-extra-kwargs', "{'fd_L_frame': False}"])
    args, extras = calibration_assembled_options(calibration_cli_from_manual(manual, use_gwsignal=False))
    assert not args.use_gwsignal
    assert extras['extra_waveform_args'] == {'PhenomXPrecVersion': 310}
    assert extras['fd_L_frame'] is False
    assert 'gwsignal_extra_kwargs' not in extras


@pytest.mark.parametrize('payload', ['{}', "{'PhenomXPrecVersion': 310}"])
def test_gwsignal_bridge_rejects_lal_options(payload):
    manual = shlex.join(['--internal-waveform-extra-lalsuite-args', payload])
    with pytest.raises(SystemExit) as exc:
        calibration_cli_from_manual(manual)
    assert exc.value.code == 2


@pytest.mark.parametrize('payload', ['[]', '1', "'[]'"])
def test_invalid_calibration_option_dictionary_is_not_reinterpreted(payload):
    with pytest.raises(TypeError, match='must be a dictionary'):
        calibration_assembled_options(['--extra-waveform-kwargs', payload])


def test_direct_calibration_lmax_zero_is_retained():
    _, extras = calibration_assembled_options(['--use-gwsignal-lmax-nyquist', '0'])
    assert extras['lmax_nyquist'] == 0


@pytest.mark.parametrize("dictionary_lmax", [None, 1])
def test_manual_routing_and_owned_controls_match_ile(dictionary_lmax):
    requested = {"enable_antisymmetric_modes": True, "fd_L_frame": "backend meaning"}
    if dictionary_lmax is not None:
        requested["lmax_nyquist"] = dictionary_lmax
    manual = shlex.join(["--use-gwsignal", "--use-gwsignal-lmax-nyquist", "0",
                         "--internal-waveform-fd-L-frame", "--internal-waveform-fd-no-condition",
                         "--internal-waveform-extra-kwargs", repr(requested)])
    args, extras = calibration_assembled_options(calibration_cli_from_manual(manual, use_gwsignal=False))
    assert args.use_gwsignal
    assert extras["fd_L_frame"] is True
    assert extras["no_condition"] is True
    # The dictionary value wins in ILE (old: update() overwrites the flag; new: the explicit
    # dictionary wins), so calmarg's legacy slot carries it too, for older calibration workers.
    assert extras["lmax_nyquist"] == (0 if dictionary_lmax is None else dictionary_lmax)
    assert extras["gwsignal_extra_kwargs"] == requested
    effective = GWSignal._with_generator_options({}, extras)
    assert effective["lmax_nyquist"] == (0 if dictionary_lmax is None else dictionary_lmax)


def test_manual_gwsignal_route_rejects_lal_payload():
    manual = shlex.join(["--use-gwsignal", "--internal-waveform-extra-lalsuite-args", "{}"])
    with pytest.raises(SystemExit) as exc:
        calibration_cli_from_manual(manual, use_gwsignal=False)
    assert exc.value.code == 2


def test_native_route_does_not_turn_into_gwsignal_from_lmax():
    args, extras = calibration_assembled_options(calibration_cli_from_manual(
        "--use-gwsignal-lmax-nyquist 0", use_gwsignal=False))
    assert not args.use_gwsignal
    assert args.use_gwsignal_lmax_nyquist is None
    assert extras["lmax_nyquist"] == 0


def test_native_highlevel_extra_waveform_args_keeps_existing_precedence():
    manual = shlex.join(["--internal-waveform-extra-lalsuite-args", "{'PhenomXPrecVersion': 310}",
                         "--internal-waveform-extra-kwargs", "{'extra_waveform_args': {'PhenomXPrecVersion': 223}}"])
    _, extras = calibration_assembled_options(calibration_cli_from_manual(manual, use_gwsignal=False))
    assert extras["extra_waveform_args"] == {"PhenomXPrecVersion": 223}



def test_manual_dictionary_expressions_keep_existing_ile_compatibility():
    manual = shlex.join(["--internal-waveform-extra-kwargs", "{'dA_dict': {'2,2': 1/10}}"])
    _, extras = calibration_assembled_options(calibration_cli_from_manual(manual))
    assert extras["gwsignal_extra_kwargs"]["dA_dict"] == {"2,2": 0.1}


@pytest.mark.parametrize('text,expected', [
    # Calib_reweight.sub argv written by submit hosts with older RIFT (O4c 0.0.17.9 production): the key quotes
    # are lost on the way through Condor (rift-v5PHM-calmarg-A, S241114bi, test_o4c_release).
    ('"{antisymmetric_modes_hm: True, enable_antisymmetric_modes: True}"',
     {'antisymmetric_modes_hm': True, 'enable_antisymmetric_modes': True}),
    ('{antisymmetric_modes_hm: True}', {'antisymmetric_modes_hm': True}),
    ('"{PhenomXHMReleaseVersion: 122022, label: a b}"',
     {'PhenomXHMReleaseVersion': 122022, 'label': 'a b'}),
])
def test_existing_dag_unquoted_dictionary_still_parses(text, expected):
    args, extras = calibration_assembled_options(
        ['--use-gwsignal', '--use-gwsignal-lmax-nyquist', '1', '--extra-waveform-kwargs', text])
    assert args.use_gwsignal
    assert extras['lmax_nyquist'] == 1
    assert {k: extras[k] for k in expected} == expected
    assert all(type(extras[k]) is type(v) for k, v in expected.items())


def test_lmax_nyquist_flag_kept_for_older_calibration_containers():
    """Older calibration workers read lmax_nyquist only from the dedicated flag."""
    manual = shlex.join(['--internal-waveform-extra-kwargs',
                         "{'antisymmetric_modes_hm': True, 'lmax_nyquist': 1}"])
    argv = calibration_cli_from_manual(manual)
    assert argv[argv.index('--use-gwsignal-lmax-nyquist') + 1] == '1'
    args, extras = calibration_assembled_options(argv)
    assert extras['lmax_nyquist'] == 1
    assert extras['gwsignal_extra_kwargs'] == {'antisymmetric_modes_hm': True, 'lmax_nyquist': 1}
