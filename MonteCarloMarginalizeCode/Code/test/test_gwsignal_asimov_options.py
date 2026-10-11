"""Render the actual Asimov policy and follow executable argument handling.

The driver imports load data, so extract their actual narrow option-handling
statements; invoke the real submit writer, likelihood and waveform wrappers.
No model of RIFT's shell quoting is used: inspect the emitted Condor arguments.
"""
import argparse
import ast
import configparser
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest
liquid = pytest.importorskip('liquid')
pytest.importorskip('glue.pipeline')
from RIFT.misc import dag_utils
from test_gwsignal_ile_options import assembled_options, lal, lsu, fl, rgws

CODE = Path(__file__).resolve().parents[1]


def render_manual_options(waveform, manual=None):
    source = (CODE / 'RIFT/asimov/rift.ini').read_text()
    start = source.index('# Keep the waveform option namespaces')
    end = source.index('#{%- if sampler', start)
    ile = {} if manual is None else {'manual extra args': manual}
    text = liquid.Liquid(source[start:end], from_file=False).render(
        waveform=waveform, sampler={'ile': ile})
    parser = configparser.RawConfigParser()
    parser.read_string('[rift-pseudo-pipe]\n' + text)
    return parser


def pseudo_ile_line(config, use_gwsignal, tmp_path):
    tree = ast.parse((CODE / 'bin/util_RIFT_pseudo_pipe.py').read_text())
    parser = argparse.ArgumentParser()
    flags = {'--use-gwsignal', '--approx', '--manual-extra-ile-args'}
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == 'add_argument' and any(
                    isinstance(x, ast.Constant) and x.value in flags for x in call.args):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<pseudo parser>', 'exec'), {'parser': parser})
    # The scheduler.pipeline forwarding loop used by Rift.build_dag.
    asimov = ast.parse((CODE / 'RIFT/asimov/rift.py').read_text())
    loop = next(node for node in ast.walk(asimov) if isinstance(node, ast.For)
                and ast.unparse(node.target) == 'key'
                and "['pipeline']" in ast.unparse(node.iter))
    production = SimpleNamespace(meta={'scheduler': {'pipeline': {'use-gwsignal': use_gwsignal}}})
    scope = {'self': SimpleNamespace(production=production), 'command': []}
    exec(compile(ast.Module(body=[loop], type_ignores=[]), '<Asimov build>', 'exec'), scope)
    opts = parser.parse_args(scope['command'] + ['--approx', 'SEOBNRv5PHM' if use_gwsignal else 'IMRPhenomXPHM'])
    # Execute the exact pseudo-pipe INI assignment loop, preserving manual text.
    ini_loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and ast.unparse(node.target) == 'item'
                    and ast.unparse(node.iter) == 'rift_items')
    scope = {'config_dict': vars(opts), 'rift_items': dict(config['rift-pseudo-pipe'])}
    exec(compile(ast.Module(body=[ini_loop], type_ignores=[]), '<pseudo INI>', 'exec'), scope)
    # Execute actual generator selection and manual-option append statements.
    scope = {'opts': opts, 'line': 'integrate_likelihood_extrinsic_batchmode', 'sys': __import__('sys')}
    selected = next(node for node in tree.body if isinstance(node, ast.If)
                    and ast.unparse(node.test) == 'opts.use_gwsignal'
                    and 'line +=' in ast.unparse(node))
    manual = next(node for node in tree.body if isinstance(node, ast.If)
                  and 'opts.manual_extra_ile_args is None' in ast.unparse(node.test))
    exec(compile(ast.Module(body=[selected, manual], type_ignores=[]), '<pseudo args_ile>', 'exec'), scope)
    args_path = tmp_path / 'args_ile.txt'
    args_path.write_text(scope['line'])
    # Actual BasicIteration file-reading/normalization statements.
    basic = ast.parse((CODE / 'bin/create_event_parameter_pipeline_BasicIteration').read_text())
    start = next(i for i, n in enumerate(basic.body) if isinstance(n, ast.With)
                 and 'open(opts.ile_args)' in ast.unparse(n))
    end = next(i for i in range(start, len(basic.body)) if isinstance(basic.body[i], ast.If))
    scope = {'opts': SimpleNamespace(ile_args=str(args_path))}
    exec(compile(ast.Module(body=basic.body[start:end], type_ignores=[]), '<BasicIteration args>', 'exec'), scope)
    return scope['ile_args']


def condor_argv(text):
    """Decode HTCondor new-syntax arguments independently of RIFT's encoder."""
    assert text.startswith('"') and text.endswith('"')
    text = text[1:-1].replace('""', '"')
    argv, token, quoted, active = [], '', False, False
    i = 0
    while i < len(text):
        char = text[i]
        if char == "'":
            if quoted and i + 1 < len(text) and text[i + 1] == "'":
                token += "'"
                i += 2
                continue
            quoted = not quoted
            active = True
        elif char.isspace() and not quoted:
            if active:
                argv.append(token)
                token, active = '', False
        else:
            token += char
            active = True
        i += 1
    assert not quoted
    if active:
        argv.append(token)
    return argv


def emit_worker_argv(line, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job, name = dag_utils.write_ILE_sub_simple(tag='ILE', arg_str=line, exe='/bin/true')
    job.write_sub_file()
    arguments = next(line.partition('=')[2].strip() for line in Path(name).read_text().splitlines()
                     if line.startswith('arguments ='))
    argv = condor_argv(arguments)
    # Automatically injected Condor event macro is not needed by narrow parser.
    argv = argv[:argv.index('--event')]
    return argv


@pytest.mark.parametrize('manual', [['--internal-waveform-fd-L-frame'], '--internal-waveform-fd-L-frame'])
def test_asimov_to_condor_to_generator_preserves_generic_options(monkeypatch, tmp_path, manual):
    requested = {'enable_antisymmetric_modes': True, 'antisymmetric_modes_hm': True,
                 'lmax_nyquist': 1, 'future_false': False, 'future_zero': 0,
                 'dA_dict': {'2,2': 0.1}, 'future_nested': {'modes': [(2, 2), (3, 3)],
                 'text': 'a "quoted" value\\path--option'}}
    config = render_manual_options({'gwsignal arguments': requested}, manual)
    line = pseudo_ile_line(config, True, tmp_path)
    opts, extras = assembled_options(emit_worker_argv(line, tmp_path, monkeypatch))
    assert extras['gwsignal_extra_kwargs'] == requested
    class Reached(Exception):
        pass
    def capture(params, gen):
        for key, value in requested.items():
            assert params[key] == value
        assert 'fd_L_frame' not in params
        raise Reached
    monkeypatch.setattr(rgws.wfm, 'GenerateTDModes', capture)
    monkeypatch.setattr(rgws.gws.models, 'gwsignal_get_waveform_generator', lambda name: name)
    monkeypatch.setattr(fl, 'has_GWS', True)
    monkeypatch.setattr(fl, 'rgws', rgws, raising=False)
    with pytest.raises(Reached):
        fl.internal_hlm_generator(lsu.ChooseWaveformParams(), 4, quiet=True,
                                  use_gwsignal=opts.use_gwsignal, use_gwsignal_approx=opts.approximant,
                                  extra_waveform_kwargs=extras)


def test_asimov_lalsuite_dictionary_reaches_actual_lal_dict(monkeypatch, tmp_path):
    config = render_manual_options({'arguments': {'PhenomXPrecVersion': 310}})
    line = pseudo_ile_line(config, False, tmp_path)
    opts, extras = assembled_options(emit_worker_argv(line, tmp_path, monkeypatch))
    assert not opts.use_gwsignal
    class Reached(Exception):
        pass
    def capture(*args):
        assert lsu.lalsim.SimInspiralWaveformParamsLookupPhenomXPrecVersion(args[-2]) == 310
        raise Reached
    monkeypatch.setattr(lsu.lalsim, 'SimInspiralChooseFDModes', capture)
    p = lsu.ChooseWaveformParams()
    p.approx = lsu.lalsim.IMRPhenomXPHM
    p.deltaF = 1 / 4
    with pytest.raises(Reached):
        fl.internal_hlm_generator(p, 4, quiet=True, use_gwsignal=False, extra_waveform_kwargs=extras)


def test_asimov_both_namespaces_are_not_silently_prioritized(monkeypatch, tmp_path):
    config = render_manual_options({'gwsignal arguments': {'enable_antisymmetric_modes': True},
                                    'arguments': {'PhenomXPrecVersion': 310}})
    line = pseudo_ile_line(config, True, tmp_path)
    with pytest.raises(SystemExit) as exc:
        assembled_options(emit_worker_argv(line, tmp_path, monkeypatch))
    assert exc.value.code == 2


def test_asimov_omitted_options_keep_plain_worker_arguments(monkeypatch, tmp_path):
    config = render_manual_options({})
    assert not config.has_option('rift-pseudo-pipe', 'manual-extra-ile-args')
    line = pseudo_ile_line(config, True, tmp_path)
    assert emit_worker_argv(line, tmp_path, monkeypatch) == ['--use-gwsignal', '--approx', 'SEOBNRv5PHM']


def test_real_basiciteration_dag_build_preserves_asimov_options(monkeypatch, tmp_path):
    import os
    import subprocess
    import sys
    requested = {'enable_antisymmetric_modes': True, 'antisymmetric_modes_hm': True,
                 'lmax_nyquist': 1, 'future_option': {'modes': [(2, 2)],
                 'text': 'quote" and backslash\\ and --option', 'false': False}}
    config = render_manual_options({'gwsignal arguments': requested}, '--internal-waveform-fd-L-frame')
    # Materialize real rendered INI and pseudo-pipe args as the Asimov route does.
    with (tmp_path / 'production.ini').open('w') as stream:
        config.write(stream)
    pseudo_ile_line(config, True, tmp_path)
    p = lsu.ChooseWaveformParams()
    p.m1, p.m2 = 60 * lal.MSUN_SI, 40 * lal.MSUN_SI
    p.fmin = p.fref = 30
    lsu.ChooseWaveformParams_array_to_xml([p], fname=str(tmp_path / 'input-grid'))
    (tmp_path / 'args_cip.txt').write_text('X --parameter m1 --parameter m2 --n-output-samples 10')
    env = dict(os.environ)
    env['PATH'] = str(CODE / 'bin') + os.pathsep + env['PATH']
    env['PYTHONPATH'] = str(CODE)
    env['GW_SURROGATE'] = ''
    command = [sys.executable, str(CODE / 'bin/create_event_parameter_pipeline_BasicIteration'),
               '--working-directory', str(tmp_path), '--input-grid', str(tmp_path / 'input-grid.xml.gz'),
               '--ile-args', str(tmp_path / 'args_ile.txt'), '--cip-args', str(tmp_path / 'args_cip.txt'),
               '--ile-exe', str(CODE / 'bin/integrate_likelihood_extrinsic_batchmode'),
               '--n-iterations', '1', '--n-samples-per-job', '1', '--n-copies', '1']
    built = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stdout + '\n' + built.stderr
    assert (tmp_path / 'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag').is_file()
    submission = (tmp_path / 'ILE.sub').read_text()
    arguments = next(line.partition('=')[2].strip() for line in submission.splitlines()
                     if line.startswith('arguments ='))
    argv = condor_argv(arguments)
    # Parse only the waveform arguments; retain the complete .sub for external
    # condor_submit -dry-run validation without launching parameter estimation.
    end = argv.index('--sim-xml')
    _, extras = assembled_options(argv[:end])
    assert extras['gwsignal_extra_kwargs'] == requested
