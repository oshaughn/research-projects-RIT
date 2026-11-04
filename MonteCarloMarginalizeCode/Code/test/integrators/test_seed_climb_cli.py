"""Inherited ILE settings and setup guards honor ordinary argparse spellings."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

BIN = Path(__file__).resolve().parents[2] / "bin"


def module():
    spec = importlib.util.spec_from_file_location("seed_climb_cli", BIN / "util_TransverseSeedClimb.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_inherited_equals_and_separate_forms_match(tmp_path):
    separate = tmp_path / "separate"
    inline = tmp_path / "inline"
    separate.write_text('X --srate 1024 --fmax 448 --psd-file "H1=psd one.xml" '
                        '--psd-file L1=two.xml --channel-name H1=STRAIN '
                        '--internal-waveform-fd-L-frame --fmin-template 13.33')
    inline.write_text('X --srate=1024 --fmax=448 --psd-file="H1=psd one.xml" '
                      '--psd-file=L1=two.xml --channel-name=H1=STRAIN '
                      '--internal-waveform-fd-L-frame --fmin-template=13.33')
    mod = module()
    parsed = mod.parse_ile_args(str(inline))
    assert parsed == mod.parse_ile_args(str(separate))
    assert parsed['srate'] == 1024
    assert parsed['psd_file'] == ['H1=psd one.xml', 'L1=two.xml']


@pytest.mark.parametrize('text', ['X --srate', 'X --srate --fmax 448'])
def test_missing_inherited_value_fails(tmp_path, text):
    f = tmp_path / 'args'
    f.write_text(text)
    with pytest.raises(ValueError, match='requires a value'):
        module().parse_ile_args(str(f))


@pytest.mark.parametrize('option', ['--psd-file ', '--psd-file=', '-p ', '-p'])
def test_setup_rejects_missing_quoted_psd_for_both_spellings(tmp_path, monkeypatch, option):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'args_ile.txt').write_text('X ' + option + '"H1=missing psd.xml"')
    tree = ast.parse((BIN / 'util_RIFT_pseudo_pipe.py').read_text())
    block = next(n for n in tree.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "opts.transverse_seed_climb == 'setup'")
    import os
    monkeypatch.setattr(os, 'system', lambda command: pytest.fail('climb must not start with missing PSDs'))
    namespace = dict(opts=SimpleNamespace(transverse_seed_climb='setup',
                        transverse_seed_climb_args='', fake_data_cache=None),
                     _use_hpip_pp=False, os=os, shlex=__import__('shlex'))
    with pytest.raises(SystemExit, match='missing psd.xml'):
        exec(compile(ast.Module(body=[block], type_ignores=[]), '<setup>', 'exec'), namespace)


@pytest.mark.parametrize('attached', [False, True])
def test_true_ile_aliases_match_canonical_settings(tmp_path, attached):
    aliases = {'-c': ('cache', 'cache.dat'), '-C': ('channel-name', 'H1=STRAIN'),
               '-p': ('psd-file', 'H1=psd.xml'), '-t': ('event-time', '100'),
               '-s': ('data-start-time', '90'), '-e': ('data-end-time', '110'),
               '-w': ('window-shape', '0.2'), '-F': ('fmax', '128'),
               '-f': ('reference-freq', '35'), '-a': ('approx', 'IMRPhenomXPHM')}
    short = tmp_path / 'short'; canonical = tmp_path / 'canonical'
    short.write_text('X ' + ' '.join(k + ('' if attached else ' ') + v
                                   for k, (_, v) in aliases.items()))
    canonical.write_text('X ' + ' '.join('--' + k + ' ' + v for k, v in aliases.values()))
    assert module().parse_ile_args(str(short)) == module().parse_ile_args(str(canonical))
    full = tmp_path / 'full'
    full.write_text('X --cache-file=cache.dat --approximant=IMRPhenomXPHM')
    assert module().parse_ile_args(str(full)) == {'cache': 'cache.dat', 'approx': 'IMRPhenomXPHM'}
