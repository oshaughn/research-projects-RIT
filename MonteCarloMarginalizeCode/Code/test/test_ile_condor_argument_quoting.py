"""Shell argument boundaries survive the real ILE Condor submit writer."""
import shlex
import json
import subprocess
import sys

import pytest

pytest.importorskip("glue.pipeline")
pytest.importorskip("numpy")
from RIFT.misc import dag_utils, dag_utils_generic


def decode_condor_new_arguments(value):
    """Decode documented new-syntax quoting from an actual arguments attribute.

    Condor quotes the whole value in double quotes, doubles embedded double
    quotes, groups argv with single quotes, and doubles literal single quotes
    inside those groups. This is intentionally independent of shell shlex.
    It does not expand submit macros or submit jobs.
    """
    value = value.strip()
    assert value.startswith('"') and value.endswith('"')
    value = value[1:-1].replace('""', '"')
    argv, token = [], []
    quoted = started = False
    i = 0
    while i < len(value):
        char = value[i]
        if char == "'":
            if quoted and i + 1 < len(value) and value[i + 1] == "'":
                token.append("'")
                i += 2
                continue
            quoted = not quoted
            started = True
        elif char.isspace() and not quoted:
            if started:
                argv.append(''.join(token))
                token = []
                started = False
        else:
            token.append(char)
            started = True
        i += 1
    assert not quoted, "unbalanced Condor quote groups"
    if started:
        argv.append(''.join(token))
    return argv


PAYLOAD = {'antisymmetric_modes_hm': True, 'enable_antisymmetric_modes': True,
           'lmax_nyquist': 1, 'future_option': {'label': 'a -- b', 'quote': '"x"',
                                                'apostrophe': "it's a value"}}
CASES = [
    '--use-gwsignal --approx SEOBNRv5PHM --internal-waveform-extra-kwargs '
    '"{\'antisymmetric_modes_hm\': True, \'enable_antisymmetric_modes\': True, \'lmax_nyquist\': 1}"',
    shlex.join(['--use-gwsignal', '--internal-waveform-extra-kwargs', repr(PAYLOAD),
                '--label', 'second quoted value', '--empty', '', '--event', '$(macroevent)']),
    shlex.join(['--internal-waveform-extra-lalsuite-args', "{'PhenomXPrecVersion': 310}",
                '--internal-waveform-extra-kwargs', "{'fd_L_frame': False}"]),
    shlex.join(['--use-gwsignal', '--internal-waveform-extra-kwargs', repr(repr(PAYLOAD))]),
    '--approx IMRPhenomXPHM --event $(macroevent) --n-events-to-analyze 1',
    shlex.join(['--label', 'a\tb c', '--path', '/a path/with spaces']),
]


@pytest.mark.parametrize('command', CASES)
@pytest.mark.parametrize('writer', [dag_utils, dag_utils_generic])
def test_real_submit_writer_preserves_every_argument(tmp_path, monkeypatch, command, writer):
    monkeypatch.chdir(tmp_path)
    job, filename = writer.write_ILE_sub_simple(
        exe='/bin/true', tag='test', log_dir=str(tmp_path), arg_str=command)
    job.write_sub_file()
    arguments = next(line.partition('=')[2].strip()
                     for line in (tmp_path / filename).read_text().splitlines()
                     if line.startswith('arguments ='))
    event_args = ['--event=$(macroevent)'] if writer is dag_utils_generic else ['--event', '$(macroevent)']
    assert decode_condor_new_arguments(arguments) == shlex.split(command) + event_args


def test_unquoted_helper_returns_text_and_empty_values_survive():
    assert dag_utils.safely_quote_arg_str('--x 0') == '--x 0'
    assert decode_condor_new_arguments('"' + dag_utils.quote_arguments(['', 'a\tb']) + '"') == ['', 'a\tb']


@pytest.mark.parametrize("value", ["a\nb", "a\rb"])
def test_literal_submit_line_breaks_are_rejected(value):
    with pytest.raises(ValueError, match="literal line breaks"):
        dag_utils.quote_arguments([value])


@pytest.mark.parametrize('command', CASES)
def test_generic_slurm_script_preserves_literal_argv_without_condor_escaping(tmp_path, command):
    recorder = tmp_path / 'record_argv.py'
    recorded = tmp_path / 'argv.json'
    recorder.write_text("#!" + sys.executable + "\nimport json,sys\nfrom pathlib import Path\n"
                        "Path(" + repr(str(recorded)) + ").write_text(json.dumps(sys.argv[1:]))\n")
    recorder.chmod(0o755)
    job, _ = dag_utils_generic.write_ILE_sub_simple(
        exe=str(recorder), tag=str(tmp_path / 'ILE'), log_dir=str(tmp_path), arg_str=command)
    script = tmp_path / 'ILE.sbatch'
    dag_utils_generic.SlurmBackend().emit_job(job, str(script))
    completed = subprocess.run(['bash', str(script)], capture_output=True, text=True,
                               env={'PATH': '/usr/bin:/bin', 'SLURM_VAR_EVENT': '7'}, timeout=10)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(recorded.read_text()) == shlex.split(command) + ['--event=7']


@pytest.mark.parametrize('command', CASES)
def test_explicit_generic_glue_emission_preserves_literal_argv(tmp_path, command):
    job, _ = dag_utils_generic.write_ILE_sub_simple(
        exe='/bin/true', tag=str(tmp_path / 'ILE'), log_dir=str(tmp_path), arg_str=command)
    submit = tmp_path / 'ILE_glue.sub'
    dag_utils_generic.GluePipelineBackend().emit_job(job, str(submit))
    arguments = next(line.partition('=')[2].strip() for line in submit.read_text().splitlines()
                     if line.startswith('arguments ='))
    assert decode_condor_new_arguments(arguments) == shlex.split(command) + ['--event', '$(macroevent)']
