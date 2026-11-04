"""ILE's SHORT option spellings must mean the same thing in the JAX driver.

Drop-in compatibility is an argv-level contract, and it was long-option only:
integrate_likelihood_extrinsic_jax defined NO short options at all, so a
production command line written with ILE's short forms (-o, -S, -P, -c, -t, ...)
died at optparse before any of the long-option compatibility could help it.

The parity check below reads both parsers rather than a hand-written list, so a
short form added to ILE cannot quietly go missing here.

Run:
  PYTHONPATH=<...>/Code  python -m pytest -q test/jax/test_jax_ile_short_option_forms.py
"""

import importlib.machinery
import importlib.util
import optparse
import os
import re

import pytest


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JAX_DRIVER = os.path.join(CODE_DIR, "bin", "integrate_likelihood_extrinsic_jax")
ILE_DRIVER = os.path.join(CODE_DIR, "bin", "integrate_likelihood_extrinsic_batchmode")


def load_driver():
    loader = importlib.machinery.SourceFileLoader("_ile_jax_driver_shortopts",
                                                  JAX_DRIVER)
    spec = importlib.util.spec_from_loader("_ile_jax_driver_shortopts", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


drv = load_driver()
PARSER = drv.build_parser()


def ile_short_forms():
    """ILE's (short, long) pairs, read out of its source.

    integrate_likelihood_extrinsic_batchmode builds its parser at import and
    then runs, so it cannot be imported for this; the option lines are
    unambiguous enough to read directly, and the count is asserted below so a
    regex that stops matching fails loudly instead of passing on nothing.
    """
    pattern = re.compile(r'optp\.add_option\(\s*"(-[A-Za-z])"\s*,\s*"(--[-a-zA-Z0-9]+)"')
    with open(ILE_DRIVER) as f:
        return dict((m.group(1), m.group(2)) for m in pattern.finditer(f.read()))


ILE_SHORTS = ile_short_forms()


def test_the_reference_parse_found_ile_s_short_forms():
    # A regex that silently matched nothing would make every test below vacuous.
    assert len(ILE_SHORTS) >= 20, ILE_SHORTS
    assert ILE_SHORTS["-o"] == "--output-file"
    assert ILE_SHORTS["-S"] == "--save-samples"
    assert ILE_SHORTS["-P"] == "--save-P"


@pytest.mark.parametrize("short", sorted(ILE_SHORTS))
def test_every_ile_short_form_resolves_to_the_same_long_option(short):
    opt = PARSER._short_opt.get(short)
    assert opt is not None, "%s (%s in ILE) is not accepted" % (short, ILE_SHORTS[short])
    assert ILE_SHORTS[short] in opt._long_opts, (
        "%s means %s here and %s in ILE" % (short, opt._long_opts, ILE_SHORTS[short]))


def test_the_driver_s_table_is_not_a_superset_that_invents_meanings():
    """A short form here that ILE does not define would be a new interface."""
    extra = {s: o for s, o in drv._ILE_SHORT_FORMS.items() if s not in ILE_SHORTS}
    assert not extra, extra


def test_the_extrinsic_stage_command_line_parses_in_short_form():
    """The options create_event_parameter_pipeline_BasicIteration appends.

    Its ILE_extr job writes "--save-P 0.01 --save-samples ... --output-file=..."
    in long form; a hand-written equivalent uses -P/-S/-o and must reach the same
    options.
    """
    opts, _ = PARSER.parse_args(
        ["-P", "0.01", "-S", "-o", "EXTR_out-0.xml", "-E", "3",
         "-t", "1000000014.0", "--sampler-method", "portfolio", "--n-eff", "20"])
    assert opts.save_samples is True
    assert opts.output_file == "EXTR_out-0.xml"
    assert int(opts.event) == 3
    assert float(opts.event_time) == 1000000014.0


def test_save_P_is_accepted_and_reported_as_ignored_when_passed(capsys):
    """It governs ILE's output VOLUME (igrand_threshold_p), never the estimate.

    So it is reported, not refused, which is this driver's stated rule.  Passing
    it must still be visible: a knob that evaporates without a word is the
    failure the compat layer exists to prevent.
    """
    assert "--save-P" in drv._ILE_ALL_OPTS
    parser = drv.build_parser()
    argv = ["-P", "0.01", "-S", "-o", "EXTR_out-0.xml", "-t", "1000000014.0",
            "--sampler-method", "portfolio", "--n-eff", "20"]
    opts, _ = parser.parse_args(argv)
    drv.record_supplied_options(opts, argv, parser)
    drv.check_critical_and_report(opts, parser)
    assert "--save-P" in capsys.readouterr().out
