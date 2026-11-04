"""Flow re-use is OFF by default, and re-use is still reachable.

WHY THIS EXISTS.  `--no-flow-reuse` used to be `store_true, default=False`, i.e.
flow re-use ON unless disabled.  Measured across an 8-event batch at two seeds,
re-use contracts the extrinsic posterior monotonically in slot index -- psi to
~40% of its no-re-use width by slot 7, on BOTH seeds -- while slot 0, where no
re-use has yet happened, sits at ~1.0 as a built-in control.  It buys no
measurable wall time (1589 s mean with, 1567 s without; the seed-to-seed spread
is larger than the difference and its sign flips).  So the default is now OFF.

THE TRAP THIS PINS.  A `store_true` flag cannot express its own negation: simply
setting `default=True` would have made `--no-flow-reuse` inert AND removed any
way to turn re-use back on, silently deleting a capability while appearing to
change only a default.  Hence the paired `--flow-reuse` writing the same dest.
Both directions are asserted here, and so is last-one-wins, because a paired
flag whose order does not resolve is worse than no flag at all.

Needs no lal, no GPU and no flowMC: the parser is driven directly.
"""
import importlib.machinery
import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
DRIVER = os.path.join(CODE, "bin", "integrate_likelihood_extrinsic_jax")
if CODE not in sys.path:
    sys.path.insert(0, CODE)


def _driver():
    loader = importlib.machinery.SourceFileLoader("_ile_jax_reuse", DRIVER)
    spec = importlib.util.spec_from_loader("_ile_jax_reuse", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


drv = _driver()


def _parse(*args):
    opts, _ = drv.build_parser().parse_args(list(args))
    return opts.no_flow_reuse


def test_default_is_NO_flow_reuse():
    """The whole point of the change."""
    assert _parse() is True


def test_flow_reuse_is_still_reachable():
    """Without this, flipping the default would have deleted a capability."""
    assert _parse("--flow-reuse") is False


def test_no_flow_reuse_still_accepted_for_compatibility():
    """Existing command lines and scripts pass it; it now restates the default."""
    assert _parse("--no-flow-reuse") is True


@pytest.mark.parametrize("args,expected", [
    (("--flow-reuse", "--no-flow-reuse"), True),
    (("--no-flow-reuse", "--flow-reuse"), False),
])
def test_last_flag_on_the_command_line_wins(args, expected):
    """Both write the same dest; an unresolved order would be worse than nothing."""
    assert _parse(*args) is expected


def test_both_flags_share_one_dest():
    """Structural: two dests would let the two flags disagree silently."""
    p = drv.build_parser()
    dests = {}
    for o in p._get_all_options():
        for s in (o._long_opts or []):
            if s in ("--flow-reuse", "--no-flow-reuse"):
                dests[s] = o.dest
    assert dests == {"--flow-reuse": "no_flow_reuse",
                     "--no-flow-reuse": "no_flow_reuse"}, dests


def test_the_batch_loop_still_honours_the_flag():
    """A default flip is worthless if the consumer stopped reading it.

    Pins the two call sites in analyze-the-batch: the flow state handed to the
    next event, and whether the returned state is retained at all.
    """
    with open(DRIVER) as f:
        src = f.read()
    assert "flow_state=(None if opts.no_flow_reuse else flow_state)" in src
    assert "if not opts.no_flow_reuse and new_flow_state is not None:" in src
