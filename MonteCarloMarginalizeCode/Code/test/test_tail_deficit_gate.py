"""Regression suite for the CIP composition-reweight gate (util_CIPTailDeficitGate.py and
its wrapper).  Named in .github/workflows/ci.yml -- a test file is not in CI otherwise.

Three layers:
  1. Measured-value regressions (fixtures_tail_deficit_gate.json): the SHIPPED gate's
     decisions on the events that defined its validation.  The three production controls
     are the events the UNGATED fix drove to 1.543/1.175x the reference width -- they are
     the permanent guard against that regression.  The T5-family toys must abstain VIA THE
     VALIDITY FLOOR, not the threshold (asserted: each has R < threshold).
  2. The validity floor is unbypassable: property scan over decide(), and the CLI refuses
     any attempt to weaken it below one expected count.
  3. End-to-end CLI on synthetic data: severe deficit fires; healthy does not; unresolvable
     tail abstains; the wrapper's fname-rebuild contract stays intact.
"""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BIN = os.path.join(_HERE, "..", "bin")
_GATE = os.path.abspath(os.path.join(_BIN, "util_CIPTailDeficitGate.py"))
_FIX = os.path.join(_HERE, "fixtures_tail_deficit_gate.json")

_spec = importlib.util.spec_from_file_location("tail_deficit_gate", _GATE)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

FIXTURES = json.load(open(_FIX))["cases"]


# --------------------------------------------------------------------------- layer 1
@pytest.mark.parametrize("case", FIXTURES, ids=[c["case"] for c in FIXTURES])
def test_shipped_gate_decision_on_measured_events(case):
    """The shipped gate (default threshold + floor) must reproduce the validated decision
    on every event that defined its validation."""
    decision, floor = gate.decide(case["implied"], case["R"], case["n_post"])
    assert decision == case["expect"], (
        f"{case['case']} ({case['kind']}): shipped gate says {decision}, "
        f"validated decision is {case['expect']}")


def test_controls_are_noops_and_exemplars_fire():
    """Group-level restatement, so a failure names the physics not just one event."""
    by = {c["case"]: gate.decide(c["implied"], c["R"], c["n_post"])[0] for c in FIXTURES}
    for ev in ("S240921cw", "S240930aa", "S240621dy"):
        assert by[ev] == "NO-FIRE", f"control {ev} must be a NO-OP (ungated fix broke it)"
    for ev in ("S241109bn", "S240413p", "S241102br", "S240629by", "S241225c"):
        assert by[ev] == "FIRE", f"exemplar {ev} must fire (severe deficit)"


def test_t5_family_abstains_via_floor_not_threshold():
    """The reason matters: every T5-family chain has R < threshold, so the THRESHOLD alone
    would false-fire on known-truth healthy-narrow cases; only the floor prevents it."""
    t5 = [c for c in FIXTURES if c["expect"] == "ABSTAIN-FLOOR"]
    assert len(t5) >= 6
    for c in t5:
        assert c["R"] < gate.THRESHOLD, f"{c['case']}: fixture no longer exercises the floor"
        decision, floor = gate.decide(c["implied"], c["R"], c["n_post"])
        assert decision == "ABSTAIN-FLOOR"
        assert c["implied"] < floor, f"{c['case']}: abstain must be BECAUSE of the floor"


# --------------------------------------------------------------------------- layer 2
def test_floor_is_unbypassable_property_scan():
    """No (R, implied, n_post) with implied below the floor may ever FIRE -- including
    R = 0, the degenerate maximum-deficit reading the floor exists to intercept."""
    rng = np.random.default_rng(20260819)
    for _ in range(2000):
        n_post = int(rng.integers(100, 200000))
        floor = gate.FLOOR_COUNTS / n_post
        implied = floor * rng.uniform(0, 1.0 - 1e-12)   # strictly below the floor
        R = float(rng.choice([0.0, rng.uniform(0, 2.0), np.nan]))
        decision, _ = gate.decide(implied, R, n_post)
        assert decision == "ABSTAIN-FLOOR", (implied, R, n_post, decision)
    # and the exact worst case
    assert gate.decide(0.0, 0.0, 20000)[0] == "ABSTAIN-FLOOR"
    assert gate.decide(float("nan"), 0.0, 20000)[0] == "ABSTAIN-FLOOR"


def test_fire_requires_floor_and_threshold_jointly():
    rng = np.random.default_rng(7)
    for _ in range(2000):
        n_post = int(rng.integers(100, 200000))
        implied = float(rng.uniform(0, 0.5))
        R = float(rng.uniform(0, 1.5))
        decision, floor = gate.decide(implied, R, n_post)
        if decision == "FIRE":
            assert implied >= floor and R < gate.THRESHOLD
        elif decision == "NO-FIRE":
            assert implied >= floor and not (R < gate.THRESHOLD)


def test_cli_refuses_to_disable_floor(tmp_path):
    """--floor-counts below 1 is refused (exit 2), and no bypass flag exists at all."""
    train, post = _synthetic(tmp_path, deficit="severe")
    r = subprocess.run([sys.executable, _GATE, str(train), str(post), "--floor-counts", "0"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "mandatory" in r.stderr
    h = subprocess.run([sys.executable, _GATE, "--help"], capture_output=True, text=True)
    assert h.returncode == 0
    low = h.stdout.lower()
    for bad in ("--no-floor", "--skip-floor", "--disable-floor", "--force-fire"):
        assert bad not in low


# --------------------------------------------------------------------------- layer 3
def _synthetic(tmp_path, deficit):
    """Small synthetic (training, posterior) pair with a controllable tail state.
    Training: 6000 rows, 13 cols, uniform-ish chi1_perp coverage; lnL flat near-peak in the
    core.  'severe': tail lnL also near peak (implied large) but posterior confined ->
    delivered tiny -> R tiny -> FIRE.  'healthy': posterior tracks the implied tail mass ->
    NO-FIRE.  'confined': tail lnL ~ 25 nats down -> implied unresolvable -> ABSTAIN-FLOOR."""
    rng = np.random.default_rng(42)
    n = 6000
    a1 = rng.uniform(0, 0.99, n)
    ct = rng.uniform(-1, 1, n)
    cp = a1 * np.sqrt(1 - ct ** 2)
    lnl = 100.0 - 0.5 * rng.uniform(0, 1, n)            # near-peak core everywhere
    tail = cp > 0.5
    if deficit == "confined":
        lnl[tail] -= 25.0                               # tail is truly dead
    X = np.zeros((n, 13))
    X[:, 1] = 10.0; X[:, 2] = 8.0                       # m1 m2 (unused by the gate)
    X[:, 3] = cp; X[:, 4] = 0.0; X[:, 5] = a1 * ct      # s1x s1y s1z
    X[:, 9] = lnl
    train = tmp_path / f"train_{deficit}.net"
    np.savetxt(train, X)
    m = 20000
    if deficit == "healthy":
        pa1 = rng.uniform(0, 0.99, m); pct = rng.uniform(-1, 1, m)
        pcp = pa1 * np.sqrt(1 - pct ** 2)               # prior-wide: delivered ~ implied
    else:
        pcp = rng.uniform(0, 0.10, m)                   # confined posterior
    post = tmp_path / f"post_{deficit}.dat"
    with open(post, "w") as f:
        f.write("# a1x a1y lnL\n")
        np.savetxt(f, np.column_stack([pcp, np.zeros(m), np.zeros(m)]))
    return train, post


@pytest.mark.parametrize("deficit,expect", [("severe", "FIRE"), ("healthy", "NO-FIRE"),
                                            ("confined", "ABSTAIN-FLOOR")])
def test_cli_end_to_end_synthetic(tmp_path, deficit, expect):
    train, post = _synthetic(tmp_path, deficit)
    r = subprocess.run([sys.executable, _GATE, str(train), str(post),
                        "--json", str(tmp_path / "g.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    last = r.stdout.strip().splitlines()[-1]
    assert last.startswith(f"GATE DECISION={expect} "), (last, r.stderr)
    rec = json.load(open(tmp_path / "g.json"))
    assert rec["decision"] == expect
    # the stderr log must name the deciding condition -- silent no-ops are the bug class
    assert expect.split("-")[0] in r.stderr or expect in r.stderr


def test_wrapper_gates_and_falls_back(tmp_path):
    """Wrapper e2e with both hooks faked: FIRE path swaps --fname to the thinned set for the
    FINAL pass only; a broken gate falls back to the original file loudly."""
    wrapper = os.path.abspath(os.path.join(_BIN, "util_CIPCompositionReweightWrapper.sh"))
    train, post = _synthetic(tmp_path, "severe")
    fake_cip = tmp_path / "fake_cip.sh"
    fake_cip.write_text(
        "#!/bin/bash\n"
        "out=''; nxt=0\n"
        "for a in \"$@\"; do\n"
        "  if [[ $nxt == 1 ]]; then out=$a; nxt=0; fi\n"
        "  [[ $a == --fname-output-samples ]] && nxt=1\n"
        "  [[ $a == --fname-output-samples=* ]] && out=${a#--fname-output-samples=}\n"
        "done\n"
        "[[ -n $out ]] && echo fake > \"$out.xml.gz\"\n"
        "echo \"FAKECIP ARGS: $@\"\n")
    fake_cip.chmod(0o755)
    fake_conv = tmp_path / "fake_conv.sh"
    fake_conv.write_text(f"#!/bin/bash\ncat {post}\n")
    fake_conv.chmod(0o755)
    # the wrapper invokes the gate/reweight tools via their `#!/usr/bin/env python3`
    # shebangs; guarantee that python3 resolves to THIS interpreter (numpy-capable)
    pybin = tmp_path / "pybin"
    pybin.mkdir()
    (pybin / "python3").symlink_to(sys.executable)
    env = dict(os.environ, CIP_REWEIGHT_REAL_CIP=str(fake_cip),
               CIP_REWEIGHT_CONVERT=str(fake_conv),
               PATH=str(pybin) + os.pathsep + _BIN + os.pathsep + os.environ.get("PATH", ""))
    r = subprocess.run(["bash", wrapper, "--fname", str(train),
                        "--fname-output-samples", str(tmp_path / "final"),
                        "--n-output-samples", "138"],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr
    assert "GATE DECISION=FIRE" in r.stderr
    final_args = [ln for ln in r.stdout.splitlines() if ln.startswith("FAKECIP ARGS")][-1]
    assert "all_comp_" in final_args, "final CIP must train on the thinned set after a FIRE"
    # fallback: converter emits garbage -> gate cannot evaluate -> original file, loudly
    fake_conv.write_text("#!/bin/bash\necho not-a-posterior\n")
    r2 = subprocess.run(["bash", wrapper, "--fname", str(train),
                         "--fname-output-samples", str(tmp_path / "final2"),
                         "--n-output-samples", "138"],
                        capture_output=True, text=True, env=env, timeout=600)
    assert r2.returncode == 0
    assert "falling back" in r2.stderr
    final2 = [ln for ln in r2.stdout.splitlines() if ln.startswith("FAKECIP ARGS")][-1]
    assert str(train) in final2 and "all_comp_" not in final2
