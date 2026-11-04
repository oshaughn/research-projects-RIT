"""The (local, reserve) pair is CHOSEN from analysis, before any row is evaluated.

RO, 2026-09-08: rely on analysis and the known physics to pick the pair, rather
than try-then-decline-then-refine.  Everything the selector uses is computable
from the precomputed inputs, so the choice and its reasons are printed in the
run's first lines instead of being discovered from a ledger at the end.

The tests below pin the RULE, not a run.  The most important one is
`test_the_predictor_reproduces_the_measured_refine4_failure`: an independent
session MEASURED the whole-window reserve failing its own convergence warrant at
the lowest rung, and the predictor says the same thing from the physics alone.
"""

import os

import numpy as np
import pytest

from RIFT.likelihood.jax_ile import direct_marginalization_policy as DP
from RIFT.likelihood.jax_ile.anglemarg import ANGLE_MARG_CROSSOVER_AMPLITUDE

from test_angle_marg_exact import make_synth

_DRIVER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "bin",
    "integrate_likelihood_extrinsic_jax")

# Ladder-2 network SNRs.
RHO_40, RHO_160, RHO_320, RHO_640 = 40.7691, 163.0766, 326.1531, 652.3062


@pytest.fixture(scope="module")
def data():
    return make_synth(scale=1.0, npts=614)


def _pick(data, rho, refine_max=32, **kw):
    kw.setdefault("max_time_nodes", 64)
    return DP.predict_reserve_pair(
        data, rho, reserve_time_refine_max=refine_max,
        crossover_amplitude=ANGLE_MARG_CROSSOVER_AMPLITUDE, **kw)


def test_the_bandwidth_comes_from_the_complex_Q_spectrum(data):
    """Q (rholm) is complex.  A real-input transform rejects it outright, and
    taking only the real part would discard half the phase structure, so this
    pins that the second moment is finite, positive and Nyquist-bounded."""
    sigma_f = DP.q_effective_bandwidth_hz(data)
    assert np.isfinite(sigma_f) and sigma_f > 0.0
    f_nyq = 0.5 / (data.deltaT / data.q_time_pregrid_factor)
    assert sigma_f < f_nyq


def test_the_angle_scheme_follows_the_validated_crossover(data):
    """A = rho^2/2 against ANGLE_MARG_CROSSOVER_AMPLITUDE, which is a measured
    accuracy crossover in anglemarg, not a tuning constant introduced here."""
    lo, _ = _pick(data, 1.0)                      # A = 0.5, far below 450
    assert lo == "exact"
    hi, info = _pick(data, RHO_40)                # A = 831, above 450
    assert hi == "laplace"
    assert info["amplitude_A"] > info["crossover_amplitude"]


def test_a_peak_the_ceiling_cannot_resolve_is_refused_not_refined(data):
    """The whole point.  When the escalation ceiling cannot resolve the peak and
    no peak-local time reserve is available, the selector returns None so the
    caller REFUSES.  Silently falling back to whole-window refinement is the
    failure mode this exists to prevent."""
    scheme, info = _pick(data, RHO_640)
    assert scheme is None
    assert not info["time_peak_resolvable_whole_window"]
    assert info["whole_window_nodes_needed"] > info["whole_window_nodes_available"]
    assert "NOT IMPLEMENTED" in info["reason"]


def test_the_same_signal_selects_peaklocal_once_it_exists(data):
    """The refusal is about availability, not about the signal.  With a
    peak-local time reserve on the menu the identical inputs select it, so the
    day that kernel lands the selector starts choosing it with no rule change."""
    scheme, info = _pick(data, RHO_640, available=("exact", "laplace", "peaklocal"))
    assert scheme == "peaklocal"
    assert "peak-local" in info["reason"]


def test_the_predictor_reproduces_the_measured_refine4_failure(data):
    """Independent confirmation, and the reason to trust the rule at all.

    The reserve-scheme session MEASURED the whole-window refined reserve at
    refine=4 failing its OWN half-refined convergence warrant at the LOWEST
    ladder rung, rho 40.77: 0.0018, 0.0040 and 0.0114 nats on three rows
    against a 1e-3 target.  The predictor is told nothing about that.  From the
    physics alone it says the refine-4 rule affords 2453 nodes on this window
    while the peak needs more, i.e. under-resolved at the bottom of the ladder,
    which is what they saw.  At the ceiling of 32 the same rung is comfortable.
    """
    _, tight = _pick(data, RHO_40, refine_max=4)
    assert not tight["time_peak_resolvable_whole_window"]
    assert tight["whole_window_nodes_available"] == pytest.approx(
        (data.npts - 1) * 4 + 1)

    scheme, loose = _pick(data, RHO_40, refine_max=32)
    assert loose["time_peak_resolvable_whole_window"]
    assert scheme == "laplace"


def test_an_explicit_request_bypasses_the_analysis(data):
    """Overrides stay overrides: 'auto' analyses, anything else is obeyed and
    labelled as such so a run log cannot be misread as an analysed choice."""
    scheme, info = _pick(data, RHO_640, requested="exact")
    assert scheme == "exact"
    assert "explicit request" in info["reason"]


def test_the_pair_line_is_printable_and_names_the_refusal(data):
    scheme, info = _pick(data, RHO_640)
    line = DP.format_reserve_pair(scheme, info)
    assert line.startswith("RESERVE-PAIR local=four-axis reserve=REFUSED")
    for token in ("rho=", "sigma_f=", "A=", "peak=", "cover_needs=",
                  "reserve_needs="):
        assert token in line


def test_the_local_cover_verdict_is_reported_and_leads(data):
    """The FIRST quantity: can the local branch's time cover hold the peak?

    Measured elsewhere at rho 163: raising the cover 64 -> 256 took acceptance
    31% -> 75%.  That is why the start cap appeared to plateau -- time capacity
    was binding, not the start cap saturating -- so a selector that reports only
    the reserve's budget would predict the wrong lever.  The local branch is the
    thing under test; the reserve is only what it falls back to.
    """
    _, tight = _pick(data, RHO_640, max_time_nodes=64)
    _, loose = _pick(data, RHO_640, max_time_nodes=4096)
    assert tight["cover_nodes_needed"] == pytest.approx(loose["cover_nodes_needed"])
    assert not tight["local_cover_resolves_peak"]
    assert loose["local_cover_resolves_peak"]
    assert "local cover" in loose["reason"]


def test_the_bandwidth_definition_is_pinned_not_just_its_plausibility(data):
    """A synthetic signal with a KNOWN analytic bandwidth, so a definition swap
    fails here rather than shifting every threshold quietly.

    The earlier tests in this file only asserted sigma_f was finite, positive and
    Nyquist-bounded.  Every one of them passes with the RAW two-sided moment,
    which is the wrong quantity for a timing width and was what this function
    originally returned.  Plausibility is not a definition.

    Construction: a complex tone at f0 with a Gaussian AMPLITUDE of width bw in
    frequency.  The moments weight |Qtilde|^2, so the POWER spectrum has width
    bw/sqrt(2), and that -- not bw -- is what the central moment must return.
    Getting this wrong was my first version of this test: the expectation, not
    the code, was off by sqrt(2).  The two-sided RAW moment must instead land
    near f0, which is the error the test exists to catch.
    """
    import numpy as np

    n, dt = 4096, 1.0 / 4096
    f0, bw = 200.0, 20.0
    freqs = np.fft.fftfreq(n, d=dt)
    amp = np.exp(-0.5 * ((freqs - f0) / bw) ** 2)      # positive-side only
    x = np.fft.ifft(amp)

    class _D:
        deltaT = dt
        q_time_pregrid_factor = 1
        detector_names = ["H1"]
        detectors = {"H1": {"Q": np.asarray(x)[:, None],
                            "q_time_pregrid_factor": 1}}

    central = DP.q_effective_bandwidth_hz(_D, moment="central")
    raw = DP.q_effective_bandwidth_hz(_D, moment="raw")
    default = DP.q_effective_bandwidth_hz(_D)

    bw_power = bw / np.sqrt(2.0)
    assert central == pytest.approx(bw_power, rel=0.05), central
    assert raw == pytest.approx(np.hypot(f0, bw_power), rel=0.05), raw
    assert raw > 5.0 * central          # the two are not interchangeable
    # The DEFAULT must be the narrow one: sizing a time rule on the envelope
    # bandwidth under-resolves every row whose likelihood is carrier-modulated.
    assert default == pytest.approx(raw)


def test_the_selector_sizes_on_the_narrow_bandwidth(data):
    """sigma_t must be built from the RAW moment -- the narrowest peak the
    primitive can produce.  If the envelope moment is ever wired back in, the
    predicted width widens by the ratio of the two and the selector starts
    calling a whole-window rule adequate when it is not."""
    _, info = _pick(data, RHO_160)
    sigma_f = info["sigma_f_hz"]
    assert sigma_f == pytest.approx(
        DP.q_effective_bandwidth_hz(data, moment="raw"))
    assert info["sigma_f_envelope_hz"] == pytest.approx(
        DP.q_effective_bandwidth_hz(data, moment="central"))
    expect = 1.0 / (2.0 * np.pi * RHO_160 * sigma_f)
    assert info["sigma_t_s"] == pytest.approx(expect, rel=1e-9)


def test_the_gate_roster_lists_every_file_once_and_covers_this_one():
    """The CI roster is an explicit list, so a new test file is unrun until it
    is added -- and a DUPLICATE entry runs the file twice and inflates the
    collection floor, which then hides a later removal.

    Both halves are things I got wrong on this branch within one hour: a rebase
    resolved the roster conflict by taking upstream's side and silently dropped
    my entry, and the fix then added a second copy of an entry that was already
    there because my check used a broken grep pattern.
    """
    import collections
    import os
    import re

    # test/jax/<file> -> test/jax -> test -> Code -> MonteCarloMarginalizeCode
    # -> repo root: FIVE levels, not four.
    root = os.path.abspath(__file__)
    for _ in range(5):
        root = os.path.dirname(root)
    gate = os.path.join(root, ".travis", "test-jax.sh")
    src = open(gate, encoding="utf-8").read()
    block = src[src.index("FILES=("):src.index("\n)", src.index("FILES=("))]
    listed = re.findall(r'\$\{JAXDIR\}/(test_[A-Za-z0-9_]+\.py)', block)

    dupes = {n: c for n, c in collections.Counter(listed).items() if c > 1}
    assert not dupes, "roster lists a file more than once: %r" % (dupes,)
    assert os.path.basename(__file__) in listed, (
        "this file is not in the gate roster, so CI would not run it")


def test_the_gate_sets_its_floor_exactly_once():
    """Bash keeps the LAST assignment, so a duplicated constant leaves earlier
    ones dead while they still read as authoritative in review.

    This file carried FIVE consecutive unconditional EXPECTED_TESTS= lines
    (648, 629, 657, 648, 705) accumulated by parallel merges.  Only 705 was
    live.  A reviewer checking "is the floor right?" would most likely read the
    first, which had been dead for three merges.
    """
    import os
    import re

    root = os.path.abspath(__file__)
    for _ in range(5):
        root = os.path.dirname(root)
    src = open(os.path.join(root, ".travis", "test-jax.sh"), encoding="utf-8").read()
    assigns = re.findall(r"(?m)^EXPECTED_TESTS=(\d+)", src)
    assert len(assigns) == 1, (
        "EXPECTED_TESTS assigned %d times (%s); bash keeps the last and the "
        "rest are dead" % (len(assigns), ", ".join(assigns)))


def test_the_roster_is_honoured_on_the_ANGULAR_branch_too(data):
    """A roster without laplace must not yield laplace.

    The roster was checked only where the selector chooses ``peaklocal`` -- the
    branch that could not have chosen it anyway, since 'peaklocal' is never on
    the roster today.  On the branch that CAN choose laplace, the caller's
    roster was ignored, so a run whose data cannot support the laplace reserve
    still selected it the moment A cleared the crossover.

    The roster is not a preference.  It says which schemes this data and this
    distance quadrature can support at all -- for laplace, that the adaptive
    node placement's A0 == 0 / B1 == 0 premise holds -- so ignoring it means
    running exact under laplace's name, or worse.
    """
    with_lap, _ = _pick(data, RHO_40, available=("exact", "laplace"))
    assert with_lap == "laplace"

    scheme, info = _pick(data, RHO_40, available=("exact",))
    assert scheme is None, (
        "laplace selected off a roster that does not offer it: %r"
        % (info["reason"],))
    assert "on the roster" in info["reason"]
    assert "laplace" in info["reason"]


def test_a_roster_absence_is_not_overridable_by_an_explicit_request(data):
    """An explicit request overrides the ANALYSIS, not the ROSTER.

    Forcing a scheme whose premise is absent is not an override; it is an
    unnoticed wrong answer.  The distinction matters because the driver's
    refusal message invites the user to pass an explicit scheme -- that must
    let them overrule the crossover, and must not let them overrule a measured
    identity failure.
    """
    ok, _ = _pick(data, RHO_40, requested="exact", available=("exact",))
    assert ok == "exact"

    scheme, info = _pick(data, RHO_40, requested="laplace",
                         available=("exact",))
    assert scheme is None
    assert "not overridable" in info["reason"]


def test_an_unwired_reserve_scheme_is_refused_not_run_as_exact():
    """A config field the composite never reads is worse than a missing one.

    ``PolicyConfig.reserve_scheme`` is validated against the CHOICES tuple, and
    the composite dispatches through ``reserve_pair``.  A scheme declared in
    CHOICES but absent from the pair table must be refused, not accepted,
    reported in the policy line, and computed as exact.
    """
    DP.validate_policy_config(DP.PolicyConfig(reserve_scheme="exact"))
    # 'auto' is resolved before the composite sees it, so it is admitted here.
    DP.validate_policy_config(DP.PolicyConfig(reserve_scheme="auto"))

    # laplace and peaklocal are wired through the pair table (#304), so every
    # declared scheme executes today; the refusal is exercised by declaring a
    # scheme the pair table does not carry.
    for scheme in ("laplace", "peaklocal"):
        assert scheme in DP.RESERVE_SCHEME_CHOICES
        assert scheme in DP.RESERVE_SCHEME_EXECUTABLE
        DP.validate_policy_config(DP.PolicyConfig(reserve_scheme=scheme))
    saved = DP.RESERVE_SCHEME_CHOICES
    DP.RESERVE_SCHEME_CHOICES = saved + ("nonesuch-declared",)
    try:
        assert "nonesuch-declared" not in DP.RESERVE_SCHEME_EXECUTABLE
        with pytest.raises(ValueError, match="NOT WIRED"):
            DP.validate_policy_config(
                DP.PolicyConfig(reserve_scheme="nonesuch-declared"))
    finally:
        DP.RESERVE_SCHEME_CHOICES = saved

    with pytest.raises(ValueError, match="must be one of"):
        DP.validate_policy_config(DP.PolicyConfig(reserve_scheme="nonesuch"))


def test_the_executable_roster_is_a_subset_of_the_choices():
    """Guards the pair as a pair: a scheme becomes executable by being wired,
    and this fails if RESERVE_SCHEME_EXECUTABLE ever names something the
    choices tuple does not, which would mean the two lists were edited apart."""
    assert set(DP.RESERVE_SCHEME_EXECUTABLE) <= set(DP.RESERVE_SCHEME_CHOICES)
    assert "auto" not in DP.RESERVE_SCHEME_EXECUTABLE, (
        "'auto' is a resolution mode, not something the composite executes")
    assert DP.RESERVE_SCHEME_DEFAULT in DP.RESERVE_SCHEME_EXECUTABLE, (
        "the default must be executable or every bare run refuses")


def test_auto_is_admitted_only_because_something_resolves_it():
    """``validate_policy_config`` admits 'auto' on a PREMISE: that the driver
    writes the resolved pair back onto the config before the composite runs.

    Found by the #304 session -- the driver announced a pair and left
    ``policy_config.reserve_scheme == "auto"``.  Nothing dispatched on the
    string yet, so nothing broke; the printed line was a claim about a value
    the composite never read, and the admission's stated reason was false.

    A COUPLING GUARD, not a behaviour test, and labelled as one.  The behaviour
    -- 'auto' reaching the composite and being executed as something -- cannot
    be observed until #304 dispatches on the string, because today the
    composite ignores the field entirely.  So this checks the two tokens that
    have to co-exist, tolerant of formatting, and says what to do if it fires:
    if the driver stops resolving, 'auto' must stop being admitted.
    """
    import re
    with open(_DRIVER) as fh:
        src = fh.read()
    assert re.search(r"_replace\(\s*reserve_scheme=_pair\s*\)", src), (
        "the driver no longer writes the resolved pair onto policy_config, so "
        "validate_policy_config must stop admitting 'auto' -- an unresolved "
        "'auto' reaching the composite is the silently-inert field this branch "
        "refuses everywhere else")
    DP.validate_policy_config(DP.PolicyConfig(reserve_scheme="auto"))


def test_a_selected_reserve_the_composite_cannot_run_is_refused():
    """Selecting is not running.  The roster says what the DATA supports; the
    executable tuple says what the composite DISPATCHES, and the two are not
    the same list.  A pair that clears the first and fails the second must
    refuse, not run exact under the selected scheme's name."""
    for scheme in DP.RESERVE_SCHEME_CHOICES:
        if scheme == "auto":
            continue
        if scheme in DP.RESERVE_SCHEME_EXECUTABLE:
            DP.validate_policy_config(DP.PolicyConfig(reserve_scheme=scheme))
        else:
            with pytest.raises(ValueError, match="NOT WIRED"):
                DP.validate_policy_config(DP.PolicyConfig(reserve_scheme=scheme))
