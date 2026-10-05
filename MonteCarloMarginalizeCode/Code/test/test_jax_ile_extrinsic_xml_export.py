"""The JAX ILE driver must export sim_inspiral XML, not only the .dat sidecar.

create_event_parameter_pipeline_BasicIteration's --last-iteration-extrinsic
stage hands its convert_extr job ``EXTR_out-<event>.xml_<k>_.xml.gz``.  The
driver wrote only ``<output>_<k>_samples.dat``, so a DAG driven by it built
cleanly, ran its terminal stage and collected nothing.  These tests pin the
XML's existence, its row count against the ``.dat``, and the round trip of every
column both files carry.

They also pin what the resamplers do with the file.  The exported rows are an
equal-weight fair draw, so the weight
``exp(lnL - lnLmax) * (alpha2/alpha3) / Npts`` that
util_ResampleILEOutputWithExtrinsic.py and util_BatchConvertResampleILEOutput.py
form must come out CONSTANT over them; constant prior columns do not do that,
and the spread test below measures the difference rather than asserting it.

The executable parses options at import, so the writers are extracted by AST --
the same technique as test_jax_template_finalization.py, and for the same
reason: it exercises the real implementation rather than a copy of it.
"""
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import lal
import numpy as np
import pytest

from igwn_ligolw import ligolw, lsctables, utils as ligolw_utils

from RIFT.likelihood import factored_likelihood


DRIVER = (Path(__file__).resolve().parents[1] /
          "bin" / "integrate_likelihood_extrinsic_jax")

# The export path and everything it calls, plus the module constants those read.
# Named explicitly rather than taken wholesale: the driver has functions whose
# DEFAULT ARGUMENTS reference module state, and a default is evaluated at `def`
# time, so exec-ing the whole file's functions fails on names this test has no
# reason to supply.
_WANTED_FUNCS = ("dat_path", "samples_path", "xml_path", "_remove_stale_artifact",
                 "fairdraw_indices", "fairdraw_size", "_target_ess_was_given",
                 "was_supplied", "_xml_extrinsic_columns", "_xml_prior_columns",
                 "write_samples_xml", "write_samples")
_WANTED_ASSIGNS = ("_TEMPERED_MODES", "_FAIRDRAW_MODES", "_FAIRDRAW_N_MAX_DEFAULT",
                   "_XML_PS_LOG_CLIP", "_USABLE_EXPORT_ESS")


def _driver_namespace():
    """The driver's real export functions, in a namespace that can run them."""
    source = DRIVER.read_text()
    tree = ast.parse(source, filename=str(DRIVER))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED_FUNCS:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in _WANTED_ASSIGNS
                for t in node.targets):
            body.append(node)
    found = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    missing = set(_WANTED_FUNCS) - found
    assert not missing, (
        "the driver no longer defines %s; this test would silently exercise "
        "less than it claims" % sorted(missing))
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "sys": sys, "os": __import__("os"),
                 "MSUN": lal.MSUN_SI, "PC": lal.PC_SI}
    exec(compile(module, str(DRIVER), "exec"), namespace)
    return namespace


NS = _driver_namespace()


def _P(m1=1.5, m2=1.3):
    """A minimal template with the intrinsic attributes the export reads."""
    return SimpleNamespace(
        m1=m1 * lal.MSUN_SI, m2=m2 * lal.MSUN_SI,
        s1x=0.0, s1y=0.0, s1z=0.11, s2x=0.0, s2y=0.0, s2z=-0.07,
        lambda1=300.0, lambda2=250.0, eccentricity=0.0, meanPerAno=0.0,
        a6c=0.0, E0=0.0, p_phi0=0.0)


def _opts(tmp_path, mode="laplace-is", **updates):
    values = dict(
        output_file=str(tmp_path / "EXTR_out-0.xml"), save_samples=True,
        mode=mode, seed=7, phase_marginalization=False,
        fairdraw_extrinsic_output=False, fairdraw_extrinsic_output_n_max=None,
        n_fairdraw_extrinsic_samples=None, n_eff=100,
        target_export_ess_frac=None, _supplied_options=set())
    values.update(updates)
    return SimpleNamespace(**values)


def _theta(n, ndim, rng):
    """A physical extrinsic cloud in the driver's column order."""
    cols = [rng.uniform(0.0, 2 * np.pi, n),              # ra
            np.arcsin(rng.uniform(-1.0, 1.0, n)),        # dec
            rng.uniform(0.0, np.pi, n),                  # psi
            np.arccos(rng.uniform(-1.0, 1.0, n)),        # incl
            rng.uniform(0.0, 2 * np.pi, n),              # phiref
            rng.uniform(50.0, 500.0, n)]                 # distance
    if ndim == 3:        # flowmc-phipsimarg: ra, dec, incl
        return np.column_stack([cols[0], cols[1], cols[3]])
    if ndim == 4:        # phimarg: ra, dec, psi, incl -- dpsimarg reorders below
        return np.column_stack(cols[:4])
    return np.column_stack(cols[:ndim])


def _run(opts, theta, lnL, with_distance, epoch=1126259462.0,
         logZ=41.25, sigma_lnL=0.031, neff=88.5, ntotal=4096):
    NS["write_samples"](opts, 0, theta, lnL, with_distance, P=_P(),
                        fiducial_epoch=epoch, logZ=logZ, sigma_lnL=sigma_lnL,
                        report_neff=neff, ntotal=ntotal)
    return (Path(NS["samples_path"](opts, 0)), Path(NS["xml_path"](opts, 0)))


def _load(xml):
    return ligolw_utils.load_filename(
        str(xml), contenthandler=ligolw.LIGOLWContentHandler)


def _sim_rows(xml):
    return lsctables.SimInspiralTable.get_table(_load(xml))


def _dat(path):
    return np.atleast_1d(np.genfromtxt(str(path), names=True))


# ---------------------------------------------------------------------------
# The file the pipeline reads
# ---------------------------------------------------------------------------
def test_xml_name_is_the_one_convert_extr_opens(tmp_path):
    opts = _opts(tmp_path)
    assert NS["xml_path"](opts, 3).endswith("EXTR_out-0.xml_3_.xml.gz")


def test_one_xml_per_event_alongside_the_unchanged_dat(tmp_path):
    rng = np.random.default_rng(0)
    opts = _opts(tmp_path)
    theta = _theta(64, 6, rng)
    lnL = rng.normal(40.0, 1.0, 64)
    dat, xml = _run(opts, theta, lnL, with_distance=True)
    assert dat.is_file() and xml.is_file()
    # The sidecar keeps its exact columns: nothing that reads it may change.
    assert list(_dat(dat).dtype.names) == [
        "right_ascension", "declination", "distance", "inclination", "psi",
        "phi_orb", "loglikelihood"]


def test_xml_row_count_matches_the_dat(tmp_path):
    rng = np.random.default_rng(1)
    opts = _opts(tmp_path)
    theta = _theta(37, 6, rng)
    lnL = rng.normal(40.0, 1.0, 37)
    dat, xml = _run(opts, theta, lnL, with_distance=True)
    assert len(_sim_rows(xml)) == len(_dat(dat)) == 37


def test_nonfinite_lnL_rows_are_dropped_from_both_products(tmp_path):
    rng = np.random.default_rng(2)
    opts = _opts(tmp_path)
    theta = _theta(20, 6, rng)
    lnL = rng.normal(40.0, 1.0, 20)
    lnL[[3, 11]] = np.nan
    dat, xml = _run(opts, theta, lnL, with_distance=True)
    assert len(_sim_rows(xml)) == len(_dat(dat)) == 18


def test_no_xml_without_save_samples(tmp_path):
    rng = np.random.default_rng(3)
    opts = _opts(tmp_path, save_samples=False)
    NS["write_samples"](opts, 0, _theta(8, 6, rng), np.zeros(8), True, P=_P(),
                        fiducial_epoch=0.0, logZ=1.0, sigma_lnL=0.1)
    assert not Path(NS["xml_path"](opts, 0)).exists()
    assert not Path(NS["samples_path"](opts, 0)).exists()


# ---------------------------------------------------------------------------
# Round trip: the XML columns must be the .dat columns
# ---------------------------------------------------------------------------
def test_angles_and_distance_round_trip_to_the_dat(tmp_path):
    rng = np.random.default_rng(4)
    opts = _opts(tmp_path)
    theta = _theta(50, 6, rng)
    lnL = rng.normal(40.0, 2.0, 50)
    dat, xml = _run(opts, theta, lnL, with_distance=True)
    tab, ref = _sim_rows(xml), _dat(dat)
    got = {name: np.array([getattr(r, name) for r in tab])
           for name in ("longitude", "latitude", "inclination", "polarization",
                        "coa_phase", "distance", "alpha1")}
    np.testing.assert_allclose(got["longitude"], ref["right_ascension"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(got["latitude"], ref["declination"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(got["inclination"], ref["inclination"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(got["polarization"], ref["psi"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(got["coa_phase"], ref["phi_orb"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(got["distance"], ref["distance"], rtol=0, atol=1e-4)
    # lnL rides alpha1, as it does in batchmode's export.
    np.testing.assert_allclose(got["alpha1"], ref["loglikelihood"], rtol=0, atol=1e-5)


def test_time_is_the_fiducial_epoch_on_every_row(tmp_path):
    rng = np.random.default_rng(5)
    opts = _opts(tmp_path)
    epoch = 1187008882.43
    _, xml = _run(opts, _theta(12, 6, rng), np.full(12, 30.0), True, epoch=epoch)
    for row in _sim_rows(xml):
        t = row.geocent_end_time + 1e-9 * row.geocent_end_time_ns
        assert math.isclose(t, epoch, rel_tol=0, abs_tol=1e-6)


def test_intrinsic_columns_and_evidence_row(tmp_path):
    rng = np.random.default_rng(6)
    opts = _opts(tmp_path)
    _, xml = _run(opts, _theta(9, 6, rng), np.full(9, 12.0), True,
                  logZ=41.25, sigma_lnL=0.031, neff=88.5, ntotal=4096)
    doc = _load(xml)
    for row in lsctables.SimInspiralTable.get_table(doc):
        assert math.isclose(row.mass1, 1.5, rel_tol=1e-9)
        assert math.isclose(row.mass2, 1.3, rel_tol=1e-9)
        assert math.isclose(row.spin1z, 0.11, rel_tol=1e-9)
        assert math.isclose(row.alpha5, 300.0, rel_tol=1e-9)
    sngl = lsctables.SnglInspiralTable.get_table(doc)
    assert len(sngl) == 1
    # append_likelihood_result_to_xmldoc: snr=logZ, tau0=neff, tau3=converged.
    assert math.isclose(sngl[0].snr, 41.25, rel_tol=1e-9)
    assert math.isclose(sngl[0].tau0, 88.5, rel_tol=1e-9)
    assert math.isclose(sngl[0].mass1, 1.5, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Every --mode layout, or an explicit refusal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,ndim", [
    ("flowmc-phimarg", 4),
    ("nuts-phimarg", 4),
    ("flowmc-dpsimarg", 4),
    ("flowmc-phipsimarg", 3),
    ("laplace-is", 5),
])
def test_marginalized_layouts_export_and_agree_with_their_dat(tmp_path, mode, ndim):
    rng = np.random.default_rng(7)
    opts = _opts(tmp_path, mode=mode)
    theta = _theta(16, ndim, rng)
    if mode == "flowmc-dpsimarg":       # ra, dec, phiref, incl
        theta = np.column_stack([theta[:, 0], theta[:, 1], theta[:, 3], theta[:, 2]])
    lnL = rng.normal(20.0, 1.0, 16)
    dat, xml = _run(opts, theta, lnL, with_distance=False)
    tab, ref = _sim_rows(xml), _dat(dat)
    assert len(tab) == len(ref) == 16
    np.testing.assert_allclose([r.longitude for r in tab],
                               ref["right_ascension"], rtol=0, atol=1e-6)
    np.testing.assert_allclose([r.inclination for r in tab],
                               ref["inclination"], rtol=0, atol=1e-6)
    # Distance was marginalized analytically: no per-sample draw exists, so the
    # reference distance the templates were built at is written, exactly as
    # batchmode does.  A silent per-sample value here would be fabricated.
    np.testing.assert_allclose([r.distance for r in tab],
                               factored_likelihood.distMpcRef, rtol=0, atol=1e-9)
    if "psi" in (ref.dtype.names or ()):
        np.testing.assert_allclose([r.polarization for r in tab],
                                   ref["psi"], rtol=0, atol=1e-6)
    else:
        # psi was integrated out: NaN, never a fiducial 0.0 that would read as a
        # polarization measurement downstream.
        assert all(math.isnan(r.polarization) for r in tab)


def test_six_d_distance_is_the_per_sample_draw_not_the_reference(tmp_path):
    rng = np.random.default_rng(8)
    opts = _opts(tmp_path)
    theta = _theta(24, 6, rng)
    _, xml = _run(opts, theta, np.full(24, 15.0), True)
    got = np.array([r.distance for r in _sim_rows(xml)])
    np.testing.assert_allclose(np.sort(got), np.sort(theta[:, 5]), rtol=0, atol=1e-4)
    assert got.std() > 1.0                      # a real posterior, not a constant
    assert not np.allclose(got, factored_likelihood.distMpcRef)


def test_phase_marginalized_writes_the_reference_phase(tmp_path):
    rng = np.random.default_rng(9)
    opts = _opts(tmp_path, phase_marginalization=True)
    theta = _theta(10, 6, rng)
    dat, xml = _run(opts, theta, np.full(10, 15.0), True)
    assert "phi_orb" not in (_dat(dat).dtype.names or ())
    assert all(r.coa_phase == 0.0 for r in _sim_rows(xml))


def test_unmappable_layout_refuses_rather_than_guessing(tmp_path):
    opts = _opts(tmp_path, mode="laplace-is")
    with pytest.raises(RuntimeError, match="no sim_inspiral column mapping"):
        NS["_xml_extrinsic_columns"](opts, np.zeros((4, 7)), False)
    with pytest.raises(RuntimeError, match="not 6"):
        NS["_xml_extrinsic_columns"](opts, np.zeros((4, 5)), True)


def test_missing_provenance_arguments_skip_the_xml_but_keep_the_dat(tmp_path):
    """A caller without an evidence row loses the XML, loudly -- not the .dat.

    Raising here instead took test/jax/test_jax_fairdraw_export.py from 34
    passed to 22 failed: it drives write_samples directly to check the fair
    draw and has no intrinsic point or evidence row to hand it.
    """
    rng = np.random.default_rng(10)
    opts = _opts(tmp_path)
    NS["write_samples"](opts, 0, _theta(4, 6, rng), np.zeros(4), True)
    assert not Path(NS["xml_path"](opts, 0)).exists()
    assert Path(NS["samples_path"](opts, 0)).is_file()


def test_skipped_xml_names_every_missing_input_on_stderr(tmp_path, capsys):
    rng = np.random.default_rng(101)
    opts = _opts(tmp_path)
    NS["write_samples"](opts, 0, _theta(4, 6, rng), np.zeros(4), True,
                        P=_P(), logZ=3.0)          # epoch and sigma_lnL missing
    err = capsys.readouterr().err
    assert "SKIPPED" in err
    named = err.split("without", 1)[1]
    assert "fiducial_epoch" in named and "sigma_lnL" in named
    # the two that WERE supplied are not named as missing
    assert "logZ" not in named and "P," not in named


def test_skipping_the_xml_removes_an_earlier_runs_file(tmp_path):
    """The path is what convert_extr opens, so a stale file is collected as ours."""
    rng = np.random.default_rng(102)
    opts = _opts(tmp_path)
    stale = Path(NS["xml_path"](opts, 0))
    stale.write_bytes(b"not this run's output")
    NS["write_samples"](opts, 0, _theta(4, 6, rng), np.zeros(4), True)
    assert not stale.exists()


def test_an_attempted_xml_that_cannot_be_mapped_still_raises(tmp_path):
    """Skipping is only for ABSENT inputs; a bad layout stays a hard failure."""
    rng = np.random.default_rng(103)
    opts = _opts(tmp_path)
    theta = np.column_stack([_theta(6, 6, rng), rng.normal(size=6)])   # 7 columns
    with pytest.raises(RuntimeError, match="no sim_inspiral column mapping"):
        NS["write_samples"](opts, 0, theta, np.zeros(6), False, P=_P(),
                            fiducial_epoch=0.0, logZ=1.0, sigma_lnL=0.1)


# ---------------------------------------------------------------------------
# The prior columns: what the resamplers actually do with alpha2 / alpha3
# ---------------------------------------------------------------------------
def _consumer_weights(*xmls):
    """``exp(lnL - lnLmax) * (p/ps) / Npts``, as the resamplers form it.

    util_ResampleILEOutputWithExtrinsic.py reads ONE --fname, which may be a
    concatenation of several intrinsic points, and takes a single GLOBAL lnLmax
    over it; util_BatchConvertResampleILEOutput.py loops file by file with a
    per-file lnLmax.  Pass one path for the per-file case, several for the
    pooled one.  Npts is max(simulation_id)+1, which xmlutils makes the row
    count, and convert_output_format_ile2inference copies alpha1/alpha2/alpha3
    into lnL/p/ps unchanged.
    """
    lnL, p, ps, npts = [], [], [], []
    for xml in xmls:
        rows = _sim_rows(xml)
        n = max(int(r.simulation_id) for r in rows) + 1
        lnL += [r.alpha1 for r in rows]
        p += [r.alpha2 for r in rows]
        ps += [r.alpha3 for r in rows]
        npts += [n] * len(rows)
    lnL, p, ps, npts = (np.array(x, dtype=float) for x in (lnL, p, ps, npts))
    return np.exp(lnL - lnL.max()) * (p / ps) / npts


def test_resampler_weights_are_uniform_over_the_equal_weight_rows(tmp_path):
    """The exported rows are ALREADY a fair draw, so the consumer weight is flat.

    With p/ps = 1 it was not: the resampler tilted by exp(lnL) a second time and
    returned samples proportional to prior * L^2.
    """
    rng = np.random.default_rng(11)
    opts = _opts(tmp_path)
    lnL = rng.normal(300.0, 6.0, 500)         # realistic magnitude and spread
    _, xml = _run(opts, _theta(500, 6, rng), lnL, True, logZ=297.5)
    w = _consumer_weights(xml)
    assert np.all(np.isfinite(w)) and np.all(w > 0)
    # Not exact, and the residual is pinned rather than waved at: alpha1 and
    # alpha3 are real_4 in sim_inspiral and ligolw's ASCII round trip is not an
    # exact float32 one, so the cancellation survives the file to ~4e-5 at
    # lnL ~ 300 (measured 3.9e-5).  1e-4 is that with headroom, and 3 orders
    # below the 29% narrowing the p/ps = 1 export cost.
    np.testing.assert_allclose(w, w[0], rtol=1e-4, atol=0)


def test_the_double_likelihood_tilt_is_gone(tmp_path):
    """Measured, not asserted: resample the export and compare the spread.

    The rows are draws from a Gaussian posterior in ra of width 0.05.  A
    consumer that applies exp(lnL) once more narrows them by exactly
    1/sqrt(2); the reviewer measured 0.049845 -> 0.035384, ratio 0.7099, under
    p/ps = 1.
    """
    rng = np.random.default_rng(12)
    n, width = 40000, 0.05
    theta = _theta(n, 6, rng)
    theta[:, 0] = rng.normal(1.2, width, n)                   # ra ~ posterior
    lnL = 300.0 - 0.5 * ((theta[:, 0] - 1.2) / width) ** 2
    opts = _opts(tmp_path)
    _, xml = _run(opts, theta, lnL, True, logZ=float(lnL.max()) - 1.0)
    w = _consumer_weights(xml)
    ra = np.array([r.longitude for r in _sim_rows(xml)])
    drawn = rng.choice(ra, size=n, p=w / w.sum())
    ratio = drawn.std() / ra.std()
    assert 0.97 < ratio < 1.03, "spread ratio %.4f (0.7071 = tilted twice)" % ratio


def test_pooled_weight_of_an_intrinsic_point_tracks_its_evidence(tmp_path):
    """The offset is logZ, and this is the case that decides it.

    util_ResampleILEOutputWithExtrinsic.py may be handed several intrinsic
    points in one file.  Each point's share of the total must be its evidence --
    the same relative weight conventional ILE's retained cloud carries, since
    sum_rows exp(lnL) (p/ps) / Npts is the importance-sampling estimate of Z.
    A per-file lnLmax offset would make the share track the point's PEAK
    likelihood instead: correct within a file, wrong between files.
    """
    rng = np.random.default_rng(13)
    logZ_a, logZ_b = 312.0, 305.5
    xmls = []
    for k, (logZ, n) in enumerate(((logZ_a, 40), (logZ_b, 90))):
        sub = tmp_path / str(k)
        sub.mkdir()
        opts = _opts(sub)
        # Different row counts AND different peak likelihoods, so neither Npts
        # nor lnLmax can stand in for the evidence by accident.
        lnL = rng.normal(logZ + 3.0 * (1 + k), 2.0, n)
        xmls.append(_run(opts, _theta(n, 6, rng), lnL, True, logZ=logZ)[1])
    w = _consumer_weights(*xmls)
    n_a = len(_sim_rows(xmls[0]))
    got = w[:n_a].sum() / w[n_a:].sum()
    assert math.isclose(got, math.exp(logZ_a - logZ_b), rel_tol=1e-4)


def test_each_events_rows_are_numbered_from_zero_within_its_own_file(tmp_path):
    """Npts must be THIS event's row count, in a batch as well as alone.

    samples_to_siminsp_row takes simulation_id from lsctables' get_next_id(),
    a CLASS attribute shared by every SimInspiralTable in the process.  One ILE
    process writes one XML per --n-events-to-analyze event, so without an
    explicit sample_n the second event's ids continue from the first, both
    resamplers read Npts = max(simulation_id)+1 as the running total, and that
    event's share of the pooled posterior is scaled by its batch position.
    """
    rng = np.random.default_rng(141)
    ids = []
    for k, n in enumerate((13, 21)):
        sub = tmp_path / ("event%d" % k)
        sub.mkdir()
        opts = _opts(sub)
        _, xml = _run(opts, _theta(n, 6, rng), rng.normal(30.0, 1.0, n), True)
        ids.append([int(r.simulation_id) for r in _sim_rows(xml)])
    assert ids[0] == list(range(13))
    assert ids[1] == list(range(21))


def test_weights_stay_finite_when_rows_sit_far_below_the_evidence(tmp_path):
    """Clipped, deliberately, to zero weight -- never to a nan.

    util_BatchConvertResampleILEOutput.py rewrites a nan weight to 1e-2, which
    would put a LARGE weight on a row carrying no posterior mass.
    """
    rng = np.random.default_rng(14)
    opts = _opts(tmp_path)
    lnL = np.full(40, 300.0)
    lnL[:5] = -5000.0                      # |lnL - logZ| far past the clip
    _, xml = _run(opts, _theta(40, 6, rng), lnL, True, logZ=300.0)
    tab = _sim_rows(xml)
    ps = np.array([r.alpha3 for r in tab])
    assert np.all(np.isfinite(ps)) and np.all(ps > 0)
    w = _consumer_weights(xml)
    assert np.all(np.isfinite(w))
    assert np.all(w[:5] == 0.0)
    np.testing.assert_allclose(w[5:], w[5], rtol=1e-4, atol=0)


def test_nonfinite_evidence_falls_back_to_the_file_lnLmax_and_says_so(tmp_path, capsys):
    rng = np.random.default_rng(15)
    opts = _opts(tmp_path)
    lnL = rng.normal(100.0, 1.0, 30)
    _, xml = _run(opts, _theta(30, 6, rng), lnL, True, logZ=float("-inf"))
    assert "not finite" in capsys.readouterr().err
    w = _consumer_weights(xml)
    assert np.all(np.isfinite(w))
    np.testing.assert_allclose(w, w[0], rtol=1e-4, atol=0)


# ---------------------------------------------------------------------------
# --distance-marginalization --phase-marginalization: the 5-D branch
# ---------------------------------------------------------------------------
def test_five_d_phase_marginalized_writes_the_reference_phase(tmp_path):
    """The real --distance-marginalization --phase-marginalization combination.

    Only the 6-D branch was covered, so replacing the 5-D branch's
    ``phi_reference if phase_marginalized else theta[:, 4]`` with plain
    ``theta[:, 4]`` left all 18 tests green while exporting a SAMPLED phase
    column from a run whose phase was integrated out analytically.
    """
    rng = np.random.default_rng(16)
    opts = _opts(tmp_path, phase_marginalization=True)
    theta = _theta(24, 5, rng)
    theta[:, 4] = rng.uniform(0.5, 6.0, 24)      # nothing near the reference 0.0
    dat, xml = _run(opts, theta, rng.normal(30.0, 1.0, 24), with_distance=False)
    got = np.array([r.coa_phase for r in _sim_rows(xml)])
    np.testing.assert_allclose(got, 0.0, rtol=0, atol=0)
    assert not np.allclose(got, theta[:, 4])
    # and the sidecar agrees: no phi_orb column when phase was marginalized
    assert "phi_orb" not in (_dat(dat).dtype.names or ())


def test_five_d_without_phase_marginalization_keeps_the_sampled_phase(tmp_path):
    rng = np.random.default_rng(17)
    opts = _opts(tmp_path, phase_marginalization=False)
    theta = _theta(24, 5, rng)
    dat, xml = _run(opts, theta, rng.normal(30.0, 1.0, 24), with_distance=False)
    got = np.array([r.coa_phase for r in _sim_rows(xml)])
    np.testing.assert_allclose(got, theta[:, 4], rtol=0, atol=1e-6)
    np.testing.assert_allclose(got, _dat(dat)["phi_orb"], rtol=0, atol=1e-6)


def test_analyze_one_still_supplies_every_xml_input():
    """Structural: the loud skip must never become the production behaviour.

    write_samples now SKIPS the XML when an input is missing instead of raising,
    so a call site that quietly stopped passing one would print a warning into a
    Condor log and produce a DAG that collects nothing -- the exact failure the
    export was added to fix.  Pin the call site instead.
    """
    tree = ast.parse(DRIVER.read_text(), filename=str(DRIVER))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "write_samples"]
    assert len(calls) == 1, "expected one write_samples call, found %d" % len(calls)
    passed = {k.arg for k in calls[0].keywords if k.arg}
    assert {"P", "fiducial_epoch", "logZ", "sigma_lnL"} <= passed, sorted(passed)


# ---------------------------------------------------------------------------
# --sampler-method AV: the WEIGHTED cloud, with its own prior columns
# ---------------------------------------------------------------------------
def _weighted(opts, theta, lnL, log_p, log_ps, **kw):
    NS["write_samples"](opts, 0, theta, lnL, True, P=_P(),
                        fiducial_epoch=1126259462.0, logZ=41.25,
                        sigma_lnL=0.031, report_neff=88.5, ntotal=4096,
                        log_prior=log_p, log_s_prior=log_ps, **kw)
    return (Path(NS["samples_path"](opts, 0)), Path(NS["xml_path"](opts, 0)))


def test_av_publishes_the_real_prior_pair_not_a_cancellation(tmp_path):
    """samplers.adaptive_volume_sample keeps the retained population under AV.

    igrand_fairdraw_samples is False there, so log_joint_prior and
    log_joint_s_prior still exist per row.  Write them: exp(lnL) * p/ps is then
    the true importance weight, with nothing cancelling against lnL.
    """
    rng = np.random.default_rng(20)
    n = 300
    theta = _theta(n, 6, rng)
    lnL = rng.normal(300.0, 4.0, n)
    log_p = rng.normal(-9.0, 0.5, n)
    log_ps = rng.normal(-7.0, 0.8, n)
    _, xml = _weighted(_opts(tmp_path), theta, lnL, log_p, log_ps)
    tab = _sim_rows(xml)
    got = np.array([r.alpha2 / r.alpha3 for r in tab])
    np.testing.assert_allclose(got, np.exp(log_p - log_ps), rtol=1e-5, atol=0)
    # NOT the cancelling pair: these weights vary, because the cloud does.
    w = _consumer_weights(xml)
    assert w.std() / w.mean() > 0.1


def test_av_xml_is_the_weighted_cloud_while_the_dat_stays_a_fair_draw(tmp_path):
    """The two products stop being the same rows here, on purpose.

    util_ConvertJAXILEFairdraws.py and every hand-rolled collector read the
    .dat as equal-weight rows, so it keeps its fair draw; the XML's consumer
    reweights, so it gets the cloud the weights belong to.
    """
    rng = np.random.default_rng(21)
    n = 4000
    theta = _theta(n, 6, rng)
    lnL = rng.normal(300.0, 3.0, n)
    log_p = rng.normal(-9.0, 0.5, n)
    log_ps = rng.normal(-7.0, 0.8, n)
    opts = _opts(tmp_path, n_fairdraw_extrinsic_samples=50)
    logw = lnL + log_p - log_ps
    dat, xml = _weighted(opts, theta, lnL, log_p, log_ps, logw=logw)
    # The sidecar is the fair draw, at or under the requested count (ILE's
    # 1.5*ESS clamp can cut it further); the XML is the whole weighted cloud.
    n_dat = len(_dat(dat))
    assert 0 < n_dat <= 50
    assert len(_sim_rows(xml)) == n
    assert n_dat < n


def test_av_prior_pair_must_line_up_with_the_rows(tmp_path):
    rng = np.random.default_rng(22)
    theta = _theta(12, 6, rng)
    with pytest.raises(RuntimeError, match="disagree in length"):
        _weighted(_opts(tmp_path), theta, np.zeros(12),
                  np.zeros(11), np.zeros(12))


def test_out_of_range_priors_are_shifted_together_so_the_ratio_survives(tmp_path):
    """alpha2 and alpha3 are real_4, and only their ratio is ever read.

    A common shift on both logs is weight preserving; scaling one alone would
    change this file's weight against the other intrinsic points.
    """
    rng = np.random.default_rng(23)
    n = 200
    log_p = rng.normal(-400.0, 2.0, n)      # exp() underflows float32
    log_ps = rng.normal(-398.0, 2.0, n)
    p, ps = NS["_xml_prior_columns"](log_p, log_ps)
    assert np.all(np.isfinite(p)) and np.all(p > 0)
    assert np.all(np.isfinite(ps)) and np.all(ps > 0)
    np.testing.assert_allclose(p / ps, np.exp(log_p - log_ps), rtol=1e-9, atol=0)


def test_ordinary_priors_are_written_verbatim(tmp_path):
    """No shift in the normal case: the columns are the classic ILE's values."""
    rng = np.random.default_rng(24)
    log_p = rng.normal(-9.0, 0.4, 50)
    log_ps = rng.normal(-7.0, 0.4, 50)
    p, ps = NS["_xml_prior_columns"](log_p, log_ps)
    np.testing.assert_allclose(p, np.exp(log_p), rtol=0, atol=0)
    np.testing.assert_allclose(ps, np.exp(log_ps), rtol=0, atol=0)


def test_analyze_one_hands_the_av_prior_pair_to_the_export():
    """Structural: the AV branch's log_joint_prior must reach write_samples.

    Losing the keyword would fall back to the fair-drawn cancellation silently,
    which is a different (and for AV, wrong) export.
    """
    tree = ast.parse(DRIVER.read_text(), filename=str(DRIVER))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "write_samples"]
    passed = {k.arg for k in calls[0].keywords if k.arg}
    assert {"log_prior", "log_s_prior"} <= passed, sorted(passed)
    src = DRIVER.read_text()
    assert 'res.get("log_joint_prior")' in src
    assert 'res.get("log_joint_s_prior")' in src
