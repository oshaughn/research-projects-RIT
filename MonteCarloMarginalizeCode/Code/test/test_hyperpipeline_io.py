"""
Tests for RIFT/misc/hyperpipeline_io.py.

These intentionally avoid importing lalsuite or any of the heavy ILE/CIP
modules so the round-trip can be exercised in a minimal environment::

    python test/test_hyperpipeline_io.py
    pytest test/test_hyperpipeline_io.py
"""

import os
import sys
import tempfile
import numpy as np

# Load hyperpipeline_io directly from its file so this test does not
# trigger the RIFT.__init__ import chain (which pulls in lalsimutils +
# scipy + lalsuite -- not needed for testing pure-numpy I/O code).
HERE = os.path.dirname(os.path.abspath(__file__))
_HPIO_PATH = os.path.normpath(os.path.join(
    HERE, "..", "RIFT", "misc", "hyperpipeline_io.py"))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("hyperpipeline_io", _HPIO_PATH)
hpio = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(hpio)


def _roundtrip(columns, values):
    """Write *values* under *columns*, read back, return structured array."""
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as fp:
        fname = fp.name
    try:
        hpio.write_table(fname, columns, np.asarray(values, dtype=float))
        arr, hdr = hpio.read_table(fname)
        assert hdr == tuple(columns), (hdr, tuple(columns))
        return arr
    finally:
        os.unlink(fname)


def test_default_roundtrip():
    cols = hpio.build_column_list()  # default: 10-col bbh
    rows = [
        [-12.34, 0.05, 30.0, 25.0, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
        [-13.50, 0.07, 28.0, 27.0, 0.0, 0.0, 0.4, 0.0, 0.0, -0.4],
    ]
    arr = _roundtrip(cols, rows)
    assert len(arr) == 2
    np.testing.assert_allclose(arr["lnL"], [-12.34, -13.50])
    np.testing.assert_allclose(arr["sigma_lnL"], [0.05, 0.07])
    np.testing.assert_allclose(arr["m1"], [30.0, 28.0])
    np.testing.assert_allclose(arr["a2z"], [-0.3, -0.4])
    print("test_default_roundtrip: OK")


def test_eccentricity_columns():
    cols = hpio.build_column_list(use_eccentricity=True, use_meanPerAno=True)
    assert cols[-2:] == ("eccentricity", "meanPerAno")
    rows = [[-1.0, 0.01, 1.4, 1.4, 0,0,0, 0,0,0, 0.3, 1.7]]
    arr = _roundtrip(cols, rows)
    np.testing.assert_allclose(arr["eccentricity"], [0.3])
    np.testing.assert_allclose(arr["meanPerAno"], [1.7])
    print("test_eccentricity_columns: OK")


def test_tides_with_eos_index():
    cols = hpio.build_column_list(use_tides=True, use_eos_index=True)
    assert "lambda1" in cols and "eos_table_index" in cols
    rows = [[-0.5, 0.02, 1.4, 1.35, 0,0,0, 0,0,0, 500.0, 480.0, 7.0]]
    arr = _roundtrip(cols, rows)
    np.testing.assert_allclose(arr["lambda1"], [500.0])
    np.testing.assert_allclose(arr["eos_table_index"], [7.0])
    print("test_tides_with_eos_index: OK")


def test_eccentric_tides_eob_hyperbolic_columns():
    """All JL physics groups survive named I/O and legacy adaptation together."""
    kw = dict(use_eccentricity=True, use_meanPerAno=True,
              use_tides=True, use_eob_parameters=True,
              use_hyperbolic=True)
    cols = hpio.build_column_list(**kw)
    values = {
        "lnL": -3.5, "sigma_lnL": 0.02,
        "m1": 2.0, "m2": 1.4,
        "a1x": 0.0, "a1y": 0.0, "a1z": 0.1,
        "a2x": 0.0, "a2y": 0.0, "a2z": -0.1,
        "eccentricity": 0.2, "meanPerAno": 1.1,
        "lambda1": 0.0, "lambda2": 300.0,
        "a6c": -45.0, "E0": 1.05, "p_phi0": 4.2,
    }
    arr = _roundtrip(cols, [[values[c] for c in cols]])
    legacy = hpio.to_legacy_dat(arr, **kw)
    ix = hpio.legacy_column_indices(**kw)
    for name in ("lambda1", "a6c", "E0", "p_phi0",
                 "eccentricity", "meanPerAno", "lnL", "sigma_lnL"):
        assert ix[name] is not None, (name, ix)
        np.testing.assert_allclose(legacy[:, ix[name]], [values[name]])
    assert ix["lambda1"] < ix["a6c"] < ix["E0"] < ix["eccentricity"] < ix["lnL"]
    print("test_eccentric_tides_eob_hyperbolic_columns: OK")


def test_sky_columns():
    cols = hpio.build_column_list(use_sky=True)
    assert cols[-2:] == ("ecliptic_longitude", "ecliptic_latitude")
    rows = [[-0.5, 0.02, 1.4, 1.35, 0,0,0, 0,0,0, 1.25, -0.4]]
    arr = _roundtrip(cols, rows)
    np.testing.assert_allclose(arr["ecliptic_longitude"], [1.25])
    np.testing.assert_allclose(arr["ecliptic_latitude"], [-0.4])

    legacy = hpio.to_legacy_dat(arr, use_sky=True)
    ix = hpio.legacy_column_indices(use_sky=True)
    np.testing.assert_allclose(legacy[:, ix["ecliptic_longitude"]], [1.25])
    np.testing.assert_allclose(legacy[:, ix["ecliptic_latitude"]], [-0.4])
    assert ix["lnL"] == 11
    print("test_sky_columns: OK")


def test_to_legacy_dat_default():
    """Hyperpipeline -> legacy positional matrix, default 10-col case."""
    cols = hpio.build_column_list()
    rows = [[-12.34, 0.05, 30.0, 25.0, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3]]
    arr = _roundtrip(cols, rows)
    legacy = hpio.to_legacy_dat(arr)
    # Layout: event_id, m1, m2, a1xyz, a2xyz, lnL, sigma_lnL = 11 columns
    assert legacy.shape == (1, 11), legacy.shape
    # event_id is synthesized to -1
    assert legacy[0, 0] == -1.0
    # m1, m2
    np.testing.assert_allclose(legacy[0, 1:3], [30.0, 25.0])
    # spins
    np.testing.assert_allclose(legacy[0, 3:9], [0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    # lnL, sigma_lnL at the end
    np.testing.assert_allclose(legacy[0, 9:11], [-12.34, 0.05])
    print("test_to_legacy_dat_default: OK")


def test_legacy_column_indices_consistency():
    """The col_lnL etc. indices must match the actual layout produced by
    to_legacy_dat across all combinations of optional groups."""
    combos = [
        dict(),
        dict(use_distance=True),
        dict(use_tides=True),
        dict(use_tides=True, use_eos_index=True),
        dict(use_eccentricity=True),
        dict(use_eccentricity=True, use_meanPerAno=True),
        dict(use_distance=True, use_tides=True),
        dict(use_eob_parameters=True),
        dict(use_hyperbolic=True),
        dict(use_tides=True, use_eob_parameters=True, use_hyperbolic=True,
             use_eccentricity=True, use_meanPerAno=True),
    ]
    for kw in combos:
        cols = hpio.build_column_list(**kw)
        # synthesize a single row of zeros
        rows = [[0.0] * len(cols)]
        arr = _roundtrip(cols, rows)
        legacy = hpio.to_legacy_dat(arr, **kw)
        ix = hpio.legacy_column_indices(**kw)
        assert ix["lnL"] < legacy.shape[1], (kw, ix, legacy.shape)
        assert ix["sigma_lnL"] < legacy.shape[1], (kw, ix, legacy.shape)
        if kw.get("use_distance"):
            assert ix["distance"] is not None
            assert ix["distance"] < legacy.shape[1]
        if kw.get("use_tides"):
            assert ix["lambda1"] is not None
        if kw.get("use_eccentricity"):
            assert ix["eccentricity"] is not None
        if kw.get("use_eob_parameters"):
            assert ix["a6c"] is not None
        if kw.get("use_hyperbolic"):
            assert ix["E0"] is not None and ix["p_phi0"] is not None
    print("test_legacy_column_indices_consistency: OK")


def test_sniff_distinguishes_legacy():
    """sniff() must return False on a legacy-style file (no header)."""
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as fp:
        # Plain numeric line -- looks like a legacy ILE shard.
        fp.write("0 30.0 25.0 0 0 0.5 0 0 -0.3 -12.34 0.05 1000 200\n")
        fname = fp.name
    try:
        assert not hpio.sniff(fname), "legacy file misidentified as hyperpipeline"
    finally:
        os.unlink(fname)
    print("test_sniff_distinguishes_legacy: OK")


def test_sniff_recognizes_new_format():
    cols = hpio.build_column_list()
    rows = [[0.0] * len(cols)]
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as fp:
        fname = fp.name
    try:
        hpio.write_table(fname, cols, rows)
        assert hpio.sniff(fname), "hyperpipeline file not recognized"
    finally:
        os.unlink(fname)
    print("test_sniff_recognizes_new_format: OK")


def test_env_flag():
    saved = os.environ.pop(hpio.ENV_FLAG, None)
    try:
        assert not hpio.is_active()
        for v in ("1", "true", "YES", "On"):
            os.environ[hpio.ENV_FLAG] = v
            assert hpio.is_active(), v
        for v in ("0", "false", "no", "", "wat"):
            os.environ[hpio.ENV_FLAG] = v
            assert not hpio.is_active(), v
    finally:
        os.environ.pop(hpio.ENV_FLAG, None)
        if saved is not None:
            os.environ[hpio.ENV_FLAG] = saved
    print("test_env_flag: OK")


def test_concatenated_shards():
    """Concatenate two single-row hyperpipeline files (as ILE produces) and
    confirm the reader still parses them.  Each shard carries its own
    magic+header lines, both of which must be treated as comments."""
    cols = hpio.build_column_list()
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as fp:
        a = fp.name
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as fp:
        b = fp.name
    with tempfile.NamedTemporaryFile("w", suffix=".composite", delete=False) as fp:
        c = fp.name
    try:
        hpio.write_row(a, cols, [-1.0, 0.1, 30, 25, 0,0,0, 0,0,0])
        hpio.write_row(b, cols, [-2.0, 0.2, 31, 26, 0,0,0, 0,0,0])
        with open(c, "w") as out:
            for src in (a, b):
                with open(src) as inp:
                    out.write(inp.read())
        arr, hdr = hpio.read_table(c)
        assert len(arr) == 2, len(arr)
        np.testing.assert_allclose(sorted(arr["lnL"]), [-2.0, -1.0])
    finally:
        for f in (a, b, c):
            os.unlink(f)
    print("test_concatenated_shards: OK")


def test_read_many_skips_empties_and_mismatches():
    cols_a = hpio.build_column_list()
    cols_b = hpio.build_column_list(use_eccentricity=True)
    good1 = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    good2 = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    bad_cols = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    empty = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    missing = "/tmp/this_file_does_not_exist_12345.dat"
    try:
        hpio.write_row(good1, cols_a, [-1, 0.1, 30, 25, 0,0,0, 0,0,0])
        hpio.write_row(good2, cols_a, [-2, 0.2, 31, 26, 0,0,0, 0,0,0])
        hpio.write_row(bad_cols, cols_b, [-3, 0.3, 30, 25, 0,0,0, 0,0,0, 0.1])
        # `empty` already exists with zero bytes
        arr, hdr = hpio.read_many([good1, empty, good2, bad_cols, missing])
        assert hdr == cols_a
        assert len(arr) == 2  # bad/empty/missing all skipped
        np.testing.assert_allclose(sorted(arr["lnL"]), [-2.0, -1.0])
    finally:
        for f in (good1, good2, bad_cols, empty):
            if os.path.exists(f):
                os.unlink(f)
    print("test_read_many_skips_empties_and_mismatches: OK")


def test_consolidate_weighted_average():
    """Two rows with identical intrinsic params should weighted-average."""
    cols = hpio.build_column_list()
    f1 = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    f2 = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    f3 = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    try:
        # Two rows at the SAME intrinsic point with very different sigma --
        # the consolidated lnL should be pulled toward the lower-sigma one.
        hpio.write_row(f1, cols, [-10.0, 0.01, 30, 25, 0,0,0.5, 0,0,-0.3])
        hpio.write_row(f2, cols, [-12.0, 0.50, 30, 25, 0,0,0.5, 0,0,-0.3])
        # And one row at a DIFFERENT intrinsic point; should pass through.
        hpio.write_row(f3, cols, [-5.0, 0.05, 28, 27, 0,0,0.4, 0,0,-0.4])
        arr, hdr = hpio.read_many([f1, f2, f3])
        assert len(arr) == 3
        out, out_cols = hpio.consolidate(arr, hdr)
        # Two unique intrinsic points -> 2 consolidated rows.
        assert len(out) == 2, len(out)
        # Sorted by lnL desc; check the order.
        assert out["lnL"][0] >= out["lnL"][1]
        # The weighted-average for the duplicated point: with sigma=0.01 vs
        # 0.50, the weight is overwhelmingly on the first row (~2500x).
        # So the consolidated lnL should be very close to -10.0, not the
        # arithmetic mean of -11.0.
        cons = out[(out["m1"] == 30.0) & (out["m2"] == 25.0)][0]
        assert abs(cons["lnL"] - (-10.0)) < 0.05, cons["lnL"]
        # The pass-through row should retain its original values.
        passthru = out[(out["m1"] == 28.0) & (out["m2"] == 27.0)][0]
        np.testing.assert_allclose(passthru["lnL"], -5.0)
        np.testing.assert_allclose(passthru["sigma_lnL"], 0.05)
    finally:
        for f in (f1, f2, f3):
            os.unlink(f)
    print("test_consolidate_weighted_average: OK")


class _FakeP(object):
    """Stand-in for lalsimutils.ChooseWaveformParams.

    Carries m1, m2 in kg and dist in metres (matching the real class's
    SI-internal convention).  Spin / lambda / eccentricity components are
    plain dimensionless floats.  Only the attribute / setattr surface used
    by hyperpipeline_io is implemented.
    """
    def __init__(self):
        self.m1 = 0.0; self.m2 = 0.0
        self.s1x = 0.0; self.s1y = 0.0; self.s1z = 0.0
        self.s2x = 0.0; self.s2y = 0.0; self.s2z = 0.0
        self.lambda1 = 0.0; self.lambda2 = 0.0
        self.eccentricity = 0.0; self.meanPerAno = 0.0
        self.a6c = 10000.0; self.E0 = 0.0; self.p_phi0 = 0.0
        self.eos_table_index = 0.0
        self.dist = 0.0
        self.phi = 0.0
        self.theta = 0.0
        self.fref = 20.0


class _FakeLal(object):
    """Two SI scale constants matched to lal.MSUN_SI / lal.PC_SI."""
    MSUN_SI = 1.98892e30
    PC_SI = 3.085677581e16


def test_grid_write_read_roundtrip_with_units():
    """Write a P_list as a hyperpipeline grid, read it back, and verify
    that mass values round-trip via the on-disk solar-mass <-> kg
    conversion declared in PARAM_DISK_TO_SI."""
    fake_lal = _FakeLal()
    P_list = []
    for m1_solar, m2_solar, s1z, ecc in [
        (30.0, 25.0, 0.4, 0.1),
        (15.0, 12.0, -0.3, 0.05),
        (1.4, 1.35, 0.0, 0.0),
    ]:
        P = _FakeP()
        P.m1 = m1_solar * fake_lal.MSUN_SI  # kg in P
        P.m2 = m2_solar * fake_lal.MSUN_SI
        P.s1z = s1z
        P.eccentricity = ecc
        P_list.append(P)

    cols = hpio.build_column_list(use_eccentricity=True)
    fname = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    try:
        hpio.write_grid_from_P_list(fname, P_list, cols, lal_module=fake_lal,
                                    lalsimutils_module=None)
        # On-disk file should have m1 in solar masses.
        arr, hdr = hpio.read_table(fname)
        assert hdr == cols
        np.testing.assert_allclose(arr["m1"], [30.0, 15.0, 1.4])
        np.testing.assert_allclose(arr["m2"], [25.0, 12.0, 1.35])
        np.testing.assert_allclose(arr["eccentricity"], [0.1, 0.05, 0.0])
        # lnL/sigma_lnL filled with 0 (no values supplied).
        np.testing.assert_allclose(arr["lnL"], [0.0, 0.0, 0.0])

        # Now read back into fresh P objects and verify mass is in kg.
        P_back, hdr_back = hpio.read_grid_to_P_list(
            fname, P_factory=_FakeP, lal_module=fake_lal,
            valid_params={"m1","m2","s1x","s1y","s1z","s2x","s2y","s2z",
                          "lambda1","lambda2","eccentricity","meanPerAno",
                          "eos_table_index","dist"})
        assert len(P_back) == 3
        for orig, back in zip(P_list, P_back):
            assert abs(orig.m1 - back.m1) < 1e10, (orig.m1, back.m1)  # kg
            assert abs(orig.m2 - back.m2) < 1e10, (orig.m2, back.m2)
            assert abs(orig.s1z - back.s1z) < 1e-9
            assert abs(orig.eccentricity - back.eccentricity) < 1e-9
    finally:
        os.unlink(fname)
    print("test_grid_write_read_roundtrip_with_units: OK")


def test_grid_distance_unit_conversion():
    """Distance round-trips: stored in Mpc on disk, in metres in P."""
    fake_lal = _FakeLal()
    P = _FakeP()
    P.m1 = 30.0 * fake_lal.MSUN_SI
    P.m2 = 25.0 * fake_lal.MSUN_SI
    P.dist = 500.0 * 1e6 * fake_lal.PC_SI  # 500 Mpc in metres
    cols = ("lnL", "sigma_lnL", "m1", "m2",
            "a1x", "a1y", "a1z", "a2x", "a2y", "a2z", "dist")
    fname = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    try:
        hpio.write_grid_from_P_list(fname, [P], cols, lal_module=fake_lal)
        arr, _ = hpio.read_table(fname)
        np.testing.assert_allclose(arr["dist"], [500.0])  # Mpc on disk
        # Read back into a P
        P_back_list, _ = hpio.read_grid_to_P_list(
            fname, P_factory=_FakeP, lal_module=fake_lal,
            valid_params={"m1","m2","dist","a1x","a1y","a1z","a2x","a2y","a2z"})
        # dist should be back in metres; tolerance is generous because
        # 1 Mpc = ~3e22 m so absolute differences are huge.
        rel = abs(P.dist - P_back_list[0].dist) / P.dist
        assert rel < 1e-12, rel
    finally:
        os.unlink(fname)
    print("test_grid_distance_unit_conversion: OK")


def test_grid_auto_suffix_append():
    """write_grid_from_P_list mirrors ChooseWaveformParams_array_to_xml's
    auto-suffix-append behaviour: callers can pass a basename without
    extension and the writer adds '.dat' for them.  This is what makes
    the BasicIteration call sites (which pass `overlap-grid-N` to both
    the legacy XML writer and the hyperpipeline writer) work uniformly."""
    P = _FakeP()
    P.m1 = 30.0 * _FakeLal.MSUN_SI
    P.m2 = 25.0 * _FakeLal.MSUN_SI
    cols = hpio.build_column_list()
    base = tempfile.NamedTemporaryFile("w", suffix="", delete=False).name
    try:
        hpio.write_grid_from_P_list(base, [P], cols, lal_module=_FakeLal())
        # The actual file should exist with a .dat suffix appended.
        assert os.path.exists(base + ".dat"), "auto-suffix not appended"
        assert not os.path.exists(base) or os.path.getsize(base) == 0, \
            "writer wrote to the un-suffixed path"
        arr, _ = hpio.read_table(base + ".dat")
        np.testing.assert_allclose(arr["m1"], [30.0])
    finally:
        for f in (base, base + ".dat"):
            if os.path.exists(f):
                os.unlink(f)
    # And: when caller already includes .dat, it's not double-appended.
    base2 = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    try:
        hpio.write_grid_from_P_list(base2, [P], cols, lal_module=_FakeLal())
        assert not os.path.exists(base2 + ".dat"), "double-appended .dat"
        arr, _ = hpio.read_table(base2)
        np.testing.assert_allclose(arr["m1"], [30.0])
    finally:
        for f in (base2, base2 + ".dat"):
            if os.path.exists(f):
                os.unlink(f)
    print("test_grid_auto_suffix_append: OK")


def test_column_alias_bridge():
    """The on-disk column name 'a1x' bridges to ChooseWaveformParams.s1x."""
    P = _FakeP()
    P.s1x = 0.42; P.s1y = -0.17; P.s2z = 0.6
    cols = hpio.build_column_list()
    fname = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    try:
        hpio.write_grid_from_P_list(fname, [P], cols, lal_module=_FakeLal())
        arr, _ = hpio.read_table(fname)
        # On disk: a1x carries the s1x value
        np.testing.assert_allclose(arr["a1x"], [0.42])
        np.testing.assert_allclose(arr["a1y"], [-0.17])
        np.testing.assert_allclose(arr["a2z"], [0.6])
        # Round-trip back into a P
        P_back, _ = hpio.read_grid_to_P_list(
            fname, P_factory=_FakeP, lal_module=_FakeLal(),
            valid_params={"s1x","s1y","s1z","s2x","s2y","s2z","m1","m2"})
        assert abs(P_back[0].s1x - 0.42) < 1e-12
        assert abs(P_back[0].s2z - 0.6) < 1e-12
    finally:
        os.unlink(fname)
    print("test_column_alias_bridge: OK")


def test_sky_column_alias_bridge():
    """Sky columns use explicit ecliptic names on disk and theta/phi on P."""
    P = _FakeP()
    P.phi = 1.3
    P.theta = -0.2
    cols = hpio.build_column_list(use_sky=True)
    fname = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    try:
        hpio.write_grid_from_P_list(fname, [P], cols, lal_module=_FakeLal())
        arr, hdr = hpio.read_table(fname)
        assert hdr == cols
        np.testing.assert_allclose(arr["ecliptic_longitude"], [1.3])
        np.testing.assert_allclose(arr["ecliptic_latitude"], [-0.2])
        P_back, _ = hpio.read_grid_to_P_list(
            fname, P_factory=_FakeP, lal_module=_FakeLal(),
            valid_params={"m1", "m2", "s1x", "s1y", "s1z", "s2x", "s2y",
                          "s2z", "phi", "theta"})
        assert abs(P_back[0].phi - 1.3) < 1e-12
        assert abs(P_back[0].theta + 0.2) < 1e-12
    finally:
        os.unlink(fname)
    print("test_sky_column_alias_bridge: OK")


def test_grid_no_lal_module_passthrough():
    """When lal_module is None, no unit conversion happens (raw passthrough)."""
    P = _FakeP()
    P.m1 = 30.0  # NOT scaled
    P.m2 = 25.0
    cols = hpio.build_column_list()
    fname = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    try:
        hpio.write_grid_from_P_list(fname, [P], cols, lal_module=None)
        arr, _ = hpio.read_table(fname)
        np.testing.assert_allclose(arr["m1"], [30.0])  # passthrough
    finally:
        os.unlink(fname)
    print("test_grid_no_lal_module_passthrough: OK")


def test_consolidate_drops_high_sigma():
    cols = hpio.build_column_list()
    f1 = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    f2 = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
    try:
        hpio.write_row(f1, cols, [-1.0, 0.01, 30, 25, 0,0,0, 0,0,0])
        hpio.write_row(f2, cols, [-2.0, 1.5, 30, 25, 0,0,0, 0,0,0])  # sigma>0.9
        arr, hdr = hpio.read_many([f1, f2])
        out, _ = hpio.consolidate(arr, hdr, sigma_cut=0.9)
        assert len(out) == 1
        np.testing.assert_allclose(out["lnL"][0], -1.0)
    finally:
        for f in (f1, f2):
            os.unlink(f)
    print("test_consolidate_drops_high_sigma: OK")


if __name__ == "__main__":
    test_default_roundtrip()
    test_eccentricity_columns()
    test_tides_with_eos_index()
    test_eccentric_tides_eob_hyperbolic_columns()
    test_sky_columns()
    test_to_legacy_dat_default()
    test_legacy_column_indices_consistency()
    test_sniff_distinguishes_legacy()
    test_sniff_recognizes_new_format()
    test_env_flag()
    test_concatenated_shards()
    test_read_many_skips_empties_and_mismatches()
    test_consolidate_weighted_average()
    test_consolidate_drops_high_sigma()
    test_grid_write_read_roundtrip_with_units()
    test_grid_distance_unit_conversion()
    test_grid_auto_suffix_append()
    test_column_alias_bridge()
    test_sky_column_alias_bridge()
    test_grid_no_lal_module_passthrough()
    print("\nAll hyperpipeline_io tests passed.")
