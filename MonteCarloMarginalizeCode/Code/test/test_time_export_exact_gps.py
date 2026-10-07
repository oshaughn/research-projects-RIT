"""Nanosecond serialization regression tests ported from O4d 94840482."""
import numpy as np
from RIFT.misc import xmlutils


def test_exact_gps_addition_preserves_sub_float_ulp_offsets_and_carry():
    epoch_s = 1400000000
    epoch_ns = 0
    offset = np.array([60e-9, -60e-9, 1.000000060])
    seconds, nanoseconds = xmlutils.gps_add_seconds_exact(
        epoch_s, epoch_ns, offset)
    np.testing.assert_array_equal(seconds,
                                  [1400000000, 1399999999, 1400000001])
    np.testing.assert_array_equal(nanoseconds, [60, 999999940, 60])
    # At this epoch float64 cannot represent a 60 ns increment.  The exact pair
    # must therefore be the serialization source, not the compatibility float.
    assert float(epoch_s) + 60e-9 == float(epoch_s)


def test_exact_xml_time_fields_override_the_legacy_float_mapping():
    keys = list(xmlutils.CMAP)
    assert keys.index("t_ref") < keys.index("t_ref_gps_seconds")
    assert keys.index("t_ref_gps_seconds") < keys.index("t_ref_gps_nanoseconds")
    assert xmlutils.CMAP["t_ref_gps_seconds"] == "geocent_end_time"
    assert xmlutils.CMAP["t_ref_gps_nanoseconds"] == "geocent_end_time_ns"

    class Row(object):
        pass

    class Table(object):
        RowType = Row

        @staticmethod
        def get_next_id():
            return 17

    row = xmlutils.samples_to_siminsp_row(
        Table(), t_ref=float(1400000000),
        t_ref_gps_seconds=np.int64(1400000000),
        t_ref_gps_nanoseconds=np.int64(60))
    assert row.geocent_end_time == 1400000000
    assert row.geocent_end_time_ns == 60


def test_legacy_float_time_mapping_is_unchanged_without_exact_fields():
    class Row:
        pass

    class Table:
        RowType = Row

        @staticmethod
        def get_next_id():
            return 17

    time = 1400000000.125
    row = xmlutils.samples_to_siminsp_row(Table(), t_ref=time)
    assert row.geocent_end_time == int(time)
    assert row.geocent_end_time_ns == int((time - int(time)) * 1e9)
