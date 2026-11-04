"""Convert Sangria A/E/T time series into RIFT LISA HDF5 frames."""

from argparse import ArgumentParser
import os

import h5py
import lal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import RIFT.lalsimutils as lsu

__author__ = "A. Jan"

DEFAULT_DELTA_T = 8
DEFAULT_DURATION = 31536000
DEFAULT_LENGTH = 4194304


def _import_gwpy_timeseries():
    try:
        from gwpy.timeseries import TimeSeries
    except ImportError as exc:
        raise ImportError("gwpy is required to read Sangria frame files") from exc
    return TimeSeries


def _read_gwpy_frame(frame_path, channel):
    TimeSeries = _import_gwpy_timeseries()
    return TimeSeries.read(frame_path, channel=channel)


def _as_lal_time_series_input(time_series):
    """Return data, delta_t, and epoch from either gwpy or PyCBC time series."""
    if hasattr(time_series, "value"):
        data = np.asarray(time_series.value)
        delta_t = float(time_series.dt.value)
        epoch = float(time_series.t0.value)
        return data, delta_t, epoch
    data = np.asarray(time_series.data)
    return data, float(time_series.delta_t), float(time_series._epoch)


def create_lal_COMPLEX16TimeSeries(
    time_series,
    delta_t=DEFAULT_DELTA_T,
    duration=DEFAULT_DURATION,
    target_length=DEFAULT_LENGTH,
):
    """Resample a gwpy/PyCBC time series onto the LISA/RIFT cadence."""
    data, input_delta_t, epoch = _as_lal_time_series_input(time_series)
    old_tvals = np.arange(0, input_delta_t * len(data), input_delta_t)
    new_tvals = np.arange(0, duration, delta_t)
    func = interp1d(old_tvals, data, fill_value=tuple([0, 0]), bounds_error=False)
    new_data = func(new_tvals)

    ht_lal = lal.CreateCOMPLEX16TimeSeries(
        "ht_lal",
        epoch,
        0,
        delta_t,
        lal.DimensionlessUnit,
        len(new_data),
    )
    ht_lal.data.data = new_data + 0j
    ht_lal = lal.ResizeCOMPLEX16TimeSeries(ht_lal, 0, target_length)
    print(
        f" Delta T = {ht_lal.deltaT} s, size = {ht_lal.data.length}, "
        f"time = {ht_lal.data.length * ht_lal.deltaT / 3600 / 24:2f} days"
    )
    return ht_lal


def _plot_time_series(tvals, series, channel, save_path):
    plt.plot(tvals, series.data.data)
    plt.xlabel("Time [s]")
    plt.savefig(f"{save_path}/{channel}_time.png")
    plt.cla()


def _plot_frequency_series(fvals, series, channel, save_path):
    plt.loglog(fvals, 2 * fvals * np.abs(series.data.data))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Characteristic Strain")
    plt.savefig(f"{save_path}/{channel}_frequency.png")
    plt.cla()


def _write_h5_frame(channel, series, save_path):
    frame_path = f"{save_path}/{channel}-fake_strain-1000000-10000.h5"
    with h5py.File(frame_path, "w") as h5_file:
        h5_file.create_dataset("data", data=series.data.data)
        h5_file.attrs["deltaF"] = series.deltaF
        h5_file.attrs["epoch"] = float(series.epoch)
        h5_file.attrs["length"] = series.data.length
        h5_file.attrs["f0"] = series.f0
    return frame_path


def create_injection_from_time_series(time_series, save_path):
    """Write A/E/T diagnostic plots and RIFT HDF5 frames from time series objects."""
    os.makedirs(save_path, exist_ok=True)
    time_domain = {
        channel: create_lal_COMPLEX16TimeSeries(time_series[channel])
        for channel in ["A", "E", "T"]
    }

    tvals = np.arange(0, time_domain["A"].data.length * time_domain["A"].deltaT, time_domain["A"].deltaT)
    for channel, series in time_domain.items():
        _plot_time_series(tvals, series, channel, save_path)

    data_dict = {
        channel: lsu.DataFourier(series)
        for channel, series in time_domain.items()
    }
    fvals = -data_dict["A"].deltaF * np.arange(
        data_dict["A"].data.length // 2,
        -data_dict["A"].data.length // 2,
        -1,
    )

    frame_paths = {}
    for channel, series in data_dict.items():
        _plot_frequency_series(fvals, series, channel, save_path)
        frame_paths[channel] = _write_h5_frame(channel, series, save_path)
    return frame_paths


def create_injection_from_pycbc(pycbc_tseries, save_path):
    """Backward-compatible alias for older scripts using PyCBC time series."""
    return create_injection_from_time_series(pycbc_tseries, save_path)


def read_gwpy_channels(frame_path, channels=("A", "E", "T")):
    """Read named channels from a frame file using gwpy."""
    return {channel: _read_gwpy_frame(frame_path, channel) for channel in channels}


def read_pycbc_channels(frame_path, channels=("A", "E", "T")):
    """Read named channels with gwpy; kept as a compatibility alias."""
    return read_gwpy_channels(frame_path, channels=channels)


def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument("frame_path", help="Input Sangria frame file.")
    parser.add_argument("--save-path", default=os.getcwd(), help="Directory for generated RIFT products.")
    parser.add_argument("--channels", default="A,E,T", help="Comma-separated channels to read from the frame.")
    return parser.parse_args(argv)


def main(argv=None):
    opts = parse_args(argv)
    channels = tuple(channel.strip() for channel in opts.channels.split(",") if channel.strip())
    time_series = read_gwpy_channels(opts.frame_path, channels=channels)
    create_injection_from_time_series(time_series, opts.save_path)


if __name__ == "__main__":
    main()
