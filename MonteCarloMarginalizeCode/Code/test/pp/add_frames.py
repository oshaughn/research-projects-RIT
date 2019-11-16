#! /usr/bin/env python
from __future__ import division, print_function


def get_args(raw_args):
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "channel",
        help="Channel to add.",
    )

    parser.add_argument(
        "input_frame",
        nargs="+",
        help="One of the frame files to be added.",
    )

    parser.add_argument(
        "output_frame",
        help="Frame file to store result in.",
    )

    return parser.parse_args(raw_args)


def main(raw_args):
    args = get_args(raw_args)

    import pycbc.frame


    output_timeseries = None
    print(args.input_frame)
    for frame in args.input_frame:
        print(frame, args.channel)
        timeseries = pycbc.frame.read_frame(
            frame, args.channel
        )

        start_time = timeseries.start_time
        delta_t = timeseries.delta_t
        end_time = timeseries.end_time
        print("Start Time: ",start_time)
        print("End Time: ",end_time)
        # A bit of a hack.
        # I've encountered frames that appear to have their first
        # sample discarded. As a result, the start_time is delta_t
        # greater than an integer value. So we take the fractional
        # part of the start time (t - int(t)) and compare to the
        # timestep delta_t. If they're the same, we've hit that case,
        # and we simply prepend a zero to the timeseries, making it
        # start at an integer time.
        if delta_t == (start_time - int(start_time)):
            print("Doing Hack (start)")
            timeseries.prepend_zeros(1)
        if  abs(end_time - int(end_time)) > 0:
            print("Doing Hack (end)")
            timeseries.append_zeros(1)
        print(timeseries,len(timeseries))

        if output_timeseries is None:
            output_timeseries = timeseries
        else:
            output_timeseries += timeseries

    pycbc.frame.write_frame(
        args.output_frame,
        args.channel,
        output_timeseries,
    )


if __name__ == "__main__":
    import sys
    raw_args = sys.argv[1:]
    sys.exit(main(raw_args))
