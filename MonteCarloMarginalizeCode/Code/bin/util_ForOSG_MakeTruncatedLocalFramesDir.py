#!/usr/bin/env python
"""Crop cached strain into frames_dir, replacing the cache only on success."""
import argparse
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
from urllib.parse import unquote, urlparse

from RIFT.misc.pipeline_arguments import parse_submit_arguments
from gwpy.timeseries import TimeSeries, TimeSeriesList


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rundir', type=Path)
    opts = parser.parse_args()
    rundir = opts.rundir.resolve()
    cache = rundir / 'local.cache'
    output = rundir / 'frames_dir'
    if output.exists():
        parser.error('frames_dir already exists; refusing to overwrite staged frames')
    args_file = rundir / 'args_ile.txt'
    if args_file.exists():
        argv = shlex.split(args_file.read_text())
    else:
        submit = (rundir / 'ILE.sub').read_text()
        assignment = re.search(r'^\s*arguments\s*=\s*(.*?)\s*$', submit, re.I | re.M)
        if not assignment:
            parser.error('neither args_ile.txt nor ILE.sub arguments are available')
        argv = parse_submit_arguments(assignment.group(1))
    ile = argparse.ArgumentParser(add_help=False)
    ile.add_argument('--data-start-time', required=True, type=float)
    ile.add_argument('--data-end-time', required=True, type=float)
    ile.add_argument('--channel-name', required=True, action='append')
    args, _ = ile.parse_known_args(argv)
    start, end = math.floor(args.data_start_time)-1, math.ceil(args.data_end_time)+1
    if end <= start or args.data_end_time <= args.data_start_time:
        parser.error('data end time must exceed start time')
    channels = {}
    for specification in args.channel_name:
        if '=' not in specification:
            parser.error('channel names must use DETECTOR=CHANNEL')
        ifo, channel = specification.split('=', 1)
        if channel.startswith(ifo + ':'):
            channel = channel[len(ifo)+1:]
        if ifo in channels or not channel or not re.fullmatch(r'[A-Z][0-9]', ifo):
            parser.error('channel names must specify one nonempty channel per detector')
        channels[ifo] = channel
    original = cache.read_text()
    inputs = {ifo: [] for ifo in channels}
    for line in original.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 5:
            parser.error('cache rows must contain five fields')
        url = urlparse(fields[-1])
        if url.scheme not in ('', 'file') or url.netloc not in ('', 'localhost'):
            parser.error('automatic truncation requires local frame files')
        filename = Path(unquote(url.path))
        if not filename.is_absolute():
            filename = rundir / filename
        for ifo in channels:
            if fields[0] == ifo[0] or fields[0] == ifo:
                inputs[ifo].append((str(filename), float(fields[2]), float(fields[2])+float(fields[3])))
    if any(not files for files in inputs.values()):
        parser.error('cache has no frames for one or more requested detectors')
    with tempfile.TemporaryDirectory(prefix='.frames-staging-', dir=rundir) as temp:
        staged = Path(temp)
        rows = []
        for ifo, channel in channels.items():
            name = re.sub(r'[^A-Za-z0-9_]', '_', channel)
            filename = f'{ifo}-{name}-{start}-{end-start}.gwf'
            segments = TimeSeriesList()
            for path, file_start, file_end in sorted(inputs[ifo], key=lambda item: item[1]):
                left, right = max(start, file_start), min(end, file_end)
                if left < right:
                    # Read paths individually: GWPy treats a list containing
                    # whitespace paths as malformed cache text.
                    segments.append(TimeSeries.read(path, ifo + ':' + channel, start=left, end=right))
            if not segments:
                raise ValueError(f'{ifo}: no frames overlap the padded interval')
            data = segments.join(gap='raise')
            if float(data.t0.value) != start or float(data.span.end) != end:
                raise ValueError(f'{ifo}: frames do not cover the requested padded interval')
            data.write(str(staged / filename))
            rows.append(f'{ifo} {name} {start} {end-start} {(output / filename).as_uri()}\n')
        # Every detector has been written before changing either public output.
        with tempfile.NamedTemporaryFile(mode='w', prefix='.cache-', dir=rundir, delete=False) as handle:
            handle.writelines(rows)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_cache = Path(handle.name)
        try:
            backup = rundir / 'local_orig.cache'
            if not backup.exists():
                shutil.copyfile(cache, backup)
            os.rename(staged, output)
            try:
                os.replace(temporary_cache, cache)
            except BaseException:
                shutil.rmtree(output)
                raise
        finally:
            temporary_cache.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
