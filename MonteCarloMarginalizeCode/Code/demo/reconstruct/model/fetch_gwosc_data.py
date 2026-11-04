#!/usr/bin/env python3
"""fetch_gwosc_data.py -- download GW150914 O1 open data + build the lal cache.

Self-contained, no assumptions imported from other events.  Uses gwosc.locate to
resolve the exact .gwf frame URLs for the 4096-s O1 4 kHz open-data block that
covers GPS 1126259462, curls them, writes data/event.cache over them, and reads
the REAL in-frame channel name from the frame table (never assumed).

For O1 open data the in-frame channel is <IFO>:LOSC-STRAIN (NOT the O4-era
GWOSC-4KHZ_R1_STRAIN).  We store it WITHOUT the leading '<IFO>:' because
ILE/pseudo_pipe form '<IFO>:<channel>' themselves.

Usage:  fetch_gwosc_data.py --outdir data --event-time 1126259462.4
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys


def resolve_urls(ifos, gps):
    from gwosc.locate import get_urls
    urls = {}
    for ifo in ifos:
        u = get_urls(ifo, int(gps), int(gps) + 1, format="gwf", sample_rate=4096)
        if not u:
            raise SystemExit("no GWOSC .gwf URL for %s at %d" % (ifo, gps))
        urls[ifo] = u[0]
    return urls


def curl(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print("  have", os.path.basename(dest), "(skip)")
        return
    print("  curl", url)
    subprocess.check_call(["curl", "-s", "-f", "-L", "-o", dest, url])


def detect_channel(gwf, ifo):
    """Read the REAL strain channel from the frame table (do not assume)."""
    from gwpy.io.gwf import iter_channel_names
    chans = list(iter_channel_names(gwf))
    strain = [c for c in chans if c.endswith("STRAIN") or c.endswith("STRAIN".lower())]
    strain = [c for c in chans if "STRAIN" in c.upper()]
    if not strain:
        raise SystemExit("no STRAIN channel in %s: %s" % (gwf, chans))
    full = strain[0]                       # e.g. 'H1:LOSC-STRAIN'
    bare = full.split(":", 1)[1] if ":" in full else full
    return full, bare


def write_cache(frames, cache):
    with open(cache, "w") as f:
        for ifo, p in frames.items():
            base = os.path.basename(p)
            m = re.match(r"([A-Z])-(\w+)-(\d+)-(\d+)\.gwf", base)
            if not m:
                raise SystemExit("cannot parse frame name %s" % base)
            obs, tag, gps, dur = m.groups()
            f.write("%s %s %s %s file://localhost%s\n"
                    % (obs, tag, gps, dur, os.path.abspath(p)))
    print("  wrote", cache)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--event-time", type=float, default=1126259462.4)
    ap.add_argument("--ifos", nargs="+", default=["H1", "L1"])
    a = ap.parse_args(argv)

    os.makedirs(a.outdir, exist_ok=True)
    print("[fetch] GW150914 O1 open data ->", a.outdir)
    urls = resolve_urls(a.ifos, a.event_time)
    frames, channels_full, channels_bare = {}, {}, {}
    for ifo in a.ifos:
        dest = os.path.join(a.outdir, os.path.basename(urls[ifo]))
        curl(urls[ifo], dest)
        frames[ifo] = dest
    write_cache(frames, os.path.join(a.outdir, "event.cache"))
    for ifo in a.ifos:
        full, bare = detect_channel(frames[ifo], ifo)
        channels_full[ifo] = full
        channels_bare[ifo] = bare
        print("  %s channel: %s  (bare: %s)" % (ifo, full, bare))

    params = {"event": "GW150914", "event_time": a.event_time,
              "ifos": a.ifos, "channels_full": channels_full,
              "channels_bare": channels_bare, "frames": {k: os.path.abspath(v) for k, v in frames.items()}}
    with open(os.path.join(a.outdir, "event_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print("  wrote", os.path.join(a.outdir, "event_params.json"))
    print("[fetch] done.")


if __name__ == "__main__":
    main()
