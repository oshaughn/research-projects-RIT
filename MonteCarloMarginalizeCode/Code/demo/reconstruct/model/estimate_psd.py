#!/usr/bin/env python3
"""estimate_psd.py -- off-source PSD from the fetched GW150914 strain.

This is the ONE part of the pipeline not copied from a working run.  We read the
downloaded frame with gwpy, crop an off-source stretch (a few hundred s away from
the event, near the start of the block), estimate a median-Welch PSD, write a
2-column (freq, psd) ascii per IFO, then convert it to the ILE-readable xml.gz
with RIFT's helper `convert_psd_ascii2xml` (we do NOT hand-roll the xml: RIFT
assumes the PSD starts at f=0 and the helper zero-pads down to 0).  Finally we
sanity-check that the resulting xml loads via RIFT.lalsimutils and starts at f=0.

Usage:
  estimate_psd.py --cache data/event.cache \
      --channel H1=H1:LOSC-STRAIN --channel L1=L1:LOSC-STRAIN \
      --seg-start 1126256704 --seg-len 400 --fftlen 4 --srate 4096 --outdir .
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys, tempfile
import numpy as np


def read_cache(cache):
    urls = {}
    with open(cache) as f:
        for line in f:
            obs, tag, gps, dur, url = line.split()
            urls.setdefault(obs, []).append(url.replace("file://localhost", ""))
    return urls


def estimate(cache, ifo, channel, seg_start, seg_len, fftlen, srate):
    from gwpy.timeseries import TimeSeries
    frames = read_cache(cache).get(ifo[0], [])
    if not frames:
        raise SystemExit("no frame for %s in %s" % (ifo, cache))
    ts = TimeSeries.read(frames, channel, start=seg_start, end=seg_start + seg_len)
    nan_frac = float(np.mean(~np.isfinite(np.asarray(ts.value))))
    if nan_frac > 0.001:
        raise SystemExit(
            "%s PSD window [%d,%d) is %.0f%% NaN (data-quality gap); pick a clean "
            "off-source window (see PSD_SEG_START in config.sh)"
            % (ifo, seg_start, seg_start + seg_len, 100 * nan_frac))
    if ts.sample_rate.value != srate:
        ts = ts.resample(srate)
    # median-averaged Welch PSD, 4-s FFT, 50% overlap -> robust off-source PSD
    psd = ts.psd(fftlength=fftlen, overlap=fftlen / 2.0, method="median")
    freq = psd.frequencies.value
    val = psd.value
    good = np.isfinite(val) & (val > 0)
    return freq[good], val[good]


def to_xml(freq, psd, ifo, outdir, rift_bin):
    helper = shutil.which("convert_psd_ascii2xml")
    if not helper and rift_bin:
        cand = os.path.join(rift_bin, "convert_psd_ascii2xml")
        if os.path.exists(cand):
            helper = cand
    if not helper:
        raise SystemExit("convert_psd_ascii2xml not on PATH (activate RIFT env)")
    tmp = tempfile.mkdtemp(prefix="psdconv_")
    ascii_path = os.path.join(tmp, "%s-psd-ascii.dat" % ifo)
    np.savetxt(ascii_path, np.column_stack([freq, psd]))
    r = subprocess.run([sys.executable, helper,
                        "--fname-psd-ascii", os.path.abspath(ascii_path),
                        "--ifo", ifo, "--conventional-postfix"],
                       cwd=tmp, capture_output=True, text=True)
    tmp_xml = os.path.join(tmp, "%s-psd.xml.gz" % ifo)
    out_xml = os.path.join(outdir, "%s-psd.xml.gz" % ifo)
    if r.returncode != 0 or not os.path.exists(tmp_xml):
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit("convert_psd_ascii2xml FAILED for %s: %s"
                         % (ifo, (r.stderr or r.stdout).strip()[:400]))
    shutil.copyfile(tmp_xml, out_xml)
    shutil.rmtree(tmp, ignore_errors=True)
    return out_xml


def sanity_check(xml, ifo):
    import RIFT.lalsimutils as lalsimutils
    ps = lalsimutils.get_psd_series_from_xmldoc(xml, ifo)
    assert ps.f0 == 0.0, "%s PSD does not start at f=0 (f0=%s)" % (ifo, ps.f0)
    fmax = ps.f0 + ps.deltaF * (ps.data.length - 1)
    print("  [ok] %s: f0=%.3f df=%.4f n=%d fmax=%.1f"
          % (ifo, ps.f0, ps.deltaF, ps.data.length, fmax))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--channel", action="append", required=True,
                    help="IFO=IFO:CHANNEL (repeat per detector)")
    ap.add_argument("--seg-start", type=float, required=True)
    ap.add_argument("--seg-len", type=float, default=400.0)
    ap.add_argument("--fftlen", type=float, default=4.0)
    ap.add_argument("--srate", type=float, default=4096.0)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--rift-bin",
                    default=os.path.join(os.environ.get("RIFT_CODE", ""), "bin"))
    a = ap.parse_args(argv)
    chans = dict(c.split("=", 1) for c in a.channel)
    os.makedirs(a.outdir, exist_ok=True)
    print("[psd] off-source PSD from [%d, %d) (fft=%gs)"
          % (a.seg_start, a.seg_start + a.seg_len, a.fftlen))
    for ifo, ch in chans.items():
        freq, psd = estimate(a.cache, ifo, ch, a.seg_start, a.seg_len, a.fftlen, a.srate)
        xml = to_xml(freq, psd, ifo, a.outdir, a.rift_bin)
        print("  PSD %s -> %s" % (ifo, os.path.basename(xml)))
        sanity_check(xml, ifo)
    print("[psd] done.")


if __name__ == "__main__":
    main()
