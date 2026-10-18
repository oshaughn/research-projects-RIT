"""Seeded conditional-distance GPU ILE, two points per worker and saved lnL parity."""
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from igwn_ligolw import ligolw, lsctables, utils

code, fixtures, out = [Path(p).resolve() for p in sys.argv[1:]]
out.mkdir(parents=True, exist_ok=True)


class Handler(ligolw.LIGOLWContentHandler):
    pass


lsctables.use_in(Handler)


def samples(path):
    rows = lsctables.SimInspiralTable.get_table(utils.load_filename(str(path), contenthandler=Handler))
    cols = ("longitude", "latitude", "polarization", "coa_phase", "inclination", "distance",
            "geocent_end_time", "geocent_end_time_ns")
    return {tuple(getattr(row, col) for col in cols): float(row.alpha1) for row in rows}


results = {}
for backend in ["lal", "numpy", "cupy"]:
    dest = out/backend
    dest.mkdir(exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).with_name("seeded_cli.py")),
           str(code/"bin/integrate_likelihood_extrinsic_batchmode"),
           "--precompute-backend", backend, "--precompute-batch-size", "4",
           "--time-marginalization", "--vectorized", "--gpu", "--force-gpu-only",
           "--sampler-method", "AV",
           "--right-ascension", "1.0", "--declination", "0.4", "--inclination", "0.6",
           "--psi", "0.4", "--phi-orb", "0.3",
           "--sim-xml", str(fixtures/"grid.xml.gz"), "--n-events-to-analyze", "2",
           "--event-time", "1000000000", "--data-start-time", "999999896", "--data-end-time", "1000000024",
           "--cache-file", str(fixtures/"zero_noise.cache"),
           "--channel-name", "H1=FAKE-STRAIN", "--channel-name", "L1=FAKE-STRAIN",
           "--psd-file", "H1="+str(fixtures/"psd_O4.xml.gz"),
           "--psd-file", "L1="+str(fixtures/"psd_O4.xml.gz"),
           "--fmin-template", "25", "--reference-freq", "25", "--fmax", "1700", "--srate", "4096",
           "--inv-spec-trunc-time", "8", "--l-max", "4", "--approximant", "IMRPhenomXHM",
           "--d-max", "300", "--n-max", "100000", "--n-eff", "50", "--n-chunk", "2048",
           "--seed", "19", "--save-samples", "--output-file", str(dest/"output")]
    (dest/"command.json").write_text(json.dumps(cmd, indent=2))
    begin = time.perf_counter()
    with (dest/"ile.log").open("w") as log:
        proc = subprocess.run(cmd, cwd=dest, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    assert proc.returncode == 0, (backend, (dest/"ile.log").read_text()[-6000:])
    assert len(list(dest.glob("output_*_.dat"))) == 2, backend
    rows = [np.loadtxt(p).tolist() for p in sorted(dest.glob("output_*_.dat"))]
    assert all(np.isfinite(row[-4]) and row[-1] >= 50 for row in rows), (backend, rows)
    results[backend] = dict(seconds=time.perf_counter()-begin, rows=rows)
    if backend != "lal":
        assert "Full-rate precompute: backend "+backend in (dest/"ile.log").read_text()
        matched = []
        for i in range(2):
            ref = samples(out/"lal"/("output_%d_.xml.gz" % i))
            actual = samples(dest/("output_%d_.xml.gz" % i))
            keys = ref.keys() & actual.keys()
            assert len(keys) >= 10, (backend, i, len(keys))
            assert all(np.isfinite(ref[k]) == np.isfinite(actual[k]) for k in keys)
            errors = [abs(ref[k]-actual[k]) for k in keys if np.isfinite(ref[k])]
            assert max(errors) < 1e-4, (backend, i, max(errors))
            matched.append(dict(samples=len(keys), max_abs_saved_lnL_error=max(errors)))
        results[backend]["matched"] = matched
        delta = [abs(a[-4]-b[-4]) for a,b in zip(results["lal"]["rows"], rows)]
        assert max(delta) < 1e-8, (backend, delta)
        results[backend]["abs_dln_integral"] = delta
    (out/"results.json").write_text(json.dumps(results, indent=2))
    print(backend, results[backend], flush=True)
print("CLI_CHECK_PASS", flush=True)
