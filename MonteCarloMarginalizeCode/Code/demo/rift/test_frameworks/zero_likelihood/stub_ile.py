import argparse
import os

from RIFT.misc import hyperpipeline_io as hpio


parser = argparse.ArgumentParser()
parser.add_argument("--sim-grid")
parser.add_argument("--output-file", required=True)
parser.add_argument("--n-events-to-analyze", type=int, default=1)
parser.add_argument("--event", type=int, default=0)
parser.add_argument("--n-max")
parser.add_argument("--n-eff")
parser.add_argument("--save-samples", action="store_true")
args, _ = parser.parse_known_args()

if os.environ.get("RIFT_HYPERPIPELINE_FORMAT") != "1":
    raise SystemExit("stub_ile: missing RIFT_HYPERPIPELINE_FORMAT=1")

columns = hpio.DEFAULT_BASE_COLUMNS
for event in range(args.n_events_to_analyze):
    out = "{}_{}_.dat".format(args.output_file, event)
    m1 = 30.0 + args.event + event
    m2 = 20.0 + args.event + event
    hpio.write_table(out, columns, [
        [0.0, 0.1, m1, m2, 0.0, 0.0, 0.1, 0.0, 0.0, -0.1],
        [0.0, 0.1, m1 + 1.0, m2 + 1.0, 0.0, 0.0, 0.1, 0.0, 0.0, -0.1],
    ])
