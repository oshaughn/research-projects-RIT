import argparse
import os

from RIFT.misc import hyperpipeline_io as hpio


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--fname")
parser.add_argument("--fname-output-samples")
parser.add_argument("--fname-output-integral")
parser.add_argument("--output-file", default="overlap-grid")
args, _ = parser.parse_known_args()

if os.environ.get("RIFT_HYPERPIPELINE_FORMAT") != "1":
    raise SystemExit("stub_cip: missing RIFT_HYPERPIPELINE_FORMAT=1")

fname = args.fname_output_samples or args.output_file
if not fname.endswith(".dat"):
    fname += ".dat"

columns = hpio.DEFAULT_BASE_COLUMNS
hpio.write_table(fname, columns, [
    [0.0, 0.1, 30.0, 20.0, 0.0, 0.0, 0.1, 0.0, 0.0, -0.1],
    [0.0, 0.1, 31.0, 21.0, 0.0, 0.0, 0.1, 0.0, 0.0, -0.1],
    [0.0, 0.1, 32.0, 22.0, 0.0, 0.0, 0.1, 0.0, 0.0, -0.1],
    [0.0, 0.1, 33.0, 23.0, 0.0, 0.0, 0.1, 0.0, 0.0, -0.1],
])
