#!/usr/bin/env python
"""Create an exact unique ILE grid; leave the weighted posterior XML untouched."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from RIFT import lalsimutils
from RIFT.misc.intrinsic_grid import unique_intrinsic_indices, INTRINSIC_FIELDS

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-xml',required=True,type=Path)
    parser.add_argument('--output-file',required=True,type=Path,help='Output XML basename (without .xml.gz)')
    parser.add_argument('--min-points',required=True,type=int)
    parser.add_argument('--receipt',required=True,type=Path)
    args = parser.parse_args()
    if args.min_points < 1: parser.error('--min-points must be positive')
    output = Path(str(args.output_file)+'.xml.gz')
    if output.resolve() == args.input_xml.resolve():
        parser.error('Grid output must differ from the posterior input')
    before = hashlib.sha256(args.input_xml.read_bytes()).hexdigest()
    points = lalsimutils.xml_to_ChooseWaveformParams_array(str(args.input_xml))
    indices = unique_intrinsic_indices(points)
    record = dict(input=str(args.input_xml),input_sha256=before,input_rows=len(points),
                  unique_rows=len(indices),duplicate_rows=len(points)-len(indices),
                  required_points=args.min_points,identity_fields=list(INTRINSIC_FIELDS),
                  unique_source_indices=indices,output=str(output))
    if len(indices) < args.min_points:
        record['status']='insufficient_unique_grid'
        args.receipt.write_text(json.dumps(record,indent=2)+'\n')
        raise RuntimeError('Only {} distinct points for {} scheduled ILE evaluations'.format(len(indices),args.min_points))
    temp = str(args.output_file)+'.tmp-'+str(os.getpid())
    lalsimutils.ChooseWaveformParams_array_to_xml([points[k] for k in indices],temp)
    os.replace(temp+'.xml.gz',output)
    if hashlib.sha256(args.input_xml.read_bytes()).hexdigest() != before:
        raise RuntimeError('Posterior input changed during grid construction')
    record.update(status='verified',output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    args.receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='unique_source_indices'}))

if __name__ == '__main__': main()
