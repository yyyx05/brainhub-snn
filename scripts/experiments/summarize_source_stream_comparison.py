"""Descriptive arithmetic for the new three-way measurement, not new simulation."""
import csv
import json
from pathlib import Path

from run_srnn_resident_remote import ROOT, RESULT, local_path


def read_rows(path):
    with path.open(newline='') as f:
        return [{key: int(value) for key, value in row.items()} for row in csv.DictReader(f)]


def main():
    archive=Path(local_path(ROOT))/RESULT/'source_stream_compare_r1'
    out=archive/'derived_metrics.json'
    if out.exists():raise FileExistsError('already summarized')
    rows=read_rows(archive/'frame_measurements.csv')
    fixture=read_rows(archive/'fixture/frames.csv')
    result=dict(scope='Existing new three-way measurements. Descriptive sequence aggregates; not IID, P95, author RTL or whole-network speedup.',scenarios=[],boundaries=[])
    for scenario in range(2):
        selected=[r for r in rows if r['scenario']==scenario and r['frame']<588]
        lookup={(r['policy'],r['frame']):r for r in selected}
        totals=[sum(r['total'] for r in selected if r['policy']==p) for p in range(3)]
        sequences=[]
        for seq in sorted({r['sequence'] for r in fixture}):
            frames=[i for i,r in enumerate(fixture) if r['sequence']==seq]
            values=[sum(lookup[p,f]['total'] for f in frames) for p in range(3)]
            sequences.append(dict(dataset_index=fixture[frames[0]]['dataset_index'],frames=len(frames),cycles=values,
                dense_over_stream=values[0]/values[2],buffered_over_stream=values[1]/values[2]))
        comparisons=[]
        for base in (0,1):
            diffs=[lookup[base,f]['total']-lookup[2,f]['total'] for f in range(588)]
            comparisons.append(dict(base_policy=base,interval_ratio=totals[base]/totals[2],
                reduction_percent=100*(1-totals[2]/totals[base]),saved_cycles=totals[base]-totals[2],
                shorter=sum(d>0 for d in diffs),equal=sum(d==0 for d in diffs),longer=sum(d<0 for d in diffs)))
        result['scenarios'].append(dict(scenario=scenario,cycles=totals,comparisons=comparisons,sequences=sequences))
        for f in range(588,592):
            values=[next(r['total'] for r in rows if r['scenario']==scenario and r['policy']==p and r['frame']==f) for p in range(3)]
            result['boundaries'].append(dict(scenario=scenario,frame=f,cycles=values))
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
