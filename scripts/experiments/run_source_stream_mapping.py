"""Mapping-only recovery: explicit input nets for Vivado, no RTL rerun."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import time

import run_source_stream_comparison as common

NAME='source_stream_mapping_r2'
EXPERIMENT=Path('verification/experiments/nearest_neighbor/stream_mapping_r2')


def prepare():
    from run_srnn_resident_remote import ROOT,RESULT,local_path
    root=Path(local_path(ROOT));out=root/EXPERIMENT
    out.mkdir(exist_ok=False)
    source=root/RESULT/'source_stream_compare_r1'
    changes={}
    for name in ('source_scan.sv','source_stream.sv'):
        text=(source/name).read_text()
        changes[name]=text.count('input logic')
        assert changes[name]>0
        (out/name).write_text(text.replace('input logic','input wire'))
    for name in ('clock.xdc','mapping.tcl','run_fc_bitmap_mapping.py'):
        shutil.copy2(source/name,out/name)
    shutil.copy2(source/'launch_worker.py',out/'run_source_stream_comparison.py')
    shutil.copy2(Path(__file__),out/'launch_worker.py')
    (out/'preparation.json').write_text(json.dumps(dict(
        source_run='source_stream_compare_r1',changes=changes,
        transformation='Only input logic -> input wire; widths, signedness, ports, combinational/sequential logic unchanged.',
        reason='Vivado2022.2 Synth8-6735 under default_nettype none; original simulation inputs and FAIL left untouched.',
        scope='Three static producer synthesis mappings, not routed STA or full-core cost; no repeated RTL simulation.'),indent=2)+'\n')
    files=[p for p in out.rglob('*') if p.is_file()]
    print(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files))))


def worker():
    from run_fc_bitmap_mapping import vivado_environment
    root=Path.cwd();scratch=Path('/tmp/snn_'+NAME);scratch.mkdir(exist_ok=False)
    command=[common.VIVADO,'-mode','batch','-nojournal','-nolog','-source','mapping.tcl',
             '-tclargs',str(root),str(scratch)]
    state=dict(status='mapping_running',start=dt.datetime.now().astimezone().isoformat(),scratch=str(scratch),command=command)
    def save():(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    save();begin=time.monotonic()
    try:
        with (root/'mapping.log').open('x') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                     env=vivado_environment(common.VIVADO))
            try:state['exit']=process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                state.update(exit=None,pid=process.pid,timed_out=True);save();raise
        text=(root/'mapping.log').read_text(errors='replace')
        state['diagnostics']=common.diagnostics(text)
        if state['exit'] or state['diagnostics']['errors'] or state['diagnostics']['critical_warnings'] or 'SOURCE_STREAM_MAPPING_COMPLETE' not in text:
            raise RuntimeError('mapping failed or critical diagnostics')
        state['status']='PASS_three_producer_synthesis_mappings_not_routed_STA'
    except Exception as error:state.update(status='FAIL',error=repr(error))
    for path in scratch.rglob('*'):
        rel=path.relative_to(scratch)
        if path.is_file() and not set(rel.parts)&{'obj','elab_obj','.Xil','__pycache__'}:
            target=root/'scratch_originals'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    state.update(end=dt.datetime.now().astimezone().isoformat(),wall_seconds=time.monotonic()-begin,scratch_preserved=True);save()


def launch():
    common.NAME=NAME;common.EXPERIMENT=EXPERIMENT
    common.launch()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','launch','worker'))
    args=parser.parse_args();{'prepare':prepare,'launch':launch,'worker':worker}[args.action]()
