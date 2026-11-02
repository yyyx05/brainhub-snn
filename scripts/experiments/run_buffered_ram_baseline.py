"""Use a RAM-inferable buffered baseline, with targeted cycle equivalence."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import run_source_stream_comparison as common

NAME='source_buffered_ram_r1'
EXPERIMENT=Path('verification/experiments/nearest_neighbor/buffered_ram_r1')


def transform(text):
    old='                frame[load_q]<=counts;\n'
    assert text.count(old)==1
    text=text.replace(old,'')
    anchor='    always_ff @(posedge clk or negedge rst_n) begin\n'
    replacement=('    always_ff @(posedge clk) begin\n'
                 '        if (rst_n && phase==LOAD && count_valid && count_ready) frame[load_q]<=counts;\n'
                 '    end\n'+anchor)
    assert text.count(anchor)==1
    return text.replace(anchor,replacement)


def prepare():
    from run_srnn_resident_remote import ROOT,RESULT,local_path
    root=Path(local_path(ROOT));out=root/EXPERIMENT
    if (out/'launch_worker.py').exists():raise FileExistsError('already prepared')
    r2=root/RESULT/'source_stream_mapping_r2'
    source=(r2/'source_scan.sv').read_text()
    (out/'source_scan.sv').write_text(transform(source))
    (out/'source_scan_original.sv').write_text(source.replace('module source_scan (','module source_scan_original ('))
    for name in ('source_stream.sv','clock.xdc','run_fc_bitmap_mapping.py'):
        shutil.copy2(r2/name,out/name)
    shutil.copy2(r2/'run_source_stream_comparison.py',out/'run_source_stream_comparison.py')
    tcl=(r2/'mapping.tcl').read_text().replace('foreach policy {0 1 2}', 'foreach policy {0 1}')
    (out/'mapping.tcl').write_text(tcl)
    shutil.copy2(root/RESULT/'source_stream_compare_r1/fixture/counts.mem',out/'counts.mem')
    shutil.copy2(Path(__file__),out/'launch_worker.py')
    (out/'preparation.json').write_text(json.dumps(dict(
        change='Move only frame write to posedge-only block with identical rst_n/LOAD/valid/ready enable; control unchanged.',
        reason='Async-reset process prevents Vivado RAM inference even though frame payload is not explicitly reset.',
        scope='Cycle equivalence of original and RAM-inferable buffered producers, both policies, 588 real frames plus four boundaries and local resets. Synthesis only of two buffered producers; streaming mapping not repeated.'),indent=2)+'\n')
    files=[p for p in out.rglob('*') if p.is_file()]
    print(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files))))


def parse_log(text):
    m=re.search(r'^BUFFER_RAM_PASS frames=(\d+) resets=(\d+) compared=(\d+) stalled=(\d+)$',text,re.M)
    assert m and text.count('BUFFER_RAM_PASS')==1
    result=dict(zip(('frames','resets','compared','stalled'),map(int,m.groups())))
    assert result['frames']==2380 and result['resets']==13 and result['compared']>0 and result['stalled']>0
    return result


def worker():
    from run_fc_bitmap_mapping import vivado_environment
    root=Path.cwd();scratch=Path('/tmp/snn_'+NAME);scratch.mkdir(exist_ok=False)
    state=dict(status='running',start=dt.datetime.now().astimezone().isoformat(),scratch=str(scratch))
    def save():(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    def execute(stage,command,env=None):
        state['status']=stage+'_running';save();begin=time.monotonic()
        with (root/(stage+'.log')).open('x') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
            try:code=process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                state[stage]=dict(exit=None,pid=process.pid,timed_out=True,command=command);save();raise
        diag=common.diagnostics((root/(stage+'.log')).read_text(errors='replace'))
        state[stage]=dict(exit=code,command=command,wall_seconds=time.monotonic()-begin,diagnostics=diag);save()
        if code or diag['errors'] or diag['critical_warnings'] or diag['fatal'] or diag['structural']:
            raise RuntimeError(stage+' failed')
    try:
        execute('build',[common.VERILATOR,'--binary','--timing','--assert','-j','2','-Wall','-Wno-fatal',
            '--top-module','tb_buffered_ram','--Mdir',str(scratch/'obj'),
            'source_scan_original.sv','source_scan.sv','tb_buffered_ram.sv'])
        execute('run',[str(scratch/'obj/Vtb_buffered_ram')])
        state['checks']=parse_log((root/'run.log').read_text());save()
        execute('mapping',[common.VIVADO,'-mode','batch','-nojournal','-nolog','-source','mapping.tcl',
            '-tclargs',str(root),str(scratch)],vivado_environment(common.VIVADO))
        assert 'SOURCE_STREAM_MAPPING_COMPLETE' in (root/'mapping.log').read_text()
        state['status']='PASS_buffered_RAM_equivalence_and_synthesis_not_routed_STA'
    except Exception as error:state.update(status='FAIL',error=repr(error))
    for path in scratch.rglob('*'):
        rel=path.relative_to(scratch)
        if path.is_file() and not set(rel.parts)&{'obj','elab_obj','.Xil','__pycache__'}:
            target=root/'scratch_originals'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    state.update(end=dt.datetime.now().astimezone().isoformat(),scratch_preserved=True);save()


def launch():
    common.NAME=NAME;common.EXPERIMENT=EXPERIMENT;common.launch()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','worker','launch'))
    args=parser.parse_args();{'prepare':prepare,'worker':worker,'launch':launch}[args.action]()
