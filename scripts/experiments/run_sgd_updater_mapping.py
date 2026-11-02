"""Same-device scalar updater mapping; sequential synthesis, no implementation."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import time

import run_source_stream_comparison as common

NAME='sgd_updater_mapping_r1'
EXPERIMENT=Path('verification/experiments/sgd_updater_mapping_r1')


def parse_mapping(log):
    diag=common.diagnostics(log)
    if diag['errors'] or diag['critical_warnings'] or diag['structural']:
        raise ValueError('tool diagnostics')
    if log.count('SGD_UPDATER_MAPPING_COMPLETE') != 1:
        raise ValueError('incomplete mappings')
    return diag


def prepare():
    from run_srnn_resident_remote import ROOT,RESULT,local_path
    root=Path(local_path(ROOT));out=root/EXPERIMENT
    accepted=root/RESULT/'sgd_updater_local_r1'
    state=json.loads((accepted/'terminal_audit.json').read_text())
    assert state['status']=='PASS_independent_component_RTL_transcript_acceptance'
    out.mkdir(exist_ok=False)
    for name in ('fixed_sgd_update.sv','direct_update_wrapper.sv','unified_tp_direct_residual_update.sv'):
        shutil.copy2(accepted/name,out/name)
    shutil.copy2(Path(__file__).with_name('sgd_updater_mapping.tcl'),out/'mapping.tcl')
    for name in ('run_source_stream_comparison.py','run_fc_bitmap_mapping.py'):
        shutil.copy2(Path(__file__).with_name(name),out/name)
    (out/'clock.xdc').write_text('create_clock -name clk -period 15.000 [get_ports i_clk]\n')
    shutil.copy2(Path(__file__),out/'launch_worker.py')
    files=[p for p in out.rglob('*') if p.is_file()]
    print(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files))))


def worker():
    from run_fc_bitmap_mapping import vivado_environment
    root=Path.cwd();scratch=Path('/tmp/snn_'+NAME);scratch.mkdir(exist_ok=False)
    command=[common.VIVADO,'-mode','batch','-nojournal','-nolog','-source','mapping.tcl','-tclargs',str(root),str(scratch)]
    state=dict(status='mapping_running',start=dt.datetime.now().astimezone().isoformat(),scratch=str(scratch),command=command)
    def save():(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    save();begin=time.monotonic()
    try:
        with (root/'mapping.log').open('x') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=vivado_environment(common.VIVADO))
            try:state['exit']=process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                state.update(exit=None,pid=process.pid,timed_out=True);save();raise
        state['diagnostics']=parse_mapping((root/'mapping.log').read_text(errors='replace'))
        if state['exit']:raise RuntimeError('tool exit')
        state['status']='PASS_updater_synthesis_mappings_not_routed_STA'
    except Exception as error:state.update(status='FAIL',error=repr(error))
    for path in scratch.rglob('*'):
        rel=path.relative_to(scratch)
        if path.is_file() and not set(rel.parts)&{'obj','elab_obj','.Xil','__pycache__'}:
            target=root/'scratch_originals'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    state.update(end=dt.datetime.now().astimezone().isoformat(),wall_seconds=time.monotonic()-begin,scratch_preserved=True);save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','launch','worker'))
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='worker':worker()
    else:
        common.NAME=NAME;common.EXPERIMENT=EXPERIMENT;common.launch()
