"""Map signed44 operators after their functional run, with registered inputs."""
import argparse
import json
from pathlib import Path
import re
import shutil

import run_sgd_updater_mapping as previous
import run_source_stream_comparison as common

NAME='sgd_bounded_mapping_r1'
EXPERIMENT=Path('verification/experiments/sgd_bounded_mapping_r1')


def parse_mapping(log):
    diag=common.diagnostics(log)
    if diag['errors'] or diag['critical_warnings'] or diag['structural']:
        raise ValueError('tool diagnostics')
    if len(re.findall(r'^BOUNDED_MAPPING_COMPLETE$',log,re.M))!=1:
        raise ValueError('incomplete mappings')
    rows=re.findall(r'^INPUT_REG_PATH top=(\w+) kind=(\w+) registers=(\d+) paths=(\d+) slack=(-?[\d.]+)$',log,re.M)
    if [(t,k) for t,k,_,_,_ in rows]!=[(t,k) for t in ('direct_registered44','sgd_registered44') for k in ('gradient','code','residual')]:
        raise ValueError('missing input register path report')
    if any(int(n)==0 or int(p)==0 for _,_,n,p,_ in rows):
        raise ValueError('input arithmetic not timed')
    return dict(diagnostics=diag,input_register_paths=[dict(top=t,kind=k,registers=int(n),paths=int(p),slack=float(s)) for t,k,n,p,s in rows])


def prepare():
    from run_srnn_resident_remote import ROOT,RESULT,local_path
    root=Path(local_path(ROOT));out=root/EXPERIMENT
    accepted=root/RESULT/'sgd_bounded_local_r1'
    state=json.loads((accepted/'status.json').read_text())
    assert state['status']=='PASS_bounded_signed44_registered_component_RTL'
    out.mkdir(exist_ok=False)
    for name in ('bounded_sgd_update.sv','sgd_registered44.sv','direct_update_wrapper.sv','unified_tp_direct_residual_update.sv'):
        shutil.copy2(accepted/name,out/name)
    shutil.copy2(Path(__file__).with_name('sgd_bounded_mapping.tcl'),out/'mapping.tcl')
    for name in ('run_source_stream_comparison.py','run_fc_bitmap_mapping.py','run_sgd_updater_mapping.py'):
        shutil.copy2(Path(__file__).with_name(name),out/name)
    (out/'clock.xdc').write_text('create_clock -name clk -period 15.000 [get_ports i_clk]\n')
    shutil.copy2(Path(__file__),out/'launch_worker.py')
    files=list(out.iterdir())
    print(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files))))


def worker():
    previous.NAME=NAME
    previous.parse_mapping=parse_mapping
    previous.worker()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','launch','worker'))
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='worker':worker()
    else:
        common.NAME=NAME;common.EXPERIMENT=EXPERIMENT;common.launch()
