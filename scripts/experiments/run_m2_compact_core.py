#!/usr/bin/env python3
"""Launch one isolated selected-capacity native M2 test, never a completed run."""
import argparse
import json
from pathlib import Path
import shlex

from prepare_m2_compact_core import prepare, OUTPUT
from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, upload, local_path


def launch(name):
    prepared=prepare()
    bootstrap=r'''
import json,pathlib,shutil
base=pathlib.Path(ROOT);old=base/'srnn_specialized_r1';target=base/NAME
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','Vscgraph','verilator_bin','verilator','xsim','xelab') or comm.startswith('Vtb_'):active.append((p.name,comm))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
if active:raise RuntimeError(active)
free=shutil.disk_usage(base).free
if free<750000000:raise RuntimeError('insufficient space '+str(free))
target.mkdir(exist_ok=False)
for folder in ('scripts','tb','results','build'):
 shutil.copytree(old/folder,target/folder,ignore=shutil.ignore_patterns('__pycache__','obj','elab_obj','.Xil'))
for file in ('tb_srnn_profile_measurement.sv','srnn_profile_measurement_generated_body.svh','input_commands.json'):
 shutil.copy2(old/file,target/file)
shutil.copy2(old/'status.json',target/'accepted_profile_status.json')
tb=target/'tb_srnn_profile_measurement.sv';text=tb.read_text()
anchor='    unified_h40_final_composition_top_v1 #(\n'
if text.count(anchor)!=1:raise ValueError('profile top instantiation')
text=text.replace(anchor,anchor+'        .M2_CACHE_DEPTH(512),\n')
text=text.replace('endmodule','    initial begin\n        if (u_final.M2_CACHE_DEPTH != 512) $fatal(1,"M2 capacity selection");\n        $display("M2_COMPACT_SELECTED depth=%0d",u_final.M2_CACHE_DEPTH);\n    end\nendmodule')
tb.write_text(text)
(target/'input_origin.json').write_text(json.dumps(dict(rtl_source=SOURCE,input_source=str(old),free_before=free,capacity=512,default_capacity=578,scope='Native T73 unchanged inputs/reference/service; TB only selects capacity and reports it.'),indent=2)+'\n')
print(json.dumps(dict(root=str(target),free_before=free)))
'''
    client=connect()
    try:
        setup='ROOT='+repr(REMOTE)+'\nNAME='+repr(name)+'\nSOURCE='+repr(str(ROOT/RESULT/'scnn_issue_banked_synth_r1'))+'\n'+bootstrap
        print(run(client,'python3 -c '+shlex.quote(setup)))
        remote=REMOTE+'/'+name
        with client.open_sftp() as s:
            for file in prepared.rglob('*'):
                if file.is_file(): upload(s,file,remote+'/'+file.relative_to(prepared).as_posix())
            for file in ('m2_compact_core_worker.py','srnn_equal_storage_worker.py'):
                upload(s,Path(__file__).with_name(file),remote+'/scripts/experiments/'+file)
        configure="from pathlib import Path; p=Path("+repr(remote)+"); fs=sorted(x.relative_to(p).as_posix() for x in (p/'rtl').rglob('*.sv')); assert len(fs)==108; (p/'sources.f').write_text(''.join(chr(34)+x+chr(34)+'\\n' for x in fs+['tb_srnn_profile_measurement.sv','tb/monitors/supplement_h40_activity_monitor.sv']))"
        run(client,'python3 -c '+shlex.quote(configure))
        shell='cd '+shlex.quote(remote)+' && exec python3 scripts/experiments/m2_compact_core_worker.py'
        run(client,'screen -dmS '+shlex.quote('snn_'+name)+' bash -lc '+shlex.quote(shell))
        print(json.dumps(dict(launched=True,root=remote)))
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='m2_compact_core_r1')
    launch(parser.parse_args().name)
