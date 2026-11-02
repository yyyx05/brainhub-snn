"""Measure only the selected, functionally checked compact full-core image."""
import json
from pathlib import Path
import shlex

from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, upload, local_path
from prepare_m2_compact_core import prepare

NAME='m2_compact_synth_r2'
TCL='vivado/run_unified_h40_final_composition_v1_synth.tcl'


def selected_tcl(text):
    old='-generic [list "ENABLE_ONLINE_LEARNING=$enable_online_learning"]'
    if text.count(old)!=1:
        raise ValueError('synthesis generic anchor')
    text=text.replace(old,old[:-1]+' "M2_CACHE_DEPTH=512"]')
    anchor='puts $status_handle "enable_online_learning=$enable_online_learning"'
    if text.count(anchor)!=1:
        raise ValueError('status anchor')
    return text.replace(anchor,anchor+'\nputs $status_handle "m2_cache_depth=512"')


def main():
    prepared=prepare()
    native=Path(local_path(ROOT/RESULT/'m2_compact_core_r1/rtl'))
    for path in (prepared/'rtl').rglob('*.sv'):
        assert path.read_bytes()==(native/path.relative_to(prepared/'rtl')).read_bytes()
    source=Path(local_path(ROOT/RESULT/'scnn_issue_banked_synth_r1'/TCL))
    dest=Path(local_path(ROOT/'verification/experiments/m2_compact_core/synthesis_r2'/TCL))
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(selected_tcl(source.read_text()))
    bootstrap=r"""
import json,pathlib,shutil
base=pathlib.Path(ROOT); source=base/'m2_compact_core_r1'; old=base/'scnn_issue_banked_synth_r1'; target=base/NAME
status=json.loads((source/'status.json').read_text())
assert status['status']=='PASS_compact512_native_T73_pair_not_physical'
assert all(status[s]['exit']==0 for s in ('build','run_off','run_on'))
bounds=json.loads((base/'m2_capacity_bounds_r1/status.json').read_text())
assert bounds['status']=='PASS_public_capacity_bounds_32_cases'
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','Vscgraph','verilator','verilator_bin','xsim','xelab') or comm.startswith('Vtb_'): active.append([p.name,comm])
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert not active,active
free=shutil.disk_usage(base).free
assert free>750000000,free
target.mkdir(exist_ok=False)
for folder,origin in (('rtl',source),('server',old),('vivado',old)):
 shutil.copytree(origin/folder,target/folder)
source_data=[]
for original in (old/'rtl').rglob('*'):
 if original.is_file() and original.suffix!='.sv':
  relative=original.relative_to(old); destination=target/relative
  destination.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(original,destination)
  source_data.append(dict(path=relative.as_posix(),bytes=original.stat().st_size))
shutil.copy2(source/'preparation.json',target/'native_preparation.json')
shutil.copy2(source/'compact_core.diff',target/'compact_core.diff')
shutil.copy2(source/'status.json',target/'accepted_native_functional_status.json')
shutil.copy2(base/'m2_capacity_bounds_r1/status.json',target/'accepted_capacity_status.json')
(target/'input_origin.json').write_text(json.dumps(dict(rtl_source=str(source),recipe_source=str(old),source_data=source_data,repaired_omission_from='m2_compact_synth_r1',m2_cache_depth=512,default_depth=578,free_before=free,scope='Same108 RTL and Tcl as R1, restoring original non-SV source data omitted from R1 packaging. Original15ns/protection; not new SCNN full-graph functional proof.'),indent=2)+'\n')
print(json.dumps(dict(root=str(target),free_before=free)))
"""
    client=connect()
    try:
        code='ROOT='+repr(REMOTE)+'\nNAME='+repr(NAME)+'\n'+bootstrap
        print(run(client,'python3 -c '+shlex.quote(code)))
        remote=REMOTE+'/'+NAME
        with client.open_sftp() as s:
            upload(s,dest,remote+'/'+TCL)
            upload(s,prepared/'preparation.json',remote+'/preparation.json')
            for path in (prepared/'rtl').rglob('*'):
                if path.is_file() and path.suffix!='.sv':
                    with s.open(remote+'/'+path.relative_to(prepared).as_posix(),'rb') as stream:
                        assert stream.read()==path.read_bytes()
        shell='bash '+shlex.quote(remote+'/server/run_unified_h40_final_composition_v1_synth.sh')+' '+shlex.quote(remote)+' 1'
        run(client,'screen -dmS snn_resident_synth bash -lc '+shlex.quote(shell))
        print(json.dumps(dict(launched=True,root=remote,stage='synth')))
    finally:
        client.close()


if __name__=='__main__':
    main()
