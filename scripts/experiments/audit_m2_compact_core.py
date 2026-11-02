"""Fetch and independently verify the isolated compact core native run."""
import collections
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import stat
import sys

from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, fetch_tree, local_path
from prepare_m2_compact_core import OUTPUT, SOURCE, transform
from srnn_equal_storage_worker import residency

sys.path.insert(0, str(ROOT/'scripts'))
from run_srnn_profile_measurement import parse_log, inspect_build_log

NAME = 'm2_compact_core_r1'


def main():
    archive = Path(local_path(ROOT/RESULT))
    folder = archive/NAME
    client = connect()
    inventory = []
    try:
        with client.open_sftp() as s:
            with s.open(REMOTE+'/'+NAME+'/status.json') as stream:
                status = json.load(stream)
            if status['status'] != 'PASS_compact512_native_T73_pair_not_physical':
                raise RuntimeError(status)
            fetch_tree(s, REMOTE+'/'+NAME, ROOT/RESULT/NAME)
            def visit(relative=''):
                for entry in s.listdir_attr(REMOTE+'/'+NAME+('/'+relative if relative else '')):
                    if entry.filename in ('obj','elab_obj','__pycache__','.Xil'):
                        continue
                    child = (relative+'/' if relative else '')+entry.filename
                    if stat.S_ISDIR(entry.st_mode):
                        visit(child)
                    else:
                        assert stat.S_ISREG(entry.st_mode)
                        assert (folder/child).stat().st_size == entry.st_size, child
                        inventory.append(dict(path=child,bytes=entry.st_size))
            visit()
        remote = json.loads(run(client, 'python3 -c '+shlex.quote(
            "import json,shutil,subprocess; d=subprocess.run(['dmesg'],capture_output=True,text=True); "
            "print(json.dumps(dict(free_bytes=shutil.disk_usage('/root/autodl-tmp').free,dmesg_readable=d.returncode==0,dmesg_error=d.stderr.strip())))")))
    finally:
        client.close()
    assert not list(folder.rglob('*.part'))
    assert status == json.loads((folder/'status.json').read_text())
    assert all(status[x]['exit']==0 for x in ('build','run_off','run_on'))
    rtls = sorted((folder/'rtl').rglob('*.sv'))
    assert len(rtls)==108
    changed = []
    for path in rtls:
        rel = path.relative_to(folder/'rtl')
        assert path.read_bytes() == Path(local_path(OUTPUT/'rtl'/rel)).read_bytes(), str(rel)
        old = Path(local_path(SOURCE/'rtl'/rel))
        if old.is_file():
            assert path.read_text()==transform(rel.as_posix(),old.read_text())
            if path.read_bytes()!=old.read_bytes():
                changed.append(rel.as_posix())
    assert len(changed)==11
    expected = archive/'srnn_specialized_r1'
    inherited = []
    for path in folder.rglob('*'):
        rel = path.relative_to(folder)
        old = expected/rel
        if path.is_file() and rel.parts[0] in ('scripts','tb','results','build') and old.is_file():
            assert path.read_bytes()==old.read_bytes(), str(rel)
            inherited.append(rel.as_posix())
    for name in ('srnn_profile_measurement_generated_body.svh','input_commands.json'):
        assert (folder/name).read_bytes()==(expected/name).read_bytes()
        inherited.append(name)
    original = (expected/'tb_srnn_profile_measurement.sv').read_text()
    modified = original.replace('    unified_h40_final_composition_top_v1 #(\n',
        '    unified_h40_final_composition_top_v1 #(\n        .M2_CACHE_DEPTH(512),\n')
    modified = modified.replace('endmodule',
        '    initial begin\n        if (u_final.M2_CACHE_DEPTH != 512) $fatal(1,"M2 capacity selection");\n'
        '        $display("M2_COMPACT_SELECTED depth=%0d",u_final.M2_CACHE_DEPTH);\n    end\nendmodule')
    assert (folder/'tb_srnn_profile_measurement.sv').read_text()==modified
    assert (folder/'sources.f').read_text().splitlines()==[
        '"'+p.relative_to(folder).as_posix()+'"' for p in rtls]+[
        '"tb_srnn_profile_measurement.sv"','"tb/monitors/supplement_h40_activity_monitor.sv"']
    build = (folder/'build.log').read_text()
    assert inspect_build_log(build)==status['build_warnings']
    assert not status['build_warnings']['blocking']
    sample = json.loads((expected/'status.json').read_text())['sample_definition']
    measurements = {}
    for mode in ('off','on'):
        log = (folder/('run_'+mode+'.log')).read_text()
        assert not re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b|\bERROR\b|Out of memory|\bOOM\b|\bKilled\b',build+log,re.I)
        assert re.findall(r'^M2_COMPACT_SELECTED depth=(\d+)$',log,re.M)==['512']
        assert len(re.findall(r'^H40_ACTIVITY tile=',log,re.M))==12
        parsed = parse_log(log,sample,15)
        parsed['residency']=residency(log)
        parsed['simulation_wall_seconds']=status['run_'+mode]['wall_seconds']
        assert parsed==json.loads((folder/mode/'measurements.json').read_text())
        baseline=json.loads((archive/'srnn_equal_storage_r1/baseline'/mode/'measurements.json').read_text())
        for key in ('counters','residency','initial_load_and_state_cycles','sample_service_cycles_including_refill_excluding_tb_checks'):
            assert parsed[key]==baseline[key], (mode,key)
        assert {k:v for k,v in parsed['terminal'].items() if k!='end'}=={
            k:v for k,v in baseline['terminal'].items() if k!='end'}
        measurements[mode]=parsed
    report=dict(status='PASS_independent_compact_core_native_audit',checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        raw_files=len(inventory),raw_bytes=sum(x['bytes'] for x in inventory),all_raw_sizes_match=True,
        stages={k:status[k] for k in ('build','run_off','run_on')},actual_rtl_count=108,changed=changed,
        added=['memory/m2_compact_store.sv'],unchanged_input_files=inherited,
        warnings=dict(collections.Counter(re.findall(r'%Warning-(\w+)',build))),measurements=measurements,
        remote_at_audit=remote,raw_inventory=inventory,
        scope='One native B1T73 sample, explicit512 Context1 and generic578 default. Context0 unchanged. Same cycles/counters as accepted banked baseline. Behavioral RAM simulation, not new XPM full-core or physical closure.')
    (folder/'terminal_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('raw_inventory','measurements')},indent=2))


if __name__=='__main__':
    main()
