"""Fetch actual boundary evidence and check the public request results."""
import collections
import datetime as dt
import json
from pathlib import Path
import re
import stat
from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, fetch_tree, local_path
from run_m2_capacity_bounds import NAME


def parse(log):
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b|\bERROR\b|Out of memory|\bOOM\b|\bKilled\b',log,re.I):
        raise ValueError('diagnostic failure')
    cases=re.findall(r'^CAPACITY_PUBLIC_PASS depth=(\d+) cases=(\d+)$',log,re.M)
    if sorted(cases)!=[('512','16'),('578','16')]:
        raise ValueError('capacity cases')
    if len(re.findall(r'^CAPACITY_BOUNDS_PASS depths=2 cases=32$',log,re.M))!=1:
        raise ValueError('terminal result')
    if log.index('CAPACITY_BOUNDS_PASS')<max(log.index('CAPACITY_PUBLIC_PASS depth='+d) for d in ('512','578')):
        raise ValueError('early completion')
    return dict(depths=[512,578],cases=32,public_consumers=['WH1 configuration','Wout configuration','refill request'])


def main():
    folder=Path(local_path(ROOT/RESULT/NAME))
    inventory=[]
    client=connect()
    try:
        with client.open_sftp() as s:
            with s.open(REMOTE+'/'+NAME+'/status.json') as stream:
                status=json.load(stream)
            assert status['status']=='PASS_public_capacity_bounds_32_cases'
            fetch_tree(s,REMOTE+'/'+NAME,ROOT/RESULT/NAME)
            def visit(rel=''):
                for entry in s.listdir_attr(REMOTE+'/'+NAME+('/'+rel if rel else '')):
                    if entry.filename in ('obj','elab_obj','__pycache__','.Xil'):continue
                    child=(rel+'/' if rel else '')+entry.filename
                    if stat.S_ISDIR(entry.st_mode):visit(child)
                    else:
                        assert stat.S_ISREG(entry.st_mode)
                        assert (folder/child).stat().st_size==entry.st_size
                        inventory.append(dict(path=child,bytes=entry.st_size))
            visit()
    finally:
        client.close()
    assert not list(folder.rglob('*.part'))
    assert status==json.loads((folder/'status.json').read_text())
    assert all(status[x]['exit']==0 for x in ('build','run'))
    rtls=list((folder/'rtl').rglob('*.sv'))
    assert len(rtls)==108
    for path in rtls:
        assert path.read_bytes()==(Path(local_path(ROOT/RESULT/'m2_compact_core_r1'))/path.relative_to(folder)).read_bytes()
    assert (folder/'tb_m2_capacity_bounds.sv').read_bytes()==(
        ROOT/'verification/experiments/m2_capacity_bounds/candidate_r1/tb_m2_capacity_bounds.sv').read_bytes()
    for name in ('m2_capacity_bounds_worker.py','srnn_resident_server_worker.py'):
        assert (folder/'scripts/experiments'/name).read_bytes()==(ROOT/'scripts/experiments'/name).read_bytes()
    build=(folder/'build.log').read_text()
    assert not re.search(r'%Warning-(LATCH|MULTIDRIVEN|UNOPTFLAT)',build)
    checks=parse(build+'\n'+(folder/'run.log').read_text())
    report=dict(status='PASS_independent_public_capacity_audit',checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        raw_files=len(inventory),raw_bytes=sum(x['bytes'] for x in inventory),all_raw_sizes_match=True,
        stages={x:status[x] for x in ('build','run')},checks=checks,
        warnings=dict(collections.Counter(re.findall(r'%Warning-(\w+)',build))),raw_inventory=inventory,
        scope='Actual public configuration/refill acceptance for zero counts, nonzero bases, exact upper bounds and overflow at depths512/578. Legal requests intentionally stop at first memory request and reset between cases. Not data storage, abort recovery or XPM proof.')
    (folder/'terminal_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='raw_inventory'},indent=2))


if __name__=='__main__':
    main()
