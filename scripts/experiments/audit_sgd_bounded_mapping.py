"""Accept one terminal bounded-operator mapping archive, not routed STA."""
import csv
import importlib.util
import json
from pathlib import Path
import re

from audit_source_stream_comparison import inventory
from run_srnn_resident_remote import ROOT,RESULT,connect,local_path
from run_sgd_bounded_mapping import NAME,EXPERIMENT


def main():
    root=Path(local_path(ROOT));archive=root/RESULT/NAME;prepared=root/EXPERIMENT
    if (archive/'terminal_audit.json').exists():raise FileExistsError('already accepted')
    state=json.loads((archive/'status.json').read_text())
    assert state['exit']==0 and state['end']
    assert state['status']=='PASS_updater_synthesis_mappings_not_routed_STA'
    client=connect()
    try:observation=inventory(client,NAME,state['scratch'])
    finally:client.close()
    files=observation.pop('files')
    assert not observation['active_eda'] and not list(archive.rglob('*.part'))
    assert all((archive/p).stat().st_size==n for p,n in files.items())
    assert all((archive/'scratch_originals'/p).stat().st_size==n for p,n in observation['scratch_noncache'].items())
    inputs=[p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    assert all(p.read_bytes()==(archive/p.relative_to(prepared)).read_bytes() for p in inputs)
    local=root/RESULT/'sgd_bounded_local_r1'
    for name in ('bounded_sgd_update.sv','sgd_registered44.sv','direct_update_wrapper.sv','unified_tp_direct_residual_update.sv'):
        assert (archive/name).read_bytes()==(local/name).read_bytes()
    assert (archive/'launch_worker.py').read_bytes()==Path(__file__).with_name('run_sgd_bounded_mapping.py').read_bytes()
    spec=importlib.util.spec_from_file_location('actual_bounded_worker',archive/'launch_worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    log=(archive/'mapping.log').read_text()
    parsed=worker.parse_mapping(log)
    assert parsed==state['diagnostics']
    for bad in (re.sub(r'^BOUNDED_MAPPING_COMPLETE$', 'MISSING', log, flags=re.M),
                log+'\nBOUNDED_MAPPING_COMPLETE\n',
                re.sub(r'^INPUT_REG_PATH top=direct_registered44 kind=gradient .+$','',log, count=1,flags=re.M),
                log+'\nERROR: synthetic parser test\n'):
        try:worker.parse_mapping(bad)
        except ValueError:pass
        else:raise AssertionError('corrupt mapping accepted')
    rows=[]
    for name in ('direct_registered44','sgd_registered44'):
        util=(archive/(name+'_utilization.rpt')).read_text()
        timing=(archive/(name+'_timing.rpt')).read_text()
        row=dict(top=name)
        for key,label in dict(lut='Slice LUTs*',ff='Slice Registers',bram='Block RAM Tile',dsp='DSPs').items():
            row[key]=float(re.search(r'^\|\s*'+re.escape(label)+r'\s*\|\s*([\d.]+)',util,re.M)[1])
        lines=timing.splitlines();i=next(i for i,line in enumerate(lines) if line.strip().startswith('WNS(ns)'))
        values=lines[i+2].split()
        for key,index in (('wns_estimate',0),('whs_estimate',4),('wpws_estimate',8)):row[key]=float(values[index])
        for key in ('no_clock','unconstrained_internal_endpoints','no_input_delay','no_output_delay','loops'):
            row[key]=int(re.search(r'checking '+key+r' \((\d+)\)',timing)[1])
        for kind in ('gradient','code','residual'):
            item=next(x for x in parsed['input_register_paths'] if x['top']==name and x['kind']==kind)
            row[kind+'_path_slack']=item['slack']
        rows.append(row)
    with (archive/'updater_resources.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(status='PASS_bounded_updater_mapping_archive_not_routed_STA',raw_files=len(files),raw_bytes=sum(files.values()),
                sizes_match=True,no_part=True,prepared_files_byte_equal=len(inputs),prepared_bytes=sum(p.stat().st_size for p in inputs),
                simulated_RTL_byte_equal=True,actual_worker_parser_used=True,corrupt_variants_rejected=4,
                checks=parsed,resources=rows,remote_observation=observation,
                dcp_crc='not applicable: no DCP',
                scope='Signed44 scalar operators with matched elastic input registers; different numerical residual representations and total pipeline capacities. No memory, learning-system, routed timing or Fmax claim.')
    (archive/'terminal_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
