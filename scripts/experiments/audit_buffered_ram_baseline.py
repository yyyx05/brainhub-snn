"""Accept the targeted RAM-write refactor evidence and its producer-only costs."""
import csv
import importlib.util
import json
from pathlib import Path
import re

from audit_source_stream_comparison import inventory
from audit_source_stream_mapping import resources
from run_buffered_ram_baseline import NAME, EXPERIMENT
from run_srnn_resident_remote import ROOT, RESULT, connect, local_path


def main():
    root=Path(local_path(ROOT));archive=root/RESULT/NAME;prepared=root/EXPERIMENT
    if (archive/'terminal_audit.json').exists():raise FileExistsError('already audited')
    state=json.loads((archive/'status.json').read_text())
    assert state['end'] and state['status']=='PASS_buffered_RAM_equivalence_and_synthesis_not_routed_STA'
    assert all(state[s]['exit']==0 for s in ('build','run','mapping'))
    client=connect()
    try:observation=inventory(client,NAME,state['scratch'])
    finally:client.close()
    files=observation.pop('files')
    assert not observation['active_eda'] and not list(archive.rglob('*.part'))
    assert all((archive/p).stat().st_size==n for p,n in files.items())
    assert all((archive/'scratch_originals'/p).stat().st_size==n for p,n in observation['scratch_noncache'].items())
    inputs=[p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    assert all(p.read_bytes()==(archive/p.relative_to(prepared)).read_bytes() for p in inputs)
    assert (archive/'launch_worker.py').read_bytes()==(root/'scripts/experiments/run_buffered_ram_baseline.py').read_bytes()
    spec=importlib.util.spec_from_file_location('actual_ram_worker',archive/'launch_worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    r2=root/RESULT/'source_stream_mapping_r2'
    source=(r2/'source_scan.sv').read_text()
    assert (archive/'source_scan.sv').read_text()==worker.transform(source)
    assert (archive/'source_scan_original.sv').read_text()==source.replace('module source_scan (','module source_scan_original (')
    for name in ('source_stream.sv','clock.xdc','run_fc_bitmap_mapping.py','run_source_stream_comparison.py'):
        assert (archive/name).read_bytes()==(r2/name).read_bytes()
    assert (archive/'mapping.tcl').read_text()==(r2/'mapping.tcl').read_text().replace('foreach policy {0 1 2}','foreach policy {0 1}')
    assert (archive/'counts.mem').read_bytes()==(root/RESULT/'source_stream_compare_r1/fixture/counts.mem').read_bytes()
    assert worker.parse_log((archive/'run.log').read_text())==state['checks']
    for stage in ('build','run','mapping'):
        assert worker.common.diagnostics((archive/(stage+'.log')).read_text())==state[stage]['diagnostics']
    mapping=(archive/'mapping.log').read_text()
    assert 'SOURCE_STREAM_MAPPING_COMPLETE' in mapping
    assert 'Synth 8-4767' not in mapping and 'Synth 8-7137' not in mapping
    rows=[resources(archive,p) for p in range(2)]
    for row in rows:
        util=(archive/f"producer_{row['policy']}_utilization.rpt").read_text()
        row['lut_as_memory']=int(re.search(r'^\|\s*LUT as Memory\s*\|\s*(\d+)',util,re.M)[1])
        assert row['lut_as_memory']==132
    with (archive/'producer_resources.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(status='PASS_RAM_inferable_buffered_baseline_archive',raw_files=len(files),raw_bytes=sum(files.values()),
        all_sizes_match=True,no_part=True,prepared_files_byte_equal=len(inputs),prepared_bytes=sum(p.stat().st_size for p in inputs),
        actual_worker_strict_reparse=True,only_frame_write_process_changed=True,unchanged_count_fixture=True,
        checks=state['checks'],resources=rows,stages={s:state[s] for s in ('build','run','mapping')},
        warning_lines=re.findall(r'^WARNING:.*$',mapping,re.M),remote_observation=observation,
        dcp_crc='not applicable: no DCP',
        boundary='Producer-only cycle comparison, not formal equivalence or whole-network reset recovery. Synthesis estimates, not routed STA. R2 register-array results retained but not used as the primary cost baseline. No cleanup.')
    (archive/'terminal_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
