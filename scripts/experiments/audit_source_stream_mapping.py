"""Archive the completed mapping-only run without treating synthesis as routed STA."""
import csv
import json
from pathlib import Path
import re

from audit_source_stream_comparison import inventory
from run_source_stream_comparison import diagnostics
from run_source_stream_mapping import NAME, EXPERIMENT
from run_srnn_resident_remote import ROOT, RESULT, connect, local_path


def resources(archive, policy):
    text = (archive / f'producer_{policy}_utilization.rpt').read_text()
    labels = dict(lut='Slice LUTs*', ff='Slice Registers', bram='Block RAM Tile', dsp='DSPs')
    result = dict(policy=policy)
    for key, label in labels.items():
        result[key] = float(re.search(r'^\|\s*' + re.escape(label) + r'\s*\|\s*([\d.]+)', text, re.M)[1])
    timing = (archive / f'producer_{policy}_timing.rpt').read_text()
    lines = timing.splitlines()
    header = next(i for i, line in enumerate(lines) if line.strip().startswith('WNS(ns)'))
    values = lines[header + 2].split()
    for key, index in (('wns_estimate', 0), ('whs_estimate', 4), ('wpws_estimate', 8)):
        result[key] = float(values[index])
    for key in ('no_clock', 'unconstrained_internal_endpoints', 'no_input_delay', 'no_output_delay', 'loops'):
        result[key] = int(re.search(r'checking ' + key + r' \((\d+)\)', timing)[1])
    return result


def main():
    root = Path(local_path(ROOT)); archive = root / RESULT / NAME; prepared = root / EXPERIMENT
    if (archive / 'terminal_audit.json').exists():
        raise FileExistsError('Already audited')
    state = json.loads((archive / 'status.json').read_text())
    assert state['exit'] == 0 and state['end'] and state['status'].startswith('PASS_')
    client = connect()
    try:
        observation = inventory(client, NAME, state['scratch'])
    finally:
        client.close()
    files = observation.pop('files')
    assert not observation['active_eda'] and not list(archive.rglob('*.part'))
    assert all((archive / p).stat().st_size == n for p, n in files.items())
    assert all((archive / 'scratch_originals' / p).stat().st_size == n for p, n in observation['scratch_noncache'].items())
    inputs = [p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    assert all(p.read_bytes() == (archive / p.relative_to(prepared)).read_bytes() for p in inputs)
    r1 = root / RESULT / 'source_stream_compare_r1'
    for name in ('source_scan.sv', 'source_stream.sv'):
        assert (archive / name).read_text() == (r1 / name).read_text().replace('input logic', 'input wire')
    for name in ('clock.xdc', 'mapping.tcl', 'run_fc_bitmap_mapping.py'):
        assert (archive / name).read_bytes() == (r1 / name).read_bytes()
    assert (archive / 'run_source_stream_comparison.py').read_bytes() == (r1 / 'launch_worker.py').read_bytes()
    assert (archive / 'launch_worker.py').read_bytes() == (root / 'scripts/experiments/run_source_stream_mapping.py').read_bytes()
    log = (archive / 'mapping.log').read_text()
    diag = diagnostics(log)
    assert diag == state['diagnostics'] and diag['errors'] == diag['critical_warnings'] == 0
    assert 'SOURCE_STREAM_MAPPING_COMPLETE' in log
    rows = [resources(archive, p) for p in range(3)]
    with (archive / 'producer_resources.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = dict(status='PASS_mapping_archive_not_routed_STA', raw_files=len(files), raw_bytes=sum(files.values()),
                  all_sizes_match=True, no_part=True, prepared_files_byte_equal=len(inputs),
                  prepared_bytes=sum(p.stat().st_size for p in inputs), input_net_only_transformation=True,
                  actual_worker_equal=True, diagnostics=diag, resources=rows, remote_observation=observation,
                  dcp_crc='not applicable: no DCP',
                  boundary='Buffered memory failed RAM inference because its write is in an asynchronous-reset block. Do not use this as a strong resource baseline. Original R1 FAIL retained; no repeated RTL simulation.')
    (archive / 'terminal_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
