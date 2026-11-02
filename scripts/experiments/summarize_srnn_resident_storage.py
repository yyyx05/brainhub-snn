#!/usr/bin/env python3
"""Account actual context-1 banks without double-counting hierarchy totals."""
import csv
import json
from pathlib import Path
import re

from run_srnn_resident_remote import local_path

ROOT = Path(local_path(Path(__file__).resolve().parents[2]))
RESULT = ROOT / 'results/paper_evidence/supplement_20261015/performance_optimization'
REPORT = RESULT / 'scnn_issue_banked_synth_r1/server_results/final_composition_synth/utilization_hierarchical_post_synth.rpt'


def banks_from_report(text):
    stack, banks = [], []
    for line in text.splitlines():
        cells = line.split('|')[1:-1]
        if len(cells) != 10 or not all(re.fullmatch(r'\s*\d+\s*', x) for x in cells[2:]):
            continue
        name = cells[0].strip()
        if name.startswith('('):
            continue
        indent = len(cells[0]) - len(cells[0].lstrip())
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, name))
        path = '/'.join(x[1] for x in stack)
        if 'g_context[1].g_enabled.u_store/' not in path or not re.fullmatch(r'g_bank\[\d+\].u_weight_memory', name):
            continue
        region = int(re.search(r'g_region\[(\d+)\]', path)[1])
        tile = int(re.search(r'g_tile\[(\d+)\]', path)[1])
        bank = int(re.search(r'g_bank\[(\d+)\]', name)[1])
        banks.append(dict(path=path, region=region, tile=tile, global_tile=region*4+tile,
                          bank=bank, ramb36=int(cells[7]), ramb18=int(cells[8]),
                          lut=int(cells[2]), ff=int(cells[6])))
    if len(banks) != 32 or len({(x['global_tile'], x['bank']) for x in banks}) != 32:
        raise ValueError('expected 32 unique context-1 banks')
    return banks


def account(banks):
    rows = []
    for bank in banks:
        row = dict(bank)
        tile, index = row['global_tile'], row['bank']
        if tile not in range(8) or index not in range(4):
            raise ValueError('unexpected mapped tile/bank')
        row['matrix'] = 'WH1' if tile < 7 else 'Wout'
        row['source_count'] = 450 if tile < 7 else 256
        outputs = min(40, 256 - tile*40) if tile < 7 else 20
        row['codes_per_published_row'] = max(0, min(12, outputs - index*12))
        row['published_rows'] = row['source_count'] if row['codes_per_published_row'] else 0
        row['cache_row_base'] = 0
        row['declared_depth'] = 578
        row['declared_width'] = 200
        row['payload_bits'] = row['published_rows'] * row['codes_per_published_row'] * 16
        row['declared_bits'] = 578 * 200
        row['published_row_padding_bits'] = row['published_rows'] * 200 - row['payload_bits']
        row['unpublished_row_bits'] = (578-row['published_rows']) * 200
        row['physical_bits'] = row['ramb36']*36864 + row['ramb18']*18432
        row['mapping_capacity_gap_bits'] = row['physical_bits'] - row['declared_bits']
        if row['physical_bits'] != sum(row[key] for key in ('payload_bits','published_row_padding_bits','unpublished_row_bits','mapping_capacity_gap_bits')):
            raise ValueError('bank accounting does not close')
        rows.append(row)
    fields = ('ramb36','ramb18','payload_bits','declared_bits','published_rows',
              'published_row_padding_bits','unpublished_row_bits','physical_bits','mapping_capacity_gap_bits')
    total = {key: sum(row[key] for row in rows) for key in fields}
    total['payload_fraction_of_physical'] = total['payload_bits']/total['physical_bits']
    total['payload_fraction_of_declared'] = total['payload_bits']/total['declared_bits']
    return dict(banks=rows, totals=total,
        matrix_totals={matrix:{key:sum(row[key] for row in rows if row['matrix']==matrix) for key in fields} for matrix in ('WH1','Wout')},
        evidence=dict(report=str(REPORT.relative_to(ROOT)), dimensions='Actual wrapper 4 banks/tile, 200 bits x 578 rows; region<2 enables context1; cache bases 0; refill scheduler 12+12+12+4 lanes.',
                      observed='32 bank rows each report 6 RAMB36 and 0 RAMB18. Parent and child XPM totals are not added.'),
        boundary='Physical capacity uses nominal RAMB36/RAMB18 bit capacities, not usable payload or measured bandwidth. Mapping gap is aggregate spare capacity; no unobserved primitive aspect-ratio decomposition is asserted. Payload is the fixed 450x256 plus 256x20 workload, not the configurable maximum. Context0, optimizer storage and other system memories are excluded.')


def main():
    result = account(banks_from_report(REPORT.read_text(encoding='utf-8')))
    folder = RESULT / 'srnn_equal_storage_r1'
    folder.mkdir(exist_ok=True)
    (folder / 'storage_ledger.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    with (folder / 'storage_banks.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['banks'][0]))
        writer.writeheader()
        writer.writerows(result['banks'])
    print(json.dumps(result['totals'], indent=2))


if __name__ == '__main__':
    main()
