"""Small local RTL experiment: source availability and burst backpressure."""
import csv
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

from run_srnn_resident_remote import ROOT, RESULT, local_path
from run_source_stream_comparison import SOURCES, diagnostics, replace_once

NAME = 'source_service_sweep_local_r1'
SCENARIOS = ('word_every_4_cycles', 'word_every_16_cycles',
             'consumer_8_off_8_on', 'word_every_4_and_consumer_burst')
SCOPE = ('Local Verilator component RTL, four new deterministic service conditions, '
         'same588 real frames plus4 boundaries and three producers. Original12 '
         'weight stores and H40 consumers; pre-LIF only. Not a new distribution, '
         'whole-network/author-system comparison, replay, DDR or routed timing.')


def parse_log(text, fixture):
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b|\bERROR\b', text):
        raise ValueError('runtime diagnostic failure')
    rows = [{k: int(v) for k, v in re.findall(r'(\w+)=(-?\d+)', line)}
            for line in re.findall(r'^SCAN_FRAME (.+)$', text, re.M)]
    if [(r['scenario'], r['policy'], r['frame']) for r in rows] != [
            (s, p, f) for s in range(4) for p in range(3) for f in range(592)]:
        raise ValueError('incomplete or duplicated ordered frame set')
    if text.count('SERVICE_SWEEP_PASS frames=7104 current_checks=3196800 policies=3 scenarios=4') != 1:
        raise ValueError('missing final completion')
    with (fixture/'frames.csv').open(newline='') as handle:
        reference = list(csv.DictReader(handle))
    for row in rows:
        if min(row.values()) < 0 or row['checked'] != 450 or row['input_words'] != 175 or row['delimiters'] != 1:
            raise ValueError('numeric or ingress accounting')
        if row['memory'] != row['groups']+1 or row['accum'] != row['memory']:
            raise ValueError('consumer accounting')
        if row['total'] != sum(row[k] for k in ('load', 'scan', 'emit', 'finish', 'drain')) or row['total'] != row['ingress']+row['service']:
            raise ValueError('cycle closure')
        if row['policy'] == 2 and row['scans']:
            raise ValueError('streaming rescanned')
        if row['frame'] < 588:
            ref = reference[row['frame']]
            key = 'dense_issue_groups' if row['policy'] == 0 else 'packed_issue_groups'
            expected = int(ref[key]) - int(row['policy'] != 0 and int(ref['whole_vector_zero']) != 0)
            if row['groups'] != expected:
                raise ValueError('independent group count')
    totals = []
    for s in range(4):
        for p in range(3):
            chosen = [r for r in rows if r['scenario'] == s and r['policy'] == p and r['frame'] < 588]
            total = dict(scenario=s, condition=SCENARIOS[s], policy=p, frames=len(chosen))
            total.update({k: sum(r[k] for r in chosen) for k in (
                'total', 'load', 'scan', 'emit', 'finish', 'drain', 'input_words',
                'input_idle', 'input_stalls', 'groups', 'delimiters', 'group_stalls',
                'result_stalls', 'memory', 'accum', 'checked')})
            if s in (0, 1, 3) and total['input_idle'] == 0:
                raise ValueError('source scarcity not exercised')
            if s in (2, 3) and (not total['group_stalls'] or not total['result_stalls']):
                raise ValueError('consumer backpressure not exercised')
            totals.append(total)
    comparisons = []
    for s in range(4):
        a, b, c = [r['total'] for r in totals if r['scenario'] == s]
        paired = [[r for r in rows if r['scenario'] == s and r['policy'] == p and r['frame'] < 588] for p in (1, 2)]
        comparisons.append(dict(condition=SCENARIOS[s], scan_cycles=a, buffered_cycles=b,
            streaming_cycles=c, buffered_over_streaming=b/c, scan_over_streaming=a/c,
            streaming_saved_fraction=1-c/b,
            shorter=sum(y['total'] < x['total'] for x, y in zip(*paired)),
            equal=sum(y['total'] == x['total'] for x, y in zip(*paired)),
            longer=sum(y['total'] > x['total'] for x, y in zip(*paired))))
    return dict(scope=SCOPE, frames=rows, real_frame_totals=totals, comparisons=comparisons,
                current_checks=3196800, shared_preload_writes=16800)


def prepare(root, out):
    old = root/'verification/experiments/nearest_neighbor/stream_r1'
    # Keep measured sources untouched; the new TB changes only supply/service.
    for rel in ('source_scan.sv', 'source_stream.sv', 'scan_consumer.sv'):
        shutil.copy2(old/rel, out/rel)
    shutil.copytree(old/'rtl', out/'rtl')
    shutil.copytree(old/'fixture', out/'fixture')
    tb = (old/'tb_source_scan.sv').read_text()
    edits = [
        ('integer scenario=0;', 'integer scenario=0;\n    integer input_period=1;\n    logic input_was_stalled=0;\n    logic [31:0] held_input;'),
        ('assign request_allow=scenario==0 || (cycles-origin)%7!=0;', 'assign request_allow=1;'),
        ('assign consume_enable=scenario==0 || (cycles-origin)%11>=3;',
         'assign consume_enable=scenario<2 || (cycles-origin)%16>=8;'),
        ('assign rr=scenario==0 || result_age>=4;', 'assign rr=scenario<2 || result_age>=8;'),
        ('if (rst_n) begin', 'if (rst_n) begin\n            if(input_was_stalled && (!count_valid || counts!==held_input))\n                $fatal(1,"input changed under backpressure");\n            input_was_stalled=count_valid && !count_ready; held_input=counts;'),
        ('if (!count_valid) count_valid=scenario==0 || (cycles-origin)%5!=0;',
         'if (!count_valid) count_valid=(cycles-origin)>=word_index*input_period;'),
        ('for(int s=0;s<2;s++) begin\n            scenario=s;',
         'for(int s=0;s<4;s++) begin\n            scenario=s; input_period=(s==1 ? 16 : (s==2 ? 1 : 4));'),
        ('SCAN_PASS frames=3552 current_checks=1598400 policies=3 scenarios=2',
         'SERVICE_SWEEP_PASS frames=7104 current_checks=3196800 policies=3 scenarios=4'),
        ('initial begin #100000000;', 'initial begin #1000000000;'),
    ]
    for old_text, new_text in edits:
        tb = replace_once(tb, old_text, new_text)
    (out/'tb_source_scan.sv').write_text(tb, encoding='ascii')
    shutil.copy2(Path(__file__), out/'experiment_runner.py')
    manifest = []
    for path in old.rglob('*'):
        rel = path.relative_to(old)
        if path.is_file() and (rel.parts[0] in ('rtl', 'fixture') or str(rel) in ('source_scan.sv', 'source_stream.sv', 'scan_consumer.sv')):
            target = out/rel
            if path.read_bytes() != target.read_bytes():
                raise ValueError('copied original differs: '+str(rel))
            manifest.append(dict(path=rel.as_posix(), bytes=path.stat().st_size))
    (out/'preparation.json').write_text(json.dumps(dict(scope=SCOPE, unchanged_inputs=manifest,
        conditions=list(SCENARIOS), source_schedule='word i available at relative cycle i*period; valid/payload held until accepted, no dropping',
        burst='consumer accepts during phases8..15 of each16-cycle interval; result held8cycles',
        scope_note='No replay added; no unseen source input; all original588 frames retained in original order; four directed boundaries separate',
        resources='one compile job, 3GiB per-process virtual-memory cap, no Vivado/XSim/full-core simulation'), indent=2)+'\n')


def main():
    root = Path(local_path(ROOT))
    out = root/RESULT/NAME
    out.mkdir(exist_ok=False)
    prepare(root, out)
    state = dict(status='prepared', start=dt.datetime.now().astimezone().isoformat(), scope=SCOPE)
    windows_path = str(out).removeprefix('\\\\?\\')
    linux = '/mnt/'+windows_path[0].lower()+windows_path[2:].replace('\\', '/')
    def save():
        (out/'status.json').write_text(json.dumps(state, indent=2)+'\n')
    def execute(stage, command, timeout=1800):
        state['status'] = stage+'_running'; save(); begin = time.monotonic()
        full = ['wsl', '-e', 'sh', '-lc', 'cd '+shlex.quote(linux)+' && ulimit -v 3145728 && '+command]
        with (out/(stage+'.log')).open('xb') as log:
            process = subprocess.Popen(full, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage] = dict(exit=None, pid=process.pid, command=full, timeout=True)
                save(); raise
        log_text = (out/(stage+'.log')).read_text(errors='replace')
        diag = diagnostics(log_text)
        state[stage] = dict(exit=code, wall_seconds=time.monotonic()-begin, command=full, diagnostics=diag)
        save()
        if code or diag['errors'] or diag['fatal'] or diag['structural']:
            raise RuntimeError(stage+' failed')
    try:
        execute('environment', 'verilator --version && /usr/bin/time --version && head -3 /proc/meminfo', 180)
        sources = [shlex.quote('rtl/'+p) for p in SOURCES]
        execute('build', '/usr/bin/time -v -o build_resources.txt verilator --binary --timing --assert -j 1 -Wall -Wno-fatal --top-module tb_source_scan --Mdir obj '+
                ' '.join(sources)+' source_scan.sv source_stream.sv scan_consumer.sv tb_source_scan.sv')
        execute('run', '/usr/bin/time -v -o run_resources.txt obj/Vtb_source_scan +FIXTURE=fixture')
        checks = parse_log((out/'run.log').read_text(errors='replace'), out/'fixture')
        for name, rows in (('frame_measurements.csv', checks['frames']), ('comparison.csv', checks['comparisons'])):
            with (out/name).open('x', newline='', encoding='ascii') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (out/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
        state.update(status='PASS_local_pre_LIF_service_sweep', comparisons=checks['comparisons'],
                     current_checks=checks['current_checks'])
    except Exception as error:
        state.update(status='FAIL', error=repr(error))
    state['end'] = dt.datetime.now().astimezone().isoformat(); save()
    print(json.dumps(state, indent=2))
    if state['status'] == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
