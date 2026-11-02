"""Remove the up-front sparse FF count from an isolated real layer client.

This is command/protocol equivalence, not full-core throughput measurement.
The counted reference is allowed oracle counts only for this equivalence test.
"""
import collections
import datetime as dt
import difflib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('\\\\?\\' + str(Path(__file__).resolve().parents[2]))
EXP = ROOT/'verification/experiments/srnn_endmarked_layer_r1'
OUT = ROOT/'results/paper_evidence/supplement_20261015/performance_optimization/srnn_endmarked_layer_local_r1'
SOURCE = ROOT/'verification/experiments/scnn_control_payload/candidate_r4/rtl/control/unified_fc_srnn_h40_p4_layer_client_v1.sv'


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('ambiguous anchor: '+old)
    return text.replace(old, new, 1)


def candidate(source):
    text = source
    for old, new in [
        ('    input  wire logic                           i_group_valid,',
         '    input  wire logic                           i_streamed_feedforward,\n'
         '    input  wire logic                           i_group_feedforward_end,\n'
         '    input  wire logic                           i_group_valid,'),
        ('    logic descriptor_legal;',
         '    logic streamed_q, feedforward_done_q;\n'
         '    logic streamed, end_legal, count_legal;\n'
         '    logic descriptor_legal;'),
        ('        descriptor_total_groups =',
         '        streamed = i_streamed_feedforward === 1\'b1;\n'
         '        descriptor_total_groups ='),
        ('            i_feedforward_group_count != 0 &&',
         '            (streamed ? (i_mode == MODE_SRNN && i_feedforward_group_count == 0) :\n'
         '                i_feedforward_group_count != 0) &&'),
        ('        expected_recurrent =\n            issued_group_count_q >= feedforward_group_count_q;',
         '        expected_recurrent = streamed_q ? feedforward_done_q :\n'
         '            issued_group_count_q >= feedforward_group_count_q;\n'
         '        end_legal = !streamed_q || !i_group_feedforward_end ||\n'
         '            (!i_group_recurrent && i_group_source_valid == 0);\n'
         '        count_legal = !streamed_q || feedforward_done_q ||\n'
         '            issued_group_count_q != {GROUP_COUNT_WIDTH{1\'b1}};'),
        ('        group_legal = group_kind_legal && group_address_legal;',
         '        group_legal = group_kind_legal && group_address_legal && end_legal && count_legal;'),
        ('        o_array_command_first = issued_group_count_q == 0;\n'
         "        o_array_command_last = issued_group_count_q + 1'b1 ==\n            total_group_count_q;",
         '        o_array_command_first = issued_group_count_q == 0 &&\n'
         '            (!streamed_q || !feedforward_done_q);\n'
         '        o_array_command_last = streamed_q ?\n'
         '            (feedforward_done_q ? (issued_group_count_q + 1\'b1 == total_group_count_q) :\n'
         '                (i_group_feedforward_end && total_group_count_q == 0)) :\n'
         '            (issued_group_count_q + 1\'b1 == total_group_count_q);'),
        ("            state_q <= STATE_IDLE;\n            mode_q <= MODE_FC;",
         "            state_q <= STATE_IDLE;\n            streamed_q <= 0; feedforward_done_q <= 0;\n            mode_q <= MODE_FC;"),
        ('                                mode_q <= i_mode;',
         '                                mode_q <= i_mode;\n'
         '                                streamed_q <= streamed; feedforward_done_q <= 0;'),
        ('                            error_code_q <= group_kind_legal ?\n                                ERROR_WEIGHT_ADDRESS : ERROR_GROUP_KIND;',
         '                            error_code_q <= !end_legal ? 8\'h06 : !count_legal ? 8\'h07 :\n'
         '                                group_kind_legal ? ERROR_WEIGHT_ADDRESS : ERROR_GROUP_KIND;'),
        ("                            issued_group_count_q <=\n                                issued_group_count_q + 1'b1;",
         '                            // The empty FF marker is a real command, not a free side channel.\n'
         '                            if (streamed_q && !feedforward_done_q && i_group_feedforward_end) begin\n'
         '                                issued_group_count_q <= 0;\n'
         '                                feedforward_done_q <= 1;\n'
         '                            end else issued_group_count_q <= issued_group_count_q + 1\'b1;'),
    ]:
        text = once(text, old, new)
    return text


def fixture():
    source = ROOT/'verification/experiments/nearest_neighbor/stream_r1/fixture/counts.mem'
    counts = [int(x, 16) for x in source.read_text().split()]
    if len(counts) != 588*700:
        raise ValueError('expected 588 real frames')
    counts += [0]*700
    counts += [255]+[0]*699
    counts += [2 if s == 1 else 4 if s == 2 else 3 if s == 699 else 0 for s in range(700)]
    counts += [1]*700
    headers, packets = [], []
    for frame in range(592):
        start = len(packets)
        for block in range(175):
            values = counts[frame*700+block*4:frame*700+block*4+4]
            for repeat in range(max(values)):
                mask = sum((v > repeat) << lane for lane, v in enumerate(values))
                packets.append((mask, block*2, 0, 0))
        packets.append((0, 0, 0, 1))
        nf = len(packets)-start
        # Recurrent suffix affects the layer protocol; these masks are directed,
        # not a new measurement of recurrent neural activity.
        nr = (0, 1, 113)[frame % 3]
        for k in range(nr):
            packets.append(((k+frame) % 16, 350+2*k, 1, 0))
        headers.append((start, nf, nr, 1, 1))
    # Counted FC, SRNN and M2 traffic must retain original descriptor behavior.
    for mode in (0, 1, 3):
        for nr in ((0, 3) if mode == 1 else (0,)):
            start = len(packets)
            packets.extend((mask, 2*k, 0, 0) for k, mask in enumerate((0, 1, 15, 4)))
            packets.extend((3, 350+2*k, 1, 0) for k in range(nr))
            headers.append((start, 4, nr, mode, 0))
    (OUT/'headers.txt').write_text(''.join(' '.join(map(str, x))+'\n' for x in headers), encoding='ascii')
    (OUT/'packets.txt').write_text(''.join(' '.join(map(str, x))+'\n' for x in packets), encoding='ascii')
    return dict(cases=len(headers), commands=len(packets), source=str(source.relative_to(ROOT)),
                source_bytes=source.stat().st_size, source_mtime_ns=source.stat().st_mtime_ns,
                real_input_frames=588, boundary_input_frames=4,
                recurrent_suffix='Directed masks and lengths 0/1/113, not measured recurrent spikes')


def parse(text, info):
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b', text):
        raise ValueError('runtime diagnostic')
    rows = [dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
            for line in re.findall(r'^ENDMARKED_CASE (.+)$', text, re.M)]
    if len(rows) != info['cases'] or [x['case'] for x in rows] != list(range(info['cases'])):
        raise ValueError('missing/duplicate cases')
    if sum(x['commands'] for x in rows) != info['commands']:
        raise ValueError('command coverage')
    if len(re.findall(r'^ENDMARKED_PASS negatives=4$', text, re.M)) != 1:
        raise ValueError('missing terminal/negative coverage')
    if not sum(x['stalls'] for x in rows):
        raise ValueError('no command backpressure')
    return rows


def main():
    OUT.mkdir(exist_ok=False); EXP.mkdir(exist_ok=False)
    state = dict(start=dt.datetime.now().astimezone().isoformat(), status='preparing')
    def save():
        (OUT/'status.json').write_text(json.dumps(state, indent=2)+'\n')
    save()
    try:
        source = SOURCE.read_text()
        changed = candidate(source)
        (EXP/'endmarked.sv').write_text(changed, encoding='ascii')
        (EXP/'counted_reference.sv').write_text(once(source,
            'module unified_fc_srnn_h40_p4_layer_client_v1 #(', 'module counted_reference #('), encoding='ascii')
        (EXP/'changes.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True), changed.splitlines(True),
            fromfile='exact_A_layer_client', tofile='isolated_endmarked_layer_client')), encoding='ascii')
        shutil.copy2(Path(__file__).with_name('tb_srnn_endmarked_layer.sv'), EXP/'tb.sv')
        for path in EXP.iterdir():
            shutil.copy2(path, OUT/path.name)
        shutil.copy2(Path(__file__), OUT/'executed_runner.py')
        state['fixture'] = fixture(); save()
        normal = str(OUT).removeprefix('\\\\?\\')
        linux = '/mnt/'+normal[0].lower()+normal[2:].replace('\\', '/')
        for stage, command in (
            ('build', 'verilator --binary --timing --assert -j 1 -Wall -Wno-fatal --top-module tb_srnn_endmarked_layer --Mdir obj tb.sv endmarked.sv counted_reference.sv'),
            ('run', 'obj/Vtb_srnn_endmarked_layer')):
            shell = 'cd '+shlex.quote(linux)+' && ulimit -v 3145728 && /usr/bin/time -v -o '+stage+'_resources.txt '+command+' > '+stage+'.log 2>&1'
            state['status'] = stage+'_running'; begin = time.monotonic()
            with (OUT/(stage+'_launcher.log')).open('xb') as handle:
                process = subprocess.Popen(['wsl', '-e', 'sh', '-lc', shell], stdout=handle, stderr=subprocess.STDOUT)
                state[stage] = dict(pid=process.pid, exit=None, command=shell); save()
                code = process.wait(timeout=900)
            state[stage].update(exit=code, wall_seconds=time.monotonic()-begin); save()
            if code:
                raise RuntimeError(stage+' failed')
        log = (OUT/'run.log').read_text()
        rows = parse(log, state['fixture'])
        for bad in (log.replace('ENDMARKED_PASS', 'MISSING'), log.replace('ENDMARKED_CASE', 'MISSING', 1), log+'\n%Error: synthetic\n'):
            try:
                parse(bad, state['fixture'])
            except ValueError:
                pass
            else:
                raise AssertionError('damaged transcript accepted')
        state['case_count'] = len(rows)
        state['commands_equal'] = sum(x['commands'] for x in rows)
        state['stalled_cycles_equal'] = sum(x['stalls'] for x in rows)
        state['damaged_transcripts_rejected'] = 3
        state['diagnostics'] = {stage:dict(warnings=dict(collections.Counter(re.findall(r'%Warning-([A-Z0-9_]+):',
                    (OUT/(stage+'.log')).read_text()))), errors=len(re.findall(r'%Error|%Fatal', (OUT/(stage+'.log')).read_text())))
                    for stage in ('build', 'run')}
        for path in EXP.iterdir():
            assert path.read_bytes() == (OUT/path.name).read_bytes()
        assert candidate(SOURCE.read_text()) == (EXP/'endmarked.sv').read_text()
        src = ROOT/state['fixture']['source']
        assert (src.stat().st_size, src.stat().st_mtime_ns) == (state['fixture']['source_bytes'], state['fixture']['source_mtime_ns'])
        state['status'] = 'PASS_endmarked_real_layer_command_equivalence_not_full_core'
    except Exception as error:
        state.update(status='FAIL', error=repr(error))
    state['end'] = dt.datetime.now().astimezone().isoformat(); save()
    print(json.dumps(state, indent=2))
    if state['status'] == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
