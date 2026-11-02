#!/usr/bin/env python3
"""Scope the original natural B4T12 test to public Conv8 service and RTL query."""
import argparse
import difflib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
if os.name == 'nt' and not str(ROOT).startswith('\\\\?\\'):
    ROOT = Path('\\\\?\\'+str(ROOT))
EXPERIMENT = Path('verification/experiments/scnn_final_visibility')
ORIGINAL = EXPERIMENT / ('original_b4t12_r2/results/paper_evidence/'
    'supplement_20261004/rtl/scnn_real_final_graph/server_b4t12_r2/'
    'tb_scnn_real_final_b4t12.sv')
SCOPE = ('Final combined banked scanner and issue publisher. Original cold B4T12 '
    'plus query4, Conv1..7 replaced by explicit CPU integer Conv8 service input. '
    'Actual final-top Conv8, capture, shared learner, publication, version and RTL '
    'classifier query. Not full raw Conv1..8, end-to-end throughput, DDR, or abort evidence.')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('nonunique source anchor: '+old[:100])
    return text.replace(old, new, 1)


def transform(source):
    text = replace_once(source, 'module tb_scnn_real_final_b4t12;',
                        'module tb_scnn_tail_visibility;')
    start = text.index('        cfg_in_width = 32;')
    end = text.index('        cfg_in_width = CONV7_SPATIAL;', start)
    # Fill only the TB service memory. No expected state/trace is written to DUT.
    text = text[:start]+'''        for (int channel = 0; channel < 512; channel++)
            for (int row = 0; row < 4; row++)
                for (int col = 0; col < 4; col++)
                    feature_memory[0][CONV7_WRITE_BASE+channel*4+row][col] =
                        tail_input[timestep_phase*8192+channel*16+row*4+col];
        progress_stage = 5+timestep_phase;

'''+text[end:]
    for old, new in (
        ('fg_descriptors != (timestep_phase+1)*8', 'fg_descriptors != timestep_phase+1'),
        ('tv_probes_checked != (timestep_phase+1)*24', 'tv_probes_checked != (timestep_phase+1)*3'),
        ('fg_descriptors != 416', 'fg_descriptors != 52'),
        ('tv_probes_checked != 1248', 'tv_probes_checked != 156'),
        ('SCNN_TEMPORAL_FINAL_PASS', 'SCNN_TAIL_VISIBILITY_PASS'),
        ('SUPPLEMENT_FULLGRAPH_INITIALIZATION', 'SCNN_TAIL_INITIALIZATION'),
        ('SUPPLEMENT_FULLGRAPH_TIMESTEP', 'SCNN_TAIL_TIMESTEP'),
        ('Conv1-to-classifier timeout', 'Conv8-to-classifier timeout'),
        ('// Only the raw frames and original banks drive inference. All expected\n'
         '    // arrays below are read-only oracles, never restored into DUT state.',
         '// Explicit Conv7 bits and original banks drive public services.\n'
         '    // Expected state/trace arrays are read-only; actual spills feed restores.'),
    ):
        text = replace_once(text, old, new)
    if text.count('.u_weight_store.g_bank') != 2:
        raise ValueError('packing diagnostics changed')
    text = text.replace('.u_weight_store.g_bank',
                        '.u_weight_store.g_context[0].g_enabled.u_store.g_bank')
    text = replace_once(text, '    task automatic fg_load_expected;\n        string fixture, control;', '''
    logic tail_input [0:52*8192-1];
    logic signed [15:0] tail_pre [0:52*8192-1];
    logic signed [15:0] tail_post [0:52*8192-1];
    logic tail_spike [0:52*8192-1];
    int tail_scalars, tail_forward_responses, tail_learning_responses;
    int tail_code_writes, tail_residual_writes, tail_invalidations;
    bit tail_window_invalidated [0:2];
    task automatic fg_load_expected;
        string fixture, control, tail_fixture;
        if (!$value$plusargs("TAIL_FIXTURE=%s", tail_fixture))
            $fatal(1, "TAIL_FIXTURE required");
        $readmemh({tail_fixture,"/conv8_input_bits.hex"}, tail_input);
        $readmemh({tail_fixture,"/conv8_pre.hex"}, tail_pre);
        $readmemh({tail_fixture,"/conv8_post.hex"}, tail_post);
        $readmemh({tail_fixture,"/conv8_spike.hex"}, tail_spike);''')
    text = replace_once(text, "        post = int'(pre) - (spike ? 256 : 0);", '''        post = int'(pre) - (spike ? 256 : 0);
        if (layer != 8 || pre !== tail_pre[timestep_phase*8192+index] ||
            16'(post) !== tail_post[timestep_phase*8192+index] ||
            spike !== tail_spike[timestep_phase*8192+index])
            $fatal(1, "independent full Conv8 scalar mismatch frame=%0d index=%0d",timestep_phase,index);
        tail_scalars++;''')
    text = replace_once(text, '            if (ft_o_store_load_valid)', '''            if (fg_forward_valid && ft_o_store_forward_response_ready) tail_forward_responses++;
            if (fg_learning_valid && ft_o_store_learning_response_ready) tail_learning_responses++;
            if (ft_o_store_learning_write_valid && ft_i_store_learning_write_ready) begin
                tail_code_writes += int'(ft_o_store_learning_write_code_enable);
                tail_residual_writes += int'(ft_o_store_learning_write_residual_enable);
            end
            if (ft_o_conv_cache_invalidate && tv_window_end && !tv_query) begin
                tail_window_invalidated[tv_window] = 1;
                tail_invalidations++;
            end
            if (ft_o_store_load_valid)''')
    text = replace_once(text, '        for (int bank = 0; bank < 9; bank++) begin\n'
                        '            if (fg_forward_reads[bank] == 0) $fatal(1, "missing DDR bank");', '''        if (tail_scalars != 425984 || tail_code_writes != 662 || tail_residual_writes != 60273 ||
            tail_forward_responses != 122773504 || tail_learning_responses != 60273 ||
            fg_forward_valid || fg_learning_valid || fg_rmw_outstanding ||
            fg_forward_reads[7] != 122683392 || fg_forward_reads[8] != 90112 ||
            fg_fresh_reads != 9437184)
            $fatal(1, "tail scalar/traffic completeness");
        for (int window = 0; window < 3; window++)
            if (!tail_window_invalidated[window]) $fatal(1, "missing post-publication invalidation");
        $display("SCNN_TAIL_SERVICE scalars=%0d forward_responses=%0d learning_responses=%0d code_writes=%0d residual_writes=%0d invalidations=%0d versions=3 query_uses_updated_rtl=1",
            tail_scalars,tail_forward_responses,tail_learning_responses,tail_code_writes,tail_residual_writes,tail_invalidations);
        for (int bank = 0; bank < 9; bank++) begin
            if (bank < 7 && fg_forward_reads[bank] != 0) $fatal(1, "omitted Conv1..7 bank was read");''')
    return text


def parse_log(log):
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b', log):
        raise ValueError('runtime error')
    def rows(marker):
        return [{k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)', line)}
                for line in log.splitlines() if line.startswith(marker+' ')]
    final = rows('SCNN_TAIL_VISIBILITY_PASS')
    expected = dict(frames=52,descriptors=52,features=106496,currents=572,logits=572,
                    traces=98304,probes=156,rmw=60273,commits=3,publications=7568,
                    code_changes=662,query_currents=4,query_logits=4,fresh_reads=9437184)
    if len(final) != 1 or any(final[0].get(k) != v for k,v in expected.items()) or final[0].get('cycles',0)<=0:
        raise ValueError('incomplete or duplicate tail completion')
    layers=rows('SCNN_TEMPORAL_LAYER')
    frames=rows('SCNN_TEMPORAL_FRAME_STATUS')
    frame_keys=[(i,0 if i>=48 else (i%16)//4, i-48 if i>=48 else i//16*4+i%4) for i in range(52)]
    if [(r.get('frame'),r.get('sample'),r.get('timestep')) for r in frames] != frame_keys:
        raise ValueError('incomplete frame sequence')
    if [(r.get('frame'),r.get('sample'),r.get('timestep')) for r in layers] != frame_keys or any(
            r.get('layer')!=8 or r.get('descriptors')!=i+1 or r.get('cycles',0)<=0 for i,r in enumerate(layers)):
        raise ValueError('incomplete Conv8 descriptor sequence')
    windows=rows('SCNN_TEMPORAL_WINDOW_START')
    if [(r.get('window'),r.get('query'),r.get('base'),r.get('batch'),r.get('version')) for r in windows] != [
            (1,0,0,4,4609),(2,0,4,4,4610),(3,0,8,4,4611),(4,1,0,1,4612)]:
        raise ValueError('window or version sequence')
    commits=rows('SCNN_TEMPORAL_COMMIT')
    if commits != [dict(window=i+1,code_changes=c,residual_changes=r,rmw=r,version=4610+i,publications=1892*(i+2))
                   for i,(c,r) in enumerate(((0,20041),(0,20011),(662,20221)))]:
        raise ValueError('natural updates differ')
    service=rows('SCNN_TAIL_SERVICE')
    wanted=dict(scalars=425984,forward_responses=122773504,learning_responses=60273,
                code_writes=662,residual_writes=60273,versions=3,query_uses_updated_rtl=1)
    if len(service)!=1 or any(service[0].get(k)!=v for k,v in wanted.items()) or service[0].get('invalidations',0)<3:
        raise ValueError('service completeness')
    banks=rows('SCNN_TEMPORAL_DDR')
    if banks != [dict(bank=i,reads=0 if i<7 else 122683392 if i==7 else 90112) for i in range(9)]:
        raise ValueError('service read accounting')
    if len(rows('SCNN_TEMPORAL_CROSSING'))!=1:
        raise ValueError('natural crossing missing')
    # A late pass or counters cannot hide a missing/interleaved frame or commit.
    events=[]
    for line in log.splitlines():
        if line.startswith(('SCNN_TEMPORAL_WINDOW_START ', 'SCNN_TEMPORAL_COMMIT ', 'SCNN_TEMPORAL_LAYER ',
                            'SCNN_TEMPORAL_FRAME_STATUS ', 'SCNN_TAIL_VISIBILITY_PASS ')):
            events.append(line.split()[0])
    wanted_events=[]
    for i in range(52):
        if i%16==0: wanted_events.append('SCNN_TEMPORAL_WINDOW_START')
        # run_current_layer finishes before fg_check_frame waits for publication.
        wanted_events.append('SCNN_TEMPORAL_LAYER')
        if i in (15,31,47): wanted_events.append('SCNN_TEMPORAL_COMMIT')
        wanted_events.append('SCNN_TEMPORAL_FRAME_STATUS')
    wanted_events.append('SCNN_TAIL_VISIBILITY_PASS')
    if events!=wanted_events: raise ValueError('event ordering')
    return dict(status='PASS_final_top_natural_Conv8_tail_RTL_query_not_full_graph_or_abort',
                final=final[0],windows=windows,commits=commits,layers=layers,frames=frames,
                service=service[0],banks=banks,scope=SCOPE)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='candidate_r1')
    args=parser.parse_args()
    if not args.name.replace('_','').isalnum(): parser.error('simple candidate name required')
    source=(ROOT/ORIGINAL).read_text(encoding='utf-8')
    text=transform(source)
    output=ROOT/EXPERIMENT/args.name
    output.mkdir(parents=True,exist_ok=False)
    (output/'tb_scnn_tail_visibility.sv').write_text(text,encoding='utf-8')
    (output/'testbench_changes.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),text.splitlines(True),
        fromfile='original_b4t12_r2/tb',tofile='scoped_Conv8_tail/tb')),encoding='utf-8')
    report=dict(status='PREPARED_NOT_SIMULATED',scope=SCOPE,original_tb=str(ORIGINAL),
                fixture='fixture',control='control',tail_fixture='tail_reference',build_timeout=3600,run_timeout=21600)
    (output/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
