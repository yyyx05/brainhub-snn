#!/usr/bin/env python3
"""Prepare a bounded shared-learner/publisher natural-update integration test.

The accepted B4T12 trace/logit oracle becomes explicit component stimulus here.
It does not represent a new convolution, capture, final-top, or RTL query run.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path('verification/experiments/scnn_natural_component')
SERVICE = Path('rtl/learning/unified_tp_shared_output_learning_service_v1.sv')
FIXTURE = ('results/paper_evidence/supplement_20261004/rtl/scnn_same_dut/'
           'natural_visibility_prepare/seed38472_b4t12_q22_t4_finaltop_v2')


def testbench_source(source, body):
    parameters = source[source.index('    parameter bit'):source.index('\n) (')]
    parameters = re.sub(r'\bparameter\b', 'localparam', parameters)
    parameters = re.sub(r',\s*\n', ';\n', parameters).rstrip() + ';\n'
    port_text = source[source.index('\n) (')+5:source.index('\n);')]
    declarations = []
    ports = []
    for direction, datatype, name in re.findall(
            r'\b(input|output)\s+(logic(?:\s+signed)?(?:\s+\[[^\]]+\])?)\s+(\w+)', port_text):
        initialize = direction == 'input' and name not in ('i_read_ready', 'i_result_ready')
        declarations.append('    '+datatype+' '+name+(' = \'0' if initialize else '')+';')
        ports.append('.'+name+'('+name+')')
    if not ports or len(ports) != len(re.findall(r'\b(?:input|output)\s+logic', port_text)):
        raise ValueError('incomplete shared learner public interface')
    return ('`timescale 1ns/1ps\n`default_nettype none\nmodule tb_scnn_natural_component;\n'
            +parameters+'\n'+'\n'.join(declarations)+'\n'
            +'    unified_tp_shared_output_learning_service_v1 dut (\n        '
            +',\n        '.join(ports)+');\n'+body+'\nendmodule\n`default_nettype wire\n')


def parse_log(log):
    if re.search(r'%Error|%Fatal|Assertion failed', log):
        raise ValueError('runtime error')
    def rows(marker):
        return [{k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)', line)}
                for line in log.splitlines() if line.startswith(marker+' ')]
    windows=rows('NATURAL_COMPONENT_WINDOW')
    if [(r.get('window'),r.get('code'),r.get('residual'),r.get('rmw')) for r in windows] != [
            (1,0,20041,20041),(2,0,20011,20011),(3,662,20221,20221)]:
        raise ValueError('incomplete natural windows')
    pubs=rows('NATURAL_COMPONENT_PUBLICATION')
    if [(r.get('publication'),r.get('words'),r.get('rows')) for r in pubs] != [
            (i,22528,1892) for i in range(4)]:
        raise ValueError('incomplete publications')
    finals=rows('NATURAL_COMPONENT_PASS')
    expected=dict(windows=3,trace_inputs=98304,logit_inputs=528,rmw=60273,
                  code_writes=662,residual_writes=60273,publications=4,
                  published_words=90112,published_rows=7568,
                  query_current_checks=44,query_logit_checks=44,
                  query_current_changes=4,query_logit_changes=4,reset=1)
    if finals != [expected]:
        raise ValueError('missing/duplicate/incomplete component completion')
    return dict(status='PASS_natural_shared_learner_publication_component_not_final_top_or_RTL_query',
                windows=windows,publications=pubs,final=finals[0])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='candidate_r1')
    args=parser.parse_args()
    if not args.name.replace('_','').isalnum(): parser.error('simple name required')
    output=ROOT/EXPERIMENT/args.name
    tb=testbench_source((ROOT/SERVICE).read_text(),(ROOT/EXPERIMENT/'body.svh').read_text())
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()): raise FileExistsError(output)
    (output/'tb_scnn_natural_component.sv').write_text(tb,encoding='ascii')
    config=dict(status='PREPARED_NOT_SIMULATED',fixture=FIXTURE,
        scope='B4T12 original normalized trace/logit stimulus to actual shared learner; cold zero residual, three natural updates and publisher rows. Query dot/centering is a TB consumer of actual published rows, not RTL classifier or raw graph. No capture, final-top arbitration, timing or throughput claim.')
    (output/'reference.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(config,indent=2))


if __name__=='__main__': main()
