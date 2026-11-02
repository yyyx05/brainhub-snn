"""Isolate cfg-payload selection only; generate a real-core cycle miter."""
import argparse
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path('verification/experiments/scnn_control_payload')
SOURCE = Path('verification/experiments/scnn_forward_turnover/frame_r2/candidate')
SERVICE = Path('rtl/learning/unified_tp_shared_output_learning_service_v1.sv')
MODULE = SERVICE.stem


def candidate(source):
    if source.count('= select_scnn_cfg ?') != 13:
        raise ValueError('unexpected payload mux count')
    return source.replace('    logic select_scnn_cfg;',
        '    logic select_scnn_cfg;\n    logic select_scnn_payload;').replace(
        '        selected_cfg_profile_id =',
        '        // Payload is consumed only with core_cfg_valid; keep ready/owner off this data path.\n'
        '        select_scnn_payload = ENABLE_SCNN_CLIENT &&\n'
        '            !(ENABLE_FC_CLIENT && i_fc_cfg_valid);\n\n'
        '        selected_cfg_profile_id =').replace('= select_scnn_cfg ?', '= select_scnn_payload ?')


def testbench(source, body):
    params = source[source.index('    parameter bit'):source.index('\n) (')]
    params = re.sub(r'\bparameter\b', 'localparam', params)
    params = re.sub(r',\s*\n', ';\n', params).rstrip() + ';\n'
    params = params.replace("ENABLE_FC_CLIENT = 1'b1", 'ENABLE_FC_CLIENT = (MODE & 1) != 0')
    params = params.replace("ENABLE_SCNN_CLIENT = 1'b1", 'ENABLE_SCNN_CLIENT = (MODE & 2) != 0')
    for name, old, new in [('MAX_BATCH',16,2),('MAX_SOURCES',2048,4),('MAX_OUTPUTS',16,2),('MAX_GRADIENT_ELEMENTS',32768,32)]:
        params = params.replace(f'{name} = {old}', f'{name} = {new}')
    ports = re.findall(r'\b(input|output)\s+(logic(?:\s+signed)?(?:\s+\[[^\]]+\])?)\s+(\w+)',
                      source[source.index('\n) (')+5:source.index('\n);')])
    declarations, connections, checks = [], [[],[]], []
    declared = set()
    driven = {'i_read_ready','i_result_ready','i_read_response_valid','i_read_response_code',
              'i_read_response_residual','i_read_response_error'}
    for direction, datatype, name in ports:
        match = re.match(r'([io])_(fc|scnn)_(.*)',name)
        short = f'{match[1]}_{match[3]}' if match else name
        index = ('[0]' if match[2]=='fc' else '[1]') if match else ''
        if short not in declared:
            for arm in range(2 if direction=='output' else 1):
                signal = ('candidate_' if arm else '')+short
                init = " = '{default:'0}" if match else " = '0"
                if direction=='output' or name in driven: init=''
                declarations.append(f'    {datatype} {signal}'+('[2]' if match else '')+init+';')
            declared.add(short)
        for arm in range(2):
            connections[arm].append(f'.{name}('+('candidate_' if arm and direction=='output' else '')+short+index+')')
        if direction=='output':
            checks.append(f'        if ({short}{index} !== candidate_{short}{index}) $fatal(1,"miter {name} mode=%0d cycle=%0d",MODE,cycles);')
    args = ','.join('.'+name+'('+name+')' for name in re.findall(r'localparam (?:bit|int) (\w+)',params))
    inst = []
    for arm in range(2):
        module = ('reference_' if arm==0 else '')+MODULE
        inst.append(f'    {module} #({args}) '+('reference' if arm==0 else 'dut')+' (\n        '+',\n        '.join(connections[arm])+');')
    compare = '\n'.join(checks)
    core='reference.g_output_learning_core.u_shared_output_learning_path'
    diagnostic=('    if(MODE!=0) begin : diagnostic\n'
        '        always @(negedge i_clk) if(cycles==39990)\n'
        '            $display("CFG_PAYLOAD_DIAGNOSTIC mode=%0d core=%0d scheduler=%0d contribution=%0d/%0d gradient=%0d/%0d rmw=%0d reads=%0d read=%0d/%0d pending=%0d response=%0d/%0d result=%0d/%0d",\n'
        f'                MODE,{core}.state_q,{core}.u_frontend.contribution_scheduler_i.state_q,\n'
        f'                {core}.frontend_contribution_valid,{core}.frontend_contribution_ready,\n'
        f'                {core}.u_learning_slice.gradient_sign_valid,{core}.u_learning_slice.gradient_sign_ready,\n'
        f'                {core}.u_learning_slice.update_path_i.rmw_service.state_q,\n'
        '                reads,o_read_valid,i_read_ready,pending,i_read_response_valid,o_read_response_ready,o_result_valid,i_result_ready);\n'
        '    end\n')
    return ('`timescale 1ns/1ps\n`default_nettype none\n'
        'module cfg_payload_checker #(parameter int MODE=3)(output logic done);\n'+params+
        '\n'.join(declarations+inst)+'\n'+body+diagnostic+
        '\n    task automatic compare_outputs;\n'+compare+'\n    endtask\n'
        '    always @(posedge i_clk) if(i_rst_n && !done) begin\n'
        '        compare_outputs(); #1; compare_outputs();\n    end\nendmodule\n'
        'module tb_scnn_control_payload;\n    wire [3:0] done;\n'
        '    for(genvar m=0;m<4;m++) begin : modes\n'
        '        cfg_payload_checker #(.MODE(m)) checker_i(.done(done[m]));\n    end\n'
        '    initial begin wait(&done); $display("CFG_PAYLOAD_PASS modes=4 real_core=1 cycle_miter=1"); $finish; end\n'
        'endmodule\n`default_nettype wire\n')


def prepare(name):
    from run_srnn_resident_remote import local_path
    dest = Path(local_path(ROOT/EXPERIMENT/name))
    if dest.exists(): raise FileExistsError(dest)
    src = Path(local_path(ROOT/SOURCE))
    source = (src/SERVICE).read_text()
    shutil.copytree(src/'rtl',dest/'rtl')
    (dest/SERVICE).write_text(candidate(source),encoding='ascii')
    (dest/('reference_'+MODULE+'.sv')).write_text(source.replace('module '+MODULE,'module reference_'+MODULE,1),encoding='ascii')
    (dest/'tb_scnn_control_payload.sv').write_text(testbench(source,(ROOT/EXPERIMENT/'body.svh').read_text()),encoding='ascii')
    shutil.copy2(ROOT/EXPERIMENT/'body.svh',dest/'body.svh')
    sources = [p.relative_to(dest).as_posix() for p in sorted((dest/'rtl').rglob('*.sv'))]
    sources += ['reference_'+MODULE+'.sv','tb_scnn_control_payload.sv']
    (dest/'sources.f').write_text('\n'.join(sources)+'\n',encoding='ascii')
    metadata = dict(status='PREPARED_NOT_SIMULATED',source=SOURCE.as_posix(),changed_rtl=[SERVICE.as_posix()],
        dimensions=dict(batch=2,sources=4,outputs=2,gradient_elements=32),
        boundary='Real shared service, frontend, softmax, address-major gradient reducer and RMW cores; four compile-time enable combinations. Thirteen payload muxes only, unchanged transaction cycles. Not whole-core timing, throughput, full-profile regression or production adoption.')
    (dest/'preparation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='candidate_r1')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]+',args.name): parser.error('simple name required')
    prepare(args.name)
