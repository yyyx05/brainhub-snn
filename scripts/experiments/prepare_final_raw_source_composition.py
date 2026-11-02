"""Expose the actual raw-count producer in both synthesis comparison boundaries."""
import argparse
import datetime as dt
import difflib
import json
import os
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
if os.name == 'nt' and not str(ROOT).startswith('\\\\?\\'):
    ROOT = Path('\\\\?\\' + str(ROOT))
RESULT = ROOT / 'results/paper_evidence/supplement_20261015/performance_optimization'
OUT = ROOT / 'verification/experiments/final_raw_source_composition_r1'
CORE = 'unified_h40_final_composition_top_v1'
TOP = 'unified_h40_raw_source_top_v1'
ARMS = {'baseline': ('srnn_source_core_r2', 0),
        'candidate': ('srnn_gradient_retire_core_r1', 1)}
GROUP = ('valid', 'recurrent', 'source_valid', 'weight_addr0', 'weight_addr1',
         'weight_addr2', 'weight_addr3', 'feedforward_end')
EXPECTED_CHANGES = sorted((
    'learning/unified_tp_parameter_rmw_service.sv',
    'learning/unified_srnn_tp_wout_gradient_reducer.sv',
    'learning/unified_srnn_m2_forward_learning_store_slice.sv',
    'memory/unified_h40_scalar_parameter_row_publisher_v1.sv',
))


def interface(text):
    """Read this generated one-declaration-per-line ANSI header, not general SV."""
    match = re.search(r'module\s+' + CORE + r'\s*#\((.*?)\)\s*\((.*?)\);', text, re.S)
    if not match:
        raise ValueError('expected the existing generated ANSI composition header')
    parameters, ports = [], []
    for line in match[1].splitlines():
        line = line.split('//', 1)[0].strip()
        if not line:
            continue
        item = re.fullmatch(r'parameter\s+(?:int|bit)\s+(\w+)\s*=\s*(.+?)(?:,)?', line)
        if not item:
            raise ValueError('unsupported parameter declaration: ' + line)
        parameters.append(item[1])
    for line in match[2].splitlines():
        line = line.split('//', 1)[0].strip()
        if not line:
            continue
        item = re.fullmatch(r'(input|output)\s+(.+?)\s+([io]_\w+)\s*,?', line)
        if not item:
            raise ValueError('unsupported port declaration: ' + line)
        ports.append(dict(direction=item[1], type=item[2], name=item[3]))
    if len(set(parameters)) != len(parameters) or len({p['name'] for p in ports}) != len(ports):
        raise ValueError('duplicate interface name')
    return match[1], match[2], parameters, ports


def wrapper(text, policy):
    if policy not in (0, 1):
        raise ValueError('static buffered/stream policy required')
    parameters_text, ports_text, parameters, ports = interface(text)
    override = {'i_fc_srnn_group_' + suffix: 'raw_group_' + suffix for suffix in GROUP}
    names = {p['name'] for p in ports}
    if not set(override).issubset(names):
        raise ValueError('missing expected manual group input')
    declarations = []
    for port in ports:
        if port['name'] in override:
            if port['direction'] != 'input':
                raise ValueError('only group inputs may be intercepted')
            declarations.append('    ' + port['type'] + ' ' + override[port['name']] + ';')
    bridge = dict(i_clk='i_clk', i_rst_n='i_rst_n', i_raw_source_active='i_raw_source_active',
        i_raw_count_valid='i_raw_count_valid', i_raw_counts='i_raw_counts', o_raw_count_ready='o_raw_count_ready',
        i_core_group_ready='o_fc_srnn_group_ready', i_core_layer_done='o_fc_srnn_layer_done')
    for suffix in GROUP:
        bridge['i_manual_group_' + suffix] = 'i_fc_srnn_group_' + suffix
        bridge['o_group_' + suffix] = 'raw_group_' + suffix
    instance = '\n'.join('        .%s(%s)%s' % (p['name'], override.get(p['name'], p['name']),
                         ',' if n + 1 < len(ports) else '') for n, p in enumerate(ports))
    return ('`timescale 1ns/1ps\n`default_nettype none\n\n'
        '// Generated wiring only. Original core ports and parameters remain available.\n'
        '// The host still owns descriptor sequencing, mode selection and manual suffixes.\n'
        f'module {TOP} #(\n' + parameters_text.strip() +
        f',\n    parameter int RAW_SOURCE_POLICY = {policy}\n) (\n' + ports_text.strip() +
        ',\n    input wire logic i_raw_source_active,\n    input wire logic i_raw_count_valid,\n'
        '    input wire logic [31:0] i_raw_counts,\n    output logic o_raw_count_ready\n);\n' +
        '\n'.join(declarations) + '\n\n'
        '    raw_source_binding #(.POLICY(RAW_SOURCE_POLICY), .ADDR_WIDTH(WEIGHT_ADDR_WIDTH)) u_raw_source (\n' +
        ',\n'.join('        .%s(%s)' % pair for pair in bridge.items()) + '\n    );\n\n' +
        '    ' + CORE + ' #(\n' + ',\n'.join('        .%s(%s)' % (p, p) for p in parameters) +
        '\n    ) u_core (\n' + instance + '\n    );\nendmodule\n`default_nettype wire\n')


def check_wrapper(source, generated, policy):
    _, _, parameters, ports = interface(source)
    matches = re.findall(r'\.(\w+)\((\w+)\)', generated.split(') u_core (', 1)[1])
    expected = [(p['name'], 'raw_group_' + p['name'][len('i_fc_srnn_group_'):]
                 if p['name'].startswith('i_fc_srnn_group_') and
                 p['name'][len('i_fc_srnn_group_'):] in GROUP else p['name']) for p in ports]
    if matches != expected:
        raise ValueError('core port forwarding mismatch')
    param_text = generated.split('    ' + CORE + ' #(\n', 1)[1].split(') u_core', 1)[0]
    if re.findall(r'\.(\w+)\((\w+)\)', param_text) != [(p, p) for p in parameters]:
        raise ValueError('core parameter forwarding mismatch')
    if f'parameter int RAW_SOURCE_POLICY = {policy}' not in generated:
        raise ValueError('policy mismatch')
    return dict(core_ports=len(ports), core_parameters=len(parameters),
                intercepted_group_inputs=len(GROUP), new_raw_ports=4,
                unchanged_port_connections=len(ports)-len(GROUP))


def prepare():
    if OUT.exists():
        raise FileExistsError('prepared composition already exists')
    binding = ROOT / 'verification/experiments/raw_source_binding_r1/raw_source_binding.sv'
    if not binding.is_file():
        raise FileNotFoundError(binding)
    sources = {arm: RESULT / name for arm, (name, _) in ARMS.items()}
    core_files = {arm: {p.relative_to(src/'rtl').as_posix(): p for p in (src/'rtl').rglob('*') if p.is_file()}
                  for arm, src in sources.items()}
    if len(core_files['baseline']) != 109 or set(core_files['baseline']) != set(core_files['candidate']):
        raise ValueError('expected matching 108SV+1MEM source sets')
    changes = sorted(rel for rel in core_files['baseline'] if
                     core_files['baseline'][rel].read_bytes() != core_files['candidate'][rel].read_bytes())
    if changes != EXPECTED_CHANGES:
        raise ValueError('unexpected architectural source changes: ' + repr(changes))
    top_text = {arm: (src/'rtl/top'/ (CORE+'.sv')).read_text() for arm, src in sources.items()}
    if top_text['baseline'] != top_text['candidate']:
        raise ValueError('comparison must retain the identical core interface')
    recipe_root = RESULT / 'scnn_cfg_payload_core_synth_r1/vivado'
    recipe = (recipe_root / 'run_unified_h40_final_composition_v1_synth.tcl').read_text()
    if recipe.count('set top ' + CORE) != 1:
        raise ValueError('synthesis top not unique')
    synthesis = recipe.replace('set top ' + CORE, 'set top ' + TOP, 1)
    OUT.mkdir(parents=True)
    checks = {}
    for arm, (name, policy) in ARMS.items():
        target = OUT / arm
        for rel, path in core_files[arm].items():
            dest = target / 'rtl' / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            if dest.read_bytes() != path.read_bytes():
                raise ValueError('source copy changed')
        for name in ('source_scan.sv', 'source_stream.sv'):
            src = sources[arm] / name
            if src.read_bytes() != (sources['baseline']/name).read_bytes():
                raise ValueError('producer differs between accepted runs')
            shutil.copy2(src, target/'rtl/top'/name)
        shutil.copy2(binding, target/'rtl/top'/binding.name)
        generated = wrapper(top_text[arm], policy)
        checks[arm] = check_wrapper(top_text[arm], generated, policy)
        (target/'rtl/top'/(TOP+'.sv')).write_text(generated, newline='\n')
        (target/'vivado').mkdir()
        (target/'vivado/run_unified_h40_final_composition_v1_synth.tcl').write_text(synthesis, newline='\n')
        shutil.copy2(recipe_root/'unified_h40_final_composition_v1_ooc.xdc', target/'vivado/unified_h40_final_composition_v1_ooc.xdc')
        (target/'server').mkdir()
        shutil.copy2(recipe_root.parent/'server/run_unified_h40_final_composition_v1_synth.sh',
                     target/'server/run_unified_h40_final_composition_v1_synth.sh')
    (OUT/'synthesis_recipe.diff').write_text(''.join(difflib.unified_diff(
        recipe.splitlines(True), synthesis.splitlines(True), fromfile='accepted_recipe', tofile='raw_source_boundary')), newline='\n')
    report = dict(status='PREPARED_NOT_ELABORATED_OR_SYNTHESIZED', time=dt.datetime.now().astimezone().isoformat(),
        arms=ARMS, core_files_per_arm=109, core_changes=changes, wrapper_checks=checks,
        boundary='Same public/manual ports plus four raw-count ports. Actual static producer and all mux/address/tail wiring are synthesized with the core. Host descriptor scheduling and manual recurrent suffixes remain external.',
        recipe_change='Only top module; identical part, 15ns clock, XPM, M2=512, learning enabled and synthesis directives/protection. Wrapper adds hierarchy; not inherited resource/STA.',
        not_claimed=['whole-core elaboration or simulation of this wrapper', 'physical timing', 'new cycle measurement', 'DDR/host-free autonomous service', 'production integration'])
    (OUT/'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    shutil.copy2(__file__, OUT/'prepare_runner.py')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    prepare()
