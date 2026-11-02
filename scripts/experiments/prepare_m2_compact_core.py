#!/usr/bin/env python3
"""Isolated full-core compact M2 storage with explicit end-to-end capacity."""
import difflib
import json
from pathlib import Path
import re
import shutil

from run_srnn_resident_remote import ROOT, RESULT, local_path
from run_m2_xpm_context import transform_wrapper

SOURCE = ROOT/RESULT/'scnn_issue_banked_synth_r1'
OUTPUT = ROOT/'verification/experiments/m2_compact_core/candidate_r1'
REPAIRED_OUTPUT = ROOT/'verification/experiments/m2_compact_core/candidate_r2'
CHAIN = {
 'top/unified_h40_final_composition_top_v1.sv': ['unified_scnn_h40_array16_selected_layers_train_infer_system_v1'],
 'top/unified_scnn_h40_array16_selected_layers_train_infer_system_v1.sv': ['unified_h40_array16_three_topology_top_v1','unified_srnn_m2_h40_forward_engine_v1'],
 'top/unified_h40_array16_three_topology_top_v1.sv': ['unified_binary_p4_h40_array12_path_v1'],
 'tiles/unified_binary_p4_h40_array12_path_v1.sv': ['unified_dual_role_h40_stationary_array12_candidate_v1'],
 'tiles/unified_dual_role_h40_stationary_array12_candidate_v1.sv': ['unified_dual_role_h40_stationary_region4_candidate_v1'],
 'tiles/unified_dual_role_h40_stationary_region4_candidate_v1.sv': ['unified_dual_role_h40_stationary_tile_candidate_v1'],
 'tiles/unified_dual_role_h40_stationary_tile_candidate_v1.sv': ['experimental_dual_context_weight_store_p4all_v1'],
 'learning/unified_srnn_m2_h40_forward_engine_v1.sv': ['unified_srnn_m2_h40_q14_cache_refill_scheduler_v1','unified_srnn_m2_h40_forward_observation_adapter_v1'],
 'learning/unified_srnn_m2_h40_forward_observation_adapter_v1.sv': [],
 'memory/unified_srnn_m2_h40_q14_cache_refill_scheduler_v1.sv': [],
}


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('expected one anchor: '+old)
    return text.replace(old,new)


def transform(path, text):
    if path == 'memory/experimental_dual_context_weight_store_p4all_v1.sv':
        return transform_wrapper(text).replace('experimental_dual_context_weight_store_compact_m2',
            'experimental_dual_context_weight_store_p4all_v1').replace('M2_DEPTH','M2_CACHE_DEPTH')
    if path not in CHAIN:
        return text
    module = Path(path).stem
    text = once(text,'module '+module+' #(\n','module '+module+' #(\n    parameter int M2_CACHE_DEPTH = 578,\n')
    for child in CHAIN[path]:
        # Match instantiations only, never the module declaration.
        pattern = r'(?m)^([ \t]*)'+re.escape(child)+r' #\(\n'
        text, count = re.subn(pattern,lambda m:m.group(0)+m.group(1)+'    .M2_CACHE_DEPTH(M2_CACHE_DEPTH),\n',text)
        if count != 1:
            raise ValueError((path,child,count))
    if path.endswith('q14_cache_refill_scheduler_v1.sv'):
        text = once(text,"cache_last_row_ext < 11'd578", "cache_last_row_ext < 11'(M2_CACHE_DEPTH)")
    if path.endswith('forward_observation_adapter_v1.sv'):
        if text.count('                WEIGHT_DEPTH &&') != 2:
            raise ValueError('expected two adapter cache bounds')
        text = text.replace('                WEIGHT_DEPTH &&','                M2_CACHE_DEPTH &&')
    if path.endswith('final_composition_top_v1.sv'):
        text = once(text,'endmodule', '''    initial begin
        if (M2_CACHE_DEPTH != 512 && M2_CACHE_DEPTH != 578)
            $error("M2 compact core supports explicit 512 or 578 row capacity");
        if ((TOPOLOGY_MASK & 2) != 0 &&
                (SRNN_M2_BACKBONE_MAX > M2_CACHE_DEPTH || SRNN_M2_HIDDEN_MAX > M2_CACHE_DEPTH))
            $error("M2 selected maximum working set exceeds cache capacity");
    end
endmodule''')
    return text


def prepare(output_path=REPAIRED_OUTPUT):
    output = Path(local_path(output_path))
    source = Path(local_path(SOURCE/'rtl'))
    output.mkdir(parents=True,exist_ok=True)
    changed, diffs, source_data = [], [], []
    for original in source.rglob('*'):
        if not original.is_file():
            continue
        relative = original.relative_to(source).as_posix()
        target = output/'rtl'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if original.suffix != '.sv':
            shutil.copy2(original,target)
            source_data.append(dict(path='rtl/'+relative,bytes=original.stat().st_size))
            continue
        before = original.read_text(encoding='utf-8')
        after = transform(relative,before)
        if before == after:
            shutil.copy2(original,target)
        else:
            target.write_text(after,encoding='utf-8')
            changed.append(relative)
            diffs.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                fromfile='accepted/'+relative,tofile='compact/'+relative))
    compact = Path(local_path(ROOT/'verification/experiments/m2_storage_shape/candidate_r1/m2_compact_store.sv'))
    shutil.copy2(compact,output/'rtl/memory/m2_compact_store.sv')
    (output/'compact_core.diff').write_text(''.join(diffs),encoding='utf-8')
    details = dict(source=str(SOURCE),changed=sorted(changed),added=['memory/m2_compact_store.sv'],
        source_data=source_data,
        rtl_count=len(list((output/'rtl').rglob('*.sv'))),default_capacity=578,selected_capacity=512,
        scope='Context1 only; Context0 and width/latency unchanged. Capacity propagated through top/array/tile/store and M2 refill/adapter. No model-key overlay, no production adoption or inherited STA.')
    (output/'preparation.json').write_text(json.dumps(details,indent=2)+'\n')
    print(json.dumps(details,indent=2))
    return output


if __name__ == '__main__': prepare()
