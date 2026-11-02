"""Source-grounded common-profile check, not an author RTL reproduction."""
import ast
import datetime as dt
import json
import math
from pathlib import Path
import sys

import numpy as np

from run_srnn_resident_remote import ROOT, RESULT, local_path


def main():
    root = Path(local_path(ROOT))
    source = root/'verification/experiments/nearest_neighbor/spiker_author_source'
    generator = source/'spiker/spikerplus/vhdl_generator.py'
    tree = ast.parse(generator.read_text())
    functions = [node for cls in tree.body if isinstance(cls,ast.ClassDef)
                 for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='pow2_shift']
    if len(functions)!=1:
        raise ValueError('author shift function ambiguous')
    # Execute only the inspected, dependency-free scalar conversion method.
    namespace = {'log2':math.log2}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(generator),'exec'),namespace)
    sys.path.insert(0,str(root/'scripts'))
    import supplement_srnn_backbone_numeric as bone
    beta = bone.LEAK / 32768
    shift = namespace['pow2_shift'](None,1-beta)
    probes = np.array([-1024,-32,-1,0,1,32,1024],dtype=np.int64)
    actual_pre = bone.lif_step(np.zeros_like(probes),probes)[0]
    shifted = probes-(probes >> shift)
    nearest_shift = min(range(1,32),key=lambda s:abs(beta-(1-2**(-s))))
    prepared = root/'verification/experiments/srnn_representative_inputs/a_r1'
    selection = json.loads((prepared/'selection.json').read_text())
    inputs=[]
    for i,row in enumerate(selection['selected']):
        counts=np.fromiter((int(x,16) for x in (prepared/f'multi_fixture/sequence_{i}/raw700_counts.mem').read_text().split()),dtype=np.int64)
        inputs.append(dict(dataset_index=row['dataset_index'],native_steps=row['time_steps'],
                           maximum_count=int(counts.max()),bins_above_one=int(np.count_nonzero(counts>1)),
                           total_count=int(counts.sum()),occupied_bins=int(np.count_nonzero(counts))))
    result=dict(status='SOURCE_COMMON_PROFILE_INCOMPATIBLE_WITHOUT_NUMERIC_OR_INPUT_ADAPTATION',
                time=dt.datetime.now().astimezone().isoformat(),
                repository='https://github.com/smilies-polito/Spiker',
                inspected_git_revision='de32981b4f75ba2a1b30c379780dc823217181d8',
                branch='public',source_files=[dict(path=p.relative_to(source).as_posix(),bytes=p.stat().st_size)
                    for p in sorted(source.rglob('*')) if p.is_file()],
                checkout='Git clone failed with connection reset; eleven named files retrieved from the GitHub raw endpoint at the API-reported revision. No install or generator/RTL execution.',
                author_function_line=functions[0].lineno,backbone_q15_leak=bone.LEAK,backbone_beta=beta,
                author_scalar_shift_for_unchanged_beta=shift,implied_beta=1-2**(-shift),
                nearest_power_of_two_complement_shift=nearest_shift,
                nearest_implied_beta=1-2**(-nearest_shift),
                neither_exact=True,
                scalar_probes=[dict(membrane=int(v),original_pre=int(a),shift_model_pre=int(b))
                    for v,a,b in zip(probes,actual_pre,shifted)],
                selected_official_input_counts=inputs,
                scope='Actual author scalar conversion and our existing integer LIF reference executed. Shift-model arithmetic follows the inspected datapath; it is not a generated-author-RTL simulation or timing/resource measurement.',
                decision='Do not silently replace the production leak or binarize official counts. For E10 isolate source scheduling before the LIF update, preserve count multiplicity and accumulation precision, and compare equal memory/ports/issue budget. Label any reimplementation as literature-mechanism control, never author performance reproduction.')
    if shift!=2 or np.array_equal(actual_pre,shifted) or not all(r['bins_above_one']>0 for r in inputs):
        raise ValueError('source-level counterexample changed; inspect before reporting')
    out=root/RESULT/'nearest_neighbor_spiker_r1'
    out.mkdir(parents=True,exist_ok=True)
    (out/'common_profile_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
