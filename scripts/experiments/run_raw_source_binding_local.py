"""Local source/mux boundary verification only; no core, remote, or performance run."""
import collections
import datetime as dt
import json
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
NATIVE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path('\\\\?\\' + str(NATIVE_ROOT)) if sys.platform == 'win32' else NATIVE_ROOT
EXPERIMENT = ROOT / 'verification/experiments/raw_source_binding_r1'
ORIGINAL = ROOT / 'verification/experiments/srnn_gradient_retire_core_r1'
RESULT = ROOT / 'results/paper_evidence/supplement_20261015/performance_optimization/raw_source_binding_local_r1'
SCOPE = ('Only raw-source binding and raw/manual switching, independent input oracle and '
         'cyclewise original TB expressions. No core instance, FC/SCNN computation, '
         'whole-core EDA, SSH, Vivado, physical claim, or performance comparison.')


def linux_path(path):
    value = str(path).removeprefix('\\\\?\\')
    return '/mnt/' + value[0].lower() + value[2:].replace('\\', '/')


def fixture(out):
    frames = [[0] * 700 for _ in range(6)]
    for index, value in {0:255, 1:254, 2:1, 4:1, 7:2, 127:3, 128:4,
                         695:1, 697:1, 698:254, 699:255}.items():
        frames[1][index] = value
    frames[2] = [(i * 17 + 3) % 5 for i in range(700)]
    frames[3][699] = 255
    frames[4] = [255] * 700
    rng = random.Random(20261019)
    for index in rng.sample(range(700), 60):
        frames[5][index] = rng.randrange(1, 9)
    frames[5][0], frames[5][3], frames[5][4], frames[5][695], frames[5][696], frames[5][699] = 1, 2, 3, 4, 5, 6
    names = ('all_zero', 'boundary_255_mixed', 'dense_small', 'last_lane_255',
             'all_255', 'sparse_asymmetric')
    with (out / 'input_words.hex').open('x', encoding='ascii') as handle:
        for values in frames:
            for block in range(175):
                word = sum(values[block*4+lane] << (lane*8) for lane in range(4))
                handle.write(f'{word:08x}\n')
    metadata = []
    expected = [[], []]
    for index, values in enumerate(frames):
        records = []
        reconstructed = [0] * 700
        # Mathematical expansion of 700 independent counts, not a producer FSM.
        for base in range(0, 700, 4):
            lanes = values[base:base+4]
            for repetition in range(max(lanes)):
                mask = sum(1 << lane for lane, count in enumerate(lanes) if repetition < count)
                records.append((mask << 10) | (base // 2))
                for lane in range(4):
                    reconstructed[base+lane] += bool(mask & (1 << lane))
        assert reconstructed == values
        for policy in range(2):
            # Original buffered all-zero early finish leaves row zero; the
            # stream has consumed all 175 blocks and leaves row 348.
            marker_row = 0 if policy == 0 and not any(values) else 348
            expected[policy].extend(records + [(1 << 14) | marker_row])
        metadata.append(dict(id=index, name=names[index], counts=700, words=175,
                             data_groups=len(records), outputs=len(records)+1,
                             lane_events=sum(values)))
    for policy in range(2):
        with (out / f'oracle_{policy}.hex').open('x', encoding='ascii') as handle:
            handle.write(''.join(f'{record:04x}\n' for record in expected[policy]))
    (out / 'lengths.hex').write_text(''.join(f"{row['outputs']:08x}\n" for row in metadata), encoding='ascii')
    return dict(frames=metadata, unique_counts=4200, seed=20261019,
                oracle='Independent Python block-major expansion, mask lane iff repeat<count; every accepted group checks recurrent/mask/four addresses/end in RTL.',
                frame_order=[0,1,2,3,4,5,0],
                completed_outputs_per_policy=sum(row['outputs'] for row in metadata)+metadata[0]['outputs'])


def diagnostics(log):
    categories = collections.Counter(re.findall(r'^%Warning-([^:]+):', log, re.M))
    return dict(categories=dict(categories), ordinary=sum(categories.values()),
                structural=[key for key in categories if key in ('LATCH','MULTIDRIVEN','UNOPTFLAT')],
                errors=len(re.findall(r'^%Error|^%Fatal|Assertion failed|\bFATAL\b',log,re.M)),
                warning_lines=len(re.findall(r'(?im)^.*\bwarning\b.*$',log)))


def parse(log, info):
    if diagnostics(log)['errors']:
        raise ValueError('runtime diagnostic failure')
    results = []
    for policy in range(2):
        frame_rows = [dict((key,int(value)) for key,value in re.findall(r'(\w+)=(\d+)',line))
                      for line in re.findall(r'^BIND_FRAME (.+)$',log,re.M)]
        frame_rows = [row for row in frame_rows if row['policy'] == policy]
        wanted = [dict(policy=policy,ordinal=ordinal,fixture=index,inputs=175,
                       outputs=info['frames'][index]['outputs'],markers=1)
                  for ordinal,index in enumerate(info['frame_order'],1)]
        if frame_rows != wanted:
            raise ValueError(f'policy {policy}: ordered complete frame records differ')
        resets = re.findall(r'^BIND_RESET policy='+str(policy)+r' kind=(\w+)$',log,re.M)
        if resets != ['partial_load','held_data','held_marker','drain']:
            raise ValueError(f'policy {policy}: reset coverage differs')
        passes = re.findall(r'^BIND_PASS policy='+str(policy)+r' (.+)$',log,re.M)
        if len(passes) != 1:
            raise ValueError(f'policy {policy}: missing/duplicated final record')
        row = dict((key,int(value)) for key,value in re.findall(r'(\w+)=(\d+)',passes[0]))
        expected_counts = dict(frames=7,resets=4,raw_inputs=1225+3+175+(175 if policy==0 else 1),
                               raw_outputs=info['completed_outputs_per_policy'],manual_outputs=60,
                               fc=32,recurrent=28,manual_end=16,inactive_inputs=7,drain_blocked=25)
        if any(row.get(key) != value for key,value in expected_counts.items()):
            raise ValueError(f'policy {policy}: acceptance counters differ: {row}')
        if any(row.get(key,0)<=0 for key in ('comparisons','raw_holds','manual_holds','input_holds','tail_holds','ignored_manual_cycles')):
            raise ValueError('missing binding/hold coverage')
        results.append(dict(policy=policy,**row))
    if log.count('RAW_SOURCE_BINDING_PASS policies=2 frames=14 resets=8') != 1:
        raise ValueError('missing unique completion')
    return results


def main():
    RESULT.mkdir(exist_ok=True)
    attempt = 1
    while (RESULT / f'attempt_{attempt:03d}').exists():
        attempt += 1
    out = RESULT / f'attempt_{attempt:03d}'
    out.mkdir()
    state = dict(start=dt.datetime.now().astimezone().isoformat(),status='preparing',scope=SCOPE,
                 runner_pattern='run_gradient_retire_local.py: stdlib only; WSL Linux redirection; j1; ulimit -v 3145728; GNU time; preserve launcher and raw logs',
                 attempt=attempt,original_inputs=str(ORIGINAL),output=str(out))
    def save():
        (out / 'status.json').write_text(json.dumps(state,indent=2)+'\n',encoding='ascii')
    def execute(stage, command, timeout=1800):
        state['status'] = stage+'_running'
        full = ['wsl','-e','sh','-lc','cd '+shlex.quote(linux_path(out))+
                ' && ulimit -v 3145728 && '+command+' > '+stage+'.log 2>&1']
        start = time.monotonic()
        with (out / (stage+'_launcher.log')).open('xb') as handle:
            process = subprocess.Popen(full,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            state[stage] = dict(pid=process.pid,command=full,exit=None)
            save()
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage]['timeout'] = True
                save()
                raise
        state[stage].update(exit=code,wall_seconds=time.monotonic()-start)
        log = (out / (stage+'.log')).read_text(errors='replace') if (out / (stage+'.log')).exists() else ''
        diag = diagnostics(log)
        state[stage]['diagnostics'] = diag
        save()
        if code or diag['structural'] or diag['errors']:
            raise RuntimeError(stage+' failed; original log retained')
        return log
    protected = [ORIGINAL / name for name in ('source_scan.sv','source_stream.sv')]
    originals = [(path,path.read_bytes(),path.stat().st_size,path.stat().st_mtime_ns) for path in protected]
    save()
    try:
        for path in EXPERIMENT.iterdir():
            if path.is_file():
                shutil.copy2(path,out / path.name)
        shutil.copy2(__file__,out / 'runner.py')
        for path in protected:
            shutil.copy2(path,out / path.name)
            if path.read_bytes() != (out / path.name).read_bytes():
                raise ValueError('producer archive differs')
        state['fixture'] = fixture(out)
        save()
        execute('environment','verilator --version',180)
        sources = ' '.join(shlex.quote(linux_path(path)) for path in protected)
        execute('build','/usr/bin/time -v -o build_resources.txt verilator --binary --timing --assert '
                '-j 1 -Wall -Wno-fatal -Werror-LATCH -Werror-MULTIDRIVEN -Werror-UNOPTFLAT '
                '--top-module tb_raw_source_binding --Mdir obj '+sources+
                ' raw_source_binding.sv tb_raw_source_binding.sv')
        log = execute('run','/usr/bin/time -v -o run_resources.txt obj/Vtb_raw_source_binding',180)
        state['coverage'] = parse(log,state['fixture'])
        for path,content,size,mtime in originals:
            if path.read_bytes()!=content or (path.stat().st_size,path.stat().st_mtime_ns)!=(size,mtime):
                raise ValueError('original producer changed')
        for path in EXPERIMENT.iterdir():
            if path.is_file() and path.read_bytes()!=(out / path.name).read_bytes():
                raise ValueError('tested source changed while running')
        state.update(status='PASS_raw_source_binding_local',producer_inputs_byte_equal=2,
                     original_size_mtime_preserved=2,core_instantiated=False,
                     instantiable='Yes: raw_source_binding.sv plus original source_scan.sv and source_stream.sv; POLICY 0/1, ADDR_WIDTH default 11. No physical or whole-core validation.')
    except Exception as error:
        state.update(status='FAIL',error=repr(error))
        raise
    finally:
        state['end'] = dt.datetime.now().astimezone().isoformat()
        save()
        print(json.dumps(state,indent=2))


if __name__ == '__main__':
    main()
