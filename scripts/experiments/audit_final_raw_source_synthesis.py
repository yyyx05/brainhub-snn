"""Accept one naturally terminated paired synthesis archive without rerunning EDA."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from run_srnn_resident_remote import ROOT, RESULT, local_path

NAME = 'final_raw_source_synth_r1'
ARCHIVE_METADATA = {
    'binding_acceptance.json', 'launch_observation.json', 'preparation.json',
    'prepare_runner.py', 'README.md', 'scratch_archive.json', 'status.json',
    'synthesis_recipe.diff', 'worker.py',
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(root):
    """Return the fetched non-cache files without touching the remote host."""
    skip = {'obj', 'elab_obj', '__pycache__', '.Xil'}
    return {
        p.relative_to(root).as_posix(): p.stat().st_size
        for p in root.rglob('*')
        if p.is_file() and not set(p.relative_to(root).parts) & skip
    }


def main():
    root = Path(local_path(ROOT))
    archive = root/RESULT/NAME
    prepared = root/'verification/experiments/final_raw_source_composition_r1'
    report_path = archive/'terminal_audit.json'
    if report_path.exists():
        raise FileExistsError('already accepted or archived; do not repeat')
    state = json.loads((archive/'status.json').read_text())
    if not state.get('end') or state['status'] == 'TIMEOUT_PROCESS_RETAINED':
        raise ValueError('natural terminal required; retain a timed-out process')
    # The remote terminal was already naturally reached and the full tree was
    # fetched. DNS is unavailable in this local session, so the audit uses the
    # archived worker inventory plus local file-size/byte checks rather than
    # pretending to perform a fresh remote observation.
    archived = json.loads((archive/'scratch_archive.json').read_text())
    if not isinstance(archived, list):
        raise ValueError('scratch_archive.json must contain the worker file list')
    files = {entry['relative']: entry['bytes'] for entry in archived}
    local_files = inventory(archive)
    missing = [rel for rel, size in files.items()
               if local_files.get(rel) != size]
    unexpected = [rel for rel in local_files
                  if rel not in files and rel not in ARCHIVE_METADATA]
    if missing or unexpected:
        raise ValueError('fetched archive inventory mismatch: ' + repr(dict(missing=missing, unexpected=unexpected)))
    if list(archive.rglob('*.part')) or any(p.endswith('.part') for p in local_files):
        raise ValueError('partial archive')
    for rel, size in files.items():
        path = archive/rel
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError('raw archive size or file type: '+rel)
    observation = dict(
        evidence_source='local fetched archive after natural terminal; no remote re-launch or re-audit',
        time=state['end'], files=files, scratch_files=len(files),
        scratch_missing=[], scratch_different=[], active_eda=[],
        data_free=state.get('data_free_after'), tmp_free=state.get('tmp_free_after'),
    )
    inputs = [p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for p in inputs:
        if p.read_bytes() != (archive/p.relative_to(prepared)).read_bytes():
            raise ValueError('prepared input differs: '+str(p))
    worker = load('actual_paired_synthesis_worker', archive/'worker.py')
    generator = load('actual_composition_preparer', archive/'prepare_runner.py')
    prep = json.loads((archive/'preparation.json').read_text())
    checks, classifications, compat_checks = {}, {}, {}
    for arm, (prior, policy) in prep['arms'].items():
        original = root/RESULT/prior/'rtl'
        source_files = [p for p in original.rglob('*') if p.is_file()]
        if len(source_files) != 109:
            raise ValueError('core boundary is not 108SV+1MEM')
        for p in source_files:
            if p.read_bytes() != (archive/arm/'rtl'/p.relative_to(original)).read_bytes():
                raise ValueError('accepted core differs: '+arm+'/'+str(p.relative_to(original)))
        for producer in ('source_scan.sv', 'source_stream.sv'):
            if (original.parent/producer).read_bytes() != (archive/arm/'rtl/top'/producer).read_bytes():
                raise ValueError('producer differs')
        tested = root/RESULT/'raw_source_binding_local_r1/attempt_001/raw_source_binding.sv'
        if tested.read_bytes() != (archive/arm/'rtl/top/raw_source_binding.sv').read_bytes():
            raise ValueError('tested binding differs')
        source = (original/'top'/(generator.CORE+'.sv')).read_text()
        wrapper = (archive/arm/'rtl/top'/(generator.TOP+'.sv')).read_text()
        checks[arm] = generator.check_wrapper(source, wrapper, policy)
        stage = state['arms'].get(arm)
        if stage is None or stage.get('exit') is None:
            continue
        try:
            parsed = worker.classify(archive/arm, stage['exit'])
        except Exception as error:
            parsed = dict(status='FAIL_terminal_classification', exit=stage['exit'], classification_error=repr(error))
        if any(stage.get(k) != v for k,v in parsed.items()):
            raise ValueError('actual archived worker reparse differs: '+arm)
        classifications[arm] = parsed
        if parsed['status'].startswith('PASS'):
            sv = sorted((archive/arm/'rtl').rglob('*.sv'))
            compatibility = archive/arm/worker.REPORT/'vivado_2022_2_sources'
            if len(sv) != 112 or len(list(compatibility.glob('*.sv'))) != 112:
                raise ValueError('actual Vivado read source count')
            for index,p in enumerate(sv):
                actual = (compatibility/f'{index:03d}_{p.name}').read_text()
                expected = p.read_text().replace('`default_nettype none', '`default_nettype wire')
                if actual != expected:
                    raise ValueError('original Tcl compatibility copy differs: '+p.name)
            compat_checks[arm] = 112
    passed = state['status'] == 'PASS_paired_raw_source_synthesis_not_routed_STA'
    if passed and (set(classifications) != {'baseline','candidate'} or
                   any(not c['status'].startswith('PASS') for c in classifications.values())):
        raise ValueError('incomplete paired terminal')
    report = dict(status='PASS_paired_synthesis_archive' if passed else 'ARCHIVED_NOT_ACCEPTED_paired_synthesis',
        original_status=state['status'], start=state['start'], end=state['end'],
        raw_files=len(local_files), raw_scratch_files=len(files), raw_bytes=sum(local_files.values()),
        raw_scratch_bytes=sum(files.values()), all_sizes_match=True, no_part=True,
        prepared_inputs_byte_equal=len(inputs), prepared_bytes=sum(p.stat().st_size for p in inputs),
        complete_accepted_core_files_per_arm=109, wrapper_checks=checks,
        vivado_compatibility_copies_match=compat_checks, actual_worker_reparse_matches=True,
        arms=classifications, remote_observation=observation,
        scope='Paired synthesis only. No routing, actual clock-frequency proof, power, board or new whole-core cycle measurement. Original failure/status/logs unchanged.')
    shutil.copy2(__file__, archive/'terminal_audit_runner.py')
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
