#!/usr/bin/env python3
"""Launch/fetch isolated SRNN residency experiments over existing key-only SSH."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import time
import difflib

import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/autodl-tmp/SNN_PERFORMANCE_OPT_20261015_R1"
VERILATOR = "/root/autodl-tmp/verilator_5_048_install/bin/verilator"
EXPERIMENT = Path("verification/experiments/srnn_resident_weight_context")
RESULT = Path("results/paper_evidence/supplement_20261015/performance_optimization")


def connect():
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.load_host_keys(str(Path.home() / ".ssh/known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect("connect.westc.seetacloud.com", port=46245, username="root",
                   key_filename=str(Path.home() / ".ssh/id_ed25519_autodl_codex_20260802"),
                   look_for_keys=False, allow_agent=False, timeout=15)
    return client


def run(client, command):
    _, stdout, stderr = client.exec_command(command)
    text = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"remote exit {code}: {command}\n{text}\n{error}")
    return text + error


def upload(sftp, local, remote):
    parent = PurePosixPath(remote).parent
    missing = []
    while True:
        try:
            sftp.stat(str(parent))
            break
        except FileNotFoundError:
            missing.append(parent)
            parent = parent.parent
    for directory in reversed(missing):
        sftp.mkdir(str(directory))
    sftp.put(str(local), remote)


def fetch_tree(sftp, remote, local):
    Path(local_path(local)).mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(remote):
        if entry.filename in ("obj", "elab_obj", "__pycache__", ".Xil"):
            continue
        source = remote + "/" + entry.filename
        target = local / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            fetch_tree(sftp, source, target)
        else:
            partial = target.with_name(target.name + ".part")
            sftp.get(source, local_path(partial))
            os.replace(local_path(partial), local_path(target))


def local_path(path):
    resolved = str(path.resolve())
    return "\\\\?\\" + resolved if os.name == "nt" and not resolved.startswith("\\\\?\\") else resolved


def unit(client, name, candidate):
    destination = REMOTE + "/" + name
    sftp = client.open_sftp()
    try:
        try:
            sftp.stat(destination)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(destination)
        for relative in (
            Path("rtl/memory/unified_dual_role_weight_store_p4all_candidate_v1.sv"),
            EXPERIMENT / "rtl/experimental_dual_context_weight_store_p4all_v1.sv",
            EXPERIMENT / "tb/tb_experimental_dual_context_weight_store_p4all_v1.sv",
            Path("scripts/experiments/srnn_resident_server_worker.py"),
        ):
            specialized = ROOT / EXPERIMENT / candidate / relative
            local = Path(local_path(specialized)) if Path(local_path(specialized)).is_file() else ROOT / relative
            upload(sftp, local, destination + "/" + relative.as_posix())
        command = "python3 scripts/experiments/srnn_resident_server_worker.py unit"
        shell = f"cd {shlex.quote(destination)} && exec {command}"
        run(client, "screen -dmS snn_resident_unit " + "bash -lc " + shlex.quote(shell))
        time.sleep(1)
        print(json.dumps({"remote": destination, "screen": "snn_resident_unit", "launched": True}))
    finally:
        sftp.close()


def profile(client, name, candidate):
    destination = REMOTE + "/" + name
    accepted = "/root/autodl-tmp/SNN_CURRENT_PROFILE_MEASUREMENTS_20261009_R1/srnn"
    measured = "results/paper_evidence/supplement_20261004/performance/current_source_measurements/srnn/server_pair_r1"
    from build_srnn_resident_candidate import FILES
    with client.open_sftp() as sftp:
        try:
            sftp.stat(destination)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(destination)
        paths = [line.strip().strip('"') for line in (ROOT / "rtl/final_sources.f").read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.startswith("#")]
        mismatched = []
        for relative in paths:
            with sftp.open(accepted + "/" + relative, "rb") as stream:
                if stream.read() != (ROOT / relative).read_bytes():
                    mismatched.append(relative)
        if mismatched:
            raise ValueError("accepted remote/local production source differences: " + str(mismatched))
        # Copy declared dependencies only, excluding the original compiled objects.
        bootstrap = "import json,pathlib,shutil; a=pathlib.Path(" + repr(accepted) + ");d=pathlib.Path(" + repr(destination) + ");m=a/" + repr(measured) + ";deps=json.loads((m/'dependencies.json').read_text());d.mkdir(parents=True,exist_ok=False);\nfor rel in deps:\n p=pathlib.PurePosixPath(rel)\n if p.is_absolute() or '..' in p.parts: raise ValueError(rel)\n q=d/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a/rel,q)\nfor n in ('commands.json','sources.f','srnn_profile_measurement_generated_body.svh','tb_srnn_profile_measurement.sv'):\n shutil.copy2(m/n,d/('input_'+n if n=='commands.json' else n))\nprint(len(deps))"
        print("copied dependencies:", run(client, "python3 -c " + shlex.quote(bootstrap)).strip())
        differences = []
        overlay = ROOT / EXPERIMENT / candidate
        for relative in FILES:
            changed_path = overlay / relative
            changed = Path(local_path(changed_path)).read_text(encoding="utf-8")
            with sftp.open(accepted + "/" + relative) as stream:
                original = stream.read().decode("utf-8")
            differences.extend(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                fromfile="accepted/"+relative, tofile="candidate/"+relative))
            upload(sftp, local_path(changed_path), destination + "/" + relative)
        for relative in (EXPERIMENT / "rtl/experimental_dual_context_weight_store_p4all_v1.sv",
                         Path("scripts/experiments/srnn_resident_server_worker.py")):
            specialized = overlay / relative
            local = Path(local_path(specialized)) if Path(local_path(specialized)).is_file() else ROOT / relative
            upload(sftp, local, destination + "/" + relative.as_posix())
        with sftp.open(destination + "/source_changes.diff", "w") as stream:
            stream.write("".join(differences))
        with sftp.open(destination + "/source_comparison.json", "w") as stream:
            stream.write(json.dumps(dict(accepted_sources_compared=len(paths), byte_differences_before_overlay=mismatched,
                changed_sources=list(FILES), original_root=accepted, candidate_name=candidate), indent=2)+"\n")
    shell = "cd " + shlex.quote(destination) + " && exec python3 scripts/experiments/srnn_resident_server_worker.py profile"
    screen = "snn_resident_" + name
    run(client, "screen -dmS " + shlex.quote(screen) + " bash -lc " + shlex.quote(shell))
    print(json.dumps(dict(remote=destination, screen=screen, launched=True)))


def synthesis_inputs(functional):
    accepted = functional.get("status")
    if accepted not in ("PASS_candidate_fixed_sample_pair_not_physical_QoR",
                        "PASS_SCNN_real_sample0_B1T4_pair_candidate_compatibility"):
        raise ValueError("candidate functional pair has not passed")
    if any(functional.get(stage, {}).get("exit") != 0 for stage in ("build", "run_off", "run_on")):
        raise ValueError("candidate lacks successful actual build/off/on exits")
    if accepted == "PASS_SCNN_real_sample0_B1T4_pair_candidate_compatibility":
        return ("rtl/memory/experimental_dual_context_weight_store_p4all_v1.sv",
                ["scnn_optimization_overlay.json", "fc_scanner_changes.diff", "scnn_publisher_changes.diff"])
    return ("verification/experiments/srnn_resident_weight_context/rtl/experimental_dual_context_weight_store_p4all_v1.sv",
            ["source_changes.diff"])


def synthesis_scanner_inputs(functional):
    if functional.get("status") != "PASS_FC_B16T99_pair_scanner_pipeline_not_physical_QoR":
        raise ValueError("scanner overlay lacks accepted full FC pair")
    if any(functional.get(stage, {}).get("exit") != 0 for stage in ("build", "run_off", "run_on")):
        raise ValueError("scanner overlay lacks actual successful build/off/on exits")
    return "rtl/learning/unified_fc_tp_w2_contribution_scheduler.sv"


def synthesis(client, name, candidate_run, scanner_run=None):
    source = REMOTE + "/" + candidate_run
    target = REMOTE + "/" + name
    reference = "/root/autodl-tmp/UNIFIED_SNN_PAPER_RUN136_SCNN_FRAME_SERVICE_SYNTH_S1"
    with client.open_sftp() as sftp:
        with sftp.open(source + "/status.json") as stream:
            functional = json.load(stream)
        wrapper, evidence = synthesis_inputs(functional)
        if scanner_run:
            with sftp.open(REMOTE + "/" + scanner_run + "/status.json") as stream:
                synthesis_scanner_inputs(json.load(stream))
        for relative in [wrapper, "source_comparison.json"] + evidence:
            sftp.stat(source + "/" + relative)
        try:
            sftp.stat(target)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(target)
    active = run(client, "ps -eo comm,args").splitlines()
    if any(line.lstrip().startswith("vivado ") or "/lnx64.o/vivado" in line for line in active):
        raise RuntimeError("another Vivado is active")
    bootstrap = f'''import json,pathlib,shutil
s=pathlib.Path({source!r});t=pathlib.Path({target!r});r=pathlib.Path({reference!r})
scanner_run={scanner_run!r}
if scanner_run:
 b=pathlib.Path({REMOTE!r})/scanner_run
 scheduler='rtl/learning/unified_fc_tp_w2_contribution_scheduler.sv'
 publisher='rtl/learning/unified_scnn_classifier_cache_refill_publisher_v1.sv'
 source_files={{p.relative_to(s).as_posix() for p in (s/'rtl').rglob('*.sv')}}
 banked_files={{p.relative_to(b).as_posix() for p in (b/'rtl').rglob('*.sv')}}
 if source_files != banked_files: raise ValueError('scanner overlay RTL set mismatch')
 differences=[p for p in sorted(source_files) if (s/p).read_bytes() != (b/p).read_bytes()]
 if set(differences)-{{scheduler,publisher}}: raise ValueError('unexpected scanner source changes: '+str(differences))
t.mkdir(parents=True,exist_ok=False)
shutil.copytree(s/'rtl',t/'rtl')
shutil.copy2(s/{wrapper!r},t/'rtl/memory/experimental_dual_context_weight_store_p4all_v1.sv')
shutil.copytree(r/'vivado',t/'vivado')
shutil.copytree(r/'server',t/'server')
for rel in {evidence!r}: shutil.copy2(s/rel,t/rel)
shutil.copy2(s/'source_comparison.json',t/'source_comparison.json')
shutil.copy2(s/'status.json',t/'accepted_functional_status.json')
if scanner_run:
 shutil.copy2(b/scheduler,t/scheduler)
 shutil.copy2(b/'status.json',t/'accepted_fc_scanner_status.json')
 (t/'fc_scanner_overlay.json').write_text(json.dumps(dict(base_pair=str(s),scanner_pair=str(b),changed_rtl=[scheduler],compared_rtl=len(source_files),source_pair_differences=differences,scope='Physical measurement of assembled candidate. Base SCNN pair predates banked storage rewrite; full FC pair covers scanner. This is not a new full final-top SCNN functional pass.'),indent=2)+'\\n')
print('prepared synthesis source from '+str(s))
'''
    print(run(client, "python3 -c " + shlex.quote(bootstrap)))
    shell = "bash " + shlex.quote(target + "/server/run_unified_h40_final_composition_v1_synth.sh") + " " + shlex.quote(target) + " 1"
    screen = "snn_resident_synth"
    run(client, "screen -dmS " + screen + " bash -lc " + shlex.quote(shell))
    print(json.dumps(dict(remote=target, screen=screen, synthesis_launched=True, from_functional=candidate_run)))


def control(client, name, candidate_run, test):
    source = REMOTE + "/" + candidate_run
    target = REMOTE + "/" + name
    bootstrap = f'''import pathlib,shutil
s=pathlib.Path({source!r});t=pathlib.Path({target!r})
t.mkdir(parents=True,exist_ok=False)
shutil.copytree(s/'rtl',t/'rtl')
w='verification/experiments/srnn_resident_weight_context/rtl/experimental_dual_context_weight_store_p4all_v1.sv'
shutil.copy2(s/w,t/'rtl/memory/experimental_dual_context_weight_store_p4all_v1.sv')
paths=sorted(p.relative_to(t).as_posix() for p in (t/'rtl').rglob('*.sv'))
(t/'control_sources.f').write_text(''.join('"'+p+'"\\n' for p in paths))
'''
    run(client, "python3 -c " + shlex.quote(bootstrap))
    top = "tb_srnn_resident_control" if test == "control" else "tb_srnn_resident_store_modes"
    with client.open_sftp() as sftp:
        for relative, destination in ((EXPERIMENT / ("tb/"+top+".sv"), top+".sv"),
            (Path("scripts/experiments/srnn_resident_server_worker.py"), "scripts/experiments/srnn_resident_server_worker.py")):
            upload(sftp, local_path(ROOT / relative), target + "/" + destination)
    shell = "cd " + shlex.quote(target) + " && exec python3 scripts/experiments/srnn_resident_server_worker.py " + test
    screen = "snn_resident_" + name
    run(client, "screen -dmS " + shlex.quote(screen) + " bash -lc " + shlex.quote(shell))
    print(json.dumps(dict(remote=target, screen=screen, launched=True)))


def steady(client, name, candidate_run):
    source = REMOTE + "/" + candidate_run
    target = REMOTE + "/" + name
    dependencies = "/root/autodl-tmp/SNN_CURRENT_PROFILE_MEASUREMENTS_20261009_R1/srnn/results/paper_evidence/supplement_20261004/performance/current_source_measurements/srnn/server_pair_r1/dependencies.json"
    bootstrap = f'''import json,pathlib,shutil
s=pathlib.Path({source!r});t=pathlib.Path({target!r})
status=json.loads((s/'status.json').read_text())
if status['status']!='PASS_candidate_fixed_sample_pair_not_physical_QoR': raise ValueError('source candidate not accepted')
t.mkdir(parents=True,exist_ok=False)
for rel in json.loads(pathlib.Path({dependencies!r}).read_text()):
 p=t/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/rel,p)
for rel in ('tb_srnn_profile_measurement.sv','sources.f','input_commands.json','verification/experiments/srnn_resident_weight_context/rtl/experimental_dual_context_weight_store_p4all_v1.sv'):
 p=t/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/rel,p)
shutil.copy2(s/'status.json',t/'accepted_source_status.json')
'''
    run(client, "python3 -c " + shlex.quote(bootstrap))
    with client.open_sftp() as sftp:
        upload(sftp, ROOT / "scripts/experiments/srnn_resident_server_worker.py", target + "/scripts/experiments/srnn_resident_server_worker.py")
    shell = "cd " + shlex.quote(target) + " && exec python3 scripts/experiments/srnn_resident_server_worker.py steady"
    screen = "snn_resident_" + name
    run(client, "screen -dmS " + shlex.quote(screen) + " bash -lc " + shlex.quote(shell))
    print(json.dumps(dict(remote=target, screen=screen, launched=True)))


def recovery(client, name, candidate_run):
    source = REMOTE + "/" + candidate_run
    target = REMOTE + "/" + name
    dependencies = "/root/autodl-tmp/SNN_CURRENT_PROFILE_MEASUREMENTS_20261009_R1/srnn/results/paper_evidence/supplement_20261004/performance/current_source_measurements/srnn/server_pair_r1/dependencies.json"
    bootstrap = f'''import json,pathlib,shutil
s=pathlib.Path({source!r});t=pathlib.Path({target!r})
if json.loads((s/'status.json').read_text())['status']!='PASS_candidate_fixed_sample_pair_not_physical_QoR': raise ValueError('source candidate not accepted')
t.mkdir(parents=True,exist_ok=False)
for rel in json.loads(pathlib.Path({dependencies!r}).read_text()):
 p=t/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/rel,p)
for rel in ('sources.f','source_comparison.json','source_changes.diff','verification/experiments/srnn_resident_weight_context/rtl/experimental_dual_context_weight_store_p4all_v1.sv'):
 p=t/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/rel,p)
shutil.copy2(s/'status.json',t/'accepted_source_status.json')
'''
    run(client, "python3 -c " + shlex.quote(bootstrap))
    prepared = Path(local_path(ROOT / EXPERIMENT / name))
    with client.open_sftp() as sftp:
        for path in prepared.iterdir():
            if path.is_file():
                upload(sftp, path, target+"/recovery/"+path.name)
        upload(sftp, local_path(ROOT / "scripts/experiments/srnn_resident_server_worker.py"), target+"/scripts/experiments/srnn_resident_server_worker.py")
    run(client, "cp " + shlex.quote(target+"/recovery/tb_srnn_profile_measurement.sv") + " " + shlex.quote(target+"/tb_srnn_profile_measurement.sv"))
    shell = "cd " + shlex.quote(target) + " && exec python3 scripts/experiments/srnn_resident_server_worker.py recovery"
    screen = "snn_resident_" + name
    run(client, "screen -dmS " + shlex.quote(screen) + " bash -lc " + shlex.quote(shell))
    print(json.dumps(dict(remote=target, screen=screen, launched=True)))


def multisequence(client, name, candidate_run):
    source = REMOTE + "/" + candidate_run
    target = REMOTE + "/" + name
    bootstrap = f'''import pathlib,shutil,json
s=pathlib.Path({source!r});t=pathlib.Path({target!r})
status=json.loads((s/'status.json').read_text())
if status['status']!='PASS_candidate_fixed_sample_pair_not_physical_QoR': raise ValueError('source candidate not accepted')
t.mkdir(parents=True,exist_ok=False)
shutil.copytree(s/'rtl',t/'rtl')
shutil.copytree(s/'tb',t/'tb')
w='verification/experiments/srnn_resident_weight_context/rtl/experimental_dual_context_weight_store_p4all_v1.sv'
(t/w).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/w,t/w)
shutil.copy2(s/'sources.f',t/'sources.f')
shutil.copy2(s/'status.json',t/'accepted_source_status.json')
shutil.copy2(s/'source_comparison.json',t/'source_comparison.json')
'''
    run(client, "python3 -c " + shlex.quote(bootstrap))
    prepared = ROOT / EXPERIMENT / "multi_fixture_r1"
    with client.open_sftp() as sftp:
        for path in Path(local_path(prepared)).rglob("*"):
            if path.is_file():
                upload(sftp, path, target+"/multi_fixture/"+path.relative_to(Path(local_path(prepared))).as_posix())
        upload(sftp, ROOT / "scripts/experiments/srnn_resident_server_worker.py", target+"/scripts/experiments/srnn_resident_server_worker.py")
    run(client, "cp " + shlex.quote(target+"/multi_fixture/tb_srnn_profile_measurement.sv") + " " + shlex.quote(target+"/tb_srnn_profile_measurement.sv"))
    shell = "cd " + shlex.quote(target) + " && exec python3 scripts/experiments/srnn_resident_server_worker.py multisequence"
    screen = "snn_resident_" + name
    run(client, "screen -dmS " + shlex.quote(screen) + " bash -lc " + shlex.quote(shell))
    print(json.dumps(dict(remote=target, screen=screen, launched=True)))


def compatibility(client, name, candidate_run, workload, fc_scanner_candidate=None, scnn_publisher_candidate=None):
    if fc_scanner_candidate and (workload not in ("fc", "scnn") or not fc_scanner_candidate.replace("_", "").isalnum()):
        raise ValueError("FC scanner overlay requires FC/SCNN workload and a simple candidate name")
    if scnn_publisher_candidate and (workload != "scnn" or not scnn_publisher_candidate.replace("_", "").isalnum()):
        raise ValueError("SCNN publisher overlay requires SCNN workload and a simple candidate name")
    if workload in ("scnn", "cross_profile", "scnn_probe"):
        return scnn_compatibility(client, name, candidate_run, workload, fc_scanner_candidate, scnn_publisher_candidate)
    source = REMOTE + "/" + candidate_run
    target = REMOTE + "/" + name
    accepted_fc = "/root/autodl-tmp/SNN_CURRENT_PROFILE_MEASUREMENTS_20261009_R1/fc"
    measured_fc = "results/paper_evidence/supplement_20261004/performance/current_source_measurements/fc/server_pair_r1"
    bootstrap = f'''import json,pathlib,shutil
s=pathlib.Path({source!r});t=pathlib.Path({target!r})
if json.loads((s/'status.json').read_text())['status']!='PASS_candidate_fixed_sample_pair_not_physical_QoR': raise ValueError('source candidate not accepted')
t.mkdir(parents=True,exist_ok=False)
if {workload!r}=='fc':
 a=pathlib.Path({accepted_fc!r});m=a/{measured_fc!r};config=json.loads((m/'commands.json').read_text())
 for rel in config['minimal_dependencies']:
  p=pathlib.PurePosixPath(rel)
  if p.is_absolute() or '..' in p.parts: raise ValueError(rel)
  q=t/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a/rel,q)
 shutil.copy2(m/'tb_fc_profile_measurement.sv',t/'tb_fc_profile_measurement.sv')
 lines=[]
 for line in (m/'sources.f').read_text().splitlines():
  rel=pathlib.Path(line.strip().strip('"')).relative_to(a)
  lines.append('tb_fc_profile_measurement.sv' if rel.name=='tb_fc_profile_measurement.sv' else rel.as_posix())
 (t/'compatibility_input.json').write_text(json.dumps(config,indent=2)+'\\n')
else:
 lines=sorted(p.relative_to(s).as_posix() for p in (s/'rtl').rglob('*.sv'))
 lines+=['tb_frontend_final_top_smoke.sv']
shutil.copytree(s/'rtl',t/'rtl',dirs_exist_ok=True)
w='verification/experiments/srnn_resident_weight_context/rtl/experimental_dual_context_weight_store_p4all_v1.sv'
shutil.copy2(s/w,t/'rtl/memory/experimental_dual_context_weight_store_p4all_v1.sv')
lines+=['rtl/memory/experimental_dual_context_weight_store_p4all_v1.sv']
(t/'compatibility_sources.f').write_text(''.join('"'+p+'"\\n' for p in dict.fromkeys(lines)))
shutil.copy2(s/'source_comparison.json',t/'source_comparison.json')
'''
    run(client, "python3 -c " + shlex.quote(bootstrap))
    with client.open_sftp() as sftp:
        uploads = [(Path("scripts/experiments/srnn_resident_server_worker.py"), "scripts/experiments/srnn_resident_server_worker.py")]
        if fc_scanner_candidate:
            overlay = Path("verification/experiments/fc_contribution_pipeline") / fc_scanner_candidate
            scheduler = Path("rtl/learning/unified_fc_tp_w2_contribution_scheduler.sv")
            uploads += [(overlay / scheduler, scheduler.as_posix()),
                        (overlay / "source_changes.diff", "fc_scanner_changes.diff")]
            with sftp.open(target + "/fc_scanner_overlay.json", "w") as stream:
                stream.write(json.dumps(dict(source_candidate=candidate_run,
                    fc_scanner_candidate=fc_scanner_candidate, changed_sources=[scheduler.as_posix()],
                    scope="FC B16T99 unchanged fixture/TB; new RTL has no inherited physical timing"), indent=2) + "\n")
        if workload == "frontend":
            generated = Path("verification/experiments/frontend_switch_timing/generated")
            uploads += [(generated / "tb_frontend_final_top_smoke.sv", "tb_frontend_final_top_smoke.sv"),
                        (generated / "program.txt", "program.txt")]
            component = Path("software/autonomous_reconfiguration/snn_reconfig")
            uploads += [(p.relative_to(ROOT), p.relative_to(ROOT).as_posix())
                        for p in (ROOT / component).glob("*.py")]
        for relative, destination in uploads:
            upload(sftp, local_path(ROOT / relative), target + "/" + destination)
    shell = "cd " + shlex.quote(target) + " && exec python3 scripts/experiments/srnn_resident_server_worker.py " + workload
    screen = "snn_resident_" + name
    run(client, "screen -dmS " + shlex.quote(screen) + " bash -lc " + shlex.quote(shell))
    print(json.dumps(dict(remote=target, screen=screen, launched=True, workload=workload)))


def scnn_compatibility(client, name, candidate_run, workload="scnn", fc_scanner_candidate=None, scnn_publisher_candidate=None):
    source = REMOTE + "/" + candidate_run
    target = REMOTE + "/" + name
    accepted = "/root/autodl-tmp/SNN_CURRENT_PROFILE_MEASUREMENTS_20261009_R1/scnn_r2"
    measured = "results/paper_evidence/supplement_20261004/performance/current_source_measurements/scnn/server_pair_r2"
    bootstrap = f'''import json,pathlib,shutil
s=pathlib.Path({source!r});t=pathlib.Path({target!r});a=pathlib.Path({accepted!r});m=a/{measured!r}
if json.loads((s/'status.json').read_text())['status']!='PASS_candidate_fixed_sample_pair_not_physical_QoR': raise ValueError('source candidate not accepted')
t.mkdir(parents=True,exist_ok=False)
shutil.copy2(m/'build.status.json',t/'scnn_compatibility_build.json')
for rel in (m/'dependencies.txt').read_text().splitlines():
 p=pathlib.PurePosixPath(rel)
 if p.is_absolute() or '..' in p.parts: raise ValueError(rel)
 q=t/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a/rel,q)
lines=[]
for line in (m/'sources.f').read_text().splitlines():
 rel=pathlib.Path(line.strip().strip('"')).relative_to(a)
 lines.append('tb_scnn_profile_measurement.sv' if rel.name=='tb_scnn_profile_measurement.sv' else rel.as_posix())
shutil.copytree(s/'rtl',t/'rtl',dirs_exist_ok=True)
w='verification/experiments/srnn_resident_weight_context/rtl/experimental_dual_context_weight_store_p4all_v1.sv'
shutil.copy2(s/w,t/'rtl/memory/experimental_dual_context_weight_store_p4all_v1.sv')
lines+=['rtl/memory/experimental_dual_context_weight_store_p4all_v1.sv']
(t/'compatibility_sources.f').write_text(''.join('"'+p+'"\\n' for p in dict.fromkeys(lines)))
shutil.copy2(s/'source_comparison.json',t/'source_comparison.json')
if {workload!r} == 'cross_profile':
 fixture=json.loads((s/'input_commands.json').read_text())['fixture']
 rel=pathlib.PurePosixPath(fixture)
 if rel.is_absolute() or '..' in rel.parts: raise ValueError(fixture)
 shutil.copytree(s/fixture,t/fixture)
 for filename in ('srnn_m2_v28_wh1_initial_q14.mem','srnn_m2_v28_wout_initial_q14.mem'):
  q=t/'tb/data'/filename;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/'tb/data'/filename,q)
'''
    run(client,"python3 -c "+shlex.quote(bootstrap))
    prepared = EXPERIMENT / name
    with client.open_sftp() as sftp:
        for relative,destination in [(prepared/"tb_scnn_profile_measurement.sv","tb_scnn_profile_measurement.sv"),
                (prepared/"reference.json","reference.json"),(prepared/"testbench_changes.diff","testbench_changes.diff"),
                (Path("scripts/experiments/prepare_scnn_resident_compatibility.py"),"scripts/experiments/prepare_scnn_resident_compatibility.py"),
                (Path("scripts/experiments/srnn_resident_server_worker.py"),"scripts/experiments/srnn_resident_server_worker.py")]:
            upload(sftp,local_path(ROOT/relative),target+"/"+destination)
        if workload == "cross_profile":
            relative = Path("scripts/experiments/prepare_srnn_scnn_cross_profile.py")
            upload(sftp, local_path(ROOT/relative), target+"/"+relative.as_posix())
        if workload == "scnn_probe":
            relative = Path("scripts/experiments/prepare_scnn_schedule_probe.py")
            upload(sftp, local_path(ROOT/relative), target+"/"+relative.as_posix())
        changed_sources = []
        for selected, experiment, module, label in (
                (fc_scanner_candidate, "fc_contribution_pipeline", "unified_fc_tp_w2_contribution_scheduler.sv", "fc_scanner"),
                (scnn_publisher_candidate, "scnn_publisher_issue", "unified_scnn_classifier_cache_refill_publisher_v1.sv", "scnn_publisher")):
            if selected:
                overlay = Path("verification/experiments") / experiment / selected
                relative = Path("rtl/learning") / module
                upload(sftp, local_path(ROOT/overlay/relative), target+"/"+relative.as_posix())
                upload(sftp, local_path(ROOT/overlay/"source_changes.diff"), target+"/"+label+"_changes.diff")
                changed_sources.append(relative.as_posix())
        if changed_sources:
            with sftp.open(target+"/scnn_optimization_overlay.json", "w") as stream:
                stream.write(json.dumps(dict(source_candidate=candidate_run, fc_scanner_candidate=fc_scanner_candidate,
                    scnn_publisher_candidate=scnn_publisher_candidate, changed_sources=changed_sources,
                    scope="Original sample0 T4 TB/reference and service model. New RTL has no inherited physical timing; counters must be compared after paired completion."), indent=2)+"\n")
    shell="cd "+shlex.quote(target)+" && exec python3 scripts/experiments/srnn_resident_server_worker.py "+workload
    run(client,"screen -dmS "+shlex.quote("snn_resident_"+name)+" bash -lc "+shlex.quote(shell))
    print(json.dumps(dict(remote=target,screen="snn_resident_"+name,launched=True,workload=workload)))


def select_placement(script, directive):
    allowed = ("AltSpreadLogic_high", "ExtraNetDelay_high")
    if directive not in allowed:
        raise ValueError("unsupported placement directive")
    anchor = "place_design -directive AltSpreadLogic_high"
    if script.count(anchor) != 1:
        raise ValueError("ambiguous original placement command")
    selected = script.replace(anchor, "place_design -directive " + directive, 1)
    return selected.replace("place_directive=AltSpreadLogic_high", "place_directive=" + directive)


def implementation(client, name, synthesis_run, placement_directive="AltSpreadLogic_high"):
    source = REMOTE + "/" + synthesis_run
    target = REMOTE + "/" + name
    reference = "/root/autodl-tmp/UNIFIED_SNN_PAPER_RUN137_SCNN_FRAME_SERVICE_IMPL_R1"
    out = source + "/server_results/final_composition_synth/"
    with client.open_sftp() as sftp:
        for filename, expected in (("wrapper_status.txt", "status=pass"), ("vivado_exit_code.txt", "0"),
                                   ("error_count.txt", "0"), ("critical_warning_count.txt", "0")):
            with sftp.open(out+filename) as stream:
                if stream.read().decode("utf-8").strip() != expected:
                    raise ValueError("synthesis not accepted: " + filename)
        with sftp.open(out+"vivado_status.txt") as stream:
            status = dict(line.split("=",1) for line in stream.read().decode("utf-8").splitlines() if "=" in line)
        if status.get("black_boxes") != "0":
            raise ValueError("synthesis contains black boxes")
        if sftp.stat(out+"post_synth.dcp").st_size == 0:
            raise ValueError("empty synthesis checkpoint")
        with sftp.open(reference+"/vivado/run_unified_h40_final_composition_v1_impl_from_dcp.tcl") as stream:
            original = stream.read().decode("utf-8")
    active = run(client, "ps -eo comm,args").splitlines()
    if any(line.lstrip().startswith("vivado ") or "/lnx64.o/vivado" in line for line in active):
        raise RuntimeError("another Vivado is active")
    bootstrap = f'''import pathlib,shutil
t=pathlib.Path({target!r});r=pathlib.Path({reference!r})
t.mkdir(parents=True,exist_ok=False)
shutil.copytree(r/'vivado',t/'vivado');shutil.copytree(r/'server',t/'server')
'''
    run(client, "python3 -c " + shlex.quote(bootstrap))
    # Keep the physical recipe; omit the old optional power report, outside this request.
    start = original.index("set power_status not_available")
    end = original.index("set route_text", start)
    script = original[:start] + "set power_status not_run_out_of_scope\n\n" + original[end:]
    script = select_placement(script, placement_directive)
    pre = '''set original_protected_names [lsort [get_property NAME [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE == 1 && DONT_TOUCH == 1}]]]
set protected_handle [open [file join $out_dir protected_primitives_pre_impl.rpt] w]
puts $protected_handle "protected_primitive_count=[llength $original_protected_names]"
foreach name $original_protected_names { puts $protected_handle $name }
close $protected_handle
'''
    script = script.replace("\nopt_design\n", "\n"+pre+"\nopt_design\n", 1)
    post = '''set final_protected_names [lsort [get_property NAME [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE == 1 && DONT_TOUCH == 1}]]]
if {$final_protected_names ne $original_protected_names} { error "protected primitive set changed" }
set protected_handle [open [file join $out_dir protected_primitives_post_impl.rpt] w]
puts $protected_handle "protected_primitive_count=[llength $final_protected_names]"
puts $protected_handle "protection_preserved=1"
foreach name $final_protected_names { puts $protected_handle $name }
close $protected_handle
set final_trace_cells [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE == 1 && REF_NAME =~ FD* && (NAME =~ *trace_product_q_reg* || NAME =~ *trace_old_flat_q_reg* || NAME =~ *o_trace_next_flat_reg*)}]
set final_trace_nonconstant 0
set trace_handle [open [file join $out_dir trace_payload_control_post_impl.rpt] w]
puts $trace_handle "trace_payload_cell_count=[llength $final_trace_cells]"
foreach payload_cell $final_trace_cells {
    foreach pin_name {R S CLR PRE} {
        set pin [get_pins -quiet "[get_property NAME $payload_cell]/$pin_name"]
        if {[llength $pin] == 0} { continue }
        set net [get_nets -quiet -of_objects $pin]
        set drivers [get_cells -quiet -of_objects [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]]
        set refs [lsort -unique [get_property REF_NAME $drivers]]
        puts $trace_handle "cell=[get_property NAME $payload_cell] pin=$pin_name drivers=$refs"
        if {[lsearch -exact $refs GND] < 0} { incr final_trace_nonconstant }
    }
}
puts $trace_handle "trace_payload_nonconstant_control_count=$final_trace_nonconstant"
close $trace_handle
if {[llength $final_trace_cells] == 0 || $final_trace_nonconstant != 0} { error "final trace payload control audit failed" }
report_clocks -file [file join $out_dir clocks_post_route.rpt]
report_exceptions -file [file join $out_dir exceptions_post_route.rpt]
'''
    anchor = "write_checkpoint -force [file join $out_dir post_route_physopt.dcp]"
    if script.count(anchor) != 1:
        raise ValueError("ambiguous post-route checkpoint anchor")
    script = script.replace(anchor, post+"\n"+anchor, 1)
    with client.open_sftp() as sftp:
        with sftp.open(target+"/vivado/run_unified_h40_final_composition_v1_impl_from_dcp.tcl", "w") as stream:
            stream.write(script)
        with sftp.open(target+"/implementation_script_changes.diff", "w") as stream:
            stream.write("".join(difflib.unified_diff(original.splitlines(True), script.splitlines(True), fromfile="RUN137/implementation.tcl", tofile="resident/implementation.tcl")))
    shell = "bash " + shlex.quote(target+"/server/run_unified_h40_final_composition_v1_impl_from_dcp.sh") + " " + shlex.quote(target) + " " + shlex.quote(out+"post_synth.dcp")
    run(client, "screen -dmS snn_resident_impl bash -lc " + shlex.quote(shell))
    print(json.dumps(dict(remote=target, screen="snn_resident_impl", source_dcp=out+"post_synth.dcp", placement_directive=placement_directive, launched=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("unit", "profile", "control", "steady", "recovery", "multisequence", "compatibility", "synthesis", "implementation", "status", "fetch"))
    parser.add_argument("--name", default="unit_r1")
    parser.add_argument("--candidate", default="candidate_r2")
    parser.add_argument("--candidate-run", default="srnn_r5")
    parser.add_argument("--test", choices=("control", "store_modes"), default="control")
    parser.add_argument("--synthesis-run", default="srnn_resident_synth_r1")
    parser.add_argument("--workload", choices=("fc", "frontend", "scnn", "cross_profile", "scnn_probe"), default="frontend")
    parser.add_argument("--fc-scanner-candidate", default=None)
    parser.add_argument("--fc-scanner-run", default=None)
    parser.add_argument("--scnn-publisher-candidate", default=None)
    parser.add_argument("--stage", choices=("synth", "impl"), default="synth")
    parser.add_argument("--placement-directive", choices=("AltSpreadLogic_high", "ExtraNetDelay_high"), default="AltSpreadLogic_high")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        parser.error("name must contain only letters, digits and underscores")
    client = connect()
    try:
        if args.action == "unit":
            if not args.candidate.replace("_", "").isalnum():
                parser.error("simple candidate name required")
            unit(client, args.name, args.candidate)
        elif args.action == "profile":
            if not args.candidate.replace("_", "").isalnum():
                parser.error("simple candidate name required")
            profile(client, args.name, args.candidate)
        elif args.action == "synthesis":
            if not args.candidate_run.replace("_", "").isalnum():
                parser.error("simple candidate run required")
            if args.fc_scanner_run and not args.fc_scanner_run.replace("_", "").isalnum():
                parser.error("simple FC scanner run required")
            synthesis(client, args.name, args.candidate_run, args.fc_scanner_run)
        elif args.action == "control":
            if not args.candidate_run.replace("_", "").isalnum():
                parser.error("simple candidate run required")
            control(client, args.name, args.candidate_run, args.test)
        elif args.action == "steady":
            if not args.candidate_run.replace("_", "").isalnum():
                parser.error("simple candidate run required")
            steady(client, args.name, args.candidate_run)
        elif args.action == "multisequence":
            if not args.candidate_run.replace("_", "").isalnum():
                parser.error("simple candidate run required")
            multisequence(client, args.name, args.candidate_run)
        elif args.action == "recovery":
            if not args.candidate_run.replace("_", "").isalnum():
                parser.error("simple candidate run required")
            recovery(client, args.name, args.candidate_run)
        elif args.action == "compatibility":
            if not args.candidate_run.replace("_", "").isalnum():
                parser.error("simple candidate run required")
            compatibility(client, args.name, args.candidate_run, args.workload, args.fc_scanner_candidate, args.scnn_publisher_candidate)
        elif args.action == "implementation":
            if not args.synthesis_run.replace("_", "").isalnum():
                parser.error("simple synthesis run required")
            implementation(client, args.name, args.synthesis_run, args.placement_directive)
        elif args.action == "status":
            with client.open_sftp() as sftp:
                try:
                    with sftp.open(REMOTE + "/" + args.name + "/status.json") as stream:
                        record = json.load(stream)
                except FileNotFoundError:
                    out = REMOTE + "/" + args.name + "/server_results/final_composition_" + args.stage + "/"
                    record = {}
                    for filename in ("wrapper_status.txt", "vivado_exit_code.txt", "error_count.txt", "critical_warning_count.txt", "vivado_status.txt"):
                        try:
                            with sftp.open(out + filename) as stream:
                                record[filename] = stream.read().decode("utf-8").strip()
                        except FileNotFoundError:
                            record[filename] = None
            selected = {key: value for key, value in record.items() if key not in ("sample_definition", "lint_warnings", "build_warnings")}
            for name in ("lint", "build", "run", "run_off", "run_on"):
                if name in selected:
                    selected[name] = {key: value for key, value in selected[name].items() if key != "command"}
            print(json.dumps(selected, indent=2))
        else:
            with client.open_sftp() as sftp:
                fetch_tree(sftp, REMOTE + "/" + args.name, ROOT / RESULT / args.name)
            print(ROOT / RESULT / args.name)
    finally:
        client.close()


if __name__ == "__main__":
    main()
