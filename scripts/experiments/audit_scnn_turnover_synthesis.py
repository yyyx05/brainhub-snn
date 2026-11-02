"""Audit the terminal turnover candidate synthesis; never inherit old routed STA."""
import csv
import json
from pathlib import Path
import re
import shlex
import zipfile

from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,local_path
from run_scnn_turnover_synthesis import NAME,TCL,XDC,SHELL,OUT
from audit_m2_compact_synthesis import resources,timing,hierarchy


def audit():
    base=Path(local_path(ROOT/RESULT));local=base/NAME
    state=json.loads((local/"status.json").read_text())
    if not state.get("end"):raise ValueError("active experiment; terminal fetch required")
    if state["status"]=="FAIL_archive_space":raise ValueError("raw files remain in scratch; finish archival before auditing")
    client=connect()
    code='''import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT)
files={q.relative_to(p).as_posix():q.stat().st_size for q in p.rglob('*') if q.is_file() and '.Xil' not in q.relative_to(p).parts and '__pycache__' not in q.relative_to(p).parts}
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=files,data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free)))
'''
    try:observation=json.loads(run(client,"python3 -c "+shlex.quote("ROOT="+repr(REMOTE+"/"+NAME)+"\n"+code)))
    finally:client.close()
    files=observation.pop("files")
    if list(local.rglob("*.part")):raise ValueError("partial local archive")
    for rel,size in files.items():
        if rel.endswith(".part") or not (local/rel).is_file() or (local/rel).stat().st_size!=size:
            raise ValueError("raw archive mismatch: "+rel)
    source=base/"scnn_turnover_frame_r2/candidate"
    paths=sorted(p.relative_to(source).as_posix() for p in (source/"rtl").rglob("*") if p.is_file())
    if len(paths)!=109 or sum(p.endswith('.sv') for p in paths)!=108:raise ValueError("incomplete source tree")
    for rel in paths:
        if (local/rel).read_bytes()!=(source/rel).read_bytes():raise ValueError("actual source bytes changed: "+rel)
    for rel in (TCL,XDC,SHELL):
        if (local/rel).read_bytes()!=(base/"m2_compact_synth_r2"/rel).read_bytes():raise ValueError("recipe changed: "+rel)
    snapshot=Path(local_path(ROOT/"verification/experiments/scnn_forward_turnover/core_synthesis_r1/launch_worker.py"))
    if (local/"worker.py").read_bytes()!=snapshot.read_bytes():raise ValueError("worker bytes changed")
    result=dict(status="TERMINAL_FAIL_original_preserved",original_status=state,raw_files=len(files),raw_bytes=sum(files.values()),
        raw_sizes_match=True,no_part=True,source_files_byte_equal=109,recipe_byte_equal=True,worker_byte_equal=True,
        remote_observation=observation,
        boundary="Exact108SV+ROM turnover candidate, unchanged512/15ns/8-thread original synthesis recipe. Flow pass is not routed STA or final functional release. Source protection retained; netlist protected-set/trace controls must be checked in this candidate's own implementation. No power/bitstream; OOC IO delay limitations retained.")
    if state['status'].startswith('PASS'):
        report=local/OUT
        for filename,expected in (("wrapper_status.txt","status=pass"),("vivado_exit_code.txt","0"),("error_count.txt","0"),("critical_warning_count.txt","0")):
            if (report/filename).read_text().strip()!=expected:raise ValueError("actual terminal state: "+filename)
        log=(report/"vivado.log").read_text(errors="replace")
        if re.search(r"^ERROR:|^CRITICAL WARNING:|Abnormal program termination",log,re.M):raise ValueError("EDA diagnostic")
        for index,rel in enumerate(p for p in paths if p.endswith('.sv')):
            compat=report/"vivado_2022_2_sources"/f"{index:03d}_{Path(rel).name}"
            if compat.read_text()!=(local/rel).read_text().replace('`default_nettype none','`default_nettype wire'):
                raise ValueError("compatibility source changed")
        dcp=report/"post_synth.dcp"
        with zipfile.ZipFile(dcp) as checkpoint:
            if checkpoint.testzip() is not None:raise ValueError("DCP CRC failure")
            entries=len(checkpoint.infolist())
        comparison=[];banks=[]
        for label,folder in (("compact512_pre_turnover",base/"m2_compact_synth_r2"),("compact512_turnover",local)):
            r=folder/OUT
            use=resources((r/"utilization_post_synth.rpt").read_text())
            sta=timing((r/"timing_summary_post_synth.rpt").read_text())
            items=hierarchy((r/"utilization_hierarchical_post_synth.rpt").read_text())
            contexts={i:[x for x in items if re.fullmatch(r'g_context\['+str(i)+r'\]\.g_enabled(?:\.g_original|\.g_compact)?\.u_store',x['name'])] for i in (0,1)}
            if len(contexts[0])!=12 or len(contexts[1])!=8:raise ValueError("context hierarchy coverage")
            totals={i:sum(x['ramb36']+x['ramb18']/2 for x in contexts[i]) for i in (0,1)}
            comparison.append(dict(variant=label,stage='post_synth',**use,**sta,context0_bram36eq=totals[0],context1_bram36eq=totals[1],other_bram36eq=use['bram36eq']-totals[0]-totals[1]))
            if folder==local:
                banks=[x for x in items if 'g_context[1].g_enabled.g_compact.u_store/' in x['path'] and re.fullmatch(r'g_bank\[\d+\]\.u_memory',x['name'])]
                if len(banks)!=32:raise ValueError("Context1 bank coverage")
        worst=(report/"timing_setup_paths_post_synth.rpt").read_text()
        path={k:re.search(r'^\s*'+k+r':\s*(.+)$',worst,re.M)[1] for k in ('Source','Destination','Data Path Delay','Logic Levels','Requirement')}
        summary=(report/"timing_summary_post_synth.rpt").read_text()
        timing_checks={k:int(re.search(r'checking '+k+r' \((\d+)\)',summary)[1]) for k in ('no_clock','unconstrained_internal_endpoints','no_input_delay','no_output_delay','loops')}
        methodology=(report/'methodology_post_synth.rpt').read_text()
        rules=[dict(rule=m[1],severity=m[2],count=int(m[3])) for m in re.finditer(
            r'^\|\s*([A-Z]+-\d+)\s*\|\s*(Warning|Critical Warning|Error)\s*\|[^|]*\|\s*(\d+)\s*\|',methodology,re.M)]
        total=int(re.search(r'Violations found:\s*(\d+)',methodology)[1])
        if sum(r['count'] for r in rules)!=total:raise ValueError('methodology warning accounting')
        tcl=dict(line.split('=',1) for line in (report/'vivado_status.txt').read_text().splitlines() if '=' in line)
        result.update(status='PASS_turnover_synthesis_evidence_not_routed_STA',compatibility_sources_byte_checked=108,
            dcp=dict(bytes=dcp.stat().st_size,zip_crc_pass=True,members=entries),comparison=comparison,
            delta_candidate_minus_baseline={k:comparison[1][k]-comparison[0][k] for k in ('lut','ff','bram36eq','dsp')},
            ordinary_warnings=len(re.findall(r'^WARNING:',log,re.M)),oom_or_killed=bool(re.search(r'out of memory|\bOOM\b|\bKilled\b',log,re.I)),
            methodology_rules=rules,methodology_total=total,tcl_status=tcl,timing_checks=timing_checks,worst_path=path)
        for filename,table in (("comparison.csv",comparison),("context1_banks.csv",banks)):
            with (local/filename).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (local/"terminal_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ('original_status','remote_observation')},indent=2))


if __name__=='__main__':audit()
