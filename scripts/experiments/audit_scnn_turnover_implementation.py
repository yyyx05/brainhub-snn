"""Independently audit the fetched turnover implementation and reopened DCP."""
import csv
import argparse
import json
from pathlib import Path
import re
import shlex

from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,local_path
from run_scnn_turnover_implementation import NAME,SCRATCH,SOURCE,EXPERIMENT,ORIGINAL,OUT,implementation_tcl,run_paths
from audit_m2_compact_implementation import resources,violations,report_checks_pass
from audit_m2_compact_synthesis import hierarchy,timing
from audit_m2_compact_route_recovery import fields,route_metrics,reopened_equal,checkpoint_record


def audit():
    base=Path(local_path(ROOT/RESULT));local=base/NAME;prepared=Path(local_path(ROOT/EXPERIMENT))
    state=json.loads((local/'status.json').read_text())
    if not state.get('end') or state['status']=='running':raise ValueError('natural terminal fetch required')
    client=connect()
    try:
        code='''import datetime,json,pathlib,shutil,subprocess,zipfile
r=pathlib.Path(ROOT);s=pathlib.Path(SCRATCH);source=pathlib.Path(SOURCE)
def inventory(p):
 return {q.relative_to(p).as_posix():q.stat().st_size for q in p.rglob('*') if q.is_file() and not set(q.relative_to(p).parts)&{'obj','elab_obj','__pycache__','.Xil'}}
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
with zipfile.ZipFile(source) as z:crc=z.testzip() is None
k=subprocess.run(['dmesg','--ctime'],capture_output=True,text=True)
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=inventory(r),scratch_files=inventory(s),active_eda=active,
 source_bytes=source.stat().st_size,source_crc_pass=crc,data_free=shutil.disk_usage(r).free,tmp_free=shutil.disk_usage('/tmp').free,
 dmesg_exit=k.returncode,dmesg_error=k.stderr)))
'''
        observation=json.loads(run(client,'python3 -c '+shlex.quote('ROOT='+repr(REMOTE+'/'+NAME)+'\nSCRATCH='+repr(SCRATCH)+'\nSOURCE='+repr(REMOTE+'/'+SOURCE)+'\n'+code)))
    finally:client.close()
    if observation['active_eda']:raise ValueError('EDA remains active; defer audit')
    files=observation['files']
    if list(local.rglob('*.part')) or any(p.endswith('.part') for p in files):raise ValueError('partial archive')
    for rel,size in files.items():
        if not (local/rel).is_file() or (local/rel).stat().st_size!=size:raise ValueError('incomplete fetch: '+rel)
    if any(files.get(rel)!=size for rel,size in observation['scratch_files'].items()):raise ValueError('scratch archive incomplete')
    synthesis=json.loads((base/'scnn_turnover_core_synth_r1/terminal_audit.json').read_text())
    original=(base/'m2_compact_impl_r1'/ORIGINAL)
    checks={}
    checks['original_recipe_bytes']=(local/'original_impl.tcl').read_bytes()==original.read_bytes()
    checks['prepared_input_bytes']=all((local/rel).read_bytes()==(prepared/rel).read_bytes() for rel in
        ('original_impl.tcl','implementation.tcl','implementation_changes.diff','preparation.json','launch_worker.py'))
    checks['only_export_reopen_transform']=(local/'implementation.tcl').read_text()==implementation_tcl(original.read_text())
    source=checkpoint_record(Path(local_path(ROOT/RESULT/SOURCE)))
    checks['accepted_synthesis_source']=synthesis['status']=='PASS_turnover_synthesis_evidence_not_routed_STA' and source.get('crc_pass') is True and source.get('bytes')==synthesis['dcp']['bytes']==observation['source_bytes'] and observation['source_crc_pass'] is True
    report=local/OUT;missing=[];parse_errors=[]
    def read(name):
        p=report/name
        return p.read_text(errors='replace') if p.is_file() else ''
    def parsed(name,parser):
        text=read(name)
        if not text:missing.append(name);return None
        try:return parser(text)
        except (ValueError,IndexError,TypeError,KeyError,AttributeError) as error:
            parse_errors.append(dict(report=name,error=str(error)));return None
    log=read('vivado.log');errors=re.findall(r'^ERROR:.*',log,re.M);critical=re.findall(r'^CRITICAL WARNING:.*',log,re.M)
    checks['tool_exit_and_diagnostics']=state.get('exit')==0 and bool(log) and not errors and not critical
    tcl=fields(read('vivado_status.txt'));before=fields(read('before_checkpoint_status.txt'));reopened=fields(read('checkpoint_reopen_status.txt'))
    checks['actual_status_records']=tcl==state.get('tcl_status') and reopened==state.get('checkpoint_reopen')
    checks['tcl_acceptance']=tcl.get('status')=='pass' and tcl.get('route_complete')=='1' and all(tcl.get(k)=='0' for k in ('black_boxes','unplaced_primitives','routing_error_nets','drc_errors','drc_critical_warnings'))
    checks['reopen_counts_timing_protection_trace']=reopened_equal(before,reopened)
    checkpoints={name:checkpoint_record(report/name) for name in ('post_place_physopt.dcp','post_route_physopt.dcp')}
    checks['delivered_checkpoint_crcs']=all(row.get('crc_pass') is True and state.get('checkpoints',{}).get(name,{}).get('bytes')==row.get('bytes') for name,row in checkpoints.items())
    reports={}
    for stage,prefix in (('exported',''),('reopened','reopened/')):
        row=dict(resources=parsed(prefix+'utilization_post_route.rpt',resources),timing=parsed(prefix+'timing_summary_post_route.rpt',timing),route=parsed(prefix+'route_status_post_route.rpt',route_metrics))
        row['clocks']=re.findall(r'^(\w+)\s+(\d+\.\d+)\s+\{([^}]+)\}\s+P\s+\{([^}]+)\}',read(prefix+'clocks_post_route.rpt'),re.M)
        row['no_valid_exceptions']='No valid timing exceptions found.' in read(prefix+'exceptions_post_route.rpt')
        checks[stage+'_route_setup_hold_pulse']=bool(row['timing'] and row['route'] and report_checks_pass(row['timing'],row['route']))
        checks[stage+'_15ns_clock_no_exceptions']=row['clocks']==[('core_clk','15.000','0.000 7.500','i_clk')] and row['no_valid_exceptions']
        reports[stage]=row
    checks['reopened_reports_equal']=reports['exported']==reports['reopened']
    protected=[];traces={}
    for stage in ('pre_impl','post_impl'):
        protected.append([line for line in read('protected_primitives_'+stage+'.rpt').splitlines() if line and '=' not in line])
        text=read('trace_payload_control_'+stage+'.rpt');controls=[line for line in text.splitlines() if line.startswith('cell=')]
        traces[stage]=dict(cells=len(controls),all_gnd=bool(controls) and all(line.endswith('drivers=GND') for line in controls))
        checks[stage+'_trace']=len(controls)==6270 and traces[stage]['all_gnd'] and 'trace_payload_nonconstant_control_count=0' in text
    checks['protected_primitive_set']=len(protected[0])==len(set(protected[0]))==15 and protected[0]==protected[1]
    items=parsed('utilization_hierarchical_post_route.rpt',hierarchy) or []
    arrays=[x['path'] for x in items if x['module']=='unified_dual_role_h40_stationary_array12_candidate_v1']
    tiles=[x['path'] for x in items if re.fullmatch(r'g_tile\[\d+\]\.u_tile',x['name'])]
    checks['one_array_twelve_tiles']=len(arrays)==1 and len(tiles)==12
    contexts={i:[x for x in items if re.fullmatch(r'g_context\['+str(i)+r'\]\.g_enabled\.g_(?:original|compact)\.u_store',x['name'])] for i in (0,1)}
    totals={i:sum(x['ramb36']+x['ramb18']/2 for x in contexts[i]) for i in (0,1)}
    banks=[x for x in items if 'g_context[1].g_enabled.g_compact.u_store/' in x['path'] and re.fullmatch(r'g_bank\[\d+\]\.u_memory',x['name'])]
    checks['context_mapping']=len(contexts[0])==12 and len(contexts[1])==8 and len(banks)==32 and totals=={0:288,1:80} and all(x['ramb36']==(1 if x['name'].startswith('g_bank[3]') else 3) and x['ramb18']==0 for x in banks)
    diagnostics={name:parsed(name+'_post_route.rpt',violations) for name in ('drc','methodology')}
    checks['no_severe_drc_methodology']=all(d is not None and all(x['severity']=='Warning' for x in d['rules']) for d in diagnostics.values())
    timing_text=read('timing_summary_post_route.rpt')
    constraints={key:int(m[1]) if (m:=re.search(r'checking '+key+r' \((\d+)\)',timing_text)) else None for key in ('no_clock','unconstrained_internal_endpoints','no_input_delay','no_output_delay','loops')}
    checks['internal_constraints']=all(constraints[key]==0 for key in ('no_clock','unconstrained_internal_endpoints','loops'))
    worst={key:m[1] if (m:=re.search(r'^\s*'+key+r':\s*(.+)$',read('timing_setup_paths_post_route.rpt'),re.M)) else None for key in ('Source','Destination','Data Path Delay','Logic Levels','Requirement')}
    samples=json.loads((local/'disk_samples.json').read_text());minima={key:min(x[key] for x in samples) if samples else None for key in ('data_free','tmp_free')}
    checks['disk_samples_match']=bool(samples) and state.get('minimum_data_free')==minima['data_free'] and state.get('minimum_tmp_free')==minima['tmp_free']
    checks['all_reports_parsed']=not missing and not parse_errors
    accepted=state['status']=='PASS_candidate_internal15ns_checkpoint_reopened' and all(x is True for x in checks.values())
    result=dict(status='PASS_turnover_implementation_delivery' if accepted else 'NOT_ACCEPTED_turnover_implementation_archived',accepted_implementation=accepted,
        original_status=state,checks=checks,raw_files=len(files),raw_bytes=sum(files.values()),all_raw_sizes_match=True,no_part=True,
        source_checkpoint=source,checkpoints=checkpoints,tcl_status=tcl,before_checkpoint=before,checkpoint_reopen=reopened,
        reports=reports,protected_primitives=protected[0],trace_controls=traces,physical_array_paths=arrays,physical_tile_paths=tiles,
        context_bram36eq=totals,diagnostics=diagnostics,timing_checks=constraints,worst_setup_path=worst,
        errors=errors,critical_warnings=critical,ordinary_warnings=len(re.findall(r'^WARNING:',log,re.M)),missing_reports=missing,parse_errors=parse_errors,
        disk_sample_count=len(samples),sampled_minimum_free=minima,remote_observation=observation,
        oom_or_disk_error_lines=re.findall(r'^.*(?:out of memory|\bOOM\b|\bKilled\b|ENOSPC|No space left|Disk quota exceeded).*$',log,re.I|re.M),
        boundary='Own turnover candidate, original compact optimization/15ns/protection recipe, final DCP exported and reopened. Single internal OOC implementation, not board Fmax, multi-seed stability, final functional release or production merge. IO delays absent. One-second disk samples may miss short peaks. Old failed sites and scratch preserved.')
    if reports['exported']['resources']:
        result['other_bram36eq']=reports['exported']['resources']['bram36eq']-sum(totals.values())
        previous=base/'m2_compact_route_recovery_r1'
        previous_audit=json.loads((previous/'terminal_audit.json').read_text())
        if previous_audit.get('accepted_implementation') is not True:raise ValueError('compact physical reference is not accepted')
        table=[dict(variant='compact512_pre_turnover',stage='post_route',accepted_implementation=True,
                    **resources((previous/OUT/'utilization_post_route.rpt').read_text()),
                    **timing((previous/OUT/'timing_summary_post_route.rpt').read_text()))]
        if reports['exported']['timing']:
            table.append(dict(variant='compact512_turnover',stage='post_route',accepted_implementation=accepted,
                              **reports['exported']['resources'],**reports['exported']['timing']))
            result['same_stage_comparison']=table
            result['delta_candidate_minus_compact']={key:table[1][key]-table[0][key] for key in ('lut','ff','bram36eq','dsp')}
            with (local/'implementation_comparison.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (local/'terminal_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if banks:
        with (local/'context1_banks.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(banks[0]));writer.writeheader();writer.writerows(banks)
    print(json.dumps({key:result[key] for key in ('status','accepted_implementation','checks','raw_files','raw_bytes','errors','critical_warnings','missing_reports','parse_errors')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default=NAME);args=parser.parse_args()
    NAME,SCRATCH,EXPERIMENT=run_paths(args.name)
    audit()
