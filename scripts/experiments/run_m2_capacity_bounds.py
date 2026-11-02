"""Exercise public read-configuration and refill capacity rejection remotely."""
import json
from pathlib import Path
import re
import shlex

from prepare_m2_compact_core import OUTPUT
from run_srnn_resident_remote import ROOT, REMOTE, connect, run, upload, local_path

NAME = 'm2_capacity_bounds_r1'


def instance(relative, label, params, signals):
    source = Path(local_path(OUTPUT/'rtl'/relative)).read_text()
    header = source[:source.index(');')]
    ports = re.findall(r'^\s*(input|output)\s+.*?\b([io]_\w+)\s*,?\s*$',header,re.M)
    if not ports or len(ports)!=len(set(name for _,name in ports)):
        raise ValueError(relative)
    lines = [f"        .{name}({signals.get(name, chr(39)+'0' if direction=='input' else '')})"
             for direction,name in ports]
    return Path(relative).stem+' #('+params+') '+label+' (\n'+',\n'.join(lines)+'\n    );\n'


def prepare():
    dest=ROOT/'verification/experiments/m2_capacity_bounds/candidate_r1'
    dest.mkdir(parents=True,exist_ok=True)
    base=dict(i_clk='clk',i_rst_n='rst_n')
    adapter=dict(base,i_cfg_valid='valid',o_cfg_ready='cfg_ready',o_cfg_error='cfg_error',
        i_cfg_batch_count="8'd1",i_cfg_backbone_count='count',i_cfg_hidden_count="9'(count)",
        i_cfg_class_count="5'd1",i_cfg_wh1_cache_base='wh_base',i_cfg_wout_cache_base='wo_base')
    scheduler=dict(base,i_refill_valid='valid',o_refill_ready='refill_ready',i_refill_all="1'b1",
        i_cache_row_base='row_base',i_source_count='count',i_destination_count="10'd1",
        i_array_idle="1'b1",o_busy='busy',o_sticky_error='refill_error',o_forward_read_valid='read_valid')
    tb="""`timescale 1ns/1ps
module capacity_case #(parameter int D=578)(output logic done=0);
    logic clk=0, rst_n=0, valid=0;
    logic [9:0] row_base=0, count=0;
    logic [10:0] wh_base=0, wo_base=0;
    wire cfg_ready,cfg_error,refill_ready,busy,refill_error,read_valid;
    int tested=0;
    always #5 clk=~clk;
"""
    tb+=instance('learning/unified_srnn_m2_h40_forward_observation_adapter_v1.sv','adapter',
        '.M2_CACHE_DEPTH(D),.BATCH_MAX(2),.BACKBONE_MAX(4),.HIDDEN_MAX(4),.CLASS_MAX(2)',adapter)
    tb+=instance('memory/unified_srnn_m2_h40_q14_cache_refill_scheduler_v1.sv','scheduler',
        '.M2_CACHE_DEPTH(D)',scheduler)
    tb+="""
    task automatic check(input int row,input int n,input bit wout);
        bit legal;
        legal=n>0 && row+n<=D;
        @(negedge clk); rst_n=0; valid=0;
        repeat(2) @(negedge clk);
        row_base=10'(row); count=10'(n);
        wh_base=wout ? 0 : 11'(row); wo_base=wout ? 11'(row) : 0;
        rst_n=1;
        @(negedge clk);
        if(!cfg_ready || !refill_ready) $fatal(1,"request not ready");
        valid=1;
        @(posedge clk); #1;
        if(cfg_error !== !legal || refill_error !== !legal || busy !== legal)
            $fatal(1,"capacity D=%0d row=%0d n=%0d wout=%0d cfgerr=%0b refillerr=%0b busy=%0b",
                D,row,n,wout,cfg_error,refill_error,busy);
        @(negedge clk); valid=0;
        repeat(5) @(negedge clk);
        if(read_valid !== legal) $fatal(1,"invalid refill escaped or legal refill stalled");
        tested++;
    endtask
    initial begin
        for(int w=0;w<2;w++) begin
            check(D-2,2,1'(w)); check(D-1,2,1'(w));
            check(511,1,1'(w)); check(512,1,1'(w));
            check(577,1,1'(w)); check(578,1,1'(w));
            check(0,0,1'(w)); check(1023,2,1'(w));
        end
        $display("CAPACITY_PUBLIC_PASS depth=%0d cases=%0d",D,tested);
        done=1;
    end
endmodule
module tb_m2_capacity_bounds;
    wire a,b;
    capacity_case #(.D(512)) selected(a);
    capacity_case #(.D(578)) generic_full(b);
    initial begin
        wait(a&&b);
        $display("CAPACITY_BOUNDS_PASS depths=2 cases=32");
        $finish;
    end
    initial begin #100000; $fatal(1,"timeout"); end
endmodule
"""
    (dest/'tb_m2_capacity_bounds.sv').write_text(tb)
    return dest


def main():
    dest=prepare()
    client=connect()
    remote=REMOTE+'/'+NAME
    try:
        bootstrap="import pathlib,shutil,json; p=pathlib.Path("+repr(remote)+"); b=p.parent; s=json.loads((b/'m2_compact_core_r1/status.json').read_text()); assert s['status']=='PASS_compact512_native_T73_pair_not_physical'; assert shutil.disk_usage(b).free>500000000; p.mkdir(exist_ok=False); shutil.copytree(b/'m2_compact_core_r1/rtl',p/'rtl'); print(shutil.disk_usage(b).free)"
        print(run(client,'python3 -c '+shlex.quote(bootstrap)))
        with client.open_sftp() as s:
            upload(s,dest/'tb_m2_capacity_bounds.sv',remote+'/tb_m2_capacity_bounds.sv')
            for name in ('m2_capacity_bounds_worker.py','srnn_resident_server_worker.py'):
                upload(s,Path(__file__).with_name(name),remote+'/scripts/experiments/'+name)
        shell='cd '+shlex.quote(remote)+' && exec python3 scripts/experiments/m2_capacity_bounds_worker.py'
        run(client,'screen -dmS snn_'+NAME+' bash -lc '+shlex.quote(shell))
        print(json.dumps(dict(launched=True,root=remote)))
    finally:
        client.close()


if __name__=='__main__':
    main()
