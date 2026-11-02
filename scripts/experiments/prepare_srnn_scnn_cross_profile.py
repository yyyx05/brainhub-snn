#!/usr/bin/env python3
"""Prepare a same-DUT SRNN first-step / real SCNN T4 / SRNN first-step test."""
import argparse
import difflib
import json
import re

from prepare_scnn_resident_compatibility import ROOT, BASE, transform as scnn_prefix, parse_log as parse_scnn

RESULTS = ROOT / "results/paper_evidence/supplement_20261015/performance_optimization"


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("cross-profile anchor not unique: " + old[:100])
    return text.replace(old, new, 1)


def between(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def driver(srnn):
    declarations = "    localparam int H" + between(srnn, "    localparam int H", "    always #5")
    declarations = once(declarations, "    longint cycles = 0, initialized_cycle;",
                        "    wire [63:0] cycles = 64'(cycle_count);\n    longint initialized_cycle = 0;")
    monitor = "    `define M2" + between(srnn, "    `define M2", "    initial begin\n        wait(i_rst_n);")
    monitor = once(monitor, "        cycles <= cycles + 1;", "")
    monitor = once(monitor, "if (cycles > 367000000)", "if (cross_m2_active && cycles-initialized_cycle > 5000000)")
    monitor = once(monitor, "        else begin\n            if (i_srnn_backbone_forward_response_valid",
                   "        else if (cross_m2_active) begin\n            if (i_srnn_backbone_forward_response_valid")
    tasks = "    task automatic refill" + between(srnn, "    task automatic refill", "    initial begin\n        if (!$value$plusargs")
    main = "        if (!$value$plusargs" + between(srnn, "    initial begin\n        if (!$value$plusargs", "    // All counters sample")
    load, rest = main.split("        i_parameter_owner =", 1)
    load = once(load, ' ||\n            !$value$plusargs("LEARNING=%d",learning)', "")
    config, rest = ("        i_parameter_owner =" + rest).split('        profile_enter("initial_state");', 1)
    config = once(config, '        repeat(6) @(negedge i_clk); i_rst_n = 1; profile_enter("config");', "")
    clear, run = rest.split('        initialized_cycle = cycles;', 1)
    run = "        initialized_cycle = cycles;" + run.split("        if(bone_seen!=T*H", 1)[0]
    run = once(run, "timestep<T;", "timestep<1;")
    run = once(run, "            // M2 uses the same cache: restore the backbone working set each step.",
               "            // SCNN replaces Context0: explicitly restore both backbone matrices.")
    residency = "    integer resident_wh1_loads" + between(srnn, "    integer resident_wh1_loads", "    `undef RED")
    residency = residency.replace("if (i_rst_n) begin", "if (i_rst_n && cross_m2_active) begin")
    text = declarations + monitor + tasks + '''
    integer sequence_runs = 0;
    task automatic profile_enter(input string phase_name);
    endtask
    task automatic configure_once;
        learning = 0;
''' + load + config + '''
    endtask
    task automatic run_first_step;
        initialized_cycle = cycles;
        cross_m2_active = 1;
        i_parameter_owner = 3'b010;
        i_topology_learning_enable = 0;
        i_learning_srnn_sequence_version = 16'(sequence_runs+1);
        previous_spikes = 0; next_spikes = 0;
        bone_seen=0; hidden_seen=0; logit_seen=0; layers=0; captures=0; replays=0;
        done_count=0; commits=0; detaches=0; code_writes=0; residual_writes=0;
        gradient_seen=0; store_checks=0;
        i_state_maintenance_write_data=0; i_trace_maintenance_write_data=0;
''' + clear + run + '''
        while (!o_srnn_phase_idle || o_learning_srnn_busy || !i_store_quiescent)
            @(negedge i_clk);
        if (bone_seen!=450 || hidden_seen!=256 || logit_seen!=20 || store_checks!=120320 ||
            gradient_seen!=0 || commits!=0 || detaches!=0 || code_writes!=0 || residual_writes!=0)
            $fatal(1,"cross-profile M2 incomplete numeric/update coverage");
        sequence_runs++;
        if (resident_backbone_loads!=2*sequence_runs || resident_wh1_loads!=1 || resident_wout_loads!=1)
            $fatal(1,"cross-profile residency mismatch backbone=%0d wh1=%0d wout=%0d",
                resident_backbone_loads,resident_wh1_loads,resident_wout_loads);
        $display("CROSS_PROFILE_M2_PASS sequence=%0d bone=%0d hidden=%0d logits=%0d store=%0d backbone_loads=%0d wh1_loads=%0d wout_loads=%0d",
            sequence_runs,bone_seen,hidden_seen,logit_seen,store_checks,
            resident_backbone_loads,resident_wh1_loads,resident_wout_loads);
        $fflush();
        @(negedge i_clk);
        i_parameter_owner = 3'b100;
        i_topology_learning_enable = 3'b100;
        i_state_maintenance_result_ready = 0;
        i_trace_maintenance_result_ready = 0;
        cross_m2_active = 0;
    endtask
''' + residency + "    `undef RED\n    `undef LINK\n    `undef ENG\n    `undef M2\n"
    text = text.replace("u_final.", "dut.")
    text = text.replace("u_store.code_mem", "cross_code").replace("u_store.residual_mem", "cross_residual")
    text = re.sub(r"(?<!\.)\b([io]_[A-Za-z0-9_]+)\b", r"ft_\1", text)
    # The SCNN harness uses MEMORY_AW/WORK_AW for the same physical memories.
    text = re.sub(r"\bWORK_ADDR_WIDTH\b", "WORK_AW", text)
    return text


def transform(scnn, srnn):
    text = scnn_prefix(scnn)
    block = driver(srnn)
    aliases = dict(re.findall(r"assign (ft_i_\w+) = ([A-Za-z_]\w*);", text))
    # Reuse the existing maintenance and backbone response wires, not second drivers.
    for name, alias in aliases.items():
        if name in ("ft_i_clk", "ft_i_rst_n", "ft_i_clear_error", "ft_i_maintenance_tile") or "_maintenance_" in name:
            block = re.sub(r"\b" + name + r"\b", alias, block)
    driven = set(re.findall(r"\b(ft_i_\w+)\s*(?:<=|=(?!=))", block))
    continuous = set(re.findall(r"assign (ft_i_\w+) =", block))
    for name in sorted(driven):
        match = re.search(r"    assign " + name + r" = ([^;]+);", text)
        if match is None:
            raise ValueError("missing constant input assignment: " + name)
        value = match.group(1)
        if not re.fullmatch(r"(?:'0|1'b[01]|3'b[01]+)", value):
            raise ValueError("unexpected driven input alias: " + name + "=" + value)
        text = text.replace(match.group(0) + "\n", "", 1)
        if name in continuous:
            continue
        declaration = re.search(r"    logic[^;\n]*\b" + name + r";", text)
        if declaration is None:
            raise ValueError("missing input declaration: " + name)
        text = text.replace(declaration.group(0), declaration.group(0)[:-1] + " = " + value + ";", 1)
    # Only the M2 owner uses this separate test-host model namespace. The original
    # SCNN nine-bank responder, addresses and every SCNN assertion stay intact.
    text = once(text, "    assign ft_i_store_load_ready = 1'b0;", "    assign ft_i_store_load_ready = cross_m2_active;")
    anchor = "    always @(posedge clk) begin : temporal_external_ddr\n        int address, index, expected_index, bank;\n        if (rst_n) begin"
    text = once(text, anchor, '''    always @(posedge clk) begin : temporal_external_ddr
        int address, index, expected_index, bank;
        if (rst_n && cross_m2_active) begin
            if (fg_forward_valid && ft_o_store_forward_response_ready) fg_forward_valid <= 0;
            if (ft_o_store_load_valid && ft_i_store_load_ready) begin
                address=int'(ft_o_store_load_addr);
                if (address>=120320) $fatal(1,"M2 host load address out of range");
                cross_code[address]=ft_o_store_load_code;
                cross_residual[address]=ft_o_store_load_residual;
                cross_parameter_loads++;
            end
            if (ft_o_store_forward_read_valid && ft_i_store_forward_read_ready) begin
                address=int'(ft_o_store_forward_read_addr);
                if (address>=120320) $fatal(1,"M2 host forward address out of range");
                fg_forward_code<=cross_code[address];
                fg_forward_tag<=ft_o_store_forward_read_tag; fg_forward_valid<=1;
            end
            if (ft_o_store_learning_read_valid || ft_o_store_learning_write_valid)
                $fatal(1,"inference-only cross-profile M2 unexpectedly requested learning");
        end else if (rst_n) begin''')
    # These two observers describe SCNN-only counters. Pausing them outside SCNN
    # does not remove any SCNN numeric/protocol assertion or M2 checks above.
    for label in ("temporal_scoreboard", "profile_passive_measurement"):
        start = text.index("begin : " + label)
        tail = text[start:]
        if "        if (rst_n) begin" not in tail:
            raise ValueError("missing SCNN observer qualification: " + label)
        tail = tail.replace("        if (rst_n) begin", "        if (rst_n && !cross_m2_active) begin", 1)
        text = text[:start] + tail
    text = once(text, "        fg_load_expected();\n        fg_initialize_top();", '''        fg_load_expected();
        cross_m2_active=1;
        g_cross_m2.configure_once();
        g_cross_m2.run_first_step();
        fg_initialize_top();''')
    text = once(text, "        fg_finish();\n        $finish;", '''        fg_finish();
        g_cross_m2.run_first_step();
        if (cross_parameter_loads!=120320 || cross_reset_assertions!=1 || g_cross_m2.sequence_runs!=2)
            $fatal(1,"cross-profile used reset/reload or omitted a sequence");
        $display("CROSS_PROFILE_PASS resets=%0d m2_parameter_loads=%0d m2_sequences=%0d scnn_frames=4 scnn_descriptors=%0d scnn_learning=%0d",
            cross_reset_assertions,cross_parameter_loads,g_cross_m2.sequence_runs,fg_descriptors,fg_learn_on);
        $fflush();
        $finish;''')
    addition = '''
    bit cross_m2_active=0;
    integer cross_parameter_loads=0, cross_reset_assertions=0;
    bit cross_reset_observed=0;
    logic signed [CODE_WIDTH-1:0] cross_code[0:120319];
    logic signed [RESIDUAL_WIDTH-1:0] cross_residual[0:120319];
    always @(posedge clk) begin
        if (!rst_n && !cross_reset_observed) begin
            cross_reset_assertions++; cross_reset_observed=1;
        end else if (rst_n) cross_reset_observed=0;
    end
    generate if (1) begin : g_cross_m2
''' + block + "    end endgenerate\n"
    text = once(text, "endmodule\n", addition + "endmodule\n")
    return text


def parse_log(log, expected_scnn, learning):
    scnn = parse_scnn(log, expected_scnn, learning)
    def rows(marker):
        return [{k:int(v) for k,v in re.findall(r"(\w+)=(\d+)", line)}
                for line in log.splitlines() if line.startswith(marker + " ")]
    m2 = rows("CROSS_PROFILE_M2_PASS")
    wanted = [dict(sequence=i, bone=450, hidden=256, logits=20, store=120320,
                   backbone_loads=2*i, wh1_loads=1, wout_loads=1) for i in (1,2)]
    if m2 != wanted:
        raise ValueError("incomplete same-DUT M2 numerical/residency checks")
    final = rows("CROSS_PROFILE_PASS")
    if final != [dict(resets=1,m2_parameter_loads=120320,m2_sequences=2,
                      scnn_frames=4,scnn_descriptors=32,scnn_learning=int(learning))]:
        raise ValueError("cross-profile reset/load/sequence counts differ")
    first = log.index("CROSS_PROFILE_M2_PASS sequence=1 ")
    middle = log.index("SCNN_PROFILE_PASS ")
    last = log.index("CROSS_PROFILE_M2_PASS sequence=2 ")
    if not first < middle < last < log.index("CROSS_PROFILE_PASS "):
        raise ValueError("wrong cross-profile execution order")
    return dict(status="PASS_same_DUT_M2_off_SCNN_T4_M2_off_not_DMA_or_throughput", m2=m2, scnn=scnn, final=final[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="cross_profile_r1")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        parser.error("simple name required")
    scnn = (BASE / "server_pair_r2/tb_scnn_profile_measurement.sv").read_text(encoding="utf-8")
    srnn = (RESULTS / "srnn_specialized_r1/tb_srnn_profile_measurement.sv").read_text(encoding="utf-8")
    output = ROOT / "verification/experiments/srnn_resident_weight_context" / args.name
    text = transform(scnn, srnn)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tb_scnn_profile_measurement.sv").write_text(text, encoding="utf-8")
    (output / "testbench_changes.diff").write_text("".join(difflib.unified_diff(
        scnn.splitlines(True), text.splitlines(True), fromfile="accepted_SCNN_T20/tb", tofile="cross_profile/tb")), encoding="utf-8")
    scope = dict(status="PREPARED_NOT_SIMULATED", order=["SRNN_sample0_step0_off", "SCNN_sample0_T4_off_or_on", "SRNN_sample0_step0_off"],
                 source_candidate="srnn_specialized_r1", m2_shape=[450,256,20], m2_parameter_loads=120320,
                 resets=1, expected_backbone_refills=4, expected_wh1_refills=1, expected_wout_refills=1,
                 scope="One final-top DUT. Explicit Context0 backbone refills after SCNN; Context1 M2 weights retained. M2 inference only, SCNN cold off/on separately. Same original first-step and T4 goldens. Owner-selected host model namespaces, not a physical unified DDR/DMA experiment. No timing, throughput, full-Test or arbitrary abort claim.")
    manifest = json.loads((BASE / "fixed_sample0_b1t20_r1/manifest.json").read_text())
    original = json.loads((RESULTS / "srnn_specialized_r1/input_commands.json").read_text())
    scope.update(expected={arm:manifest["windows"][arm][0] for arm in ("off", "on")},
                 fixture="results/paper_evidence/supplement_20261004/performance/current_source_measurements/scnn/fixed_sample0_b1t20_r1",
                 srnn_fixture=original["fixture"])
    (output / "reference.json").write_text(json.dumps(scope, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scope, indent=2))


if __name__ == "__main__":
    main()
