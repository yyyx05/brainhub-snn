"""Prepare a single-credit turnover candidate from the accepted compact source."""
import difflib
import argparse
import json
from pathlib import Path

from run_srnn_resident_remote import ROOT, RESULT, local_path

EXPERIMENT = Path("verification/experiments/scnn_forward_turnover")
OUTPUT = EXPERIMENT / "candidate_r1"
PUBLISHER = "unified_scnn_classifier_cache_refill_publisher_v1"
BACKEND = "unified_scnn_selected_layers_parameter_backend_v1"
ROUTER = "unified_tp_three_client_parameter_backend_router"
SOURCES = [Path("rtl/learning") / (name + ".sv") for name in (PUBLISHER, BACKEND, ROUTER)]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("source anchor changed: " + old)
    return text.replace(old, new, 1)


def publisher(text):
    text = replace_once(text, "    logic response_fire;", "    logic response_fire;\n    logic turnover_valid;")
    text = replace_once(text, "    always_comb begin\n        // Issue the current valid scalar directly;",
        """    // One response retires before its replacement becomes outstanding.
    assign turnover_valid = state_q == STATE_READ_RESPONSE &&
        i_forward_response_valid && !i_forward_response_error &&
        i_forward_response_tag == read_tag_q && !publisher_abort &&
        subword_q < ROW_SUBWORDS - 1 &&
        feature_index_int + 2 < integer'(feature_count_q);

    always_comb begin
        // Issue the current valid scalar directly;""")
    text = replace_once(text, "feature_index_int < integer'(feature_count_q);\n        o_forward_read_addr",
        "feature_index_int < integer'(feature_count_q) || turnover_valid;\n        o_forward_read_addr")
    text = replace_once(text, "        o_forward_read_tag = read_tag_q;",
        """        o_forward_read_tag = read_tag_q;
        if (turnover_valid) begin
            o_forward_read_addr = PARAM_ADDR_WIDTH'(parameter_addr_int + 2);
            o_forward_read_tag = read_tag_q + 1'b1;
        end""")
    text = replace_once(text, "                                state_q <= STATE_GATHER;",
        "                                state_q <= read_fire ? STATE_READ_RESPONSE : STATE_GATHER;")
    return text


def backend(text):
    text = replace_once(text, "    logic conv_forward_response_fire;",
        "    logic conv_forward_response_fire;\n    logic conv_forward_slot;")
    text = replace_once(text, "    assign store_forward_read_valid = publisher_busy ?",
        """    assign conv_forward_slot = !external_forward_outstanding_q ||
        conv_forward_response_fire;
    assign store_forward_read_valid = publisher_busy ?""")
    text = replace_once(text, "!i_learning_write_valid && !external_forward_outstanding_q);",
        "!i_learning_write_valid && conv_forward_slot);")
    text = replace_once(text, "        !external_forward_outstanding_q && store_forward_read_ready;",
        "        conv_forward_slot && store_forward_read_ready;")
    return text


def router(text):
    old = "!forward_pending_q"
    new = "(!forward_pending_q || (forward_response_fire && i_active_owner == forward_owner_q))"
    for prefix in ("o_store_forward_read_valid=i_forward_read_valid[active_index]&&",
                   "o_forward_read_ready[active_index]=i_store_forward_read_ready&&"):
        text = replace_once(text, prefix + old + ";", prefix + new + ";")
    return text


def testbench(text):
    text = replace_once(text, "module publisher_checker #(parameter bit OPT = 0)",
        "module publisher_checker #(parameter bit OPT = 0, parameter bit TURNOVER = 0)")
    text = replace_once(text, "    int expected_addresses[32768];",
        "    logic [1:0] owner = 3;\n    logic router_error;\n    int replacements;\n    int expected_addresses[32768];")
    text = text.replace("unified_scnn_classifier_cache_refill_publisher_v1 #(.TAG_WIDTH(4))",
        "turnover_chain #(.OPT(1), .TAG_WIDTH(4))")
    text = text.replace("reference_scnn_publisher #(.TAG_WIDTH(4))",
        "turnover_chain #(.OPT(0), .TAG_WIDTH(4))")
    text = replace_once(text, "    .i_clk(clk),", "    .i_test_owner(owner), .o_router_error(router_error), .i_clk(clk),")
    text = replace_once(text,
        "        req_ready = !pending && !hold_request && (!stalls || cycles % 5 != 0);\n        resp_valid = pending && pending_delay == 0 && !hold_response;",
        """        resp_valid = pending && pending_delay == 0 && !hold_response;""")
    text = replace_once(text, "        #5;\n        request_fire",
        """        #1;
        req_ready = (!pending || (TURNOVER && resp_valid && resp_ready)) &&
            !hold_request && (!stalls || cycles % 5 != 0);
        #4;
        request_fire""")
    text = replace_once(text, "                if (pending || reads >= total_reads || accepted_address != expected_addresses[reads] || accepted_tag != (tag_model & 15))",
        """                if ((pending && !response_fire) || reads >= total_reads ||
                    accepted_address != expected_addresses[reads] ||
                    accepted_tag != ((tag_model + (response_fire ? 1 : 0)) & 15))""")
    text = replace_once(text, "        accepted_address = int'(req_addr);",
        "        if (request_fire && response_fire) replacements++;\n        accepted_address = int'(req_addr);")
    text = replace_once(text, 'SCNN_PUBLISHER_CASE opt=%0d case_id=',
        'SCNN_TURNOVER_CASE service=%0d opt=%0d case_id=')
    text = replace_once(text, 'OPT, cases, f, c, use_stalls, cycles-start_cycle, reads, rows);',
        'TURNOVER, OPT, cases, f, c, use_stalls, cycles-start_cycle, reads, rows);')
    text = replace_once(text, 'SCNN_PUBLISHER_CHECKER_PASS opt=%0d cases=',
        'SCNN_TURNOVER_CHECKER_PASS service=%0d replacements=%0d opt=%0d cases=')
    text = replace_once(text, 'OPT, cases, total_checked_reads, total_checked_rows);',
        'TURNOVER, replacements, OPT, cases, total_checked_reads, total_checked_rows);')
    text = replace_once(text, "        configure(0, 1, 0);",
        """        // Router ownership must not switch while a response is pending.
        configure(25, 2, 0);
        hold_response = 1;
        start();
        while (!pending) tick();
        owner = 2; tick();
        if (!router_error) $fatal(1, "missing pending owner violation");
        owner = 3; abort_req = 1; tick(); abort_req = 0;
        hold_response = 0; clear_error = 1; tick(); clear_error = 0;
        if (busy || sticky_error || router_error) $fatal(1, "owner recovery not idle");
        configure(0, 1, 0);""")
    start = text.index("module tb_scnn_publisher_issue;")
    end = text.index("`undef PUBLISHER_PORTS", start)
    text = text[:start] + """module tb_scnn_forward_turnover;
    wire [3:0] finished;
    publisher_checker #(.OPT(0), .TURNOVER(0)) a(finished[0]);
    publisher_checker #(.OPT(1), .TURNOVER(0)) b(finished[1]);
    publisher_checker #(.OPT(0), .TURNOVER(1)) c(finished[2]);
    publisher_checker #(.OPT(1), .TURNOVER(1)) d(finished[3]);
    initial begin
        wait (&finished);
        $display("SCNN_TURNOVER_UNIT_PASS variants=2 services=2 cases_per_checker=14 owner_errors_per_checker=1");
        $finish;
    end
endmodule
""" + text[end:]
    return text


def geometry_wrapper(text):
    text = text.replace("[14:0]", "[16:0]").replace("[11:0]", "[12:0]")
    text = text.replace("15'b0", "17'b0").replace("12'b0", "13'b0")
    text = text.replace("45'b0", "51'b0").replace("30'b0", "34'b0")
    text = text.replace(".PARAM_ADDR_WIDTH(15)", ".PARAM_ADDR_WIDTH(17)")
    text = text.replace("#(.TAG_WIDTH(TAG_WIDTH)) publisher",
        "#(.TAG_WIDTH(TAG_WIDTH),.MAX_FEATURES(4608),.PARAM_DEPTH(131072)) publisher")
    text = text.replace(".USE_INTERNAL_STORE(0),.MAX_CLASSES(16)",
        ".USE_INTERNAL_STORE(0),.MAX_CLASSES(16),.MAX_CLASSIFIER_FEATURES(4608),.PARAM_DEPTH(131072)")
    return text


def geometry_testbench(text):
    text = text.replace("parameter bit TURNOVER = 0", "parameter int SERVICE = 0")
    text = replace_once(text, "    logic clk = 0", """    localparam bit TURNOVER = SERVICE != 0;
    localparam int MIN_RESPONSE_LATENCY = SERVICE == 2 ? 4 : SERVICE == 3 ? 12 : 1;
    logic clk = 0""")
    text = text.replace("[14:0]", "[16:0]").replace("[11:0]", "[12:0]").replace("[18:0]", "[20:0]")
    text = text.replace("15'(", "17'(").replace("12'(", "13'(")
    text = text.replace("expected_addresses[32768]", "expected_addresses[131072]")
    text = text.replace("[4096]", "[8192]")
    text = text.replace("pending_index % 3 : 0", "pending_index % 3 : MIN_RESPONSE_LATENCY - 1")
    text = text.replace("TURNOVER, OPT, cases", "SERVICE, OPT, cases")
    text = text.replace("TURNOVER, replacements, OPT", "SERVICE, replacements, OPT")
    text = text.replace("run_case(2048, 11, 0)", "run_case(4608, 16, 0)")
    text = text.replace("cycles > 1000000", "cycles > 4000000")
    start = text.index("module tb_scnn_forward_turnover;")
    end = text.index("`undef PUBLISHER_PORTS", start)
    top = ["module tb_scnn_forward_turnover;", "    wire [7:0] finished;"]
    for service in range(4):
        for opt in (0, 1):
            index = service * 2 + opt
            top.append(f"    publisher_checker #(.OPT({opt}), .SERVICE({service})) p{index}(finished[{index}]);")
    top += ["    initial begin", "        wait (&finished);",
            '        $display("SCNN_TURNOVER_UNIT_PASS variants=2 services=4 cases_per_checker=14 owner_errors_per_checker=1");',
            "        $finish;", "    end", "endmodule", ""]
    return text[:start] + "\n".join(top) + text[end:]


def prepare(name="candidate_r1", geometry=False):
    source_root = Path(local_path(ROOT / RESULT / "m2_compact_synth_r2"))
    out = Path(local_path(ROOT / EXPERIMENT / name))
    out.mkdir(parents=True, exist_ok=False)
    changes = []
    for relative, transform, name in zip(SOURCES, (publisher, backend, router), (PUBLISHER, BACKEND, ROUTER)):
        original = (source_root / relative).read_text(encoding="utf-8")
        changed = transform(original)
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(changed, encoding="utf-8", newline="\n")
        reference = original.replace(name, "reference_" + name)
        if name == BACKEND:
            reference = reference.replace(PUBLISHER, "reference_" + PUBLISHER)
        (out / ("reference_" + name + ".sv")).write_text(reference, encoding="utf-8", newline="\n")
        changes.extend(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                       fromfile="accepted/" + relative.as_posix(), tofile="candidate/" + relative.as_posix()))
    dispatcher = Path("rtl/memory/unified_array16_weight_refill_dispatcher_v1.sv")
    (out / dispatcher.name).write_bytes((source_root / dispatcher).read_bytes())
    wrapper = Path(local_path(ROOT / EXPERIMENT / "turnover_chain.sv"))
    if geometry:
        (out / wrapper.name).write_text(geometry_wrapper(wrapper.read_text()), encoding="utf-8", newline="\n")
    else:
        (out / wrapper.name).write_bytes(wrapper.read_bytes())
    old_tb = Path(local_path(ROOT / "verification/experiments/scnn_publisher_issue/tb_scnn_publisher_issue.sv"))
    tb = testbench(old_tb.read_text(encoding="utf-8"))
    if geometry:
        tb = geometry_testbench(tb)
    (out / "tb_scnn_forward_turnover.sv").write_text(tb, encoding="utf-8", newline="\n")
    (out / "source_changes.diff").write_text("".join(changes), encoding="utf-8", newline="\n")
    record = dict(source="accepted m2_compact_synth_r2", changed_sources=[p.as_posix() for p in SOURCES],
        added_payload_bits=0, added_outstanding_entries=0, added_register_bits=0, host_payload_bits=16,
        boundary="Single-credit replacement only within a nonfinal row subword. Same scalar service per comparison arm. Source-level storage ledger, not mapped area or whole-core performance. Harness flushes service/router on module abort; no final-top abort claim.")
    record.update(max_features=4608 if geometry else 2048, address_width=17 if geometry else 15,
                  services=4 if geometry else 2, minimum_response_latencies=[1,1,4,12] if geometry else [1,1])
    (out / "preparation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name")
    parser.add_argument("--geometry",action="store_true")
    args=parser.parse_args()
    print(prepare(args.name or ("candidate_geometry_r2" if args.geometry else "candidate_r1"),args.geometry))
