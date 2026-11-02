"""Reuse the reducer's accumulator and output register as two elastic stages."""
import collections
import datetime as dt
import difflib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import run_gradient_retire_local as prior

ROOT = prior.ROOT
NAME = "gradient_elastic_local_r1"
PREP = ROOT / "verification/experiments/gradient_elastic_r1"
OUT = ROOT / prior.REL_OUT.parent / NAME
MODULE = prior.MODULE
BASE = ROOT / "verification/experiments/final_raw_source_functional_r1/rtl/learning" / MODULE
MODES = 6


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("nonunique source anchor: " + old)
    return text.replace(old, new, 1)


def candidate(text):
    text = replace_once(text, "    logic retire_and_accept;", "    logic finalize_transfer;")
    text = replace_once(text, "    logic [PARAM_ADDR_WIDTH-1:0] input_addr;\n", "")
    text = replace_once(text,
        "        retire_and_accept = pending_q && i_gradient_ready && !gradient_last_q;\n"
        "        input_addr = retire_and_accept ? expected_addr_q + 1'b1 : expected_addr_q;",
        "        finalize_transfer = finalize_q && (!pending_q || i_gradient_ready);")
    text = text.replace("input_addr", "expected_addr_q")
    text = replace_once(text,
        "        o_contribution_ready = active_q && !finalize_q &&\n"
        "            (!pending_q || retire_and_accept) &&",
        "        o_contribution_ready = active_q && (!finalize_q || finalize_transfer) &&")
    text = replace_once(text,
        "        accumulated_with_current = accumulator_q + contribution_extended;",
        "        accumulated_with_current = (finalize_q ? '0 : accumulator_q) +\n"
        "            contribution_extended;")
    begin = text.index("            if (input_fire) begin")
    end = text.index("            if (i_abort &&", begin)
    text = text[:begin] + """            if (output_fire) begin
                pending_q <= 1'b0;
                if (gradient_last_q)
                    o_done <= 1'b1;
            end

            // Drain the completed accumulator only when the result slot is free.
            // expected_addr_q already points past that completed address.
            if (finalize_transfer) begin
                finalize_q <= 1'b0;
                pending_q <= 1'b1;
                gradient_addr_q <= expected_addr_q - 1'b1;
                gradient_last_q <= expected_addr_q == element_count_q;
                gradient_q <= rounded_shifted[GRADIENT_WIDTH-1:0];
                accumulator_q <= '0;
                if (rounded_overflow)
                    o_overflow <= 1'b1;
            end

            // A simultaneous input owns the reused accumulator after transfer.
            if (input_fire) begin
                if (!input_legal) begin
                    active_q <= 1'b0;
                    o_protocol_error <= 1'b1;
                    o_done <= 1'b1;
                end else begin
                    accumulator_q <= accumulated_with_current;
                    if (input_expected_last_sample) begin
                        finalize_q <= 1'b1;
                        expected_addr_q <= expected_addr_q + 1'b1;
                        expected_sample_q <= '0;
                        if (input_expected_tensor_last)
                            active_q <= 1'b0;
                    end else begin
                        expected_sample_q <= expected_sample_q + 1'b1;
                    end
                end
            end

""" + text[end:]
    return text


def testbench(text):
    text = text.replace("GRADIENT_RETIRE", "GRADIENT_ELASTIC")
    text = replace_once(text,
        "integer sent, got, cycles, input_stalls, output_stalls, overlaps;",
        "integer sent, got, cycles, input_stalls, output_stalls, overlaps;\n"
        "        integer finalize_inputs, held_inputs, double_holds;")
    text = replace_once(text, "held = 0; last_infire = 0; expected_ov = 0;",
        "held = 0; last_infire = 0; expected_ov = 0;\n"
        "        finalize_inputs = 0; held_inputs = 0; double_holds = 0;")
    text = replace_once(text, "gr = (mode != 2) || ((cycles % 11) >= 4);",
        "gr = (mode != 2 || (cycles % 11) >= 4) &&\n"
        "                 (mode != 3 || (cycles % 3) == 0) &&\n"
        "                 (mode != 4 || (cycles % 37) >= 19);")
    text = replace_once(text,
        "iv = sent < b * 20 && ((mode != 1) || (cycles % 2 == 0));",
        "iv = sent < b * 20 && (mode != 1 || cycles % 2 == 0) &&\n"
        "                     (mode != 5 || cycles % 13 < 3);")
    text = replace_once(text, "if (infire && outfire) overlaps++;",
        "if (infire && outfire) overlaps++;\n"
        "            if (infire && dut.finalize_q) finalize_inputs++;\n"
        "            if (infire && gv && !gr) held_inputs++;\n"
        "            if (dut.finalize_q && gv && !gr) double_holds++;")
    text = replace_once(text,
        'overlaps=%0d", arm, b, mode, sent, got, cycles, input_stalls, output_stalls, overlaps);',
        'overlaps=%0d finalize_inputs=%0d held_inputs=%0d double_holds=%0d", '
        "arm, b, mode, sent, got, cycles, input_stalls, output_stalls, overlaps, "
        "finalize_inputs, held_inputs, double_holds);")
    # Check two live slots, rather than only the old single pending result.
    anchor = '        $display("GRADIENT_ELASTIC_DIRECTED arm=%0d cases=9", arm);'
    extra = """        if (arm == 1) begin
            configure(1, 3); send_first(0); wait_gradient();
            @(negedge clk); iv = 1; addr = 1; sample_id = 0; last = 0;
            #1; if (!ir) $fatal(1, "free accumulator blocked by held output");
            @(posedge clk); #1;
            @(negedge clk); iv = 0;
            repeat (4) begin
                @(posedge clk); #1;
                if (!gv || ga != 0 || gq != 1 || !dut.finalize_q || !busy)
                    $fatal(1, "two-slot hold corrupted");
            end
            @(negedge clk); abort_i = 1;
            @(posedge clk); #1;
            if (gv || busy || !done) $fatal(1, "two-slot abort failed");

            configure(1, 2); send_first(0); wait_gradient();
            @(negedge clk); iv = 1; addr = 1; sample_id = 0; last = 1;
            @(posedge clk); #1;
            @(negedge clk); iv = 0; rst_n = 0; #1;
            if (gv || busy || ov || pe) $fatal(1, "two-slot reset failed");
            $display("GRADIENT_ELASTIC_TWO_SLOT_PASS cases=2");
        end
"""
    text = replace_once(text, anchor, extra + anchor)
    text = replace_once(text, "mode < 3", "mode < 6")
    text = replace_once(text, "cases=24 directed=9 inputs=22860 outputs=480",
                        "cases=48 directed=9 inputs=45720 outputs=960")
    return text


def parse(log, arm):
    if re.search(r"%Error|%Fatal|Assertion failed|\bFATAL\b", log):
        raise ValueError("runtime diagnostic")
    rows = [dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", line))
            for line in re.findall(r"^GRADIENT_ELASTIC_CASE (.*)$", log, re.M)]
    if [(r["arm"], r["batch"], r["mode"]) for r in rows] != [
            (arm, b, m) for b in prior.BATCHES for m in range(MODES)]:
        raise ValueError("case coverage")
    if any(r["inputs"] != r["batch"] * 20 or r["outputs"] != 20 for r in rows):
        raise ValueError("transaction counts")
    if log.count(f"GRADIENT_ELASTIC_PASS arm={arm} cases=48 directed=9 inputs=45720 outputs=960") != 1:
        raise ValueError("completion marker")
    if log.count(f"GRADIENT_ELASTIC_DIRECTED arm={arm} cases=9") != 1:
        raise ValueError("directed marker")
    if arm == 1:
        if log.count("GRADIENT_ELASTIC_TWO_SLOT_PASS cases=2") != 1:
            raise ValueError("two-slot directed coverage")
        for field in ("finalize_inputs", "held_inputs", "double_holds", "output_stalls"):
            if not sum(r[field] for r in rows):
                raise ValueError("missing coverage: " + field)
    return rows


def prepare():
    PREP.mkdir(exist_ok=False)
    for arm in ("original", "candidate"):
        (PREP / arm).mkdir()
    original = BASE.read_text()
    (PREP / "original" / MODULE).write_bytes(BASE.read_bytes())
    changed = candidate(original)
    (PREP / "candidate" / MODULE).write_text(changed, newline="\n")
    tb = Path(__file__).with_name("tb_gradient_retire.sv")
    (PREP / "tb_gradient_retire.sv").write_text(testbench(tb.read_text()), newline="\n")
    (PREP / "changes.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True), fromfile="accepted/" + MODULE,
        tofile="candidate/" + MODULE)))
    shutil.copy2(__file__, PREP / "local_runner.py")
    shutil.copy2(Path(prior.__file__), PREP / "fixture_generator.py")
    print(json.dumps(dict(status="PREPARED_NOT_RUN", path=str(PREP))))


def run():
    OUT.mkdir(exist_ok=False)
    for p in PREP.rglob("*"):
        if p.is_file():
            dst = OUT / p.relative_to(PREP)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    state = dict(status="preparing", start=dt.datetime.now().astimezone().isoformat(),
        scope="New two-stage elastic reducer versus accepted retirement-overlap reducer; "
              "same accumulator/output registers, arithmetic, ports and independent integer oracle. "
              "No new memory/operator, no whole-core or STA conclusion.",
        verilator="5.048", memory_limit_kib=3145728, jobs=1)
    def save():
        (OUT / "status.json").write_text(json.dumps(state, indent=2) + "\n")
    save()
    before = (BASE.stat().st_size, BASE.stat().st_mtime_ns)
    try:
        prior.OUT = OUT
        state["fixture"] = prior.fixture()
        native = prior.NATIVE_ROOT
        linux_root = "/mnt/" + str(native)[0].lower() + str(native)[2:].replace("\\", "/")
        linux = linux_root + "/" + OUT.relative_to(ROOT).as_posix()
        runtime = linux_root + "/tools/verilator_5_048_local"
        executable = shlex.quote(runtime + "/bin/verilator")
        prefix = "VERILATOR_ROOT=" + shlex.quote(runtime + "/share/verilator") + " "
        stages = [("environment", prefix + executable + " --version")]
        for arm in ("original", "candidate"):
            stages += [
                (arm + "_build", prefix + f"/usr/bin/time -v -o {arm}_build_resources.txt " +
                 executable + " --binary --timing --assert -j 1 -Wall -Wno-fatal " +
                 f"--top-module tb_gradient_retire --Mdir {arm}_obj {arm}/{MODULE} tb_gradient_retire.sv"),
                (arm + "_run", f"/usr/bin/time -v -o {arm}_run_resources.txt " +
                 f"{arm}_obj/Vtb_gradient_retire +ARM={int(arm == 'candidate')} +VECTORS=vectors.hex")]
        for stage, command in stages:
            state["status"] = stage + "_running"
            cmd = ["wsl.exe", "-e", "sh", "-lc",
                   "cd " + shlex.quote(linux) + " && ulimit -v 3145728 && " +
                   command + " > " + stage + ".log 2>&1"]
            start = time.monotonic()
            with (OUT / (stage + "_launcher.log")).open("xb") as stream:
                process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                state[stage] = dict(command=cmd, pid=process.pid, exit=None)
                save()
                try:
                    code = process.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    state[stage]["timeout_pid_preserved"] = True
                    save()
                    raise
            state[stage].update(exit=code, wall_seconds=time.monotonic() - start)
            save()
            log = (OUT / (stage + ".log")).read_text(errors="replace")
            if code:
                raise RuntimeError(stage + " exit " + str(code))
            if stage == "environment" and not re.search(r"Verilator 5\.048\b", log):
                raise ValueError("wrong runtime version")
            if stage.endswith("_build"):
                counts = collections.Counter(re.findall(r"^%Warning-([^:]+):", log, re.M))
                state[stage]["warnings"] = dict(counts)
                if set(counts) & {"LATCH", "MULTIDRIVEN", "UNOPTFLAT"} or "%Error" in log:
                    raise ValueError("structural/build diagnostic")
            if stage.endswith("_run"):
                state[stage]["parsed"] = parse(log, int(stage.startswith("candidate")))
        state["comparisons"] = [dict(batch=a["batch"], mode=a["mode"], baseline_cycles=a["cycles"],
            candidate_cycles=b["cycles"], saved_cycles=a["cycles"] - b["cycles"])
            for a, b in zip(state["original_run"]["parsed"], state["candidate_run"]["parsed"])]
        if before != (BASE.stat().st_size, BASE.stat().st_mtime_ns):
            raise ValueError("accepted source changed")
        for p in PREP.rglob("*"):
            if p.is_file() and p.read_bytes() != (OUT / p.relative_to(PREP)).read_bytes():
                raise ValueError("source archive mismatch")
        state["status"] = "PASS_two_stage_elastic_reducer_local"
    except Exception as error:
        state.update(status="FAIL", error=repr(error))
        raise
    finally:
        state["end"] = dt.datetime.now().astimezone().isoformat()
        save()
        print(json.dumps({k: v for k, v in state.items() if k not in
                          ("original_run", "candidate_run", "comparisons")}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "run"):
        raise SystemExit("usage: run_gradient_elastic_local.py prepare|run")
    globals()[sys.argv[1]]()
