#!/usr/bin/env python3
"""Real XPM-library module characterization; not a full-core simulation."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


def transform_wrapper(source):
    old = "module experimental_dual_context_weight_store_p4all_v1 #("
    if source.count(old) != 1:
        raise ValueError("unexpected original wrapper")
    result = source.replace(old, "module experimental_dual_context_weight_store_compact_m2 #(\n    parameter int M2_DEPTH = 578,")
    anchor = "            unified_dual_role_weight_store_p4all_candidate_v1 #("
    if result.count(anchor) != 1:
        raise ValueError("expected one original context store")
    result = result.replace(anchor, """            if (ctx_idx == 1) begin : g_compact
                m2_compact_store #(.DEPTH(M2_DEPTH)) u_store (
                    .i_clk(i_clk), .i_rst_n(i_rst_n),
                    .i_fill_valid(i_fill_valid && i_fill_context == ctx_idx),
                    .o_fill_ready(fill_ready[ctx_idx]),
                    .i_fill_bank(i_fill_bank), .i_fill_addr(i_fill_addr),
                    .i_fill_data(i_fill_data),
                    .i_read_valid(i_read_valid && request_slot_ready && read_context == ctx_idx),
                    .o_read_ready(read_ready[ctx_idx]), .i_read_addr(i_read_addr0),
                    .o_read_result_valid(read_result_valid[ctx_idx]),
                    .i_read_result_ready(i_read_result_ready && read_context_q == ctx_idx),
                    .o_term0_flat(term0_flat[ctx_idx])
                );
                assign term1_flat[ctx_idx] = '0;
                assign term2_flat[ctx_idx] = '0;
                assign term3_flat[ctx_idx] = '0;
            end else begin : g_original
"""+anchor)
    end = "            end else begin : g_unused"
    if result.count(end) != 1:
        raise ValueError("unexpected original context branch")
    return result.replace(end, "            end\n"+end)


def parse_run(text, variant):
    if re.search(r"(?im)^\s*(?:fatal|error):|\$fatal|assertion failed", text):
        raise ValueError("XSim fatal/error")
    fields = ("variant", "depth", "reads", "fills", "stalls", "conflicts", "replacements",
              "held_read", "held_fill", "hazard_bad_cycles", "hazard_bad_terms")
    pattern = "^XPM_CONTEXT_RESULT " + " ".join(k+r"=(\d+)" for k in fields) + "$"
    found = re.findall(pattern, text, re.M)
    if len(found) != 1:
        raise ValueError("missing or duplicate result")
    result = dict(zip(fields, map(int, found[0])))
    depth = 512 if variant == 2 else 578
    if result["variant"] != variant or result["depth"] != depth:
        raise ValueError("wrong variant/capacity")
    # The common sequence covers every legal address in both contexts.
    if result["reads"] != 578+depth+77 or result["fills"] != (578+depth)*4+6:
        raise ValueError("incomplete transactions")
    if any(result[k] <= 0 for k in ("stalls", "conflicts", "replacements", "held_read", "held_fill")):
        raise ValueError("missing protocol exercise")
    markers = [f"XPM_CONTEXT_CAPACITY variant={variant} context0=578 context1={depth}",
               f"XPM_CONTEXT_COMMON_PASS variant={variant}",
               "XPM_CONTEXT_RESULT ", f"XPM_CONTEXT_COMPLETE variant={variant}"]
    if any(text.count(m) != 1 for m in markers) or [text.index(m) for m in markers] != sorted(text.index(m) for m in markers):
        raise ValueError("completion/event order")
    if variant and (result["hazard_bad_cycles"] or result["hazard_bad_terms"]):
        raise ValueError("candidate pending payload changed")
    result["contract_status"] = "FAIL_pending_fill_payload" if result["hazard_bad_cycles"] else "PASS_exercised_scope"
    return result


def worker():
    from run_fc_bitmap_mapping import vivado_environment
    vivado = "/root/autodl-tmp/Xilinx_2022_2/Vivado/2022.2/bin/vivado"
    env = vivado_environment(vivado)
    tool_dir = Path(vivado).parent
    root = Path.cwd()
    status = {"status": "running", "scope": "real_XPM_module_dual_context_not_core", "variants": []}

    def save():
        (root/"status.json").write_text(json.dumps(status, indent=2)+"\n")

    try:
        for variant in range(3):
            folder = root/f"variant{variant}"
            folder.mkdir()
            row = {"variant": variant, "stages": {}}
            status["variants"].append(row)
            commands = {
                "compile": [str(tool_dir/"xvlog"), "-sv", "-d", "UNIFIED_USE_XPM_MEMORY", "-d", f"TEST_VARIANT={variant}",
                    "../baseline.sv", "../original_wrapper.sv", "../m2_compact_store.sv",
                    "../candidate_wrapper.sv", "../tb_m2_xpm_context.sv"],
                "elaborate": [str(tool_dir/"xelab"), "-L", "xpm", "tb_m2_xpm_context", "-s", "context_test", "--mt", "2", "--timescale", "1ns/1ps"],
                "run": [str(tool_dir/"xsim"), "context_test", "-runall"],
            }
            for name, command in commands.items():
                status["status"] = f"variant{variant}_{name}_running"
                save()
                start = time.monotonic()
                with (folder/(name+"_stdout.log")).open("w") as stream:
                    p = subprocess.run(command, cwd=folder, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, timeout=600)
                row["stages"][name] = {"exit": p.returncode, "seconds": time.monotonic()-start, "command": command}
                save()
                log = (folder/(name+"_stdout.log")).read_text(errors="replace")
                if p.returncode or re.search(r"(?m)^ERROR:", log):
                    raise RuntimeError(f"variant{variant} {name} failed")
            row["checks"] = parse_run(log, variant)
            (folder/"checks.json").write_text(json.dumps(row["checks"],indent=2)+"\n")
        status["status"] = ("diagnostic_complete_baseline_hold_failure_candidates_pass"
            if status["variants"][0]["checks"]["hazard_bad_cycles"] else "PASS_exercised_XPM_scope")
    except Exception as error:
        status.update(status="FAIL", error=str(error))
    save()


def prepare():
    from run_srnn_resident_remote import ROOT, RESULT, local_path
    original = Path(local_path(ROOT/RESULT/"scnn_issue_banked_synth_r1/rtl/memory/experimental_dual_context_weight_store_p4all_v1.sv"))
    out = ROOT/"verification/experiments/m2_xpm_context/candidate_r1/candidate_wrapper.sv"
    Path(local_path(out)).write_text(transform_wrapper(original.read_text()), encoding="utf-8")
    return out


def launch(name):
    from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, upload, local_path
    wrapper = prepare()
    target = REMOTE+"/"+name
    source = ROOT/RESULT/"scnn_issue_banked_synth_r1/rtl/memory"
    files = {
        "baseline.sv": source/"unified_dual_role_weight_store_p4all_candidate_v1.sv",
        "original_wrapper.sv": source/"experimental_dual_context_weight_store_p4all_v1.sv",
        "m2_compact_store.sv": ROOT/"verification/experiments/m2_storage_shape/candidate_r1/m2_compact_store.sv",
        "candidate_wrapper.sv": wrapper,
        "tb_m2_xpm_context.sv": wrapper.with_name("tb_m2_xpm_context.sv"),
        "worker.py": Path(__file__),
        "run_fc_bitmap_mapping.py": Path(__file__).with_name("run_fc_bitmap_mapping.py"),
    }
    c = connect()
    try:
        process = run(c,"ps -eo comm,args")
        if any(re.search(r"^(?:vivado|xsim|xelab|xvlog)(?:\s|$)",s.strip()) for s in process.splitlines()):
            raise RuntimeError("another EDA process is active")
        disk = run(c,"df -B1 /root/autodl-tmp")
        if int(disk.splitlines()[-1].split()[3]) < 300_000_000:
            raise RuntimeError("insufficient room for small XPM test")
        with c.open_sftp() as s:
            s.mkdir(target)
            for name_on_server, file in files.items():
                upload(s, local_path(file), target+"/"+name_on_server)
            with s.open(target+"/launch_environment.txt", "w") as f:
                f.write(disk+"\n"+process)
        command = "cd "+shlex.quote(target)+" && exec python3 worker.py --worker"
        run(c,"screen -dmS "+shlex.quote("snn_"+name)+" bash -lc "+shlex.quote(command))
        print(json.dumps({"remote":target,"launched":True,"disk":disk}))
    finally:
        c.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--name", default="m2_xpm_context_r1")
    args = parser.parse_args()
    if args.worker:
        worker()
    elif args.prepare_only:
        print(prepare())
    else:
        launch(args.name)
