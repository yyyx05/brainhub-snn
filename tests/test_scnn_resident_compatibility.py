import sys
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments"))
import prepare_scnn_resident_compatibility as prep
import srnn_resident_server_worker as worker


def transcript(learning):
    enabled=int(learning)
    expected=dict(rmw=17349*enabled,code_changes=0,residual_changes=17349*enabled)
    final=dict(learning=enabled,samples=1,frames=4,descriptors=32,requests=1,features=8192,
        currents=44,logits=44,traces=8192*enabled,probes=96,commits=enabled,publications=1892*(1+enabled),rmw=expected["rmw"])
    def row(marker,values): return marker+" "+" ".join(f"{k}={v}" for k,v in values.items())
    lines=[row("SCNN_PROFILE_PASS",final)]
    for frame in range(4):
        lines.append(row("SCNN_PROFILE_FRAME",dict(frame=frame,sample=0,timestep=frame)))
        for layer in range(1,9):
            for edge,cycle in (("start",10),("end",20)):
                lines.append(row("SCNN_PROFILE_DESCRIPTOR",dict(frame=frame,layer=layer,edge=edge,cycle=cycle)))
    traffic=dict(forward_reads=9,forward_responses=9,learning_reads=expected["rmw"],
        learning_responses=expected["rmw"],learning_writes=expected["rmw"],state_spill_requests=1,
        state_spill_rows=1,trace_spill_requests=1,trace_spill_rows=1,store_loads=0,
        code_write_words=0,residual_write_words=expected["residual_changes"])
    lines.append(row("SCNN_PROFILE_TRAFFIC",traffic))
    lines.extend(row("SCNN_PROFILE_DDR",dict(bank=i,forward_reads=1,forward_responses=1)) for i in range(9))
    lines.extend(row("H40_ACTIVITY",dict(tile=i)) for i in range(12))
    return "\n".join(lines)+"\n",expected


class SCNNCompatibilityTest(unittest.TestCase):
    def test_accepted_host_build_recipe_changes_only_paths(self):
        original=json.loads((prep.BASE/"server_pair_r2/build.status.json").read_text())["command"]
        before=original[:]
        command=worker.scnn_build_command(original,"/candidate/scripts/supplement_scnn_graph_host.mk")
        self.assertEqual(original,before)
        self.assertEqual(len(original),len(command))
        changed={i for i,(a,b) in enumerate(zip(original,command)) if a!=b}
        self.assertEqual(changed,{original.index(k)+1 for k in ("--Mdir","-f","-MAKEFLAGS")})
        for key,value in (("--prefix","Vscgraph"),("--output-groups","4"),("--output-split","5000"),("--output-split-cfuncs","2000")):
            self.assertEqual(command[command.index(key)+1],value)
        self.assertNotIn("-CFLAGS",command)

    def test_timeout_records_null_exit_and_actual_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);record={};command=["compiler","--binary"]
            with patch.object(worker,"ROOT",root),patch.object(worker,"RECORD",root/"status.json"),patch.object(worker.subprocess,"run",side_effect=subprocess.TimeoutExpired(command,12)):
                with self.assertRaises(subprocess.TimeoutExpired):worker.stage(record,"build",command,12)
            result=json.loads((root/"status.json").read_text())["build"]
            self.assertIsNone(result["exit"])
            self.assertTrue(result["timed_out"])
            self.assertEqual(result["timeout_seconds"],12)
            self.assertEqual(result["command"],command)

    def test_normal_stage_exit_is_not_reclassified(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);record={};command=["compiler"]
            with patch.object(worker,"ROOT",root),patch.object(worker,"RECORD",root/"status.json"),patch.object(worker.subprocess,"run",return_value=subprocess.CompletedProcess(command,7)):
                with self.assertRaises(RuntimeError):worker.stage(record,"build",command,12)
            self.assertEqual(record["build"]["exit"],7)
            self.assertNotIn("timed_out",record["build"])

    def test_prefix_keeps_numeric_checks_and_enum_fix(self):
        source=(prep.BASE/"server_pair_r2/tb_scnn_profile_measurement.sv").read_text(encoding="utf-8")
        changed=prep.transform(source)
        self.assertEqual(source.count("$fatal"),changed.count("$fatal"))
        self.assertIn("timestep_phase < 4;",changed)
        self.assertIn("frames=4 descriptors=",changed)
        self.assertIn("complete gradient/store/mask check",changed)
        self.assertIn("Verilator 5.048 faults in deadifyModules",changed)
        self.assertNotIn("force ",changed)
        self.assertNotIn(".u_weight_store.g_bank",changed)
        self.assertEqual(changed.count(".u_weight_store.g_context[0].g_enabled.u_store.g_bank"),2)

    def test_rejects_unknown_tb(self):
        with self.assertRaises(ValueError): prep.transform("module other; endmodule")

    def test_parses_complete_directed_transcript(self):
        for learning in (False,True):
            log,expected=transcript(learning)
            self.assertTrue(prep.parse_log(log,expected,learning)["status"].startswith("PASS_"))

    def test_rejects_partial_duplicate_or_bad_traffic(self):
        log,expected=transcript(True)
        for broken in (log.replace("frames=4","frames=20"),log.replace("state_spill_rows=1","state_spill_rows=0"),
                log.replace("residual_write_words=17349","residual_write_words=0"),
                log+"SCNN_PROFILE_FRAME frame=0 sample=0 timestep=0\n",
                log.replace("H40_ACTIVITY tile=11\n",""),log+"%Error: failed\n"):
            with self.assertRaises(ValueError): prep.parse_log(broken,expected,True)


if __name__=="__main__": unittest.main()
