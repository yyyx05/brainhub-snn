import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments"))
import prepare_scnn_schedule_probe as prep
import srnn_resident_server_worker as worker


def row(marker,values):
    return marker+" "+" ".join(f"{k}={v}" for k,v in values.items())


def transcript():
    reference=prep.first_frame_reference((prep.ACCEPTED/"run_off.log").read_text())
    lines=[row("SCNN_PROFILE_DESCRIPTOR",r) for r in reference["descriptors"]]
    lines += [row("SCNN_PROFILE_FRAME",reference["frame"]),row("SCNN_PROFILE_FRAME_NUMERIC_PASS",reference["numeric"])]
    for i in range(8):
        layer=i+1
        duration=int(reference["descriptors"][2*i+1]["cycle"])-int(reference["descriptors"][2*i]["cycle"])
        buckets=dict.fromkeys(prep.BUCKETS,0)
        buckets.update(context_load=10,command_wait=duration-10)
        lines.append(row("SCNN_SCHEDULE_BUCKET",dict(layer=layer,cycles=duration,**buckets)))
        lines.append(row("SCNN_SCHEDULE_CONTEXT",dict(layer=layer,idle=0,cache=8,transfer=1,drain=1)))
        pub=dict.fromkeys(prep.PUBLISHER,0);pub["read_response"]=8
        lines.append(row("SCNN_SCHEDULE_PUBLISHER",dict(layer=layer,**pub,reads=1,responses=1,stalls=0,starts=1)))
        lines.append(row("SCNN_SCHEDULE_HANDSHAKE",dict(layer=layer,features=2,responses=2,commands=2,completions=2,loads=2,releases=2,weight_loads=1,reuses=1,forward_reads=1,forward_responses=1)))
    lines.append(row("SCNN_SCHEDULE_PROBE_PASS",dict(learning=0,frames=1,descriptors=8,features=2048,currents=11,logits=11,probes=24,commits=0,rmw=0,publications=1892)))
    return "\n".join(lines)+"\n",reference


class ScheduleProbeTest(unittest.TestCase):
    def test_transform_keeps_original_checks_and_has_no_dut_drivers(self):
        source=(prep.ACCEPTED/"tb_scnn_profile_measurement.sv").read_text(encoding="utf-8")
        monitor=(prep.EXPERIMENT/"scnn_schedule_probe.svh").read_text()
        changed=prep.transform(source,monitor)
        self.assertEqual(changed.count("$fatal"),source.count("$fatal")+monitor.count("$fatal"))
        self.assertLess(changed.index("fg_check_frame();"),changed.index("        sp_report();"))
        self.assertIn("timestep_phase < 4;",changed)
        self.assertNotIn("force ",monitor)
        self.assertNotIn("<=",monitor.replace("layer<=8","layer<8"))
        self.assertEqual(changed.count("        sp_report();"),1)
        self.assertEqual(changed.count("$finish;"),source.count("$finish;"))
        self.assertNotIn("        fg_finish();",changed)
        self.assertIn("break; // First-frame diagnostic only.",changed)
        with self.assertRaises(ValueError):prep.transform("module x; endmodule",monitor)

    def test_complete_balanced_measurement(self):
        log,reference=transcript()
        result=prep.parse_log(log,reference)
        self.assertEqual(len(result["tables"]["bucket"]),8)
        self.assertAlmostEqual(sum(result["scheduler_fraction"].values()),1)
        self.assertNotIn("speedup",result)

    def test_rejects_bad_timing_counts_modes_and_partial_logs(self):
        log,reference=transcript()
        for bad in (log.replace("cycles=231921","cycles=231922"),
                    log.replace("completions=2","completions=1"),
                    log.replace("cache=8","cache=7"),
                    log.replace("starts=1","starts=2"),
                    log.replace("read_response=8","read_response=80"),
                    log.replace("SCNN_SCHEDULE_PROBE_PASS ","MISSING "),
                    log+"SCNN_PROFILE_FRAME frame=1 sample=0 timestep=1\n",
                    log+"SCNN_PROFILE_DESCRIPTOR frame=1 layer=1 edge=start cycle=4\n",
                    log+"%Error: failure\n",
                    log+"%Fatal: Assertion failed in fg_finish: measurement final completion counts\n",
                    log+log):
            with self.assertRaises((ValueError,KeyError)):prep.parse_log(bad,reference)
        with self.assertRaises(ValueError):prep.parse_log(log,reference,True)

    def test_worker_runs_only_off_with_original_host_recipe(self):
        log,reference=transcript()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"reference.json").write_text(json.dumps(dict(scope="probe",fixture="fixture",expected=dict(off=reference))))
            (root/"scnn_compatibility_build.json").write_bytes((prep.ACCEPTED/"scnn_compatibility_build.json").read_bytes())
            calls=[]
            def stage(record,name,command,timeout):
                calls.append((name,command))
                (root/(name+".log")).write_text(log if name=="run_off" else "")
            with patch.object(worker,"ROOT",root),patch.object(worker,"stage",side_effect=stage),patch.object(worker,"save"):
                record={};worker.compatibility(record,"scnn_probe")
            self.assertEqual([name for name,_ in calls],["build","run_off"])
            self.assertIn("+SCNN_LEARNING=0",calls[-1][1])
            self.assertIn("first_frame_passive_attribution",record["status"])


if __name__=="__main__":unittest.main()
