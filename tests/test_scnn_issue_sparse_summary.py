import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments"))
from summarize_scnn_issue_sparse import compare_arm, timing


class SummaryTests(unittest.TestCase):
    def fixture(self):
        item=dict(final=dict(cycles=100,frames=1),frames=[dict(cycles="70")],
                  descriptors=[dict(frame="0",layer="1",edge="start",cycle="20"),
                               dict(frame="0",layer="1",edge="end",cycle="89")],
                  traffic=dict(forward_reads=10),banks=[])
        return item,"SUPPLEMENT_FULLGRAPH_INITIALIZATION cycles=20\n"

    def test_initialization_already_in_total(self):
        result=timing(*reversed(self.fixture()))
        self.assertEqual(result["total_cycles"],100)
        self.assertEqual(result["post_initialization_cycles"],80)
        self.assertEqual(result["inter_frame_boundary_cycles"],10)

    def test_delta_accounting_keeps_unattributed_remainder(self):
        old,text=self.fixture();new=copy.deepcopy(old)
        new["final"]["cycles"]=80;new["frames"][0]["cycles"]="60"
        new["descriptors"][1]["cycle"]="79"
        row,_,_=compare_arm(old,new,text,text.replace("20","15"),"on")
        self.assertEqual(row["saved_total_cycles"],20)
        self.assertEqual(row["saved_initialization_cycles"],5)
        self.assertEqual(row["saved_descriptor_cycles"],10)
        self.assertEqual(row["saved_outside_those_intervals"],5)

    def test_changed_traffic_rejected(self):
        old,text=self.fixture();new=copy.deepcopy(old);new["traffic"]["forward_reads"]=9
        with self.assertRaises(ValueError): compare_arm(old,new,text,text,"off")

    def test_fatal_after_pass_rejected(self):
        item,text=self.fixture()
        with self.assertRaises(ValueError): timing(text+"SCNN_PROFILE_PASS\n%Fatal: late failure",item)


if __name__ == "__main__": unittest.main()
