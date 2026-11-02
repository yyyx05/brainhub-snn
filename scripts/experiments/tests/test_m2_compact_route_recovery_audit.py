import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_m2_compact_route_recovery import reopened_equal, route_metrics, delivery_pass, checkpoint_record, nonnegative_slacks


class RecoveryAuditTests(unittest.TestCase):
    def test_route_report_counts(self):
        text = "# of routable nets : 31\n# of fully routed nets : 30\n# of nets with routing errors : 1\n"
        self.assertEqual(route_metrics(text), dict(routable=31, fully_routed=30, errors=1))

    def test_reopen_requires_all_persisted_fields(self):
        before = dict(primitive_count="100", unplaced_primitives="0", black_boxes="0", wns_ns="0.0001", whs_ns="0.07")
        after = dict(before, status="pass", protected_set_preserved="1", trace_payload_cell_count="6270", trace_payload_nonconstant_control_count="0")
        self.assertTrue(reopened_equal(before, after))
        self.assertFalse(reopened_equal(before, dict(after, wns_ns="0.0000")))
        self.assertFalse(reopened_equal(before, dict(after, trace_payload_nonconstant_control_count="1")))
        self.assertFalse(reopened_equal({}, after))

    def test_exit_or_wrapper_alone_not_acceptance(self):
        good = dict(status="PASS_route_recovery_checkpoint_reopened", exit=0)
        self.assertTrue(delivery_pass(good, dict(crc=True, reports=True)))
        self.assertFalse(delivery_pass(dict(good, exit=1), dict(crc=True)))
        self.assertFalse(delivery_pass(good, dict(crc=True, reports=False)))
        self.assertFalse(delivery_pass(good, {}))
        self.assertFalse(delivery_pass(dict(status="FAIL_archive_space", exit=0), dict(crc=True)))

    def test_no_path_or_small_negative_is_not_closed(self):
        self.assertFalse(nonnegative_slacks(dict(wns_ns="no_path", whs_ns="0.07")))
        self.assertFalse(nonnegative_slacks(dict(wns_ns="-0.00001", whs_ns="0.07")))
        self.assertFalse(nonnegative_slacks({}))
        self.assertTrue(nonnegative_slacks(dict(wns_ns="0.00001", whs_ns="0.07")))

    def test_corrupt_export_is_archived_as_failure(self):
        path = Mock()
        path.stat.return_value.st_size = 37
        with patch("audit_m2_compact_route_recovery.checkpoint", side_effect=zipfile.BadZipFile("incomplete ZIP")):
            result = checkpoint_record(path)
        self.assertFalse(result["crc_pass"])
        self.assertEqual(result["bytes"], 37)
        self.assertIn("incomplete ZIP", result["archive_error"])


if __name__ == "__main__":
    unittest.main()
