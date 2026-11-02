from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_scnn_router_owners import parse_log


def complete_log():
    lines=[]
    for owner in (1,2,3):
        for service in (0,1):
            for latency in (1,4,12):
                lines.append(f'SCNN_ROUTER_OWNER_CASE owner={owner} service={service} latency={latency} reads=64 responses=64 replacements={service*12} error_payloads=4 address_bits=24 tag_bits=16 request_stalls=8 response_stalls=9')
        lines.append(f'SCNN_ROUTER_OWNER_RECOVERY owner={owner} owner_switch=1 inactive_owner=1 unsolicited=1 nonowner=1 quiescent_clear=1 reissue=1')
    lines.append('SCNN_ROUTER_OWNERS_PASS owners=3 services=2 latency_profiles=3 cases=18 reads=1152 responses=1152 recovery_cases=12')
    return '\n'.join(lines)+'\n'


class RouterOwnersParserTest(unittest.TestCase):
    def test_complete_scope(self):
        checks=parse_log(complete_log())
        self.assertEqual(len(checks['cases']),18)
        self.assertEqual(len(checks['recoveries']),3)
        self.assertIn('not_final_top_recovery',checks['status'])

    def test_incomplete_or_diagnostic_rejected(self):
        log=complete_log()
        changes=[log.replace('reads=64','reads=63',1),
                 log.replace('tag_bits=16','tag_bits=4',1),
                 log.replace('error_payloads=4','error_payloads=0',1),
                 log.replace('request_stalls=8','request_stalls=0',1),
                 log.replace('replacements=12','replacements=0',1),
                 log.replace('quiescent_clear=1','quiescent_clear=0',1),
                 log+'%Error: test failed\n',log+log.splitlines()[-1]+'\n',
                 '\n'.join(log.splitlines()[:-1])]
        for changed in changes:
            with self.subTest(changed=changed[-160:]):
                with self.assertRaises(ValueError):parse_log(changed)

    def test_marker_order_rejected(self):
        lines=complete_log().splitlines()
        lines.insert(0,lines.pop())
        with self.assertRaises(ValueError):parse_log('\n'.join(lines))
        lines=complete_log().splitlines()
        lines.insert(0,lines.pop(6))
        with self.assertRaises(ValueError):parse_log('\n'.join(lines))


if __name__=='__main__':unittest.main()
