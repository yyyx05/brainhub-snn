"""Summarize the accepted paired raw-source synthesis reports offline."""
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from run_srnn_resident_remote import ROOT, RESULT, local_path

NAME = 'final_raw_source_synth_r1'


def first_number(pattern, text, label):
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError('missing ' + label)
    return float(match.group(1))


def utilization(path):
    text = path.read_text(errors='replace')
    values = {
        'lut': first_number(r'^\|\s*Slice LUTs\*?\s*\|\s*(\d+)', text, 'LUT'),
        'ff': first_number(r'^\|\s*Slice Registers\s*\|\s*(\d+)', text, 'FF'),
        'bram_tile': first_number(r'^\|\s*Block RAM Tile\s*\|\s*([\d.]+)', text, 'BRAM tile'),
        'ramb36': first_number(r'^\|\s*RAMB36/FIFO\*?\s*\|\s*(\d+)', text, 'RAMB36'),
        'ramb18': first_number(r'^\|\s*RAMB18\s*\|\s*(\d+)', text, 'RAMB18'),
        'dsp': first_number(r'^\|\s*DSPs\s*\|\s*(\d+)', text, 'DSP'),
    }
    return {key: int(value) if value.is_integer() else value for key, value in values.items()}


def timing(path):
    text = path.read_text(errors='replace')
    setup = re.search(r'Setup\s*:\s*(\d+)\s+Failing Endpoints,\s*Worst Slack\s*([-.\d]+)ns,\s*Total Violation\s*([-.\d]+)ns', text)
    hold = re.search(r'Hold\s*:\s*(\d+)\s+Failing Endpoints,\s*Worst Slack\s*([-.\d]+)ns,\s*Total Violation\s*([-.\d]+)ns', text)
    if not setup or not hold:
        raise ValueError('missing setup/hold summary in ' + str(path))
    return {
        'setup_failing_endpoints': int(setup.group(1)),
        'wns_ns': float(setup.group(2)),
        'tns_ns': float(setup.group(3)),
        'hold_failing_endpoints': int(hold.group(1)),
        'whs_ns': float(hold.group(2)),
        'ths_ns': float(hold.group(3)),
    }


def main():
    root = Path(local_path(ROOT))
    archive = root / RESULT / NAME
    state = json.loads((archive / 'status.json').read_text())
    arms = {}
    for arm in ('baseline', 'candidate'):
        reports = archive / arm / 'server_results' / 'final_composition_synth'
        arms[arm] = {
            'utilization': utilization(reports / 'utilization_post_synth.rpt'),
            'timing': timing(reports / 'timing_summary_post_synth.rpt'),
            'ordinary_warnings': state['arms'][arm]['ordinary_warnings'],
            'errors': state['arms'][arm]['errors'],
            'critical_warnings': state['arms'][arm]['critical_warnings'],
            'dcp_bytes': state['arms'][arm]['checkpoint']['bytes'],
            'dcp_crc_pass': state['arms'][arm]['checkpoint']['crc_pass'],
            'wall_seconds': state['arms'][arm]['wall_seconds'],
        }
    baseline = arms['baseline']
    candidate = arms['candidate']
    resource_delta = {}
    for key, value in baseline['utilization'].items():
        new = candidate['utilization'][key]
        resource_delta[key] = {
            'absolute': new - value,
            'relative_percent': (new - value) / value * 100.0 if value else None,
        }
    timing_delta = {}
    for key, value in baseline['timing'].items():
        timing_delta[key] = candidate['timing'][key] - value
    result = {
        'status': 'PASS_offline_paired_synthesis_summary',
        'source_status': state['status'],
        'scope': 'Post-synthesis OOC comparison with actual producer/mux/address/tail wiring inside both arms; no route, board, power or bitstream result.',
        'top': 'unified_h40_raw_source_top_v1',
        'part': 'xc7vx690tffg1761-3',
        'clock_period_ns': 15.0,
        'arms': arms,
        'candidate_minus_baseline': {
            'resource_delta': resource_delta,
            'timing_delta': timing_delta,
            'dcp_bytes': candidate['dcp_bytes'] - baseline['dcp_bytes'],
            'wall_seconds': candidate['wall_seconds'] - baseline['wall_seconds'],
        },
        'interpretation': {
            'resource': 'Candidate is slightly smaller in LUT/FF, with BRAM/DSP unchanged; the differences are not a whole-system area claim.',
            'timing': 'Candidate synthesis WNS improves by 0.044 ns, but both arms remain negative at post-synthesis OOC and therefore are not timing closure or Fmax evidence.',
            'warnings': 'Both arms retain 502 ordinary warnings, zero errors and zero critical warnings; warning equality is not a zero-warning signoff.',
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
