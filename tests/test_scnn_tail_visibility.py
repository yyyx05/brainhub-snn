import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/experiments'))
from prepare_scnn_tail_testbench import ROOT,ORIGINAL,transform,parse_log
from prepare_scnn_tail_visibility import train_frame_index


def line(marker,**values):
    return marker+' '+' '.join(f'{k}={v}' for k,v in values.items())


def completed_log():
    lines=[]
    for frame in range(52):
        query=frame>=48
        window=frame//16
        sample=0 if query else frame%16//4
        timestep=frame-48 if query else window*4+frame%4
        if frame%16==0:
            lines.append(line('SCNN_TEMPORAL_WINDOW_START',window=window+1,query=int(query),
                base=0 if query else window*4,batch=1 if query else 4,version=4609+window,cycle=frame+1))
        lines.append(line('SCNN_TEMPORAL_LAYER',frame=frame,sample=sample,timestep=timestep,layer=8,cycles=100,descriptors=frame+1))
        if frame in (15,31,47):
            code,residual=((0,20041),(0,20011),(662,20221))[window]
            lines.append(line('SCNN_TEMPORAL_COMMIT',window=window+1,code_changes=code,
                residual_changes=residual,rmw=residual,version=4610+window,publications=1892*(window+2)))
        lines.append(line('SCNN_TEMPORAL_FRAME_STATUS',frame=frame,sample=sample,timestep=timestep,cycles=100,commits=min((frame+1)//16,3)))
    lines.append('SCNN_TEMPORAL_CROSSING window=3 index=18433 gradient=2221286714 delta=-21184 code_before=870 code_after=869 residual_before=-16155 residual_after=-4571')
    lines.append(line('SCNN_TAIL_SERVICE',scalars=425984,forward_responses=122773504,learning_responses=60273,
        code_writes=662,residual_writes=60273,invalidations=3,versions=3,query_uses_updated_rtl=1))
    lines.extend(line('SCNN_TEMPORAL_DDR',bank=i,reads=0 if i<7 else 122683392 if i==7 else 90112) for i in range(9))
    lines.append(line('SCNN_TAIL_VISIBILITY_PASS',frames=52,descriptors=52,features=106496,currents=572,logits=572,
        traces=98304,probes=156,rmw=60273,commits=3,publications=7568,code_changes=662,
        query_currents=4,query_logits=4,fresh_reads=9437184,cycles=10000))
    return '\n'.join(lines)+'\n'


class TailVisibilityTests(unittest.TestCase):
    def test_original_frame_mapping(self):
        self.assertEqual(sorted(train_frame_index(s,t) for s in range(4) for t in range(12)),list(range(48)))
        self.assertEqual([train_frame_index(s,t) for s,t in ((0,0),(3,3),(0,4),(3,11))],[0,15,16,47])

    def test_transform_keeps_learning_oracle_and_public_input(self):
        source=(ROOT/ORIGINAL).read_text(encoding='utf-8')
        result=transform(source)
        initial=result[result.index('    initial begin\n        $readmemh('):result.index('    localparam int FG_CLASSIFIER_BASE')]
        self.assertEqual(initial.count('        run_current_layer();'),1)
        self.assertIn('layer_phase = 8;',initial)
        self.assertIn('feature_memory[0][CONV7_WRITE_BASE+channel*4+row][col]',initial)
        self.assertEqual(result[result.index('    task automatic fg_check_store;'):result.index('    task automatic fg_check_frame;')],
                         source[source.index('    task automatic fg_check_store;'):source.index('    task automatic fg_check_frame;')])
        self.assertNotIn('SCNN_TEMPORAL_FINAL_PASS',result)
        self.assertIn('SCNN_TAIL_VISIBILITY_PASS',result)
        self.assertIn('tail_scalars != 425984',result)
        self.assertNotIn('force ',result)

    def test_complete_parse(self):
        checks=parse_log(completed_log())
        self.assertEqual(checks['final']['code_changes'],662)
        self.assertEqual(len(checks['layers']),52)

    def test_rejects_partial_error_duplicate_and_wrong_service(self):
        log=completed_log()
        for bad in (log+'%Error: late failure\n',log+log.splitlines()[-1]+'\n',
                    log.replace('forward_responses=122773504','forward_responses=122773503'),
                    log.replace('query_currents=4','query_currents=0'),
                    '\n'.join(s for s in log.splitlines() if not s.startswith('SCNN_TEMPORAL_LAYER frame=12 '))):
            with self.subTest(bad=bad[-150:]),self.assertRaises(ValueError):parse_log(bad)

    def test_rejects_out_of_order_frames(self):
        lines=completed_log().splitlines()
        a=next(i for i,s in enumerate(lines) if s.startswith('SCNN_TEMPORAL_FRAME_STATUS frame=0 '))
        b=next(i for i,s in enumerate(lines) if s.startswith('SCNN_TEMPORAL_FRAME_STATUS frame=1 '))
        lines[a],lines[b]=lines[b],lines[a]
        with self.assertRaises(ValueError):parse_log('\n'.join(lines))

    def test_commit_must_follow_layer_and_precede_frame_completion(self):
        for window in (1,2,3):
            for shift in (-1,1):
                lines=completed_log().splitlines()
                at=next(i for i,s in enumerate(lines) if s.startswith(f'SCNN_TEMPORAL_COMMIT window={window} '))
                lines[at],lines[at+shift]=lines[at+shift],lines[at]
                with self.subTest(window=window,shift=shift),self.assertRaisesRegex(ValueError,'event ordering'):
                    parse_log('\n'.join(lines))

    def test_archived_real_run_order(self):
        log=ROOT/'results/paper_evidence/supplement_20261015/performance_optimization/scnn_tail_visibility_r1/run.log'
        checks=parse_log(log.read_text())
        self.assertEqual(checks['service']['scalars'],425984)
        self.assertEqual(checks['final']['code_changes'],662)


if __name__=='__main__':unittest.main()
