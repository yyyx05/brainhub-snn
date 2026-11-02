import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/experiments/run_fc_representative_inputs.py'
SPEC = importlib.util.spec_from_file_location('fc_representative_selection', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SelectionTests(unittest.TestCase):
    def rows(self):
        return [dict(dataset_index=label*30+i,label=label,valid_timesteps=90+i%3,
                     events=90+i*30,repeated_extra_events=i,rawfilename=f'Test/{label}/{i}.bin',
                     integer_raw24_correct=i%2,cycles=100-i)
                for label in range(10) for i in range(30)]

    def test_disjoint_full_batches_same_class_quota(self):
        batches, strata = MODULE.select_rows(self.rows())
        self.assertEqual([len(b) for b in batches],[16]*3)
        self.assertEqual(len({r['dataset_index'] for b in batches for r in b}),48)
        self.assertEqual(len(strata),30)
        for batch in batches:
            self.assertEqual([sum(r['label']==c for r in batch) for c in range(10)],[2]*6+[1]*4)

    def test_outcomes_and_input_order_do_not_affect_selection(self):
        rows = self.rows()
        expected = MODULE.select_rows(rows)
        for row in rows:
            row['cycles'] = -1
            row['integer_raw24_correct'] = 100
        self.assertEqual(expected,MODULE.select_rows(list(reversed(rows))))

    def test_duplicate_identity_rejected(self):
        rows = self.rows()
        rows[-1]['dataset_index'] = rows[0]['dataset_index']
        with self.assertRaises(ValueError): MODULE.select_rows(rows)

    def test_equal_density_uses_official_index(self):
        rows = self.rows()
        for row in rows:
            row.update(events=100,repeated_extra_events=0,valid_timesteps=99)
        batches,_ = MODULE.select_rows(rows)
        self.assertEqual([r['dataset_index'] for r in batches[0]][:2],[4,5])


if __name__ == '__main__': unittest.main()
