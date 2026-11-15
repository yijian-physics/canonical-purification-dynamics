"""Checks that benchmark accounting and accuracy selection preserve evidence."""
import unittest
from factor_tdvp_accelerated.repeated_scaling import Ledger, repeated_frontier, configurations

class LedgerTests(unittest.TestCase):
    def test_nested_accounting_never_double_counts(self):
        clock=[0.]
        ledger=Ledger(lambda:clock[0])
        with ledger.measure('prediction'):
            clock[0]=2.
            with ledger.measure('force'):clock[0]=7.
            clock[0]=10.
        self.assertEqual(ledger.inclusive,{'force':5.,'prediction':10.})
        self.assertEqual(ledger.exclusive,{'force':5.,'prediction':5.})
        self.assertEqual(sum(ledger.exclusive.values()),10.)
    def test_failed_call_still_records_and_restores_stack(self):
        clock=[0.];ledger=Ledger(lambda:clock[0])
        with self.assertRaises(ValueError):
            with ledger.measure('force'):
                clock[0]=3.;raise ValueError('failed solve')
        self.assertEqual(ledger.exclusive['force'],3.)
        self.assertEqual(ledger.stack,[])
    def test_classmethod_instrumentation_preserves_behavior(self):
        class Source:
            @classmethod
            def produce(cls,x):return cls.__name__,x+1
        ledger=Ledger();ledger.patch(Source,'produce','reference')
        self.assertEqual(Source.produce(2),('Source',3));ledger.restore()
        self.assertEqual(Source.produce(2),('Source',3))

class FrontierTests(unittest.TestCase):
    @staticmethod
    def row(rep,**extra):
        return dict(config_id='same',family='D',n=3,method='matrix_free',steps=5,
                    requested_repeats=3,repeat=rep,status='completed',global_cp_error=1e-8,
                    total_seconds=float(rep),**extra)
    def test_selects_median_of_three_not_luckiest_timing(self):
        result=repeated_frontier([self.row(i) for i in (1,2,3)],1e-7)
        self.assertEqual(result[0]['median_seconds'],2.)
    def test_one_failed_accuracy_repeat_disqualifies_configuration(self):
        rows=[self.row(i) for i in (1,2,3)];rows[2]['global_cp_error']=1e-4
        self.assertFalse(repeated_frontier(rows,1e-7)[0]['attained'])
    def test_missing_and_timeout_runs_are_not_measured_costs(self):
        rows=[self.row(i) for i in (1,2,3)];rows[2].update(status='timeout',total_seconds=None,global_cp_error=None)
        self.assertFalse(repeated_frontier(rows,1e-7)[0]['attained'])
        self.assertFalse(repeated_frontier(rows[:2],1e-7)[0]['attained'])
    def test_bound_is_labeled_and_never_substituted_for_existing_error(self):
        rows=[self.row(i,global_cp_bound=1e-9) for i in (1,2,3)]
        for r in rows:r['global_cp_error']=1e-3
        self.assertFalse(repeated_frontier(rows,1e-7)[0]['attained'])
        for r in rows:r['global_cp_error']=None
        self.assertEqual(repeated_frontier(rows,1e-7)[0]['evidence'],'bound')
    def test_resource_guard_does_not_mark_factor_eight_unsupported(self):
        rows=list(configurations('unsupported'))
        self.assertFalse(any(r['method']=='matrix_free' and r['n']==8 for r in rows))
        self.assertTrue(all(r['requested_repeats']==0 for r in rows))

class MemoryGuardTests(unittest.TestCase):
    def test_child_rss_guard_writes_evidence_before_guarded_exit(self):
        import subprocess,sys,tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            command=[sys.executable,'-m','factor_tdvp_accelerated.repeated_scaling',
                     '--worker',json.dumps(dict(kind='trajectory',family='D',n=3,method='root',steps=1,chi=4,work_bond=4,total=.1)),
                     '--out',directory,'--memory-mib','1']
            process=subprocess.run(command,capture_output=True,text=True,timeout=15)
            self.assertEqual(process.returncode,86,process.stderr)
            event=json.loads((Path(directory)/'memory_limit_event.json').read_text())
            self.assertEqual(event['status'],'memory_limit')
            self.assertGreater(event['peak_process_memory_mib'],event['limit_mib'])

if __name__=='__main__':unittest.main()
