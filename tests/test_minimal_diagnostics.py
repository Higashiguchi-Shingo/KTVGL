import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from ktvgl.common.diagnostics import RunLog, invoke_solver, precision_diagnostics, summarize_inner
from ktvgl.methods.kronecker_tvgl import solver
from test_run_artifacts import training


class InnerDiagnosticsTests(unittest.TestCase):
    def test_solver_reports_convergence_without_changing_solution(self):
        covariance = np.array([[[2., .3], [.3, 1.5]]] * 3)
        kwargs = dict(alpha=.05, beta=.2, max_iter=1000, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            plain = solver.time_varying_graphical_lasso(covariance, **kwargs)
            info = {}
            observed = solver.time_varying_graphical_lasso(covariance, solver_info=info, **kwargs)
        np.testing.assert_array_equal(plain, observed)
        self.assertTrue(info['converged'])
        self.assertEqual(info['stop_reason'], 'residual_tolerance')
        self.assertLessEqual(info['rnorm'], info['e_pri'])
        self.assertLessEqual(info['snorm'], info['e_dual'])
        self.assertGreater(info['n_iter'], 1)

    def test_limit_target_and_final_iteration_convergence_are_distinct(self):
        covariance = np.array([np.diag([2., .5])] * 3)
        cases = [({}, False, 'max_iter'),
                 ({'tol': 1e6, 'rtol': 1e6}, True, 'residual_tolerance'),
                 ({'tol': 0., 'rtol': 0., 'stop_at': 1., 'stop_when': 1e20}, False, 'objective_target')]
        for options, converged, reason in cases:
            info = {}
            with self.subTest(reason=reason), contextlib.redirect_stdout(io.StringIO()):
                solver.time_varying_graphical_lasso(covariance, max_iter=1, solver_info=info, **options)
            self.assertEqual(info['n_iter'], 1)
            self.assertEqual(info['converged'], converged)
            self.assertEqual(info['stop_reason'], reason)
            summary = summarize_inner([info])
            self.assertEqual(summary['max_iter_count'], int(reason == 'max_iter'))

    def test_legacy_solver_unknown_and_typeerror_is_not_retried(self):
        def legacy(emp_cov):
            return emp_cov.copy()
        info = {}
        actual = invoke_solver(legacy, {'emp_cov': np.eye(2), 'n_samples': [1]}, info)
        np.testing.assert_array_equal(actual, np.eye(2))
        self.assertIsNone(info['converged'])
        self.assertEqual(summarize_inner([info])['unknown_count'], 1)
        calls = []
        def broken(**kwargs):
            calls.append(1)
            raise TypeError('inside solver')
        with self.assertRaisesRegex(TypeError, 'inside solver'):
            invoke_solver(broken, {}, info)
        self.assertEqual(len(calls), 1)
        self.assertEqual(info['stop_reason'], 'error')


class NumericalAndLoggingTests(unittest.TestCase):
    def test_numerical_checks(self):
        good = precision_diagnostics([[np.diag([1., 2.]), np.eye(3)]])
        self.assertTrue(good['valid_precision'])
        self.assertEqual(good['min_eigenvalue'], 1.)
        indefinite = precision_diagnostics([[np.diag([-1., 2.])]])
        self.assertTrue(indefinite['all_finite'])
        self.assertFalse(indefinite['valid_precision'])
        self.assertEqual(indefinite['min_eigenvalue'], -1.)
        asymmetric = precision_diagnostics([[np.array([[1., .3], [0., 1.]])]])
        self.assertFalse(asymmetric['valid_precision'])
        nonfinite = precision_diagnostics([[np.array([[np.nan, 0.], [0., np.inf]])]])
        self.assertFalse(nonfinite['all_finite'])
        self.assertEqual(nonfinite['nan_count'], 1)
        self.assertEqual(nonfinite['inf_count'], 1)
        self.assertIsNone(nonfinite['min_eigenvalue'])

    def test_log_tees_warnings_traceback_and_restores_process_state(self):
        with tempfile.TemporaryDirectory() as directory:
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), warnings.catch_warnings():
                warnings.simplefilter('always')
                original_warning = warnings.showwarning
                with self.assertRaisesRegex(RuntimeError, 'diagnostic failure'):
                    with RunLog(Path(directory) / 'run.log') as log:
                        print('progress marker')
                        print('stderr marker', file=sys.stderr)
                        warnings.warn('warning marker', UserWarning)
                        raise RuntimeError('diagnostic failure')
                self.assertIs(sys.stdout, out)
                self.assertIs(sys.stderr, err)
                self.assertIs(warnings.showwarning, original_warning)
                self.assertEqual(log.warning_count, 1)
            text = (Path(directory) / 'run.log').read_text()
            for marker in ['progress marker', 'stderr marker', 'warning marker', 'Traceback', 'diagnostic failure']:
                self.assertIn(marker, text)


class DiagnosticWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        source = self.base / 'input.npy'
        np.save(source, np.random.default_rng(29).normal(size=(4, 2, 3)))
        self.argv = ['train', '--data-path', str(source), '--dims', '2', '3',
                     '--base-dir', str(self.base / 'runs'), '--max_iter', '2',
                     '--max_iter_tvgl', '1', '--init', 'identity', '--no-plots', '--no-early-stopping']

    def execute(self):
        with patch.object(sys, 'argv', self.argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return training.main()

    def test_limit_counts_csv_and_warning_log_agree(self):
        run = self.execute()
        summary = json.loads((run / 'summary.json').read_text())
        with (run / 'inner_summary.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertEqual([(int(r['outer_iteration']), int(r['mode'])) for r in rows], [(1,0),(1,1),(2,0),(2,1)])
        count = sum(r['converged'] == 'False' for r in rows)
        self.assertGreater(count, 0)
        self.assertEqual(summary['inner_diagnostics']['unconverged_count'], count)
        self.assertEqual(summary['inner_diagnostics']['max_iter_count'], count)
        self.assertEqual(summary['status'], 'completed')
        self.assertGreater(summary['warning_count'], 0)
        self.assertIn('RuntimeWarning', (run / 'run.log').read_text())
        self.assertTrue(summary['numerical_diagnostics']['all_finite'])

    def test_solver_failure_preserves_successful_and_failed_subproblems(self):
        calls = []
        def failing_solver(**kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise TypeError('second subproblem failed')
            kwargs['solver_info'].update(n_iter=1, converged=True, stop_reason='residual_tolerance',
                                         rnorm=0., snorm=0., e_pri=1e-4, e_dual=1e-4)
            return np.array(kwargs['prev_matrices'])
        name = 'tvgl_solver' if hasattr(training, 'tvgl_solver') else 'time_varying_graphical_lasso'
        with patch.object(training, name, failing_solver):
            with self.assertRaisesRegex(TypeError, 'second subproblem failed'):
                self.execute()
        self.assertEqual(len(calls), 2)
        run = next((self.base / 'runs').rglob('summary.json')).parent
        summary = json.loads((run / 'summary.json').read_text())
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['inner_diagnostics']['converged_count'], 1)
        self.assertEqual(summary['inner_diagnostics']['error_count'], 1)
        with (run / 'inner_summary.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertIn('second subproblem failed', rows[-1]['error'])
        self.assertIn('Traceback', (run / 'run.log').read_text())

    def test_nonfinite_result_diagnostics_survive_save_failure(self):
        def invalid(model, x, **kwargs):
            return [[np.full((2, 2), np.nan), np.eye(3)] for _ in range(4)]
        with patch.object(training.KroneckerTVGL, 'fit', invalid):
            with self.assertRaises(ValueError):
                self.execute()
        run = next((self.base / 'runs').rglob('summary.json')).parent
        summary = json.loads((run / 'summary.json').read_text())
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['error']['phase'], 'precision_save')
        self.assertEqual(summary['numerical_diagnostics']['nan_count'], 16)
        self.assertFalse(summary['numerical_diagnostics']['valid_precision'])


if __name__ == '__main__':
    unittest.main()
