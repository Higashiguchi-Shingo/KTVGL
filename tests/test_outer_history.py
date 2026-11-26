import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ktvgl.common.history import HistoryWriter, history_row, plot_history, read_history, replot_run
from ktvgl.common.results import load_precision
from ktvgl.common.synthetic import generate_kron_data
from ktvgl.methods.kronecker_tvgl import KroneckerTVGL, tvgl_solver
from ktvgl.methods.kronecker_tvgl import kroneckergl as model_module
from test_run_artifacts import training


class OuterHistoryTests(unittest.TestCase):
    def test_real_solver_history_matches_evaluation_logs_and_callback(self):
        x = np.random.default_rng(19).normal(size=(5, 2, 3))
        for psi in ('l1', 'laplacian'):
            with self.subTest(psi=psi):
                model = KroneckerTVGL([.05, .05], [1.5, 1.5], tvgl_solver,
                                     init_method='identity', psi=psi,
                                     max_iter=3, max_iter_tvgl=100, early_stopping=False)
                seen = []
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    estimated = model.fit(x, iteration_callback=seen.append)
                self.assertEqual(model.objective_history_, seen)
                self.assertEqual([row['iteration'] for row in seen], [1, 2, 3])
                self.assertIsNone(seen[0]['relative_improvement'])
                self.assertEqual(model.fit_info_['n_iter'], len(seen))
                final = model_module.objective_terms(x, estimated, [.05, .05], [1.5, 1.5], psi=psi)
                for key, value in final.items():
                    self.assertEqual(seen[-1][key], value)
                logged = [float(line.split(':', 1)[1].split('|')[0]) for line in
                          output.getvalue().splitlines() if line.startswith('Objective total:')]
                np.testing.assert_allclose(logged, [row['total'] for row in seen], rtol=1e-6)
                for index, row in enumerate(seen):
                    self.assertAlmostEqual(row['total'], row['data_fit'] + row['sparsity'] + row['temporal'])
                    self.assertGreaterEqual(row['iteration_seconds'], 0.)
                    self.assertGreaterEqual(row['relative_precision_change'], 0.)
                    if index:
                        previous = seen[index - 1]
                        self.assertGreaterEqual(row['elapsed_seconds'], previous['elapsed_seconds'])
                        self.assertAlmostEqual(row['relative_improvement'],
                            (previous['total'] - row['total']) / abs(previous['total']))

    def test_relative_change_and_csv_are_available_before_context_exit(self):
        terms = dict(total=-2., data_fit=-3., sparsity=.5, temporal=.5)
        with patch('time.perf_counter', return_value=5.):
            row = history_row(1, terms, None, [[np.eye(2)]], [[2*np.eye(2)]], 1., 3.)
        self.assertEqual(row['relative_precision_change'], 1.)
        self.assertEqual(row['elapsed_seconds'], 4.)
        self.assertEqual(row['iteration_seconds'], 2.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.csv'
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                with HistoryWriter(path) as writer:
                    writer(row)
                    self.assertEqual(read_history(path), [row])
                    raise RuntimeError('interrupted')
            self.assertEqual(read_history(path), [row])

    def test_plots_use_saved_values_including_negative_and_single_point(self):
        rows = [dict(iteration=i+1, elapsed_seconds=(i+1)*.5, iteration_seconds=.5,
                     total=-10.-i, data_fit=-13.-i, sparsity=1., temporal=2.,
                     relative_improvement=None, relative_precision_change=.1) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.csv'
            with HistoryWriter(path) as writer:
                for row in rows:
                    writer(row)
            saved = read_history(path)
            figures = []
            subplots = plt.subplots
            def capture(*a, **kw):
                fig, axes = subplots(*a, **kw)
                figures.append(fig)
                return fig, axes
            with patch.object(plt, 'subplots', side_effect=capture):
                files = plot_history(saved, Path(directory) / 'figures')
            self.assertEqual(len(files), 4)
            for file in files:
                self.assertGreater(file.stat().st_size, 1000)
            np.testing.assert_array_equal(figures[0].axes[0].lines[0].get_ydata(), [-10., -11., -12.])
            for axis in figures[1].axes:
                np.testing.assert_array_equal(axis.lines[0].get_ydata(), [-13., -14., -15.])
            np.testing.assert_array_equal(figures[1].axes[1].lines[0].get_xdata(), [.5, 1., 1.5])
            self.assertEqual(plot_history([], directory), [])
            self.assertEqual(len(plot_history(saved[:1], directory)), 4)
            self.assertEqual(plt.get_fignums(), [])


class HistoryWorkflowTests(unittest.TestCase):
    def test_synthetic_and_external_history_agree_with_final_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            x = np.random.default_rng(7).normal(size=(4, 2, 3))
            np.save(base / 'external.npy', x)
            inputs = [(['--data-path', str(base / 'external.npy')], x),
                      (['--breaks-json', '[[0,2,4],[0,2,4]]', '--seeds-json', '[[11,12],[21,22]]'], None)]
            for options, observations in inputs:
                argv = ['train', '--dims', '2', '3', '--base-dir', str(base / 'runs'),
                        '--max_iter', '2', '--max_iter_tvgl', '100', '--psi', 'l1',
                        '--no-early-stopping', '--no-plots'] + options
                with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    run = training.main()
                config = json.loads((run / 'config.json').read_text())
                summary = json.loads((run / 'summary.json').read_text())
                rows = read_history(run / 'history.csv')
                self.assertEqual(len(rows), summary['n_iter'])
                self.assertEqual(summary['history']['rows'], len(rows))
                self.assertEqual(rows[-1]['total'], summary['final_objective'])
                estimated, _ = load_precision(run / 'precision.npz')
                if observations is None:
                    observations, _ = generate_kron_data(**config['generation'])
                expected = model_module.objective_terms(observations, estimated, [.01, .01], [2., 2.], psi='l1')
                for key in expected:
                    self.assertEqual(rows[-1][key], expected[key])
                self.assertFalse((run / 'figures').exists())
                files = replot_run(run)
                self.assertEqual(len(files), 4)

    def test_failure_retains_completed_iterations(self):
        with tempfile.TemporaryDirectory() as directory:
            row = dict(iteration=1, elapsed_seconds=.1, iteration_seconds=.1,
                       total=4., data_fit=2., sparsity=1., temporal=1.,
                       relative_improvement=None, relative_precision_change=.2)
            def interrupted(model, x, *, iteration_callback, inner_callback=None):
                model.fit_info_.update(n_iter=1, final_objective=4.)
                iteration_callback(row)
                raise RuntimeError('interrupted after first sweep')
            argv = ['train', '--dims', '2', '3', '--breaks-json', '[[0,2,4]]',
                    '--base-dir', directory, '--no-plots']
            with patch.object(sys, 'argv', argv), patch.object(training.KroneckerTVGL, 'fit', interrupted):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(RuntimeError):
                    training.main()
            run = next(Path(directory).rglob('summary.json')).parent
            self.assertEqual(read_history(run / 'history.csv'), [row])
            summary = json.loads((run / 'summary.json').read_text())
            self.assertEqual(summary['status'], 'failed')
            self.assertEqual(summary['history']['rows'], 1)


if __name__ == '__main__':
    unittest.main()
