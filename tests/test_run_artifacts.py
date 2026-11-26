import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from common.results import (RunArtifacts, canonical_batches, data_hash,
                                  load_observations, load_precision, save_precision)
from common.synthetic import generate_kron_data
from methods.KroneckerTVGL import KroneckerTVGL, time_varying_graphical_lasso as tvgl_solver
from methods.KroneckerTVGL import kroneckergl as model_module

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('artifact_training', ROOT / 'exp/train_ktvgl.py')
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)
SUPPORTS_MULTI = 'samples_per_t' in __import__('inspect').signature(generate_kron_data).parameters


class RunArtifactTests(unittest.TestCase):
    def test_roundtrip_unequal_mode_sizes_without_pickle(self):
        theta = [[np.eye(2) * (t + 1), np.eye(3) * (t + 2)] for t in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'precision.npz'
            for truth in (None, theta):
                save_precision(path, theta, truth)
                actual, actual_truth = load_precision(path)
                for t in range(4):
                    for m in range(2):
                        np.testing.assert_array_equal(actual[t][m], theta[t][m])
                        if truth is not None:
                            np.testing.assert_array_equal(actual_truth[t][m], truth[t][m])
                if truth is None:
                    self.assertIsNone(actual_truth)

    def test_unique_run_paths_and_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            first = RunArtifacts(directory, 'dataset-v1', 'KTVGL', {'gauge': None})
            second = RunArtifacts(directory, 'dataset-v1', 'KTVGL', {'gauge': None})
            self.assertNotEqual(first.path, second.path)
            first.finish('completed', final_objective=np.nan)
            value = json.loads((first.path / 'summary.json').read_text())
            self.assertIsNone(value['final_objective'])
            self.assertTrue((second.path / 'config.json').exists())
            with self.assertRaises(ValueError):
                RunArtifacts(directory, '../escape', 'KTVGL', {})

    def test_variable_input_and_hash_includes_time_boundaries(self):
        batches = [np.arange(n * 6).reshape(n, 2, 3) for n in (1, 3, 2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.npz'
            np.savez(path, **{f'X_{t}': x for t, x in enumerate(batches)})
            actual, truth = load_observations(path, [2, 3])
            self.assertIsNone(truth)
            self.assertEqual(data_hash(actual), data_hash(canonical_batches(batches, [2, 3])))
            self.assertNotEqual(data_hash(actual), data_hash(actual[::-1]))
        with self.assertRaises(ValueError):
            canonical_batches(np.full((3, 2, 3), np.nan), [2, 3])


class TrainingArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.common = ['train', '--base-dir', str(self.base / 'runs'),
                       '--dataset-id', 'tiny', '--dims', '2', '3',
                       '--lambdas', '.05', '.05', '--rhos', '1.5', '1.5',
                       '--max_iter', '2', '--max_iter_tvgl', '100', '--init', 'identity',
                       '--no-early-stopping', '--no-plots']

    def run_cli(self, extra):
        with patch.object(sys, 'argv', self.common + extra), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            path = training.main()
        return path, json.loads((path / 'config.json').read_text()), json.loads((path / 'summary.json').read_text())

    def synthetic_args(self):
        result = ['--breaks-json', '[[0,2,4],[0,3,4]]']
        if SUPPORTS_MULTI:
            result += ['--samples-per-t-json', '[2,3,2,1]']
        return result

    def test_synthetic_run_reproduces_direct_fit_and_resolved_seeds(self):
        path, config, summary = self.run_cli(self.synthetic_args())
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['stop_reason'], 'max_iter')
        self.assertEqual(summary['n_iter'], 2)
        self.assertTrue(summary['ground_truth_available'])
        self.assertIsNotNone(summary['metrics'])
        for seeds in config['generation']['seeds_list']:
            self.assertTrue(all(isinstance(seed, int) for seed in seeds))
        x, truth = generate_kron_data(**config['generation'])
        self.assertEqual(config['dataset']['data_hash'], data_hash(canonical_batches(x, [2, 3])))
        model = KroneckerTVGL(tvgl_solver=tvgl_solver, **config['model'])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            expected = model.fit(x)
        estimated, actual_truth = load_precision(path / 'precision.npz')
        for t in range(4):
            for m in range(2):
                np.testing.assert_array_equal(estimated[t][m], expected[t][m])
                np.testing.assert_array_equal(actual_truth[t][m], truth[t][m])
        self.assertEqual(summary['final_objective'], model.fit_info_['final_objective'])
        self.assertIn('schema_version', config)
        self.assertIn('git', json.loads((path / 'environment.json').read_text()))
        self.assertGreaterEqual(summary['timing_seconds']['total'], summary['timing_seconds']['fit'])

    def test_external_data_has_same_schema_and_preserves_coordinates(self):
        synthetic_path, synthetic_config, synthetic_summary = self.run_cli(self.synthetic_args())
        x = np.random.default_rng(1).normal(size=(4, 2, 3))
        source = self.base / 'observations.npy'
        np.save(source, x)
        metadata = self.base / 'metadata.json'
        metadata.write_text(json.dumps(dict(dataset_name='観測データ', dataset_version='v2',
            mode_names=['sensor', 'condition'], variable_names=[['a','b'],['x','y','z']],
            time_values=['t0','t1','t2','t3'], preprocessing={'centering':'upstream'})))
        extra = ['--data-path', str(source), '--metadata-json', str(metadata)]
        path, config, summary = self.run_cli(extra)
        repeated_path, _, _ = self.run_cli(extra)
        self.assertEqual(set(config), set(synthetic_config))
        self.assertEqual(set(config['dataset']), set(synthetic_config['dataset']))
        self.assertEqual(set(summary), set(synthetic_summary))
        self.assertNotEqual(path, repeated_path)
        self.assertTrue((synthetic_path / 'precision.npz').exists())
        self.assertIsNone(summary['metrics'])
        self.assertFalse(summary['ground_truth_available'])
        self.assertIsNone(load_precision(path / 'precision.npz')[1])
        self.assertEqual(config['dataset']['dataset_name'], '観測データ')
        self.assertEqual(config['dataset']['coordinates']['time_values'], ['t0','t1','t2','t3'])
        self.assertEqual(config['dataset']['samples_per_t'], [1]*4)
        self.assertEqual(len(config['dataset']['source']['sha256']), 64)

    def test_invalid_input_records_failed_run(self):
        source = self.base / 'invalid.npy'
        np.save(source, np.full((4, 2, 3), np.nan))
        with self.assertRaises(ValueError):
            self.run_cli(['--data-path', str(source)])
        summary_path = next((self.base / 'runs').rglob('summary.json'))
        summary = json.loads(summary_path.read_text())
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['error']['phase'], 'data_preparation')
        self.assertFalse((summary_path.parent / 'precision.npz').exists())

    def test_fit_failure_is_recorded_and_reraised(self):
        with patch.object(training.KroneckerTVGL, 'fit', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                self.run_cli(self.synthetic_args())
        summary = json.loads(next((self.base / 'runs').rglob('summary.json')).read_text())
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['error']['phase'], 'fit')
        self.assertIn('fit', summary['timing_seconds'])

    def test_external_npz_can_supply_ground_truth(self):
        x = np.random.default_rng(3).normal(size=(4, 2, 3))
        source = self.base / 'known.npz'
        truth0 = np.tile(np.array([[2., .2], [.2, 2.]]), (4, 1, 1))
        truth1 = np.tile(np.eye(3) + .1 * np.ones((3, 3)), (4, 1, 1))
        np.savez(source, X=x, true_mode_0=truth0, true_mode_1=truth1)
        path, _, summary = self.run_cli(['--data-path', str(source)])
        _, truth = load_precision(path / 'precision.npz')
        self.assertTrue(summary['ground_truth_available'])
        np.testing.assert_array_equal(np.stack([row[0] for row in truth]), truth0)
        np.testing.assert_array_equal(np.stack([row[1] for row in truth]), truth1)

    def test_outer_stop_reason_distinguishes_increase_from_tolerance(self):
        sequence = [[np.eye(2)] for _ in range(2)]
        for values, reason in [([4., 4.], 'objective_tolerance'),
                               ([4., 5.], 'objective_increase')]:
            with self.subTest(reason=reason), contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                stack.enter_context(patch.object(model_module, 'update_one_mode_tvgl',
                                                 return_value=(sequence, 0.)))
                if hasattr(model_module, 'objective_terms'):
                    stack.enter_context(patch.object(model_module, 'objective_terms', side_effect=[
                        dict(total=v, data_fit=v, sparsity=0., temporal=0.) for v in values]))
                else:
                    stack.enter_context(patch.object(model_module, 'objective_function', side_effect=values))
                info = {}
                model_module.kronecker_timevarying_graphical_lasso(
                    np.zeros((2, 2)), sequence, [0.], [0.], None, None,
                    min_iter=2, max_iter=5, fit_info=info)
                self.assertEqual(info, dict(stop_reason=reason, n_iter=2, final_objective=values[-1]))


if __name__ == '__main__':
    unittest.main()
