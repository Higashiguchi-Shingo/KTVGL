import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.covariance import graphical_lasso


def load_solver(name, relative_path):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ktvgl = load_solver("ktvgl_solver", "src/methods/KroneckerTVGL/solver.py")
tvgl = load_solver("tvgl_solver", "src/methods/TVGL/model.py")


class OffDiagonalL1Tests(unittest.TestCase):
    def test_matrix_preserves_diagonal_and_input(self):
        a = np.array([[0.1, -0.8, 0.2], [-0.8, -0.1, 0.5], [0.2, 0.5, 2.]])
        original = a.copy()
        expected = np.array([[0.1, -0.5, 0.], [-0.5, -0.1, 0.2], [0., 0.2, 2.]])
        for module in (ktvgl, tvgl):
            with self.subTest(module=module.__name__):
                actual = module.soft_thresholding_off_diagonal(a, 0.3)
                np.testing.assert_allclose(actual, expected)
                np.testing.assert_array_equal(a, original)

    def test_batch_with_matrix_thresholds(self):
        a = np.array([[[0.1, -0.8], [-0.8, 2.]], [[-0.2, 0.4], [0.4, 3.]]])
        thresholds = np.array([[[10., 0.3], [0.3, 10.]], [[10., 0.5], [0.5, 10.]]])
        expected = np.array([[[0.1, -0.5], [-0.5, 2.]], [[-0.2, 0.], [0., 3.]]])
        for module in (ktvgl, tvgl):
            with self.subTest(module=module.__name__):
                np.testing.assert_allclose(
                    module.soft_thresholding_off_diagonal(a, thresholds), expected
                )

    def test_temporal_l1_still_thresholds_diagonal(self):
        a = np.array([[[0.1, -0.8], [-0.8, 2.]]])
        expected = np.array([[[0., -0.5], [-0.5, 1.7]]])
        for module in (ktvgl, tvgl):
            with self.subTest(module=module.__name__):
                _, temporal_prox = module.check_norm_prox("l1")
                np.testing.assert_allclose(temporal_prox(a, 0.3), expected)

    def test_all_solver_paths_use_off_diagonal_prox(self):
        covariance = np.array([[[2., 0.6], [0.6, 1.5]]] * 3)
        paths = (
            (ktvgl, "time_varying_graphical_lasso"),
            (tvgl, "time_varying_graphical_lasso"),
            (tvgl, "time_graphical_lasso_laplacian"),
        )
        for module, name in paths:
            with self.subTest(module=module.__name__, solver=name):
                prox = module.soft_thresholding_off_diagonal
                with patch.object(module, "soft_thresholding_off_diagonal", wraps=prox) as spy:
                    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        getattr(module, name)(
                            covariance, alpha=0.2, beta=0., max_iter=1, verbose=False
                        )
                    self.assertEqual(spy.call_count, 1)

    def test_no_temporal_penalty_matches_graphical_lasso_and_kkt(self):
        covariance = np.array([
            [[2., 0.6], [0.6, 1.5]],
            [[1.5, -0.5], [-0.5, 1.2]],
            [[1.8, 0.4], [0.4, 1.4]],
        ])
        alpha = 0.2
        expected = np.array([
            graphical_lasso(s, alpha=alpha, tol=1e-12, enet_tol=1e-12)[1]
            for s in covariance
        ])
        for module in (ktvgl, tvgl):
            with self.subTest(module=module.__name__):
                # Tight tolerances also verify the solution against KKT conditions.
                with contextlib.redirect_stdout(io.StringIO()):
                    result = module.time_varying_graphical_lasso(
                        covariance, alpha=alpha, beta=0., max_iter=1000,
                        tol=1e-11, rtol=1e-11, verbose=False,
                    )
                precision = result if module is ktvgl else result[0]
                np.testing.assert_allclose(precision, expected, rtol=1e-7, atol=1e-9)
                gradient = covariance - np.linalg.inv(precision)
                np.testing.assert_allclose(
                    np.diagonal(gradient, axis1=-2, axis2=-1), 0., atol=1e-8
                )
                np.testing.assert_allclose(
                    gradient[:, 0, 1] + alpha * np.sign(precision[:, 0, 1]),
                    0., atol=1e-8,
                )


if __name__ == "__main__":
    unittest.main()
