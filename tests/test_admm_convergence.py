import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from test_off_diagonal_l1 import ktvgl, tvgl


class DiagonalConvergenceTests(unittest.TestCase):
    def solve(self, module, covariance, beta):
        with patch.object(module, "convergence", wraps=module.convergence) as checks:
            with contextlib.redirect_stdout(io.StringIO()):
                result = module.time_varying_graphical_lasso(
                    covariance, alpha=0.2, beta=beta,
                    prev_matrices=np.broadcast_to(np.eye(covariance.shape[-1]), covariance.shape).copy(),
                    max_iter=2000, tol=1e-9, rtol=1e-9, verbose=False,
                )
        history = [call.kwargs for call in checks.call_args_list if "rnorm" in call.kwargs]
        self.assertGreater(len(history), 1)
        self.assertLess(len(history), 2000)
        self.assertLessEqual(history[-1]["rnorm"], history[-1]["e_pri"])
        self.assertLessEqual(history[-1]["snorm"], history[-1]["e_dual"])
        return result if module is ktvgl else result[0], history

    def test_diagonal_only_problem_does_not_stop_before_inverse_covariance(self):
        covariance = np.array([np.diag(d) for d in ([2., 0.5], [3., 1.5], [0.8, 4.])])
        for module in (ktvgl, tvgl):
            with self.subTest(module=module.__name__):
                precision, history = self.solve(module, covariance, beta=0.)
                self.assertGreater(history[0]["snorm"], history[0]["e_dual"])
                np.testing.assert_allclose(precision, np.linalg.inv(covariance), atol=1e-7)

    def test_temporal_penalty_checks_diagonal_consensus_and_relative_tolerances(self):
        beta = 0.4
        diagonal = np.array([[0.8, 1.2], [1., 1.], [1.1, 0.9]])
        # Construct covariance so these diagonal precisions satisfy stationarity:
        # S - 1 / x + 2 * beta * L x = 0.
        temporal_gradient = np.zeros_like(diagonal)
        temporal_gradient[:-1] += 2 * beta * (diagonal[:-1] - diagonal[1:])
        temporal_gradient[1:] += 2 * beta * (diagonal[1:] - diagonal[:-1])
        covariance = np.array([np.diag(s) for s in 1 / diagonal - temporal_gradient])
        expected = np.array([np.diag(d) for d in diagonal])
        absolute_tolerance = np.sqrt(covariance.size + 2 * covariance[:-1].size) * 1e-9
        for module in (ktvgl, tvgl):
            with self.subTest(module=module.__name__):
                precision, history = self.solve(module, covariance, beta)
                first = history[0]
                self.assertGreater(first["rnorm"], first["e_pri"])
                self.assertGreater(first["snorm"], first["e_dual"])
                self.assertGreater(first["e_pri"], absolute_tolerance)
                self.assertGreater(first["e_dual"], absolute_tolerance)
                np.testing.assert_allclose(precision, expected, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
