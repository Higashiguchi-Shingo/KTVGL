import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from ktvgl.methods.kronecker_tvgl import kroneckergl as ktvgl


class TemporalObjectiveTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31)
        self.x = rng.normal(size=(3, 2, 3))
        self.thetas = []
        for t in range(3):
            row = []
            for d in (2, 3):
                a = rng.normal(size=(d, d))
                row.append(a @ a.T + (t + 1) * np.eye(d))
            self.thetas.append(row)
        self.lambdas = [0.2, 0.3]
        self.rhos = [0.4, 0.7]

    def expected_temporal(self, psi):
        value = 0.
        for t in range(1, 3):
            for m in range(2):
                delta = self.thetas[t][m] - self.thetas[t - 1][m]
                value += self.rhos[m] * np.sum(
                    np.abs(delta) if psi == "l1" else delta ** 2
                )
        return value

    def assert_objective(self, evaluate):
        base = evaluate([0., 0.], psi="laplacian")
        for psi in ("l1", "laplacian"):
            with self.subTest(psi=psi):
                self.assertAlmostEqual(
                    evaluate(self.rhos, psi=psi), base + self.expected_temporal(psi)
                )
        self.assertEqual(evaluate(self.rhos), evaluate(self.rhos, psi="laplacian"))
        with self.assertRaises(ValueError):
            evaluate(self.rhos, psi="unsupported")

    def test_objective_uses_selected_temporal_penalty(self):
        self.assert_objective(lambda rhos, **kw: ktvgl.objective_function(
            self.x, self.thetas, self.lambdas, rhos, **kw
        ))
        if hasattr(ktvgl, "objective_terms"):
            terms = [ktvgl.objective_terms(
                self.x, self.thetas, self.lambdas, self.rhos, psi=psi
            ) for psi in ("l1", "laplacian")]
            for psi, value in zip(("l1", "laplacian"), terms):
                self.assertAlmostEqual(value["temporal"], self.expected_temporal(psi))
                self.assertAlmostEqual(value["total"], sum(value[k] for k in (
                    "data_fit", "sparsity", "temporal"
                )))
            for key in ("data_fit", "sparsity"):
                self.assertEqual(terms[0][key], terms[1][key])

    @unittest.skipUnless(hasattr(ktvgl, "objective_function_gamma"), "main-only gamma API")
    def test_gamma_objective_and_scaled_evaluation_use_selected_penalty(self):
        gamma = np.array([0.8, 1., 1.2])
        self.assert_objective(lambda rhos, **kw: ktvgl.objective_function_gamma(
            self.x, self.thetas, self.lambdas, rhos, gamma, **kw
        ))
        self.assert_objective(lambda rhos, **kw: ktvgl.scale_objective_with_gamma(
            self.x, self.thetas, self.lambdas, rhos, gauge=None, **kw
        ))

    def assert_outer_loop(self, use_gamma=False):
        # Isolate the temporal term. These updates leave L1 unchanged (2 -> 2),
        # but decrease squared Frobenius (4 -> 2), so stopping must differ.
        first = [[np.eye(2)], [np.diag([3., 1.])]]
        second = [[np.eye(2)], [np.diag([2., 2.])]]
        covariance_name = "compute_Shat_multi_sample" if hasattr(ktvgl, "compute_Shat_multi_sample") else "compute_Shat"
        for psi, expected in (("l1", [2., 2.]), ("laplacian", [4., 2., 2.])):
            with self.subTest(psi=psi, use_gamma=use_gamma):
                output = io.StringIO()
                history = []
                kwargs = {}
                if hasattr(ktvgl, "objective_terms"):
                    kwargs["objective_history"] = history
                if use_gamma:
                    kwargs["use_gamma"] = True
                with contextlib.ExitStack() as stack:
                    update = stack.enter_context(patch.object(
                        ktvgl, "update_one_mode_tvgl",
                        side_effect=[(first, 1.), (second, 1.), (second, 0.)],
                    ))
                    stack.enter_context(patch.object(ktvgl, covariance_name, return_value=np.zeros((2, 2, 2))))
                    stack.enter_context(patch.object(ktvgl, "logdet", return_value=0.))
                    if use_gamma:
                        stack.enter_context(patch.object(ktvgl, "update_gamma", return_value=np.ones(2)))
                    stack.enter_context(contextlib.redirect_stdout(output))
                    ktvgl.kronecker_timevarying_graphical_lasso(
                        np.zeros((2, 2)), first, [0.], [1.],
                        tvgl_solver=None, compute_Shat_func=None,
                        max_iter=3, min_iter=2, tol_flipflop=1e-4,
                        early_stopping=True, psi=psi, **kwargs,
                    )
                self.assertEqual(update.call_count, len(expected))
                self.assertTrue(all(call.kwargs["psi"] == psi for call in update.call_args_list))
                if hasattr(ktvgl, "objective_terms"):
                    self.assertEqual([entry["total"] for entry in history], expected)
                    self.assertEqual([entry["temporal"] for entry in history], expected)
                    logged = [float(line.split(":", 1)[1].split("|")[0]) for line in
                              output.getvalue().splitlines() if line.startswith("Objective total:")]
                else:
                    logged = [float(line.split(":", 1)[1]) for line in
                              output.getvalue().splitlines() if line.startswith("Score:")]
                self.assertEqual(logged, expected)

    def test_outer_loop_logs_and_stops_using_selected_penalty(self):
        self.assert_outer_loop()

    @unittest.skipUnless(hasattr(ktvgl, "objective_function_gamma"), "main-only gamma API")
    def test_gamma_outer_loop_logs_and_stops_using_selected_penalty(self):
        self.assert_outer_loop(use_gamma=True)


if __name__ == "__main__":
    unittest.main()
