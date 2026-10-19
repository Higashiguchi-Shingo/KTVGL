import unittest
from functools import partial
from unittest.mock import patch

import numpy as np

from ktvgl.methods.kronecker_tvgl import kroneckergl as ktvgl


class ObjectiveSampleWeightingTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)
        self.dims = (2, 3)
        self.thetas = []
        for _ in range(3):
            factors = []
            for d in self.dims:
                a = self.rng.normal(size=(d, d))
                factors.append(a @ a.T + np.eye(d))
            self.thetas.append(factors)
        self.lambdas = [0.2, 0.3]
        self.rhos = [0.4, 0.5]

    def direct_data_fit(self, samples):
        """Sum Gaussian losses using the full Kronecker precision matrix."""
        total = 0.0
        for batch, factors in zip(samples, self.thetas):
            precision = np.kron(*factors)
            logdet = np.linalg.slogdet(precision)[1]
            for sample in batch:
                x = sample.ravel()
                total += x @ precision @ x - logdet
        return total

    def assert_direct_likelihood(self, input_data, batches):
        # Exclude numerical jitter only for comparison to the exact likelihood.
        without_jitter = partial(ktvgl.compute_Shat_multi_sample, jitter_scale=0)
        with patch.object(ktvgl, "compute_Shat_multi_sample", without_jitter):
            for mode in range(len(self.dims)):
                with self.subTest(mode=mode):
                    terms = ktvgl.objective_terms(
                        input_data, self.thetas, self.lambdas, self.rhos, mode=mode
                    )
                    np.testing.assert_allclose(
                        terms["data_fit"], self.direct_data_fit(batches), rtol=1e-12
                    )

    def test_single_sample_input(self):
        samples = self.rng.normal(size=(3, *self.dims))
        self.assert_direct_likelihood(samples, samples[:, None])

    def test_variable_sample_counts(self):
        batches = [self.rng.normal(size=(n, *self.dims)) for n in (1, 3, 5)]
        self.assert_direct_likelihood(batches, batches)

    def test_logdet_is_weighted_even_for_zero_observations(self):
        batches = [np.zeros((n, *self.dims)) for n in (2, 4, 1)]
        self.assert_direct_likelihood(batches, batches)

    def test_repeated_samples_scale_only_data_fit(self):
        samples = self.rng.normal(size=(3, *self.dims))
        repeated = np.repeat(samples[:, None], 5, axis=1)
        original = ktvgl.objective_terms(
            samples, self.thetas, self.lambdas, self.rhos
        )
        scaled = ktvgl.objective_terms(
            repeated, self.thetas, self.lambdas, self.rhos
        )
        np.testing.assert_allclose(scaled["data_fit"], 5 * original["data_fit"])
        for key in ("sparsity", "temporal"):
            self.assertEqual(scaled[key], original[key])
        self.assertAlmostEqual(
            scaled["total"],
            5 * original["data_fit"] + original["sparsity"] + original["temporal"],
        )
        self.assertEqual(
            ktvgl.objective_function(repeated, self.thetas, self.lambdas, self.rhos),
            scaled["total"],
        )


if __name__ == "__main__":
    unittest.main()
