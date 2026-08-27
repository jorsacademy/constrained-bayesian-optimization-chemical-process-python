import math
import unittest

import numpy as np

from constrained_bayesian_optimization_cstr import (
    ConstrainedBayesianOptimizer,
    ParallelReactionCSTR,
    structured_reference_candidate,
)


class CSTRModelTests(unittest.TestCase):
    def test_known_point_is_feasible_and_physical(self):
        process = ParallelReactionCSTR()
        e = process.evaluate(np.array([392.0, 0.8, 1.95, 0.35]))
        self.assertTrue(e.feasible)
        self.assertGreater(e.product_rate, 0.0)
        self.assertGreater(e.byproduct_rate, 0.0)
        self.assertGreater(e.conversion, 0.0)
        self.assertLess(e.conversion, 1.0)

    def test_byproduct_fraction_increases_with_temperature(self):
        process = ParallelReactionCSTR()
        low = process.evaluate(np.array([370.0, 0.8, 1.5, 0.8]))
        high = process.evaluate(np.array([430.0, 0.8, 1.5, 0.8]))
        self.assertGreater(high.byproduct_fraction, low.byproduct_fraction)

    def test_analytical_selectivity_temperature_limit(self):
        process = ParallelReactionCSTR()
        p = process.byproduct_fraction_limit
        ratio = p / (1.0 - p)
        temperature = -(
            process.E_byproduct - process.E_desired
        ) / (
            process.R
            * math.log(
                ratio / (process.A_byproduct / process.A_desired)
            )
        )
        self.assertTrue(math.isclose(
            temperature,
            392.395905966207,
            abs_tol=1e-6,
        ))

        e = process.evaluate(np.array([temperature, 1.0, 1.5, 0.8]))
        self.assertTrue(math.isclose(
            e.byproduct_fraction,
            process.byproduct_fraction_limit,
            abs_tol=1e-10,
        ))

    def test_structured_reference_regression(self):
        process = ParallelReactionCSTR()
        x, e = structured_reference_candidate(process)

        self.assertLessEqual(e.heat_constraint, 1e-7)
        self.assertLessEqual(e.byproduct_constraint, 1e-9)
        self.assertTrue(math.isclose(
            e.profit_per_hour,
            1455.9885867,
            abs_tol=1e-3,
        ))
        self.assertTrue(math.isclose(x[2], 2.0, abs_tol=1e-12))

    def test_short_bo_is_reproducible_and_recommendation_is_feasible(self):
        process = ParallelReactionCSTR()
        kwargs = dict(
            initial_points=10,
            bo_iterations=8,
            candidate_pool=512,
        )

        opt1 = ConstrainedBayesianOptimizer(process, seed=42)
        opt2 = ConstrainedBayesianOptimizer(process, seed=42)
        run1 = opt1.run(**kwargs)
        run2 = opt2.run(**kwargs)

        np.testing.assert_allclose(run1.X, run2.X)
        np.testing.assert_allclose(
            run1.observed_profit,
            run2.observed_profit,
        )
        np.testing.assert_allclose(
            run1.observed_constraints,
            run2.observed_constraints,
        )

        recommendation = opt1.recommend(
            run1,
            min_probability_feasible=0.90,
        )
        self.assertGreaterEqual(
            recommendation.probability_feasible,
            0.90,
        )
        true_eval = process.evaluate(recommendation.x)
        self.assertTrue(true_eval.feasible)


if __name__ == "__main__":
    unittest.main()
