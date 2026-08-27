from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize_scalar
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


@dataclass(frozen=True)
class ProcessEvaluation:
    profit_per_hour: float
    heat_constraint: float
    byproduct_constraint: float
    product_rate: float
    byproduct_rate: float
    conversion: float
    byproduct_fraction: float
    heat_generation: float
    heat_removal_capacity: float

    @property
    def feasible(self) -> bool:
        return self.heat_constraint <= 0.0 and self.byproduct_constraint <= 0.0


@dataclass
class OptimizationRun:
    X: np.ndarray
    observed_profit: np.ndarray
    observed_constraints: np.ndarray
    true_profit: np.ndarray
    true_constraints: np.ndarray

    @property
    def true_feasible_mask(self) -> np.ndarray:
        return np.all(self.true_constraints <= 0.0, axis=1)

    def best_true_feasible_index(self) -> int:
        mask = self.true_feasible_mask
        if not np.any(mask):
            raise RuntimeError("no truly feasible evaluated point")
        idx = np.flatnonzero(mask)
        return int(idx[np.argmax(self.true_profit[mask])])


@dataclass(frozen=True)
class Recommendation:
    index: int
    x: np.ndarray
    predicted_profit: float
    probability_feasible: float


class ParallelReactionCSTR:
    """Stylized steady-state CSTR with A->B desired and A->C side reaction."""

    bounds = np.array([[360.0, 440.0], [0.2, 2.0], [0.8, 2.0], [0.2, 1.0]])
    R = 8.314
    reactor_volume_m3 = 10.0
    A_desired, E_desired = 2.5e7, 55_000.0
    A_byproduct, E_byproduct = 7.0e8, 70_000.0
    delta_h_desired, delta_h_byproduct = 65.0, 80.0
    byproduct_fraction_limit = 0.22
    product_value_per_kmol = 300.0
    feed_cost_per_kmol = 55.0
    byproduct_disposal_per_kmol = 90.0
    electricity_cost_per_mwh = 80.0
    coolant_cost_coefficient = 160.0
    feed_temperature_K = 300.0
    feed_heat_capacity_MJ_m3_K = 4.0
    heat_removal_base, heat_removal_slope = 500.0, 1400.0

    def evaluate(self, x: np.ndarray) -> ProcessEvaluation:
        x = np.asarray(x, dtype=float)
        if x.shape != (4,):
            raise ValueError("x must contain four decision variables")
        if np.any(x < self.bounds[:, 0]) or np.any(x > self.bounds[:, 1]):
            raise ValueError("decision vector outside process bounds")

        T, tau, c0, coolant = x
        kb = self.A_desired * math.exp(-self.E_desired / (self.R * T))
        kc = self.A_byproduct * math.exp(-self.E_byproduct / (self.R * T))
        ca = c0 / (1.0 + tau * (kb + kc))
        cb, cc = tau * kb * ca, tau * kc * ca
        flow = self.reactor_volume_m3 / tau
        b_rate, c_rate, feed_rate = flow * cb, flow * cc, flow * c0
        conversion = 1.0 - ca / c0
        by_frac = c_rate / max(b_rate + c_rate, 1e-12)
        heat = b_rate * self.delta_h_desired + c_rate * self.delta_h_byproduct
        heat_cap = self.heat_removal_base + self.heat_removal_slope * coolant
        heating_mwh_h = (
            flow * self.feed_heat_capacity_MJ_m3_K * (T - self.feed_temperature_K) / 3600.0
        )
        profit = (
            self.product_value_per_kmol * b_rate
            - self.feed_cost_per_kmol * feed_rate
            - self.byproduct_disposal_per_kmol * c_rate
            - self.electricity_cost_per_mwh * heating_mwh_h
            - self.coolant_cost_coefficient * coolant**2
        )
        return ProcessEvaluation(
            float(profit), float(heat - heat_cap),
            float(by_frac - self.byproduct_fraction_limit),
            float(b_rate), float(c_rate), float(conversion), float(by_frac),
            float(heat), float(heat_cap),
        )

    def evaluate_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals = [self.evaluate(x) for x in np.asarray(X, dtype=float)]
        return (
            np.array([v.profit_per_hour for v in vals]),
            np.array([[v.heat_constraint, v.byproduct_constraint] for v in vals]),
        )

    def noisy_observation(
        self, x: np.ndarray, rng: np.random.Generator,
        profit_noise_std: float = 20.0,
        heat_noise_std: float = 15.0,
        byproduct_noise_std: float = 0.004,
    ) -> tuple[float, np.ndarray, ProcessEvaluation]:
        truth = self.evaluate(x)
        return (
            float(truth.profit_per_hour + rng.normal(0, profit_noise_std)),
            np.array([
                truth.heat_constraint + rng.normal(0, heat_noise_std),
                truth.byproduct_constraint + rng.normal(0, byproduct_noise_std),
            ]),
            truth,
        )


class ConstrainedBayesianOptimizer:
    """Independent GPs + constrained Expected Improvement."""

    def __init__(self, process: ParallelReactionCSTR, seed: int = 42, xi: float = 0.01):
        self.process, self.seed, self.xi = process, int(seed), float(xi)
        self.rng = np.random.default_rng(seed)
        self.noise_std = np.array([20.0, 15.0, 0.004])

    def _scale(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self.process.bounds[:, 0], self.process.bounds[:, 1]
        return (X - lo) / (hi - lo)

    def _sample(self, n: int, seed: int) -> np.ndarray:
        u = qmc.LatinHypercube(d=4, seed=seed).random(n)
        return qmc.scale(u, self.process.bounds[:, 0], self.process.bounds[:, 1])

    @staticmethod
    def _fit_gp(X: np.ndarray, y: np.ndarray, noise_std: float, seed: int):
        mean, std = float(np.mean(y)), max(float(np.std(y)), 1e-10)
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=np.ones(X.shape[1]), length_scale_bounds=(0.05, 5.0), nu=2.5
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=max((noise_std / std) ** 2, 1e-8),
            random_state=seed,
            n_restarts_optimizer=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(X, (y - mean) / std)
        return gp, mean, std

    @staticmethod
    def _predict(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gp, mean, std = model
        mu, sigma = gp.predict(X, return_std=True)
        return mu * std + mean, sigma * std

    def _posterior(self, run: OptimizationRun, X: np.ndarray):
        train_x, pred_x = self._scale(run.X), self._scale(X)
        obj = self._fit_gp(train_x, run.observed_profit, self.noise_std[0], self.seed)
        mu_f, sigma_f = self._predict(obj, pred_x)
        p_feas = np.ones(len(X))
        for j in range(2):
            gp = self._fit_gp(
                train_x, run.observed_constraints[:, j], self.noise_std[j + 1], self.seed + 10 + j
            )
            mu_g, sigma_g = self._predict(gp, pred_x)
            p_feas *= norm.cdf(-mu_g / np.maximum(sigma_g, 1e-12))
        return mu_f, np.maximum(sigma_f, 1e-12), p_feas

    def _acquisition(self, run: OptimizationRun, candidates: np.ndarray) -> np.ndarray:
        mu, sigma, p_feas = self._posterior(run, candidates)
        observed_feasible = np.all(run.observed_constraints <= 0.0, axis=1)
        if not np.any(observed_feasible):
            scaled = (mu - mu.min()) / (np.ptp(mu) + 1e-12)
            return p_feas * (1.0 + 0.05 * scaled)
        incumbent = float(np.max(run.observed_profit[observed_feasible]))
        imp = mu - incumbent - self.xi
        z = imp / sigma
        ei = np.maximum(imp * norm.cdf(z) + sigma * norm.pdf(z), 0.0)
        return ei * p_feas

    def run(self, initial_points: int = 12, bo_iterations: int = 24, candidate_pool: int = 4096) -> OptimizationRun:
        if initial_points < 4 or bo_iterations < 0 or candidate_pool < 100:
            raise ValueError("invalid BO configuration")
        X, yp, yg, tp, tg = [], [], [], [], []

        def observe(x):
            y, g, truth = self.process.noisy_observation(x, self.rng)
            X.append(np.array(x, copy=True)); yp.append(y); yg.append(g)
            tp.append(truth.profit_per_hour)
            tg.append([truth.heat_constraint, truth.byproduct_constraint])

        for x in self._sample(initial_points, self.seed):
            observe(x)
        for k in range(bo_iterations):
            run = OptimizationRun(np.asarray(X), np.asarray(yp), np.asarray(yg), np.asarray(tp), np.asarray(tg))
            cand = self._sample(candidate_pool, self.seed + 1000 + k)
            observe(cand[int(np.argmax(self._acquisition(run, cand)))])
        return OptimizationRun(np.asarray(X), np.asarray(yp), np.asarray(yg), np.asarray(tp), np.asarray(tg))

    def recommend(self, run: OptimizationRun, min_probability_feasible: float = 0.95) -> Recommendation:
        if not 0 < min_probability_feasible < 1:
            raise ValueError("probability threshold must be in (0,1)")
        mu, _, p = self._posterior(run, run.X)
        eligible = np.flatnonzero(p >= min_probability_feasible)
        idx = int(eligible[np.argmax(mu[eligible])]) if len(eligible) else int(np.argmax(p))
        return Recommendation(idx, run.X[idx].copy(), float(mu[idx]), float(p[idx]))


def run_random_search(process: ParallelReactionCSTR, evaluations: int, seed: int) -> OptimizationRun:
    rng = np.random.default_rng(seed)
    u = qmc.LatinHypercube(d=4, seed=seed).random(evaluations)
    X = qmc.scale(u, process.bounds[:, 0], process.bounds[:, 1])
    yp, yg, tp, tg = [], [], [], []
    for x in X:
        y, g, truth = process.noisy_observation(x, rng)
        yp.append(y); yg.append(g); tp.append(truth.profit_per_hour)
        tg.append([truth.heat_constraint, truth.byproduct_constraint])
    return OptimizationRun(X, np.asarray(yp), np.asarray(yg), np.asarray(tp), np.asarray(tg))


def structured_reference_candidate(process: ParallelReactionCSTR) -> tuple[np.ndarray, ProcessEvaluation]:
    """Structural cross-check using the active by-product selectivity boundary."""
    p = process.byproduct_fraction_limit
    target_ratio = p / (1.0 - p)
    T = -(process.E_byproduct - process.E_desired) / (
        process.R * math.log(target_ratio / (process.A_byproduct / process.A_desired))
    )
    T = float(np.clip(T, process.bounds[0, 0], process.bounds[0, 1]))
    c0 = float(process.bounds[2, 1])

    def point(tau):
        base = process.evaluate(np.array([T, tau, c0, process.bounds[3, 0]]))
        coolant = np.clip(
            (base.heat_generation - process.heat_removal_base) / process.heat_removal_slope,
            process.bounds[3, 0], process.bounds[3, 1],
        )
        x = np.array([T, tau, c0, coolant])
        return x, process.evaluate(x)

    def f(tau):
        _, e = point(tau)
        return -e.profit_per_hour if e.heat_constraint <= 1e-8 else 1e12 + e.heat_constraint * 1e6

    res = minimize_scalar(f, bounds=tuple(process.bounds[1]), method="bounded", options={"xatol": 1e-12})
    x, e = point(float(res.x))
    if e.heat_constraint > 1e-7 or e.byproduct_constraint > 1e-9:
        raise RuntimeError("structured reference is infeasible")
    return x, e


def approximate_reference_optimum(
    process: ParallelReactionCSTR, sobol_power: int = 15, seed: int = 123
) -> tuple[np.ndarray, ProcessEvaluation]:
    """Approximate reference: Sobol + constrained differential evolution + structure."""
    sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
    X = qmc.scale(sampler.random_base2(sobol_power), process.bounds[:, 0], process.bounds[:, 1])
    profit, constraints = process.evaluate_batch(X)
    feasible = np.all(constraints <= 0.0, axis=1)
    idx = np.flatnonzero(feasible)[np.argmax(profit[feasible])]
    candidates = [(X[idx], process.evaluate(X[idx])), structured_reference_candidate(process)]

    def objective(x):
        return -process.evaluate(np.asarray(x)).profit_per_hour

    def g(x):
        e = process.evaluate(np.asarray(x))
        return np.array([e.heat_constraint, e.byproduct_constraint])

    de = differential_evolution(
        objective,
        [tuple(b) for b in process.bounds],
        constraints=(NonlinearConstraint(g, [-np.inf, -np.inf], [0.0, 0.0]),),
        rng=np.random.default_rng(seed + 1),
        maxiter=120, popsize=12, tol=1e-8, polish=False, workers=1,
    )
    de_x = np.asarray(de.x)
    de_e = process.evaluate(de_x)
    if de_e.heat_constraint <= 1e-5 and de_e.byproduct_constraint <= 1e-8:
        candidates.append((de_x, de_e))
    return max(candidates, key=lambda z: z[1].profit_per_hour)


def self_test() -> None:
    p = ParallelReactionCSTR()
    e = p.evaluate(np.array([392.0, 0.8, 1.95, 0.35]))
    assert e.feasible and 0 < e.conversion < 1
    low = p.evaluate(np.array([370.0, 0.8, 1.5, 0.8]))
    high = p.evaluate(np.array([430.0, 0.8, 1.5, 0.8]))
    assert high.byproduct_fraction > low.byproduct_fraction
    r1 = ConstrainedBayesianOptimizer(p, seed=7).run(8, 3, 256)
    r2 = ConstrainedBayesianOptimizer(p, seed=7).run(8, 3, 256)
    assert np.allclose(r1.X, r2.X)
    x, ref = structured_reference_candidate(p)
    assert math.isclose(x[0], 392.395905966207, abs_tol=1e-6)
    assert 1450 < ref.profit_per_hour < 1460
    print("CSTR + constrained Bayesian optimization self-test: OK")


def benchmark(seed=42, initial_points=12, bo_iterations=24, candidate_pool=4096, reference_sobol_power=15):
    p = ParallelReactionCSTR()
    opt = ConstrainedBayesianOptimizer(p, seed)
    bo = opt.run(initial_points, bo_iterations, candidate_pool)
    rs = run_random_search(p, initial_points + bo_iterations, seed + 5000)
    rec = opt.recommend(bo, 0.95)
    rec_true = p.evaluate(rec.x)
    bi, ri = bo.best_true_feasible_index(), rs.best_true_feasible_index()
    ref_x, ref = approximate_reference_optimum(p, reference_sobol_power, seed + 9000)
    print("=" * 76)
    print("CONSTRAINED BAYESIAN OPTIMIZATION — PARALLEL-REACTION CSTR")
    print("=" * 76)
    print(f"BO evaluations              : {len(bo.X)}")
    print(f"Random-search evaluations   : {len(rs.X)}")
    print(f"Recommendation P(feasible)  : {rec.probability_feasible:.4f}")
    print(f"Recommendation true profit  : {rec_true.profit_per_hour:.3f} $/h")
    print(f"Recommendation true feasible: {rec_true.feasible}")
    print(f"Best BO true profit         : {bo.true_profit[bi]:.3f} $/h")
    print(f"Best random true profit     : {rs.true_profit[ri]:.3f} $/h")
    print(f"Reference x                 : {np.round(ref_x, 6)}")
    print(f"Reference profit            : {ref.profit_per_hour:.3f} $/h")
    print("Reference is approximate; no global-optimality certificate is claimed.")


def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument("--self-test", action="store_true")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--initial-points", type=int, default=12)
    a.add_argument("--bo-iterations", type=int, default=24)
    a.add_argument("--candidate-pool", type=int, default=4096)
    a.add_argument("--reference-sobol-power", type=int, default=15)
    return a.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        benchmark(args.seed, args.initial_points, args.bo_iterations, args.candidate_pool, args.reference_sobol_power)
