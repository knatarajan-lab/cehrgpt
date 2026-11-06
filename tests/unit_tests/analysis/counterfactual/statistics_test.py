import math
import unittest

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

from cehrgpt.analysis.counterfactual.hazard_ratio_estimation import (
    fit_cox,
    fit_cox_details,
    kaplan_meier,
)


def simulate(n=300, seed=0, ties=False, true_log_hr=0.5):
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, n).astype(float)
    latent = rng.exponential(1.0 / np.exp(true_log_hr * treatment))
    censor = rng.exponential(1.5, n)
    time = np.minimum(latent, censor)
    event = (latent <= censor).astype(int)
    if ties:
        time = np.ceil(time * 4)
    return time, event, treatment


class KaplanMeierTest(unittest.TestCase):
    def check(self, time, event):
        times, survival = kaplan_meier(time, event)
        self.assertEqual((times[0], survival[0]), (0, 1))
        fitted = KaplanMeierFitter().fit(time, event)
        expected = fitted.survival_function_at_times(times).to_numpy()
        np.testing.assert_allclose(survival, expected, atol=1e-12)

    def test_it_matches_lifelines_without_ties(self):
        time, event, _ = simulate()
        self.check(time, event)

    def test_it_matches_lifelines_with_ties_and_censoring_at_event_times(self):
        time, event, _ = simulate(ties=True)
        self.check(time, event)

    def test_a_censored_observation_tied_with_an_event_stays_at_risk_at_that_time(self):
        times, survival = kaplan_meier(np.array([1.0, 1.0, 2.0]), np.array([1, 0, 1]))
        np.testing.assert_allclose(survival, [1.0, 2 / 3, 0.0])

    def test_without_events_survival_stays_at_one(self):
        times, survival = kaplan_meier(np.array([1.0, 2.0]), np.array([0, 0]))
        np.testing.assert_allclose(survival, [1.0])


class CoxTest(unittest.TestCase):
    def lifelines_fit(self, time, event, treatment):
        fitter = CoxPHFitter()
        fitter.fit(pd.DataFrame({"T": time, "E": event, "x": treatment}), "T", "E")
        return fitter

    def test_the_estimate_and_model_standard_error_match_lifelines(self):
        time, event, treatment = simulate()
        ours = fit_cox_details(time, event, treatment)
        theirs = self.lifelines_fit(time, event, treatment)
        self.assertAlmostEqual(ours["beta"], float(theirs.params_["x"]), places=4)
        self.assertAlmostEqual(ours["se_model"], float(theirs.standard_errors_["x"]), places=4)

    def test_the_robust_standard_error_matches_lifelines_up_to_the_small_cluster_correction(self):
        time, event, treatment = simulate(n=200, seed=3)
        clusters = np.arange(len(time)) // 4
        # make observations in a cluster share a treatment so that the clustering matters
        treatment = treatment[np.repeat(np.arange(50) * 4, 4)]
        ours = fit_cox_details(time, event, treatment, clusters=clusters)
        frame = pd.DataFrame({"T": time, "E": event, "x": treatment, "c": clusters})
        fitter = CoxPHFitter()
        fitter.fit(frame, "T", "E", robust=True, cluster_col="c")
        n_clusters = len(np.unique(clusters))
        correction = math.sqrt(n_clusters / (n_clusters - 1))
        self.assertAlmostEqual(
            ours["se_robust"], float(fitter.standard_errors_["x"]) * correction, delta=1e-3
        )
        self.assertEqual(ours["n_clusters"], n_clusters)

    def test_clustering_widens_the_interval_when_a_cluster_shares_treatment_and_frailty(self):
        rng = np.random.default_rng(5)
        n_clusters, size = 100, 5
        clusters = np.repeat(np.arange(n_clusters), size)
        frailty = np.repeat(rng.gamma(0.5, 2.0, n_clusters), size)
        treatment = np.repeat(rng.integers(0, 2, n_clusters).astype(float), size)
        time = rng.exponential(1.0 / (frailty * np.exp(0.3 * treatment)))
        event = np.ones(len(time), dtype=int)
        details = fit_cox_details(time, event, treatment, clusters=clusters)
        robust_width = math.log(details["hr_upper_95"]) - math.log(details["hr_lower_95"])
        model_width = math.log(details["hr_upper_95_model"]) - math.log(details["hr_lower_95_model"])
        self.assertGreater(robust_width, model_width)

    def test_ties_follow_breslow(self):
        # Breslow partial likelihood, brute force
        time, event, treatment = simulate(n=60, seed=1, ties=True)
        betas = np.linspace(-2, 2, 4001)

        def loglik(beta):
            total = 0.0
            for t in np.unique(time[event == 1]):
                died = (time == t) & (event == 1)
                at_risk = time >= t
                total += beta * treatment[died].sum() - died.sum() * math.log(
                    np.exp(beta * treatment[at_risk]).sum()
                )
            return total

        best = betas[np.argmax([loglik(b) for b in betas])]
        self.assertAlmostEqual(fit_cox_details(time, event, treatment)["beta"], best, delta=2e-3)

    def test_fit_cox_returns_the_hazard_ratio_interval_and_p_value(self):
        time, event, treatment = simulate()
        hr, lower, upper, p = fit_cox(time, event, treatment)
        self.assertTrue(lower < hr < upper)
        self.assertGreater(hr, 1.0)
        self.assertLess(p, 0.05)

    def test_fit_cox_with_clusters_uses_the_robust_interval(self):
        time, event, treatment = simulate()
        clusters = np.arange(len(time)) // 3
        hr, lower, upper, _ = fit_cox(time, event, treatment, clusters=clusters)
        details = fit_cox_details(time, event, treatment, clusters=clusters)
        self.assertEqual((hr, lower, upper), (details["hr"], details["hr_lower_95"], details["hr_upper_95"]))


if __name__ == "__main__":
    unittest.main()
