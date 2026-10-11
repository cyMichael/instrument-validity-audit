"""Scientific regression checks: python3 -m unittest -v test_audit.py."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.stats import norm

import audit as A
import robustness as R


class PopulationTests(unittest.TestCase):
    def test_gaussian_violation_matches_full_support_quadrature(self):
        for sep in (-3.0, 0.0, 1.0, 3.0):
            g = A.DGP(alpha=0.10, sep=sep, tau_f=2.0)
            expected = []
            for d in (0, 1):
                val = quad(lambda y: max(g.pi_f * norm.pdf(y, g.mean("f", d))
                                        - g.pi_c * norm.pdf(y, g.mean("c", d)), 0.0),
                           -np.inf, np.inf, epsabs=1e-10)[0]
                expected.append(val)
            self.assertAlmostEqual(g.violation()["V"], max(expected), places=9)

    def test_gaussian_violation_is_translation_invariant(self):
        base = A.DGP(alpha=0.10, sep=3.0).violation()["V"]
        moved = A.DGP(alpha=0.10, sep=3.0, mu_base=100.0).violation()["V"]
        self.assertAlmostEqual(base, moved, places=14)

    def test_identical_outcomes_and_empty_strata(self):
        for alpha, pi_c in ((0.0, 0.30), (0.10, 0.0), (0.30, 0.10), (0.10, 0.30)):
            g = A.DGP(alpha=alpha, pi_c=pi_c, sep=0.0)
            self.assertAlmostEqual(g.violation()["V"], max(alpha - pi_c, 0.0))

    def test_zero_first_stage_has_undefined_wald_and_influence(self):
        for tau_f in (1.0, 2.0):
            g = A.DGP(alpha=0.30, tau_f=tau_f)
            self.assertTrue(np.isnan(g.wald()))
            self.assertTrue(np.isnan(g.influence()))
            self.assertTrue(np.isnan(g.influence_closed_form()))

    def test_partition_masses_include_tails_and_coarsening_lowers_V(self):
        for family in ("gaussian", "t3", "lognormal", "binary"):
            g = R.RobustDGP(family=family)
            m = g.cell_masses()
            self.assertAlmostEqual(sum(m["m11"] + m["m01"]), 1.0)
            self.assertAlmostEqual(sum(m["m10"] + m["m00"]), 1.0)
            self.assertLessEqual(A.cellwise_V(m), g.violation() + 1e-9)
        g = R.RobustDGP(family="t3")
        self.assertLessEqual(A.cellwise_V(g.cell_masses(6)), A.cellwise_V(g.cell_masses(12)) + 1e-12)
        self.assertLessEqual(A.cellwise_V(g.cell_masses(12)), A.cellwise_V(g.cell_masses(24)) + 1e-12)

    def test_robust_gaussian_quadrature_matches_analytic_result(self):
        for sep in (0.0, 1.0, 3.0):
            self.assertAlmostEqual(R.RobustDGP(sep=sep).violation(),
                                   A.DGP(alpha=0.10, sep=sep).violation()["V"], places=9)


class SamplingTests(unittest.TestCase):
    def test_deterministic_defiance_rejects_with_zero_variance(self):
        z = np.repeat([0, 1], 10)
        sample = dict(z=z, d=1 - z, y=np.zeros(20))
        self.assertEqual(A.critic_V(sample), 1.0)
        self.assertEqual(A.gaussian_moment_test(sample)["reject"], 1.0)

    def test_deterministic_compliance_does_not_reject(self):
        z = np.repeat([0, 1], 10)
        self.assertEqual(A.gaussian_moment_test(dict(z=z, d=z, y=np.zeros(20)))["reject"], 0.0)

    def test_tail_cells_match_observed_counts(self):
        sample = dict(z=np.array([0, 0, 1, 1]), d=np.array([0, 1, 0, 1]),
                      y=np.array([-100.0, 100.0, -100.0, 100.0]))
        m = A._cell_masses(sample, np.linspace(-6, 9, 13))
        assert_allclose(m["m00"][0], 0.5)
        assert_allclose(m["m10"][-1], 0.5)
        self.assertAlmostEqual(sum(m["m11"] + m["m01"]), 1.0)

    def test_missing_arm_and_nonfinite_outcome_are_rejected(self):
        for sample in (dict(z=np.zeros(4), d=np.zeros(4), y=np.zeros(4)),
                       dict(z=np.array([0, 0, 1, 1]), d=np.zeros(4), y=np.full(4, np.nan))):
            with self.assertRaises(ValueError):
                A.critic_V(sample)


class IdentificationTests(unittest.TestCase):
    def test_sharp_defier_bounds_for_random_finite_laws(self):
        # Generate full potential-outcome laws, allowing either sign of the first
        # stage and arbitrary outcome dependence within each principal stratum.
        rng = np.random.default_rng(781)
        for _ in range(30):
            joint = rng.dirichlet(np.ones(4 * 3 * 3)).reshape(4, 3, 3)
            mu0, mu1 = joint.sum(axis=2), joint.sum(axis=1)
            m = dict(m11=mu1[0] + mu1[2], m10=mu1[0] + mu1[3],
                     m00=mu0[1] + mu0[2], m01=mu0[1] + mu0[3])
            lo, hi = A.defier_mass_identified_set(m, 3)
            self.assertAlmostEqual(lo, A.cellwise_V(m), places=7)
            self.assertAlmostEqual(hi, min(sum(m["m10"]), sum(m["m01"])), places=7)
            y = np.arange(3)
            actual_late = (mu1[2] @ y - mu0[2] @ y) / mu1[2].sum()
            late_lo, late_hi = A.complier_late_identified_set(m, y)
            self.assertLessEqual(late_lo - 1e-8, actual_late)
            self.assertGreaterEqual(late_hi + 1e-8, actual_late)
            actual_ate = (mu1.sum(axis=0) - mu0.sum(axis=0)) @ y
            ate_lo, ate_hi = A.ate_identified_set(m, y)
            self.assertLessEqual(ate_lo - 1e-8, actual_ate)
            self.assertGreaterEqual(ate_hi + 1e-8, actual_ate)

    def test_point_identified_effect_and_no_compliers(self):
        z = np.zeros(2)
        m = dict(m11=np.array([0.0, 1.0]), m10=z, m00=np.array([1.0, 0.0]), m01=z)
        assert_allclose(A.complier_late_identified_set(m, np.array([0.0, 1.0])), [1.0, 1.0])
        assert_allclose(A.ate_identified_set(m, np.array([0.0, 1.0])), [1.0, 1.0])
        pure_defiers = dict(m11=z, m10=m["m11"], m00=z, m01=m["m00"])
        self.assertTrue(np.isnan(A.complier_late_identified_set(pure_defiers, np.array([0.0, 1.0]))).all())

    def test_worked_binary_examples(self):
        example = R.worked_example()
        self.assertAlmostEqual(example["V"], 0.03)
        self.assertAlmostEqual(example["wald"], -0.55)
        hidden = example["hidden_signflip"]
        self.assertEqual(hidden["V"], 0.0)
        self.assertAlmostEqual(hidden["lp_min_defier_mass"], 0.0)
        self.assertAlmostEqual(hidden["wald"], -0.2)
        self.assertAlmostEqual(hidden["tau_c"], 0.2)


if __name__ == "__main__":
    unittest.main()
