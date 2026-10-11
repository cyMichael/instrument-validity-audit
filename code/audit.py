"""
Core routines for "Identification as an Audit".

Setting. A platform runs a randomized encouragement Z in {0,1}; users choose
D in {0,1}; an outcome Y is recorded. Principal strata T = (D(0), D(1)):

    a = (1,1) always-taker      n = (0,0) never-taker
    c = (0,1) complier          f = (1,0) defier

A collective of fractional size alpha instructs its members to invert the
platform's encouragement, so collective members are exactly the defiers.

Under (A1) random assignment and (A2) exclusion, the four observed
sub-distributions decompose as

    m11(B) = P(Y in B, D=1 | Z=1) = mu_a1(B) + mu_c1(B)
    m10(B) = P(Y in B, D=1 | Z=0) = mu_a1(B) + mu_f1(B)
    m00(B) = P(Y in B, D=0 | Z=0) = mu_n0(B) + mu_c0(B)
    m01(B) = P(Y in B, D=0 | Z=1) = mu_n0(B) + mu_f0(B)

with mu_td(B) = P(T=t, Y(d) in B). Hence the two signed measures

    S1 = m10 - m11 = mu_f1 - mu_c1
    S0 = m01 - m00 = mu_f0 - mu_c0

carry the defiers, and the violation functional

    V(P) = max( S1_+(Y) , S0_+(Y) ),   S_+ = positive part of the
                                       Jordan decomposition of S

is a sharp lower bound on P(T=f) (Theorem 1 in the paper).

Two regimes this makes explicit:
  * outcome-separated collective  -> S_+ can grow with alpha;
  * outcome-mimicking collective  -> S_+ = 0 whenever mu_f1 <= mu_c1 and
    mu_f0 <= mu_c0 as measures, so there is no population violation.

Model parameters split the two forces cleanly:
  * ``sep``  shifts the collective's outcome LEVELS  -> drives detectability;
  * ``tau_f`` vs ``tau`` is the collective's treatment-effect heterogeneity
    -> drives corruption of the platform's IV estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import linprog
from scipy.stats import norm

TYPES = ("a", "n", "c", "f")
Y_RANGE = (-6.0, 9.0)


@dataclass
class DGP:
    """Gaussian-outcome model for a platform experiment with a collective."""

    alpha: float = 0.05     # collective / defier share  P(T=f)
    pi_c: float = 0.30      # complier share
    pi_a: float = 0.25      # always-taker share
    sep: float = 1.5        # defier-vs-complier outcome LEVEL shift (mimicry knob)
    tau: float = 1.0        # complier treatment effect
    tau_f: float | None = None   # defier treatment effect; defaults to ``tau``
    mu_base: float = 0.0    # complier Y(0) mean
    mu_a1: float = 0.5      # always-taker Y(1) mean
    mu_n0: float = -0.5     # never-taker Y(0) mean
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.tau_f is None:
            self.tau_f = self.tau
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        shares = (self.alpha, self.pi_c, self.pi_a, self.pi_n)
        if any(x < -1e-12 or x > 1.0 + 1e-12 for x in shares):
            raise ValueError(f"invalid type shares: {shares}")

    # ---- type shares -----------------------------------------------------
    @property
    def pi_f(self) -> float:
        return self.alpha

    @property
    def pi_n(self) -> float:
        return 1.0 - self.alpha - self.pi_c - self.pi_a

    @property
    def shares(self) -> dict[str, float]:
        return {"a": self.pi_a, "n": self.pi_n, "c": self.pi_c, "f": self.pi_f}

    # ---- outcome means ---------------------------------------------------
    def mean(self, t: str, d: int) -> float:
        """Mean of Y(d) for stratum t."""
        if t == "c":
            return self.mu_base + (self.tau if d == 1 else 0.0)
        if t == "f":
            return self.mu_base + self.sep + (self.tau_f if d == 1 else 0.0)
        if t == "a":
            # Y(0) for always-takers is never observed; irrelevant to P.
            return self.mu_a1 if d == 1 else self.mu_base
        if t == "n":
            # Y(1) for never-takers is never observed.
            return self.mu_n0 if d == 0 else self.mu_base + self.tau
        raise ValueError(t)

    # ---- population quantities ------------------------------------------
    def sub_densities(self, grid: np.ndarray) -> dict[str, np.ndarray]:
        def phi(t: str, d: int) -> np.ndarray:
            return norm.pdf(grid, loc=self.mean(t, d), scale=self.sigma)

        return {
            "m11": self.pi_a * phi("a", 1) + self.pi_c * phi("c", 1),
            "m10": self.pi_a * phi("a", 1) + self.pi_f * phi("f", 1),
            "m00": self.pi_n * phi("n", 0) + self.pi_c * phi("c", 0),
            "m01": self.pi_n * phi("n", 0) + self.pi_f * phi("f", 0),
        }

    def violation(self, grid: np.ndarray | None = None) -> dict[str, float]:
        """Population V(P), exactly for equal-variance Gaussian marginals.

        An explicit grid requests a numerical approximation over that grid only.
        The default integrates both Gaussian tails analytically, with no outcome
        truncation and no dependence on the location of the outcome support.
        """
        if grid is not None:
            m = self.sub_densities(grid)
            v1 = float(trapezoid(np.maximum(m["m10"] - m["m11"], 0.0), grid))
            v0 = float(trapezoid(np.maximum(m["m01"] - m["m00"], 0.0), grid))
        else:
            def positive_mass(d: int) -> float:
                if self.pi_f == 0.0:
                    return 0.0
                if self.pi_c == 0.0:
                    return self.pi_f
                mf, mc = self.mean("f", d), self.mean("c", d)
                shift = mf - mc
                if shift == 0.0:
                    return max(self.pi_f - self.pi_c, 0.0)
                crossing = (mf + mc) / 2 + self.sigma ** 2 / shift * np.log(self.pi_c / self.pi_f)
                tail = norm.sf if shift > 0 else norm.cdf
                return max(float(self.pi_f * tail((crossing - mf) / self.sigma)
                                 - self.pi_c * tail((crossing - mc) / self.sigma)), 0.0)

            v1, v0 = positive_mass(1), positive_mass(0)
        return {"V": max(v1, v0), "V1": v1, "V0": v0}

    def first_stage(self) -> float:
        """P(D=1|Z=1) - P(D=1|Z=0) = pi_c - alpha."""
        return self.pi_c - self.alpha

    def wald(self) -> float:
        """Population Wald / IV estimand the platform would report."""
        num = 0.0
        for t, p in self.shares.items():
            d1 = 1 if t in ("a", "c") else 0   # D(1)
            d0 = 1 if t in ("a", "f") else 0   # D(0)
            num += p * (self.mean(t, d1) - self.mean(t, d0))
        denom = self.first_stage()
        return num / denom if abs(denom) > 1e-12 else np.nan

    def late_compliers(self) -> float:
        """The parameter the platform believes it is estimating."""
        return self.tau

    def influence(self) -> float:
        """Corruption of the platform's causal estimate, |Wald - complier LATE|.

        Closed form: alpha * |tau - tau_f| / (pi_c - alpha).
        """
        return abs(self.wald() - self.late_compliers())

    def influence_closed_form(self) -> float:
        d = self.first_stage()
        if abs(d) < 1e-12:
            return np.nan
        return self.alpha * abs(self.tau - self.tau_f) / abs(d)

    # ---- sampling --------------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        shares = np.array([self.shares[t] for t in TYPES], dtype=float)
        shares = np.clip(shares, 0.0, None)
        t_idx = rng.choice(len(TYPES), size=n, p=shares / shares.sum())
        z = rng.integers(0, 2, size=n)
        d = np.zeros(n, dtype=int)
        for i, t in enumerate(TYPES):
            sel = t_idx == i
            d[sel & (z == 1)] = 1 if t in ("a", "c") else 0
            d[sel & (z == 0)] = 1 if t in ("a", "f") else 0
        y = np.empty(n)
        for i, t in enumerate(TYPES):
            for dd in (0, 1):
                sel = (t_idx == i) & (d == dd)
                k = int(sel.sum())
                if k:
                    y[sel] = rng.normal(self.mean(t, dd), self.sigma, size=k)
        return {"z": z, "d": d, "y": y, "t": t_idx}


# --------------------------------------------------------------------------
# Class-restricted plug-in critic for V(P)
# --------------------------------------------------------------------------

def _cell_masses(
    sample: dict[str, np.ndarray], edges: np.ndarray
) -> dict[str, np.ndarray]:
    """Empirical sub-distribution masses on the cells defined by ``edges``.

    Outcomes are winsorized into the edge range so that no probability mass is
    silently discarded and each arm's masses sum to one.
    """
    z, d = sample["z"], sample["d"]
    y = np.clip(sample["y"], edges[0], edges[-1])
    if not (z.ndim == d.ndim == y.ndim == 1 and len(z) == len(d) == len(y)):
        raise ValueError("z, d and y must be one-dimensional arrays of equal length")
    if not (np.isin(z, [0, 1]).all() and np.isin(d, [0, 1]).all()
            and np.isfinite(sample["y"]).all()):
        raise ValueError("z and d must be binary and outcomes must be finite")
    out = {}
    for name, (zz, dd) in {
        "m11": (1, 1), "m10": (0, 1), "m00": (0, 0), "m01": (1, 0),
    }.items():
        sel = (z == zz) & (d == dd)
        counts, _ = np.histogram(y[sel], bins=edges)
        n_arm = int((z == zz).sum())
        if n_arm == 0:
            raise ValueError("both assignment arms must be observed")
        out[name] = counts / n_arm
    return out


def critic_V(
    sample: dict[str, np.ndarray],
    n_bins: int = 20,
    y_range: tuple[float, float] = Y_RANGE,
) -> float:
    """Plug-in estimate of V(P) over the class of unions of histogram cells.

    This is a one-sided set-indexed discrepancy: the supremum
    over unions of cells is attained by keeping every cell on which the
    empirical difference is positive. Restricting the class understates the
    population supremum, while the empirical positive part is upward-biased.
    The result is therefore reported as an effect-size estimate, not a
    confidence bound. Hypothesis testing is handled
    separately by a studentized moment-inequality statistic.
    """
    edges = np.linspace(*y_range, n_bins + 1)
    m = _cell_masses(sample, edges)
    v1 = np.clip(m["m10"] - m["m11"], 0.0, None).sum()
    v0 = np.clip(m["m01"] - m["m00"], 0.0, None).sum()
    return float(max(v1, v0))


def gaussian_moment_test(
    sample: dict[str, np.ndarray],
    level: float = 0.05,
    n_draws: int = 1000,
    n_bins: int = 20,
    y_range: tuple[float, float] = Y_RANGE,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Least-favourable Gaussian test of the finite cell inequalities.

    Under monotonicity, every component of ``m10-m11`` and ``m01-m00`` is
    nonpositive. The test studentizes their empirical differences and calibrates
    the maximum with a zero-mean Gaussian having the plug-in correlation matrix.
    Setting every null moment to zero is least favourable, so the test is
    pointwise asymptotically size-controlling for a fixed partition, positive
    population variances and nonvanishing arm proportions, using the ideal
    Gaussian quantile (or increasing calibration draws). Finite ``n_draws``
    adds simulation error. This is not an exact finite-sample test, particularly
    for sparse cells, and can be conservative inside the null.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0,1)")
    if n_draws < 2:
        raise ValueError("n_draws must be at least 2")
    edges = np.linspace(*y_range, n_bins + 1)
    m = _cell_masses(sample, edges)
    diff = np.concatenate([m["m10"] - m["m11"], m["m01"] - m["m00"]])
    n0 = int(np.sum(sample["z"] == 0))
    n1 = int(np.sum(sample["z"] == 1))
    if min(n0, n1) < 2:
        raise ValueError("both assignment arms need at least two observations")

    p0 = np.concatenate([m["m10"], m["m00"]])
    p1 = np.concatenate([m["m11"], m["m01"]])
    mu0 = np.concatenate([m["m10"], -m["m00"]])
    mu1 = np.concatenate([-m["m11"], m["m01"]])
    cov = ((np.diag(p0) - np.outer(mu0, mu0)) / n0
           + (np.diag(p1) - np.outer(mu1, mu1)) / n1)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    active = se > 0.0
    # A deterministic positive difference is a violation, not an uninformative
    # moment to drop. Zero-variance nonpositive moments cannot cause rejection.
    deterministic_violation = bool(np.any((~active) & (diff > 0.0)))
    if not np.any(active):
        if deterministic_violation:
            return {"stat": float("inf"), "crit": 0.0, "reject": 1.0}
        return {"stat": float("-inf"), "crit": float("inf"), "reject": 0.0}
    t_stat = float("inf") if deterministic_violation else float(np.max(diff[active] / se[active]))
    corr = cov[np.ix_(active, active)] / np.outer(se[active], se[active])
    corr = 0.5 * (corr + corr.T)
    eigval, eigvec = np.linalg.eigh(corr)
    root = eigvec @ np.diag(np.sqrt(np.clip(eigval, 0.0, None)))
    draws = rng.normal(size=(n_draws, active.sum())) @ root.T
    crit = float(np.quantile(np.max(draws, axis=1), 1.0 - level))
    return {"stat": t_stat, "crit": crit, "reject": float(t_stat > crit)}


def audit(
    sample: dict[str, np.ndarray],
    level: float = 0.05,
    n_draws: int = 1000,
    n_bins: int = 20,
    y_range: tuple[float, float] = Y_RANGE,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Effect-size estimate plus an asymptotic cellwise moment-inequality test."""
    v_hat = critic_V(sample, n_bins=n_bins, y_range=y_range)
    test = gaussian_moment_test(
        sample, level=level, n_draws=n_draws, n_bins=n_bins,
        y_range=y_range, rng=rng
    )
    return {
        "V_hat": v_hat,
        "stat": test["stat"],
        "crit": test["crit"],
        "reject": test["reject"],
    }


# --------------------------------------------------------------------------
# Sharp identified set for the ATE under (A1) + (A2), no monotonicity
# --------------------------------------------------------------------------

def ate_identified_set(
    m: dict[str, np.ndarray], y_mid: np.ndarray
) -> tuple[float, float]:
    """Sharp bounds on the ATE of the supplied finite-support outcome by LP.

    Decision variables theta[t, d, k] = P(T=t, Y(d) in cell k) over the four
    principal strata, d in {0,1}, and K outcome cells. Four linear equalities
    per cell match the observed sub-distributions; the remaining constraints
    are nonnegativity, equal stratum mass under d = 0 and d = 1, and total
    mass one. No monotonicity restriction is imposed.
    """
    K = len(y_mid)
    n_var = 4 * 2 * K

    def idx(t: int, d: int, k: int) -> int:
        return t * 2 * K + d * K + k

    ia, ind, ic, if_ = 0, 1, 2, 3  # a, n, c, f

    rows, rhs = [], []
    for k in range(K):
        for members, target in (
            (((ia, 1, k), (ic, 1, k)), m["m11"][k]),
            (((ia, 1, k), (if_, 1, k)), m["m10"][k]),
            (((ind, 0, k), (ic, 0, k)), m["m00"][k]),
            (((ind, 0, k), (if_, 0, k)), m["m01"][k]),
        ):
            row = np.zeros(n_var)
            for t, d, kk in members:
                row[idx(t, d, kk)] = 1.0
            rows.append(row)
            rhs.append(target)

    for t in range(4):  # equal stratum mass across d
        row = np.zeros(n_var)
        for k in range(K):
            row[idx(t, 1, k)] += 1.0
            row[idx(t, 0, k)] -= 1.0
        rows.append(row)
        rhs.append(0.0)

    row = np.zeros(n_var)  # total mass one
    for t in range(4):
        for k in range(K):
            row[idx(t, 1, k)] = 1.0
    rows.append(row)
    rhs.append(1.0)

    A_eq, b_eq = np.array(rows), np.array(rhs)

    obj = np.zeros(n_var)
    for t in range(4):
        for k in range(K):
            obj[idx(t, 1, k)] += y_mid[k]
            obj[idx(t, 0, k)] -= y_mid[k]

    lo = linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    hi = linprog(-obj, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not (lo.success and hi.success):
        return (np.nan, np.nan)
    return float(lo.fun), float(-hi.fun)


def _lp_constraints(m: dict[str, np.ndarray], K: int):
    """Shared equality system for the identified-set LPs."""
    n_var = 4 * 2 * K

    def idx(t: int, d: int, k: int) -> int:
        return t * 2 * K + d * K + k

    ia, ind, ic, if_ = 0, 1, 2, 3
    rows, rhs = [], []
    for k in range(K):
        for members, target in (
            (((ia, 1, k), (ic, 1, k)), m["m11"][k]),
            (((ia, 1, k), (if_, 1, k)), m["m10"][k]),
            (((ind, 0, k), (ic, 0, k)), m["m00"][k]),
            (((ind, 0, k), (if_, 0, k)), m["m01"][k]),
        ):
            row = np.zeros(n_var)
            for t, d, kk in members:
                row[idx(t, d, kk)] = 1.0
            rows.append(row)
            rhs.append(target)
    for t in range(4):
        row = np.zeros(n_var)
        for k in range(K):
            row[idx(t, 1, k)] += 1.0
            row[idx(t, 0, k)] -= 1.0
        rows.append(row)
        rhs.append(0.0)
    row = np.zeros(n_var)
    for t in range(4):
        for k in range(K):
            row[idx(t, 1, k)] = 1.0
    rows.append(row)
    rhs.append(1.0)
    return np.array(rows), np.array(rhs), idx, n_var


def defier_mass_identified_set(m: dict[str, np.ndarray], K: int) -> tuple[float, float]:
    """Sharp bounds on P(T=f) itself, by LP over the same feasible set.

    The lower endpoint is an independent numerical check on Theorem 1: it should
    equal the cellwise violation functional V(P) computed on the same partition.
    """
    A_eq, b_eq, idx, n_var = _lp_constraints(m, K)
    obj = np.zeros(n_var)
    for k in range(K):
        obj[idx(3, 1, k)] = 1.0        # stratum 3 = defiers, d = 1
    lo = linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    hi = linprog(-obj, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not (lo.success and hi.success):
        return (np.nan, np.nan)
    return float(lo.fun), float(-hi.fun)


def complier_late_identified_set(
    m: dict[str, np.ndarray],
    y_mid: np.ndarray,
) -> tuple[float, float]:
    """Sharp complier-LATE bounds for the supplied finite-support outcome.

    For midpoint-coded outcomes this targets their LATE, not the original
    continuous-outcome LATE. Models with zero complier mass are excluded because
    their complier LATE is undefined; if none remain, return NaN endpoints.

    The Charnes--Cooper transform solves the linear-fractional program exactly:
    for numerator ``c' theta`` and denominator ``s' theta``, set
    ``x=theta/(s'theta)`` and ``t=1/(s'theta)``. Then ``A x=b t`` and ``s'x=1``.
    """
    K = len(y_mid)
    A_eq0, b_eq0, idx, n_var = _lp_constraints(m, K)

    obj_num = np.zeros(n_var)
    for k in range(K):
        obj_num[idx(2, 1, k)] += y_mid[k]     # stratum 2 = compliers
        obj_num[idx(2, 0, k)] -= y_mid[k]

    denominator = np.zeros(n_var)
    for k in range(K):
        denominator[idx(2, 1, k)] = 1.0       # stratum 2 = compliers

    A_cc = np.hstack([A_eq0, -b_eq0[:, None]])
    denom_row = np.append(denominator, 0.0)[None, :]
    A_eq = np.vstack([A_cc, denom_row])
    b_eq = np.append(np.zeros(A_eq0.shape[0]), 1.0)
    objective = np.append(obj_num, 0.0)
    bounds = [(0.0, None)] * (n_var + 1)
    lo = linprog(objective, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    hi = linprog(-objective, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not (lo.success and hi.success):
        return (np.nan, np.nan)
    return float(lo.fun), float(-hi.fun)


def cellwise_V(m: dict[str, np.ndarray]) -> float:
    """V(P) evaluated on a finite partition (the unrestricted sup over unions)."""
    v1 = np.clip(m["m10"] - m["m11"], 0.0, None).sum()
    v0 = np.clip(m["m01"] - m["m00"], 0.0, None).sum()
    return float(max(v1, v0))


def population_cell_masses(
    dgp: DGP, n_bins: int = 12, y_range: tuple[float, float] = Y_RANGE
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Exact population cell masses, with the two extreme cells absorbing the
    tails so that every arm's masses sum to one."""
    edges = np.linspace(*y_range, n_bins + 1)

    def binned(t: str, d: int) -> np.ndarray:
        c = norm.cdf(edges, loc=dgp.mean(t, d), scale=dgp.sigma)
        c[0], c[-1] = 0.0, 1.0          # winsorize tails into the end cells
        return np.diff(c)

    m = {
        "m11": dgp.pi_a * binned("a", 1) + dgp.pi_c * binned("c", 1),
        "m10": dgp.pi_a * binned("a", 1) + dgp.pi_f * binned("f", 1),
        "m00": dgp.pi_n * binned("n", 0) + dgp.pi_c * binned("c", 0),
        "m01": dgp.pi_n * binned("n", 0) + dgp.pi_f * binned("f", 0),
    }
    y_mid = 0.5 * (edges[:-1] + edges[1:])
    return m, y_mid
