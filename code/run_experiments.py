"""
Entry point: regenerates the main simulations and both figures.
Run robustness.py as well for the robustness table and binary example.

    python3 run_experiments.py              # full run (~85 s locally, 400 replications)
    python3 run_experiments.py --quick      # coarse run for smoke testing
    python3 run_experiments.py --figs-only  # re-render figures from results.json

Outputs
    figs/fig_main.pdf      3 panels: bound, detection power, set width
    figs/fig_frontier.pdf  appendix: achievable stealth/influence pairs
    results.json                    main simulation results
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import audit as A

FIGDIR = pathlib.Path(__file__).resolve().parent / "figs"
OUT = pathlib.Path(__file__).resolve().parent / "results.json"

plt.rcParams.update({
    "font.size": 7.6,
    "axes.labelsize": 7.6,
    "axes.titlesize": 7.8,
    "legend.fontsize": 6.2,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "figure.dpi": 200,
    # NeurIPS requires Type 1 or embedded TrueType fonts; matplotlib's default
    # (Type 3) is rejected, so embed TrueType (fonttype 42).
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SEPS = [0.0, 1.0, 2.0, 3.0]
N_BINS = 12
LEVEL = 0.05

POWER_CONFIGS = [
    ("separated ($\\mathrm{sep}=3$)", dict(alpha=0.10, sep=3.0), "tab:blue"),
    ("partial ($\\mathrm{sep}=1$)", dict(alpha=0.10, sep=1.0), "tab:orange"),
    ("effect heterogeneity", dict(alpha=0.10, sep=0.0, tau_f=2.0), "tab:green"),
    ("perfect mimicry", dict(alpha=0.10, sep=0.0, tau_f=1.0), "tab:red"),
]


# ---------------------------------------------------------------- theorem 1
def verify_theorem1(n_bins: int = 14) -> dict:
    """Check V(P) equals the LP-minimal defier mass across a grid of DGPs."""
    worst, cases = 0.0, 0
    for alpha in (0.02, 0.05, 0.10, 0.20, 0.29):
        for sep in (0.0, 1.0, 2.5):
            for tau_f in (1.0, 2.0):
                g = A.DGP(alpha=alpha, sep=sep, tau_f=tau_f)
                m, ymid = A.population_cell_masses(g, n_bins=n_bins)
                worst = max(worst, abs(A.cellwise_V(m)
                                       - A.defier_mass_identified_set(m, len(ymid))[0]))
                cases += 1
    return {"n_cases": cases, "max_abs_gap": worst}


# ------------------------------------------------------------- computations
def population_bound_curves(alphas: np.ndarray) -> dict:
    return {str(sep): [A.DGP(alpha=float(a), sep=sep).violation()["V"] for a in alphas]
            for sep in SEPS}


def detection_power(dgp: A.DGP, n: int, reps: int, n_draws: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    rej = []
    for _ in range(reps):
        s = dgp.sample(n, rng)
        rej.append(A.audit(s, level=LEVEL, n_draws=n_draws,
                           n_bins=N_BINS, rng=rng)["reject"])
    return float(np.mean(rej))


def power_curves(ns: list[int], reps: int, n_draws: int) -> dict:
    out = {}
    for i, (lab, kw, _) in enumerate(POWER_CONFIGS):
        g = A.DGP(**kw)
        out[lab] = [detection_power(g, n, reps, n_draws, 1000 + i * 97 + j)
                    for j, n in enumerate(ns)]
    return out


def width_curve(alphas: np.ndarray, sep: float, n_bins: int = 12) -> list[float]:
    """Width of the sharp complier-LATE identified set as alpha varies."""
    out = []
    for a in alphas:
        g = A.DGP(alpha=float(a), sep=sep)
        m, ymid = A.population_cell_masses(g, n_bins=n_bins)
        lo, hi = A.complier_late_identified_set(m, ymid)
        out.append(hi - lo)
    return out


def frontier_points() -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    v, inf = [], []
    for alpha in np.linspace(0.01, 0.29, 29):
        for tau_f in np.linspace(1.0, 3.0, 41):
            g = A.DGP(alpha=float(alpha), sep=0.0, tau_f=float(tau_f))
            v.append(g.violation()["V"])
            inf.append(g.influence())
    v, inf = np.array(v), np.array(inf)
    grid = np.linspace(0, v.max(), 40)
    ex, ey = [], []
    for i, (a, b) in enumerate(zip(grid[:-1], grid[1:])):
        sel = (v >= a) & ((v <= b) if i == len(grid) - 2 else (v < b))
        if sel.any():
            ex.append(0.5 * (a + b))
            ey.append(float(inf[sel].max()))
    return v, inf, ex, ey


# ------------------------------------------------------------------ figures
def fig_main(alphas: np.ndarray, ns: list[int], power: dict, w_alphas: np.ndarray) -> dict:
    bounds = population_bound_curves(alphas)
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 1.85))

    ax = axes[0]
    ax.plot(alphas, alphas, "k--", lw=0.8, label=r"identity")
    for sep in SEPS:
        ax.plot(alphas, bounds[str(sep)], lw=1.2, label=rf"sep$\,=\,${sep:g}")
    ax.set_xlabel(r"collective share $\alpha$")
    ax.set_ylabel(r"$V(P)$")
    ax.set_title(r"(a) sharp bound on $\alpha$")
    ax.legend(frameon=False, loc="upper left", handlelength=1.3)

    ax = axes[1]
    for lab, _, col in POWER_CONFIGS:
        ax.plot(ns, power[lab], "o-", ms=2.4, lw=1.1, color=col, label=lab)
    ax.axhline(LEVEL, color="k", ls=":", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("experiment size $n$")
    ax.set_ylabel("rejection rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(r"(b) rejection at $\alpha=0.10$")
    ax.legend([r"sep$=3$", r"sep$=1$", "heterog. effect", "exact mimicry"],
              frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.08),
              handlelength=1.3)

    ax = axes[2]
    widths, monotone = {}, {}
    for sep, style in [(0.0, "-"), (1.5, "--"), (3.0, "-.")]:
        w = width_curve(w_alphas, sep)
        widths[str(sep)] = w
        monotone[str(sep)] = bool(np.all(np.diff(np.array(w)) >= -1e-6))
        lab = rf"sep$\,=\,${sep:g}"
        lab += " (mimic)" if sep == 0.0 else (" (separated)" if sep == 3.0 else "")
        ax.plot(w_alphas, w, style, lw=1.2, label=lab)
    ax.set_xlabel(r"collective share $\alpha$")
    ax.set_ylabel("identified-set width")
    ax.set_title("(c) information loss")
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)

    fig.tight_layout(pad=0.4)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig_main.pdf")
    plt.close(fig)
    return {"bounds": bounds, "widths": widths, "width_monotone_in_alpha": monotone}


def fig_frontier() -> dict:
    v, inf, ex, ey = frontier_points()
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    ax.scatter(v, inf, s=1.6, alpha=0.3, color="tab:gray", label="achievable",
               rasterized=True)
    ax.plot(ex, ey, "-", color="tab:red", lw=1.4, label="upper envelope")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="distortion = 1")
    ax.set_xlabel(r"detectability $V(P)$")
    ax.set_ylabel(r"influence $|\mathrm{Wald}-\tau_c|$")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig_frontier.pdf")
    plt.close(fig)
    return {"frontier_x": ex, "frontier_y": ey}


# --------------------------------------------------------------- headline nums
def headline(reps: int, n_draws: int) -> dict:
    out = {}
    g = A.DGP(alpha=0.29, sep=0.0, tau_f=1.05)
    out["stealth_signflip"] = {
        "alpha": g.alpha, "V": g.violation()["V"], "wald": g.wald(),
        "V_partition": A.cellwise_V(A.population_cell_masses(g, N_BINS)[0]),
        "tau": g.tau, "influence": g.influence(), "first_stage": g.first_stage(),
        "detection_n200000": detection_power(g, 200_000, reps, n_draws, 4242)}
    g2 = A.DGP(alpha=0.05, sep=0.0, tau_f=3.0)
    out["healthy_first_stage"] = {
        "alpha": g2.alpha, "V": g2.violation()["V"], "wald": g2.wald(),
        "V_partition": A.cellwise_V(A.population_cell_masses(g2, N_BINS)[0]),
        "influence": g2.influence(), "first_stage": g2.first_stage()}
    detect = {}
    for alpha in (0.01, 0.02, 0.05):
        g3 = A.DGP(alpha=alpha, sep=3.0)
        row = {"V": g3.violation()["V"],
               "V_partition": A.cellwise_V(A.population_cell_masses(g3, N_BINS)[0])}
        for n in (2_000, 10_000, 50_000):
            row[f"power_n{n}"] = detection_power(g3, n, reps, n_draws,
                                                 7 + int(alpha * 1000))
        detect[f"alpha={alpha}"] = row
    out["small_separated"] = detect
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figs-only", action="store_true",
                    help="re-render figures from an existing results.json")
    args = ap.parse_args()

    reps = 25 if args.quick else 400
    n_draws = 300 if args.quick else 1000
    ns = [500, 2_000, 10_000] if args.quick else [500, 2_000, 10_000, 50_000, 200_000]
    alphas = np.linspace(0.0, 0.29, 15 if args.quick else 40)
    w_alphas = np.linspace(0.0, 0.29, 25)

    if args.figs_only:
        prev = json.loads(OUT.read_text())
        prev["fig_main"] = fig_main(np.array(prev["alphas"]), prev["ns"],
                                    prev["power"], w_alphas)
        prev["fig_frontier"] = fig_frontier()
        OUT.write_text(json.dumps(prev, indent=2))
        print("re-rendered figures from cached results.json")
        return

    results = {"config": {"reps": reps, "n_gaussian_draws": n_draws,
                           "n_bins": N_BINS,
                          "level": LEVEL, "quick": args.quick},
               "alphas": alphas.tolist(), "ns": ns}

    print("verifying Theorem 1 ...")
    results["theorem1"] = verify_theorem1()
    print(f"  max |V(P) - LP min P(T=f)| over {results['theorem1']['n_cases']} DGPs: "
          f"{results['theorem1']['max_abs_gap']:.2e}")

    print("detection power ...")
    results["power"] = power_curves(ns, reps, n_draws)

    print("main figure ...")
    results["fig_main"] = fig_main(alphas, ns, results["power"], w_alphas)

    print("appendix figure ...")
    results["fig_frontier"] = fig_frontier()

    print("headline numbers ...")
    results["headline"] = headline(reps, n_draws)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    h = results["headline"]["stealth_signflip"]
    print(f"stealth sign-flip: alpha={h['alpha']}, V(P)={h['V']:.5f}, "
          f"Wald={h['wald']:.3f} vs true tau={h['tau']}, first stage={h['first_stage']:.3f}")
    for sep in ("0.0", "1.5", "3.0"):
        w = results["fig_main"]["widths"][sep]
        mono = results["fig_main"]["width_monotone_in_alpha"][sep]
        print(f"width (sep={sep}): {w[0]:.4f} -> {w[-1]:.4f}, monotone: {mono}")


if __name__ == "__main__":
    main()
