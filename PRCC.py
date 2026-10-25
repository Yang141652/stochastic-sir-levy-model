from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import qmc, rankdata, t as student_t

# ============================================================
# PRCC analysis of R0
# sigma11 and sigma33 are excluded from the PRCC analysis
# ============================================================

N_SAMPLES = 1000
SEED = 20261025
CHANGE_RATE = 0.20
OUTPUT_DIR = Path(__file__).resolve().parent

# Keep only the requested parameters
PARAM_NAMES = [
    "Lambda", "beta", "mu", "epsilon",
    "b", "alpha", "sigma22", "omega_tilde"
]
PARAM_LABELS = [
    r"$\Lambda$", r"$\beta$", r"$\mu$", r"$\varepsilon$",
    r"$b$", r"$\alpha$", r"$\sigma_{22}$", r"$\tilde{\omega}$"
]

# Baseline values from the manuscript
BASELINE = np.array([
    8.0,      # Lambda
    0.232,    # beta
    0.75,     # mu
    0.50,     # epsilon
    1.0,      # b
    0.40,     # alpha
    3.0e-3,   # sigma22
    0.50      # omega_tilde
], dtype=float)

LOWER = BASELINE * (1.0 - CHANGE_RATE)
UPPER = BASELINE * (1.0 + CHANGE_RATE)

# Fixed quantities in the manuscript
GAMMA11 = -0.3
GAMMA12 = 0.5
SIGMA11_FIXED = 3.0e-3
SIGMA33_FIXED = 4.0e-3

C_JUMP = np.array([1, 5, 5, 5, 1, 1], dtype=float)
ZETA = np.array([0.01, 0.30, 0.01], dtype=float)
N_DIRECTIONS = 6


def spectral_directions(omega):
    if not (0.0 < omega < 1.0):
        raise ValueError("omega_tilde must lie in (0, 1).")
    j = np.arange(1, N_DIRECTIONS + 1)
    theta = 2.0 * np.pi * j / N_DIRECTIONS
    radius = np.sqrt(1.0 - omega**2)
    return np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.full(N_DIRECTIONS, omega)
    ])


def calculate_R0(parameters):
    """
    Parameters passed in:
    Lambda, beta, mu, epsilon, b, alpha, sigma22, omega_tilde

    sigma11 and sigma33 are kept fixed at their baseline values.
    """
    (
        Lambda, beta, mu, epsilon,
        b, alpha, sigma22, omega
    ) = parameters

    if not (0.0 < alpha < 1.0):
        return np.nan
    if b <= 0.0:
        return np.nan

    directions = spectral_directions(omega)
    x = directions * ZETA[None, :]
    weights = C_JUMP * b**(-alpha)

    if np.any(b - x <= 0.0):
        return np.nan

    gamma_factor = gamma(-alpha)

    # J_i = \int eta_i(u) nu(du)
    J = gamma_factor * np.sum(
        weights[:, None] * ((b - x)**alpha - b**alpha),
        axis=0
    )
    J1, J2 = J[0], J[1]

    # P2 = \int [eta_2(u) - ln(1+eta_2(u))] nu(du)
    x2 = x[:, 1]
    P2 = gamma_factor * np.sum(
        weights * (
            (b - x2)**alpha - b**alpha
            + alpha * x2 * b**(alpha - 1.0)
        )
    )

    mu1 = mu - GAMMA11
    mu2 = mu + epsilon + 1.0 - GAMMA12
    a1 = mu1 - J1
    a2 = mu2 - J2

    if a1 <= 0.0 or a2 <= 0.0:
        return np.nan

    R0 = beta * Lambda / (a1 * a2) - (0.5 * sigma22 + P2) / a2
    return float(R0)


def calculate_prcc(X, y):
    n, p = X.shape
    rank_X = np.column_stack([
        rankdata(X[:, j], method="average") for j in range(p)
    ])
    rank_y = rankdata(y, method="average")

    df = n - p - 1
    if df <= 0:
        raise ValueError("Insufficient degrees of freedom.")

    prcc = np.empty(p)
    p_values = np.empty(p)

    for j in range(p):
        controls = np.delete(rank_X, j, axis=1)
        design = np.column_stack([np.ones(n), controls])

        resid_x = rank_X[:, j] - design @ np.linalg.lstsq(
            design, rank_X[:, j], rcond=None
        )[0]
        resid_y = rank_y - design @ np.linalg.lstsq(
            design, rank_y, rcond=None
        )[0]

        r = np.corrcoef(resid_x, resid_y)[0, 1]
        r = float(np.clip(r, -1.0, 1.0))
        prcc[j] = r

        r_safe = np.clip(r, -1.0 + 1e-12, 1.0 - 1e-12)
        t_stat = r_safe * np.sqrt(df / (1.0 - r_safe**2))
        p_values[j] = 2.0 * student_t.sf(abs(t_stat), df=df)

    return prcc, p_values


def main():
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=SEED)
    unit_samples = sampler.random(n=N_SAMPLES)
    X = qmc.scale(unit_samples, LOWER, UPPER)

    y = np.array([calculate_R0(row) for row in X])
    valid = np.isfinite(y)
    X = X[valid]
    y = y[valid]

    R0_baseline = calculate_R0(BASELINE)
    print(f"Baseline R0 = {R0_baseline:.8f}")

    prcc, p_values = calculate_prcc(X, y)

    results = pd.DataFrame({
        "Parameter": PARAM_NAMES,
        "PRCC": prcc,
        "p_value": p_values,
        "Significant_p_lt_0.01": p_values < 0.01
    })
    results.to_csv(
        OUTPUT_DIR / "PRCC_R0_drop_sigma11_sigma33_results.csv",
        index=False, encoding="utf-8-sig"
    )

    sample_table = pd.DataFrame(X, columns=PARAM_NAMES)
    sample_table["R0"] = y
    sample_table.to_csv(
        OUTPUT_DIR / "PRCC_R0_drop_sigma11_sigma33_LHS_samples.csv",
        index=False, encoding="utf-8-sig"
    )

    print("\nPRCC results:")
    print(results.to_string(index=False, formatters={
        "PRCC": lambda v: f"{v:.4f}",
        "p_value": lambda v: f"{v:.4e}"
    }))

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6.2, 3.45))
    x_pos = np.arange(len(PARAM_NAMES))
    colors = ["#1f77b4" if v >= 0 else "#ff7f0e" for v in prcc]

    ax.bar(
        x_pos, prcc, width=0.82,
        color=colors, edgecolor="black", linewidth=0.55,
        zorder=3
    )
    ax.axhline(0, color="black", linewidth=0.8, zorder=4)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)

    ax.set_title(r"$\mathcal{R}_0$", fontsize=12, pad=4)
    ax.set_ylabel("PRCC", fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(PARAM_LABELS, rotation=48, ha="right", fontsize=9)
    ax.tick_params(axis="y", labelsize=9, length=3, width=0.7)
    ax.tick_params(axis="x", length=3, width=0.7)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    max_abs = max(0.8, float(np.max(np.abs(prcc))) + 0.12)
    ax.set_ylim(-max_abs, max_abs)
    ax.margins(x=0.03)
    fig.tight_layout(pad=0.8)

    png_path = OUTPUT_DIR / "PRCC_R0_drop_sigma11_sigma33_600dpi.png"
    pdf_path = OUTPUT_DIR / "PRCC_R0_drop_sigma11_sigma33.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    print("\nSaved files:")
    print(png_path)
    print(pdf_path)
    print(OUTPUT_DIR / "PRCC_R0_drop_sigma11_sigma33.py")
    print(OUTPUT_DIR / "PRCC_R0_drop_sigma11_sigma33_results.csv")
    print(OUTPUT_DIR / "PRCC_R0_drop_sigma11_sigma33_LHS_samples.csv")


if __name__ == "__main__":
    main()
