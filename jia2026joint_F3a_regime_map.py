#!/usr/bin/env python3
"""F3a: analytic two-clock regime map (S24-4), Figure fig:regimes in Section 4.

Plane (k1/a2, R0). From Theorem 4.1 and Corollary 4.2:
  R0 > 1                      persistence
  max(0, 1 - k1/a2) < R0 < 1  infection clock a2(1-R0) is the slower one
  0 < R0 < 1 - k1/a2          susceptible clock k1 is the slower one
No parameter values enter; the figure is exact. Parameter points (P-ext, the P-clock
ladder, P-bio) are added in WU-N through an optional CSV with columns label,k1_over_a2,R0.
With a CSV the persistence label and the leader of the boundary label move off the ladder
(Session 32, S32-2); without one the output is byte-identical to the v1_6_0 figure.
Session 79 (WU7c2, R-4: R-F1): an optional second argument, the inset JSON written by
jia2026joint_S79_F3a_inset_paths.py, adds two panels at the right, one path at P-ext and one at
the lowest ladder point, each with ln I, ln(S_hat - S) and lines of the exact slopes, rings the
two points on the map, and writes jia2026joint_F3a_regime_map_inset.pdf. Without it the code
path and the output are unchanged.

usage: python3 jia2026joint_F3a_regime_map.py [points.csv [inset.json]]
output: jia2026joint_F3a_regime_map.pdf, or jia2026joint_F3a_regime_map_inset.pdf with inset.json (deterministic: no creation date in the metadata)
"""
import csv, json, os, sys
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE, "jia2026joint_figstyle.mplstyle"))
INSET = len(sys.argv) > 2
OUT = os.path.join(HERE, "jia2026joint_F3a_regime_map_inset.pdf" if INSET else "jia2026joint_F3a_regime_map.pdf")
XMAX, YMAX = 1.5, 1.5
PTS = len(sys.argv) > 1
PER_X, EDGE_XY = (0.36, (0.90, 0.10)) if PTS else (0.75, (0.74, 0.26))
C_INF, C_SUS, C_PER = "#0072B2", "#E69F00", "#BBBBBB"

if INSET:
    fig = plt.figure(figsize=(7.0, 3.3))
    gs = fig.add_gridspec(2, 2, width_ratios=(4.6, 2.1), wspace=0.30, hspace=0.62)
    ax = fig.add_subplot(gs[:, 0])
else:
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
x = np.linspace(0.0, XMAX, 601)
edge = np.clip(1.0 - x, 0.0, None)
ax.fill_between(x, 1.0, YMAX, color=C_PER, alpha=0.35, lw=0)
ax.fill_between(x, edge, 1.0, color=C_INF, alpha=0.30, lw=0)
ax.fill_between(x, 0.0, edge, color=C_SUS, alpha=0.35, lw=0)
ax.axhline(1.0, color="black", lw=1.0)
xb = np.linspace(0.0, 1.0, 101)
ax.plot(xb, 1.0 - xb, color="black", lw=1.0, ls="--")

ax.text(PER_X, 1.25, "persistence, $\\Re_0>1$", ha="center", va="center", fontsize=8.5)
ax.text(1.05, 0.55, "infection clock\nrate $a_2(1-\\Re_0)$", ha="center", va="center", fontsize=8.5)
ax.text(0.27, 0.28, "susceptible clock\nrate $k_1$", ha="center", va="center", fontsize=8.5)
ax.annotate("$\\Re_0=1-k_1/a_2$", xy=EDGE_XY, xytext=(0.97, 0.24), fontsize=8.5, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))

if len(sys.argv) > 1:
    with open(sys.argv[1], newline="") as fh:
        for row in csv.DictReader(fh):
            px, py = float(row["k1_over_a2"]), float(row["R0"])
            ax.plot(px, py, "o", color="black", ms=3.5, zorder=5)
            ax.annotate(row["label"], (px, py), xytext=(4, 4), textcoords="offset points", fontsize=8)

if INSET:
    J = json.load(open(sys.argv[2]))
    C_LNI, C_GAP = "#0072B2", "#D55E00"
    for j, (lab, ttl) in enumerate((("P-ext", "P-ext, $\\Re_0=%.2f$"), ("low", "lowest point, $\\Re_0=%.2f$"))):
        q = J["paths"][lab]
        ax.plot(q["k1_over_a2"], q["R0"], "o", mfc="none", mec="black", ms=7.5, mew=0.8, zorder=6)
        axi = fig.add_subplot(gs[j, 1])
        t = np.array(q["t"])
        axi.plot(t, q["lnI"], color=C_LNI, lw=1.6, label="$\\ln I$", zorder=2)
        axi.plot(t, q["lngap"], color=C_GAP, lw=0.8, label="$\\ln(\\hat S-S)$", zorder=3)
        tw = np.array(q["fit_window"])
        for k, nm in enumerate(("lnI", "lngap")):
            tm, ym = q["window_mean_point"][nm]
            axi.plot(tw, ym + q["exact"][nm] * (tw - tm), color="black", lw=0.8, ls="--",
                     label="exact slope" if k == 0 else None, zorder=4)
        axi.set_xlim(0.0, q["T"]); axi.set_title(ttl % q["R0"], fontsize=8, pad=3)
        axi.tick_params(labelsize=7); axi.set_ylabel("logarithm", fontsize=8)
        if j == 0:
            axi.legend(fontsize=6.5, frameon=False, loc="lower left", handlelength=1.6, borderaxespad=0.2)
        else:
            axi.set_xlabel("$t$", fontsize=9)

ax.set_xlim(0.0, XMAX); ax.set_ylim(0.0, YMAX)
ax.set_xlabel("$k_1/a_2$"); ax.set_ylabel("$\\Re_0$")
ax.set_xticks([0, 0.5, 1.0, 1.5]); ax.set_yticks([0, 0.5, 1.0, 1.5])
fig.savefig(OUT, metadata={"CreationDate": None, "ModDate": None})
print(OUT)
