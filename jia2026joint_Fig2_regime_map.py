#!/usr/bin/env python3
"""Figure 2 (label fig:regimes): regime map in the (k1/a2, R0)-plane with two sample paths.

Left: the regimes of Theorem 1 and Theorem 2. No parameter values enter, so this panel is exact.
  R0 > 1                      the infection persists in the mean (Theorem 2)
  max(0, 1 - k1/a2) < R0 < 1  the infection dies out; a2(1 - R0) is the slower rate
  0 < R0 < 1 - k1/a2          the infection dies out; k1 is the slower rate
Right: one path each at beta = 0.21 and beta = 0.05 (ln I and ln(S_hat - S)), with dashed lines of the exact
slopes of Theorem 1. The path data come from jia2026joint_S79_F3a_inset_paths.py (work/data/...S79.json).

This is jia2026joint_F3a_regime_map.py of the authors' working files, run in its inset mode, with the labels,
panel titles and size of the figure in the submitted manuscript (Yang Jia's revision of 2026-10-29): no marked
parameter points, "infection extinction" and "susceptible return" for the two regions, and panel titles giving
beta and R0. The plotting code is otherwise unchanged.

usage: python3 -B jia2026joint_Fig2_regime_map.py [inset.json [outdir]]   (run from work/scripts)
output: jia2026joint_F3a_regime_map_inset.pdf (the file name the manuscript includes); deterministic,
        no creation date in the metadata
"""
import json, os, sys
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE, "jia2026joint_figstyle.mplstyle"))
INSET = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "jia2026joint_F3a_inset_paths_S79.json")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "figures")
OUT = os.path.join(OUTDIR, "jia2026joint_F3a_regime_map_inset.pdf")
XMAX, YMAX = 1.5, 1.5
PER_X, EDGE_XY = 0.86, (0.73, 0.27)
C_INF, C_SUS, C_PER = "#0072B2", "#E69F00", "#BBBBBB"
FIGSIZE = (9.3, 4.8)
XLIM_INSET = (95.0, 60.0)  # time axes of the two panels in the submitted figure (the beta = 0.21 path runs to t = 98)
FS = dict(region=13, edge=12, label=14, tick=11, title=12, itick=10, ilabel=12, ixlabel=13, legend=8.5)  # sizes of the submitted figure

fig = plt.figure(figsize=FIGSIZE)
gs = fig.add_gridspec(2, 2, width_ratios=(4.2, 1.85), wspace=0.30, hspace=0.62)
ax = fig.add_subplot(gs[:, 0])
x = np.linspace(0.0, XMAX, 601)
edge = np.clip(1.0 - x, 0.0, None)
ax.fill_between(x, 1.0, YMAX, color=C_PER, alpha=0.35, lw=0)
ax.fill_between(x, edge, 1.0, color=C_INF, alpha=0.30, lw=0)
ax.fill_between(x, 0.0, edge, color=C_SUS, alpha=0.35, lw=0)
ax.axhline(1.0, color="black", lw=1.0)
xb = np.linspace(0.0, 1.0, 101)
ax.plot(xb, 1.0 - xb, color="black", lw=1.0, ls="--")

ax.text(PER_X, 1.24, "persistence, $\\Re_0>1$", ha="center", va="center", fontsize=FS["region"])
ax.text(1.02, 0.55, "infection extinction\nrate $a_2(1-\\Re_0)$", ha="center", va="center", fontsize=FS["region"])
ax.text(0.28, 0.28, "susceptible return\nrate $k_1$", ha="center", va="center", fontsize=FS["region"])
ax.annotate("$\\Re_0=1-k_1/a_2$", xy=EDGE_XY, xytext=(0.97, 0.24), fontsize=FS["edge"], ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.8, color="black"))

J = json.load(open(INSET))
C_LNI, C_GAP = "#0072B2", "#D55E00"
for j, lab in enumerate(("P-ext", "low")):
    q = J["paths"][lab]
    axi = fig.add_subplot(gs[j, 1])
    t = np.array(q["t"])
    axi.plot(t, q["lnI"], color=C_LNI, lw=1.8, label="$\\ln I$", zorder=2)
    axi.plot(t, q["lngap"], color=C_GAP, lw=1.4, label="$\\ln(\\hat S-S)$", zorder=3)
    tw = np.array(q["fit_window"])
    for k, nm in enumerate(("lnI", "lngap")):
        tm, ym = q["window_mean_point"][nm]
        axi.plot(tw, ym + q["exact"][nm] * (tw - tm), color="black", lw=1.0, ls="--",
                 label="exact slope" if k == 0 else None, zorder=4)
    axi.set_xlim(0.0, XLIM_INSET[j]); axi.set_xticks(np.arange(0, XLIM_INSET[j] + 1e-9, 20))
    axi.set_title("$\\beta=%.2f,\\ \\Re_0=%.4f$" % (q["beta"], q["R0"]), fontsize=FS["title"], pad=6)
    axi.tick_params(labelsize=FS["itick"]); axi.set_ylabel("logarithm", fontsize=FS["ilabel"])
    if j == 0:
        axi.legend(fontsize=FS["legend"], frameon=False, loc="lower left", handlelength=1.8, borderaxespad=0.2)
    else:
        axi.set_xlabel("$t$", fontsize=FS["ixlabel"])

ax.set_xlim(0.0, XMAX); ax.set_ylim(0.0, YMAX)
ax.set_xlabel("$k_1/a_2$", fontsize=FS["label"]); ax.set_ylabel("$\\Re_0$", fontsize=FS["label"])
ax.tick_params(labelsize=FS["tick"])
ax.set_xticks([0, 0.5, 1.0, 1.5]); ax.set_yticks([0, 0.5, 1.0, 1.5])
os.makedirs(OUTDIR, exist_ok=True)
fig.savefig(OUT, metadata={"CreationDate": None, "ModDate": None})
print(OUT)
