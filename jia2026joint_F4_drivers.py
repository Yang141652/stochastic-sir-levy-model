#!/usr/bin/env python3
"""F4 (E7): median and 50% and 90% bands of I(tau) for the five drivers at matched mean rates and log-scale jump
covariance (rulings S33-2, S34-2): the balanced laws G, CP6, TS6, then the one-signed pair TS1 (upward I-jumps) and
TS1d (downward). Top row: the band case, beta 0.232 (R0 < 1 < R0^det), with the quasi-extinction level 1e-3
dashed. Bottom row: P-per, beta 0.316, with the end of the burn-in (tau 50) dotted. Each panel prints the driver's R0; the bottom row has headroom so the label clears the
bands (Session 34; the Session 33 labels touched the light band).
Reads the percentiles of ln I stored by jia2026joint_S33_E7_run.py and jia2026joint_S34_E7d_run.py; writes a byte-stable PDF (no creation date).
usage: python3 -B jia2026joint_F4_drivers.py [out.pdf]   (from work/scripts; default ../numerics/S34_WUN2b2b1/jia2026joint_F4_drivers.pdf)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); S33 = os.path.join(HERE, "..", "numerics", "S33_WUN2b2a"); OUT = os.path.join(HERE, "..", "numerics", "S34_WUN2b2b1")
PDF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "jia2026joint_F4_drivers.pdf")
plt.style.use(os.path.join(HERE, "..", "data", "jia2026joint_figstyle.mplstyle"))
V = json.load(open(os.path.join(S33, "E7_validation.json"))); Vd = json.load(open(os.path.join(OUT, "E7d_validation.json")))
assert V["all_pass"] and Vd["all_pass"]; SM = dict(V["constants"]["summary"], **Vd["constants"]["summary"])
DR = ("G", "CP6", "TS6", "TS1", "TS1d")
NAMES = dict(G="Brownian", CP6="Compound Poisson", TS6="Tempered stable,\nsix directions", TS1="Tempered stable,\nupward I-jumps",
             TS1d="Tempered stable,\ndownward I-jumps")
fig, ax = plt.subplots(2, 5, figsize=(7.2, 4.0), sharex="row", sharey="row")
for r, (reg, beta) in enumerate((("band", 0.232), ("per", 0.316))):
    for c, X in enumerate(DR):
        d = np.load(os.path.join(OUT if X == "TS1d" else S33, "E7_%s_%s.npz" % (reg, X))); t = d["rec_t"]; q = np.exp(d["q_lnI"]); A = ax[r, c]
        h90 = A.fill_between(t, q[0], q[4], color="C0", alpha=0.18, lw=0)
        h50 = A.fill_between(t, q[1], q[3], color="C0", alpha=0.40, lw=0)
        hm, = A.plot(t, q[2], color="C0", lw=1.1)
        A.set_yscale("log")
        A.text(0.96, 0.95, r"$\Re_0=%.3f$" % SM[X][str(beta)]["R0"], transform=A.transAxes,
               ha="right", va="top", fontsize=7)
        if r == 0:
            hq = A.axhline(1e-3, color="0.3", ls="--", lw=0.7); A.set_title(NAMES[X], fontsize=7.5)
        else:
            A.axvline(50.0, color="0.45", ls=":", lw=0.8)
        if c == 0: A.set_ylabel(r"$I(\tau)$, $\beta=%.3f$" % beta, fontsize=8)
        if r == 1: A.set_xlabel(r"$\tau$", fontsize=8)
        A.tick_params(labelsize=7)
lo, hi = ax[1, 0].get_ylim(); ax[1, 0].set_ylim(lo, hi * (hi / lo) ** 0.22)   # headroom for the R0 labels (shared row)
fig.legend([hm, h50, h90, hq], ["median", "50% band", "90% band", r"quasi-extinction level $10^{-3}$"],
           loc="lower center", ncol=4, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(PDF, metadata={"CreationDate": None})
print("wrote", PDF)
