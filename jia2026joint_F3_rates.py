#!/usr/bin/env python3
"""F3: simulated pathwise decay rates of I, S and R against the exact curves across beta (S28-2 point).

Top row: decay rates (per unit tau = gamma* t) of I, of the S return |S - S_hat| and of R; lines are the
exact rates a2(1 - R0) for I, min(a2(1 - R0), k1) for S and min(a2(1 - R0), k3) for R (Theorem 4.1,
Corollary 4.2), from SIRTS1_ref_thresholds via `exact` (extracted unchanged from jia2026joint_S30_WUN2a.py).
Markers: means of 1000 per-path slopes at dt 0.005 with 95% intervals (work/numerics/S30_WUN2a, Part 4).
Bottom row: relative deviation of the simulated rate from the exact one, in percent, with 95% intervals.
Dotted lines: crossovers beta_clock(S) and beta_clock(R); shading: the band R0 < 1 < R0det.
usage: python3 -B jia2026joint_F3_rates.py [outdir]   (run from work/scripts; default outdir ../figures)
output: jia2026joint_F3_rates.pdf (deterministic: no creation date in the metadata)
"""
import ast, hashlib, json, os, sys, types
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import SIRTS1_ref_thresholds as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "figures")
plt.style.use(os.path.join(HERE, "..", "data", "jia2026joint_figstyle.mplstyle"))
RES = os.path.join(HERE, "..", "numerics", "S30_WUN2a", "jia2026joint_S30_WUN2a_results.json")

LAM, EPS, MU, G1 = 8.0, 0.5, 0.75, (-0.3, 0.5, 0.2)
PHI = np.array([[3, 2, 1], [2, 3, 2], [1, 2, 4]]) * 1e-3
LAW = dict(alpha=0.4, b=1.0, zeta=(0.01, 0.3, 0.01), n=6, om=0.5, c=(1, 5, 5, 5, 1, 1))
P = T.setup(**LAW); KW = dict(Lam=LAM, mu=MU, eps=EPS, g1=G1, Phi=PHI, P=P); M = T.moments(P)
src = open(os.path.join(HERE, "jia2026joint_S30_WUN2a.py")).read()
node = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "exact"][0]
mod = types.ModuleType("S30"); mod.__dict__.update(dict(np=np, T=T, KW=KW, PHI=PHI, M=M, LAM=LAM))
exec(compile(ast.Module(body=[node], type_ignores=[]), "S30", "exec"), mod.__dict__)
exact = mod.exact

e0 = exact(0.2); a1, a2, nz = e0["a1"], e0["a2"], e0["band"] * e0["a2"]
b_det1, b_R1 = a1 * a2 / LAM, a1 * (a2 + nz) / LAM
b_cS, b_cR = a1 * (a2 - e0["k1"] + nz) / LAM, a1 * (a2 - e0["k3"] + nz) / LAM
assert abs(b_cS - 0.0860) < 5e-4 and abs(b_cR - 0.1922) < 5e-4 and abs(b_det1 - 0.2124) < 5e-4

R = json.load(open(RES)); rows = R["part4"]["rows"]
for r in rows:   # the stored exact values must equal the exact curve used here
    e = exact(r["beta"]); assert abs(e["lam"] - r["lam"]) < 1e-12 and abs(e["S_rate"] - r["S_rate"]) < 1e-12
bg = np.linspace(0.04, 0.245, 400)
ex = {k: np.array([-exact(b)[k] for b in bg]) for k in ("lam", "S_rate", "R_rate")}
spec = [("I", "lam", "$I$", None), ("S", "S_rate", "$S$ return", b_cS), ("R", "R_rate", "$R$", b_cR)]
fig, axs = plt.subplots(2, 3, figsize=(6.6, 3.9), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
for c, (key, ek, title, bc) in enumerate(spec):
    a, d = axs[0, c], axs[1, c]
    bs = np.array([r["beta"] for r in rows]); lv = [r["levels"]["0.005"][key] for r in rows]
    m = -np.array([v["mean"] for v in lv]); h = np.array([v["half95"] for v in lv]); xt = -np.array([v["exact"] for v in lv])
    for ax in (a, d):
        ax.axvspan(b_det1, 0.245, color="#BBBBBB", alpha=0.35, lw=0)
        if bc is not None:
            ax.axvline(bc, color="black", lw=0.7, ls=":")
    a.plot(bg, ex[ek], color="#0072B2", lw=1.1)
    a.errorbar(bs, m, yerr=h, fmt="o", ms=3, color="black", elinewidth=0.8, capsize=1.5)
    a.set_title(title, fontsize=9); a.set_ylim(0, 1.55)
    d.axhline(0, color="#0072B2", lw=1.0)
    d.errorbar(bs, 100 * (m / xt - 1), yerr=100 * h / xt, fmt="o", ms=3, color="black", elinewidth=0.8, capsize=1.5)
    d.set_ylim(-5.5, 5.5); d.set_xlabel(r"$\beta$"); d.set_xlim(0.04, 0.245)
    if c:
        a.set_yticklabels([]); d.set_yticklabels([])
axs[0, 0].set_ylabel("decay rate")
axs[1, 0].set_ylabel(r"deviation (%)")
axs[0, 2].text(0.2285, 1.42, "band", fontsize=7.5, ha="center")
fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.4)
os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "jia2026joint_F3_rates.pdf")
fig.savefig(out, metadata={"CreationDate": None, "ModDate": None})
print(out, "crossovers %.4f %.4f band from %.4f" % (b_cS, b_cR, b_det1))
