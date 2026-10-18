#!/usr/bin/env python3
"""Session 34, WU-N2b2b1: the E7 numbers sheet with the downward driver TS1d added (ruling S34-2), read from the result files only (no number retyped, none re-simulated).
Closed forms (validation constants): R0, lambda, penalty Pen_2, k1, k3 and the stationary means per driver; the
arithmetic-scale Brownian threshold (cor:C-gauss (ii)); the small-jump series of cor:C-gauss (iii) against the exact
differences; the sign test (TS1 against TS1d: odd-order and even-order parts of the shift). Simulated endpoints: quasi-extinction probabilities (Wilson 95% intervals), the median first-passage time
to 1e-3 (order-statistic interval), per-path slopes of ln I on [20, 80] against lambda^X, quantiles of I, and at P-per the
time averages over t > 50 against the closed forms and the pooled prevalence quantiles (path-block standard errors).
Derived from jia2026joint_S33_E7_summary.py by exact edits; the S33 sheet stays as the record of Session 33.
usage: python3 -B jia2026joint_S34_E7_summary.py   (from work/scripts)
"""
import json, os
import numpy as np
import SIRTS1_ref_thresholds as T

HERE = os.path.dirname(os.path.abspath(__file__)); S33 = os.path.join(HERE, "..", "numerics", "S33_WUN2b2a"); OUT = os.path.join(HERE, "..", "numerics", "S34_WUN2b2b1")
V = json.load(open(os.path.join(S33, "E7_validation.json"))); PR = json.load(open(os.path.join(S33, "params", "E7_params.json")))
Vd = json.load(open(os.path.join(OUT, "E7d_validation.json"))); assert V["all_pass"] and Vd["all_pass"]
DR, LEV, LAM, PHI = ["G", "CP6", "TS6", "TS1", "TS1d"], PR["levels"], PR["Lam"], np.array(PR["Phi"])
P = T.setup(**PR["law"]); a = np.array(V["constants"]["a"]); assert a.tolist() == Vd["constants"]["a"]
SM = dict(V["constants"]["summary"], **Vd["constants"]["summary"]); LW = dict(V["constants"]["laws"], **Vd["constants"]["laws"])
npz = lambda reg, X: os.path.join(OUT if X == "TS1d" else S33, "E7_%s_%s.npz" % (reg, X))
Z = 1.959964; L = []; J = dict(closed={}, band={}, per={})
def out(s=""): L.append(s); print(s)
def wilson(k, n):
    p = k / n; den = 1 + Z ** 2 / n; c = (p + Z ** 2 / (2 * n)) / den; h = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)
def ms(x):
    x = np.asarray(x, float); return x.mean(), x.std(ddof=1) / np.sqrt(x.size)

out("# E7 numbers sheet (Session 34, WU-N2b2b1; S28-2 point; matched mean rates and log-scale jump covariance, ruling S33-2; TS1d added, S34-2)")
out("a1 a2 a3 = %.4f %.4f %.4f | I log-variance per unit time C22 = %.4f (Phi22 = %.4f) | dt %.3f, %d paths per run"
    % (a[0], a[1], a[2], LW["TS6"]["C"][1][1], PHI[1, 1], PR["dt"], PR["npath"]))
out(); out("## Closed forms (exact under the matching convention)")
out("| driver | Pen_2 | R0 (beta 0.232) | R0 - R0^G | lambda (0.232) | k1 | k3 | R0 (0.316) | Ibar (0.316) |")
for X in DR:
    s2, s3 = SM[X]["0.232"], SM[X]["0.316"]; d = s2["R0"] - SM["G"]["0.232"]["R0"]
    out("| %s | %.6f | %.6f | %+.6f | %.5f | %.5f | %.5f | %.6f | %.5f |" % (X, LW[X]["Pen"][1], s2["R0"], d, s2["lam"], s2["k1"], s2["k3"], s3["R0"], s3["Ibar"]))
    J["closed"][X] = dict(Pen2=LW[X]["Pen"][1], R0_232=s2["R0"], dR0_vs_G=d, lam_232=s2["lam"], k1=s2["k1"], k3=s2["k3"], R0_316=s3["R0"], Ibar_316=s3["Ibar"])
M = T.moments(P); v_ar = M[1]["int_eta2"]
for b in (0.232, 0.316):
    r_ar = (b * LAM / a[0] - 0.5 * (PHI[1, 1] + v_ar)) / a[1]
    out("arithmetic-scale Brownian (cor:C-gauss (ii), v = int eta_2^2 dnu = %.6f), beta %.3f: R0^G = %.6f; R0^TS6 - R0^G = %+.6f"
        % (v_ar, b, r_ar, SM["TS6"][str(b)]["R0"] - r_ar))
    J["closed"]["G_arith_%s" % b] = dict(v=v_ar, R0=r_ar, dR0_TS6=SM["TS6"][str(b)]["R0"] - r_ar)
sh = dict(V["T4_shape"], **Vd["T4_shape"])
for X, terms in (("TS6", (4,)), ("TS1", (3, 4)), ("TS1d", (3, 4))):
    lead = -sum(sh[X]["kappa%d_I_per_time" % k] / {3: 6, 4: 24}[k] for k in terms) / a[1]
    exact = SM[X]["0.232"]["R0"] - SM["G"]["0.232"]["R0"]
    out("small-jump series (cor:C-gauss (iii)), %s: terms k = %s give %+.6f of the exact %+.6f (%.0f%%)" % (X, terms, lead, exact, 100 * lead / exact))
    J["closed"]["series_" + X] = dict(terms=terms, leading=lead, exact=exact)

for b in (0.232, 0.316):
    g, u, dn = SM["G"][str(b)]["R0"], SM["TS1"][str(b)]["R0"], SM["TS1d"][str(b)]["R0"]
    out("sign test, beta %.3f: R0^TS1 - R0^G = %+.6f, R0^TS1d - R0^G = %+.6f; odd-order part %+.6f, even-order part %+.6f"
        % (b, u - g, dn - g, (u - dn) / 2, (u + dn) / 2 - g))
    J["closed"]["sign_%s" % b] = dict(up=u - g, down=dn - g, odd=(u - dn) / 2, even=(u + dn) / 2 - g)
out(); out("## Band case, beta 0.232, start (5.551, 0.5, 0.1), horizon 80")
out("| driver | P(I < 1e-2 by 80) | P(I < 1e-3 by 20) | P(I < 1e-3 by 40) | P(I < 1e-3 by 80) | P(I < 1e-4 by 80) | median first passage to 1e-3 | slope of ln I on [20, 80] | lambda^X | z |")
base = None
for X in DR:
    d = np.load(npz("band", X)); hit = d["hit"]; n = hit.shape[1]; e = {}
    for (li, tau) in ((0, 80), (1, 20), (1, 40), (1, 80), (2, 80)):
        e[(li, tau)] = wilson(int((hit[li] <= tau + 1e-9).sum()), n)
    s = np.sort(hit[1]); j = int(np.floor(n / 2 - Z * np.sqrt(n) / 2)); k = int(np.ceil(n / 2 + Z * np.sqrt(n) / 2))
    med = 0.5 * (s[n // 2 - 1] + s[n // 2]); lo, hi = s[j - 1], s[k - 1]
    sl = (d["lnI_at"][3] - d["lnI_at"][0]) / 60.0; m, se = ms(sl); lam = SM[X]["0.232"]["lam"]
    qI = np.percentile(np.exp(d["lnI_at"][1]), [5, 25, 50, 75, 95])
    f = lambda w: "%.3f (%.3f, %.3f)" % w
    out("| %s | %s | %s | %s | %s | %s | %s (%s, %s) | %.4f ± %.4f | %.4f | %+.2f |" % (X, f(e[(0, 80)]), f(e[(1, 20)]), f(e[(1, 40)]), f(e[(1, 80)]), f(e[(2, 80)]),
        "%.1f" % med if np.isfinite(med) else "> 80", "%.1f" % lo, "%.1f" % hi if np.isfinite(hi) else "> 80", m, Z * se, lam, (m - lam) / se))
    J["band"][X] = dict(p={"%g_by_%d" % (LEV[li], tau): v for (li, tau), v in e.items()}, median_fpt_1e3=[med, lo, hi],
                        slope=[m, se], lam=lam, z_slope=(m - lam) / se, I40_quantiles_5_25_50_75_95=qI.tolist())
out("quantiles of I at tau 40 (5, 25, 50, 75, 95%):")
for X in DR: out("  %s: %s" % (X, ", ".join("%.3g" % v for v in J["band"][X]["I40_quantiles_5_25_50_75_95"])))
pG = J["band"]["G"]["p"]["0.001_by_80"][0]
for X in DR[1:]:
    p = J["band"][X]["p"]["0.001_by_80"][0]; se = np.sqrt(p * (1 - p) / 1000 + pG * (1 - pG) / 1000)
    out("  P(I < 1e-3 by 80): %s - G = %+.3f ± %.3f" % (X, p - pG, Z * se)); J["band"][X]["dp_1e3_80_vs_G"] = [p - pG, Z * se]

out(); out("## P-per, beta 0.316, start at the TS6 stationary means, horizon 100, averages over t > 50")
out("| driver | <S> (z vs Sbar^X) | <I> (z vs Ibar^X) | <R> (z vs Rbar^X) | I quantiles 5, 25, 50, 75, 95% pooled over t > 50 (± 1.96 SE) | P(I < 1e-2 by 100) | P(I < 1e-3 by 100) |")
for X in DR:
    d = np.load(npz("per", X)); avg = d["avg"]; hit = d["hit"]; n = hit.shape[1]; s3 = SM[X]["0.316"]
    zz = []
    for i, key in enumerate(("Sbar", "Ibar", "Rbar")):
        m, se = ms(avg[:, i]); zz.append((m, Z * se, (m - s3[key]) / se))
    pool = np.exp(d["pooled"].reshape(-1, n)); qs = np.percentile(pool, [5, 25, 50, 75, 95])
    blk = np.array([np.percentile(pool[:, b * 50:(b + 1) * 50], [5, 25, 50, 75, 95]) for b in range(n // 50)])
    qse = blk.std(0, ddof=1) / np.sqrt(blk.shape[0])
    e1, e2 = wilson(int((hit[0] <= 100 + 1e-9).sum()), n), wilson(int((hit[1] <= 100 + 1e-9).sum()), n)
    out("| %s | %.4f ± %.4f (%+.2f) | %.4f ± %.4f (%+.2f) | %.4f ± %.4f (%+.2f) | %s | %.3f (%.3f, %.3f) | %.3f (%.3f, %.3f) |"
        % ((X,) + zz[0] + zz[1] + zz[2] + (", ".join("%.3f ± %.3f" % (q, Z * s) for q, s in zip(qs, qse)),) + e1 + e2))
    J["per"][X] = dict(avg={k: list(v) for k, v in zip(("S", "I", "R"), zz)}, closed=dict(Sbar=s3["Sbar"], Ibar=s3["Ibar"], Rbar=s3["Rbar"]),
                       I_pooled_quantiles=qs.tolist(), I_pooled_quantiles_se=qse.tolist(), p_1e2_100=e1, p_1e3_100=e2)
open(os.path.join(OUT, "jia2026joint_S34_E7_numbers.txt"), "w").write("\n".join(L) + "\n")
json.dump(J, open(os.path.join(OUT, "jia2026joint_S34_E7_summary.json"), "w"), indent=1, default=float)
