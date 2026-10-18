#!/usr/bin/env python3
"""Session 38, E-1: the check of the first-passage corollary against the stored E7 band runs (beta = 0.232, 1000 paths per driver,
dt 0.005, horizon 80, levels 1e-2, 1e-3, 1e-4; seeds and parameters in work/numerics/S33_WUN2b2a/params/E7_params.json).
No new simulation. The exact rates lambda^X are the validation constants of the S33 and S34 E7 gates (V and Vd), cross-checked
against the closed forms and the band entries of the S34 summary (three signals, asserted equal). Paths that do not reach a
level by the horizon are censored (time +inf); medians and their 95% intervals are order statistics, as in the S34 summary.
Writes work/numerics/S38_E1/E1_check.json and E1_check.txt.  usage (CCR root): python3 -B work/scripts/jia2026joint_S38_E1_check.py"""
import hashlib, json, os
import numpy as np
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
S33, S34, OUT = "work/numerics/S33_WUN2b2a", "work/numerics/S34_WUN2b2b1", "work/numerics/S38_E1"
PR = json.load(open(f"{S33}/params/E7_params.json")); V = json.load(open(f"{S33}/E7_validation.json")); Vd = json.load(open(f"{S34}/E7d_validation.json"))
SJ = json.load(open(f"{S34}/jia2026joint_S34_E7_summary.json"))
assert V["all_pass"] and Vd["all_pass"] and PR["regimes"]["band"]["beta"] == 0.232 and PR["regimes"]["band"]["T"] == 80.0
LEV, T80, Z = tuple(PR["levels"]), PR["regimes"]["band"]["T"], 1.959964
assert LEV == (1e-2, 1e-3, 1e-4) and PR["npath"] == 1000 and PR["dt"] == 0.005
SM = dict(V["constants"]["summary"], **Vd["constants"]["summary"]); a2 = V["constants"]["a"][1]
DR = ["G", "CP6", "TS6", "TS1", "TS1d"]
def med_ci(x):
    s = np.sort(np.asarray(x, float)); n = s.size; j = int(np.floor(n / 2 - Z * np.sqrt(n) / 2)); k = int(np.ceil(n / 2 + Z * np.sqrt(n) / 2))
    return 0.5 * (s[n // 2 - 1] + s[n // 2]), s[j - 1], s[k - 1]
J = dict(meta=dict(levels=LEV, horizon=T80, z=Z, a2=a2, npz={}), drivers={}); L = []
def out(t=""): L.append(t); print(t)
out("# E-1 check (Session 38): first passage of I below eps in the band case, beta = 0.232, stored E7 runs")
out("| driver | ln10/|lam| | median tau at 1e-2, 1e-3, 1e-4 [95%] | censored | median(tau)/ln(1/eps) | rise of the median per decade [95% bootstrap] | per-path decade time, median [95%] | fall of ln I on (20,80]: mean, SD |")
rng = np.random.default_rng(38001); NB = 2000
for X in DR:
    p = f"{S34 if X == 'TS1d' else S33}/E7_band_{X}.npz"; J["meta"]["npz"][X] = sha(p)
    lam = SM[X]["0.232"]["lam"]
    assert abs(lam - SJ["closed"][X]["lam_232"]) < 1e-12 and abs(lam - SJ["band"][X]["lam"]) < 1e-12 and lam < 0, X
    z = np.load(p); hit, lat = z["hit"], z["lnI_at"]; n = hit.shape[1]
    assert hit.shape == (3, 1000) and np.all(np.isfinite(hit) | np.isposinf(hit)) and lat.shape[0] == 4, X
    assert np.all(hit[np.isfinite(hit)] <= T80 + 1e-9) and np.all(hit[1] >= hit[0]) and np.all(hit[2] >= hit[1]), X
    cens = [int(np.isposinf(h).sum()) for h in hit]; assert max(cens) < n // 4, X
    med = [med_ci(h) for h in hit]; rat = [med_ci(h / np.log(1 / e))[0] for h, e in zip(hit, LEV)]
    rise = [med[l + 1][0] - med[l][0] for l in (0, 1)]; bs = np.empty((NB, 2))
    for b in range(NB):
        mb = np.median(hit[:, rng.integers(0, n, n)], axis=1); bs[b] = mb[1:] - mb[:-1]
    rci = [tuple(np.percentile(bs[:, l], [2.5, 97.5])) for l in (0, 1)]
    per = []
    for l in (0, 1):
        dl = np.where(np.isposinf(hit[l + 1]), np.inf, hit[l + 1] - np.where(np.isposinf(hit[l]), 0.0, hit[l])); per.append(med_ci(dl))
    fall = lat[0] - lat[3]; fm, fs = float(fall.mean()), float(fall.std(ddof=1))
    ref = np.log(10.0) / abs(lam)
    J["drivers"][X] = dict(lam=lam, R0=1 + lam / a2, inv_abs_lam=1 / abs(lam), ln10_over_abs_lam=ref, censored=cens, median_tau=med, median_ratio=rat,
                           rise_of_median=rise, rise_ci=rci, rise_over_ref=[r / ref for r in rise], per_path_decade=per, fall_20_80=[fm, fs])
    f = lambda w: "%.2f [%.2f, %.2f]" % w
    out("| %s | %.2f | %s | %s | %s | %s | %s | %.2f, %.2f |" % (X, ref, "; ".join(f(w) for w in med), cens, ", ".join("%.2f" % r for r in rat),
        "; ".join("%.2f [%.2f, %.2f]" % (r, c[0], c[1]) for r, c in zip(rise, rci)), "; ".join(f(w) for w in per), fm, fs))
fr = [r for X in DR for r in J["drivers"][X]["rise_over_ref"]]; J["meta"]["rise_over_ref_range"] = [min(fr), max(fr)]
out("rise of the median per decade / (ln10/|lam|), all drivers and both decades: %.3f to %.3f" % (min(fr), max(fr)))
out(); out("1/|lam| per driver is the limit of tau_eps/ln(1/eps) (the corollary); the decade increment / ln10 is the finite-level slope of tau in ln(1/eps).")
os.makedirs(OUT, exist_ok=True)
json.dump(J, open(f"{OUT}/E1_check.json", "w"), indent=1); open(f"{OUT}/E1_check.txt", "w").write("\n".join(L) + "\n")
print("written:", f"{OUT}/E1_check.json", f"{OUT}/E1_check.txt")
