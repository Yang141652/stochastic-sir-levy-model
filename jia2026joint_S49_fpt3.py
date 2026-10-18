"""Session 49 (row 44 part 2d; S41-18, S48-3): the three-way test of the first-passage mechanism, off the manuscript.
Band case of Section 5.4 (beta, x0, horizon, dt, paths and levels read from the S33 E7_params.json, never retyped). Three arms
driven by common random numbers, drawn in the order of the validated run():
  F  the full model (the S33 run() scheme, step for step);
  H  the drift of ln I driven by the disease-free S-hat (the S equation without -beta S I; same noise, same start);
  B  ln I alone: drift lambda - int l_2 dnu (so its mean rate is lambda) and the same noise increments of ln I.
Drivers: G (the VEES specification: the noise of ln I is purely Brownian) and TS1 (the driver Remark S4.4 reports). Seeds: the E7
seed of each driver (arm F must reproduce the stored E7 hit times bitwise) and two fresh seeds per driver. Checks before any
statistic: c1 arm F equals E7_band_<X>.npz hit times bitwise; c2 under G, arm B medians against the inverse-Gaussian median with the
discrete-monitoring shift 0.5826 (v dt)^(1/2), inside the order-statistic 95% interval; c3 the time average of S-hat over t > 20
against Lambda / a1 (Lemma S3.3) within 1%; c4 the mean rate of arm B against lambda within 4 standard errors. Statistics: medians per
level with order-statistic 95% intervals; per-decade rises with paired bootstrap 95% intervals (2000 resamples, seed 4949); the arm
differences of the rises, paired. Every argument explicit (F-S28-4). Writes work/notes/S49/S49_fpt3.json and S49_fpt3_hits.npz.
usage (CCR root): python3 -B work/scripts/jia2026joint_S49_fpt3.py"""
import hashlib, json, os, sys, time
import numpy as np
from math import erf, exp, log, sqrt
sys.path.insert(0, "work/scripts")
import SIRTS1_ref_thresholds as T
import jia2026joint_S33_E7_drivers as D
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
NUM, OUT = "work/numerics/S33_WUN2b2a", "work/notes/S49"
V = json.load(open(f"{NUM}/E7_validation.json"))
assert V["all_pass"] and V["meta"]["drivers_sha256"] == sha("work/scripts/jia2026joint_S33_E7_drivers.py"), "validation gate"
PR = json.load(open(f"{NUM}/params/E7_params.json")); assert PR["validation_sha256"] == sha(f"{NUM}/E7_validation.json")
g = PR["regimes"]["band"]; DT, NPATH, LEVELS = PR["dt"], PR["npath"], tuple(PR["levels"])
LAM, MU, EPS, G1, PHI, LW = PR["Lam"], PR["mu"], PR["eps"], tuple(PR["g1"]), np.array(PR["Phi"]), PR["law"]
P = T.setup(alpha=LW["alpha"], b=LW["b"], zeta=tuple(LW["zeta"]), n=LW["n"], om=LW["om"], c=tuple(LW["c"]))
B = D.build(LAM, MU, EPS, G1, PHI, P, PR["ts1_dir"])
SEEDS = {"G": [PR["seeds"]["band"]["G"], 49101, 49102], "TS1": [PR["seeds"]["band"]["TS1"], 49121, 49122]}


def run3(beta, T_end, dt, npath, x0, jump, mu_x, lam, lmean2, Lam, Phi, levels):
    """Arms F, H and B on common random numbers; draws in the order of D.run (dE, then the driver's jump)."""
    Gm = np.linalg.cholesky(Phi); m1, m2, m3 = mu_x
    Y = np.tile(np.log(x0), (npath, 1)); nstep = int(T_end / dt)
    Hs = np.full(npath, np.log(x0[0])); Hi = np.full(npath, np.log(x0[1])); Bi = np.full(npath, np.log(x0[1]))
    hit = {a: {L: np.full(npath, np.inf) for L in levels} for a in "FHB"}; sacc = np.zeros(npath); nacc = 0
    for k in range(nstep):
        dE = np.sqrt(dt) * D.SIM.rng.standard_normal((npath, 3)) @ Gm.T
        dJ = jump(dt, npath)
        S, I, R = np.exp(Y[:, 0]), np.exp(Y[:, 1]), np.exp(Y[:, 2])
        drift = np.stack([Lam / S - m1 - beta * I - Phi[0, 0] / 2, beta * S - m2 - Phi[1, 1] / 2,
                          I / R - m3 - Phi[2, 2] / 2], axis=1)
        Y = Y + drift * dt + dE + dJ
        Sh = np.exp(Hs)
        Hi = Hi + (beta * Sh - m2 - Phi[1, 1] / 2) * dt + dE[:, 1] + dJ[:, 1]
        Hs = Hs + (Lam / Sh - m1 - Phi[0, 0] / 2) * dt + dE[:, 0] + dJ[:, 0]
        Bi = Bi + (lam - lmean2) * dt + dE[:, 1] + dJ[:, 1]
        t = (k + 1) * dt
        if t > 20 - 1e-9: sacc += np.exp(Hs); nacc += 1
        for a, y in (("F", Y[:, 1]), ("H", Hi), ("B", Bi)):
            for L in levels:
                h = hit[a][L]; new = np.isinf(h) & (y < np.log(L)); h[new] = t
    return hit, sacc / nacc, Bi


def ci_median(x):
    """Distribution-free 95% interval of the median (normal approximation to the binomial order statistics)."""
    n = len(x); s = np.sort(x); j = int(np.floor(0.5 * n - 0.98 * sqrt(n))); k = int(np.ceil(0.5 * n + 0.98 * sqrt(n)))
    return [float(s[j - 1]), float(s[k - 1])]


def ig_median(d, lam_abs, v):
    """Median of the first passage of drifted Brownian motion over distance d (inverse Gaussian), by bisection."""
    m, s = d / lam_abs, d * d / v; N = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
    F = lambda t: N(sqrt(s / t) * (t / m - 1)) + exp(2 * s / m) * N(-sqrt(s / t) * (t / m + 1))
    lo, hi = 1e-6, 50 * m
    for _ in range(200):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if F(mid) < 0.5 else (lo, mid)
    return 0.5 * (lo + hi)


t0 = time.time(); rec = dict(meta=dict(beta=g["beta"], x0=g["x0"], T=g["T"], dt=DT, npath=NPATH, levels=LEVELS, seeds=SEEDS,
                                     drivers_sha256=V["meta"]["drivers_sha256"], params_sha256=sha(f"{NUM}/params/E7_params.json"),
                                     utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())), drivers={})
hits_out = {}; L2, L3, L4 = LEVELS
for X in ("G", "TS1"):
    sm = D.summary(B["laws"][X], g["beta"], LAM, B["a"], PHI); lam = sm["lam"]; lm2 = float(B["laws"][X]["Lmean"][1])
    v = float(PHI[1, 1] + B["laws"][X]["C"][1, 1]); target = log(10) / abs(lam)
    H = {a: {L: [] for L in LEVELS} for a in "FHB"}; checks = {}
    for i, sd in enumerate(SEEDS[X]):
        D.SIM.rng = np.random.default_rng(sd)
        hit, shat, Bi = run3(g["beta"], g["T"], DT, NPATH, tuple(g["x0"]), B["jumps"][X], B["mu_x"][X], lam, lm2, LAM, PHI, LEVELS)
        if i == 0:   # c1: arm F reproduces the stored E7 hit times bitwise
            E = np.load(f"{NUM}/E7_band_{X}.npz")["hit"]
            assert all(np.array_equal(hit["F"][L], E[j]) for j, L in enumerate(LEVELS)), f"c1 {X}"
        rel = abs(shat.mean() / (LAM / B["a"][0]) - 1); assert rel < 0.01, ("c3", X, rel)   # c3
        rate = (Bi - log(g["x0"][1])) / g["T"]; se = rate.std(ddof=1) / sqrt(NPATH)
        assert abs(rate.mean() - lam) < 4 * se, ("c4", X, rate.mean(), lam, se)                # c4
        checks[f"seed{sd}"] = dict(c3_Shat_over_Lam_a1=float(shat.mean() / (LAM / B["a"][0])), c4_rateB=float(rate.mean()), c4_se=float(se))
        for a in "FHB":
            for L in LEVELS: H[a][L].append(hit[a][L])
    H = {a: {L: np.concatenate(H[a][L]) for L in LEVELS} for a in "FHB"}; n = len(H["F"][L2])
    if X == "G":   # c2: arm B against the inverse-Gaussian median with the discrete-monitoring shift
        c2 = {}
        for L in LEVELS:
            d = log(g["x0"][1]) - log(L) + 0.5826 * sqrt(v * DT); mdl = ig_median(d, abs(lam), v); ci = ci_median(H["B"][L])
            assert ci[0] <= mdl <= ci[1], ("c2", L, mdl, ci); c2[str(L)] = dict(ig_median=mdl, sim_ci=ci)
        checks["c2"] = c2
    rng = np.random.default_rng(4949); idx = rng.integers(0, n, (2000, n))
    med = {a: {L: float(np.median(H[a][L])) for L in LEVELS} for a in "FHB"}
    bmed = {a: {L: np.median(H[a][L][idx], axis=1) for L in LEVELS} for a in "FHB"}
    rise = {a: [med[a][L3] - med[a][L2], med[a][L4] - med[a][L3]] for a in "FHB"}
    brise = {a: [bmed[a][L3] - bmed[a][L2], bmed[a][L4] - bmed[a][L3]] for a in "FHB"}
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    e7 = {a: {L: float(np.median(H[a][L][:NPATH])) for L in LEVELS} for a in "FHB"}
    out = dict(lam=lam, target_ln10_over_abs_lam=target, v=v, Lmean2=lm2, R0=sm["R0"], k1=sm["k1"], n_pooled=n, checks=checks,
               median_pooled={a: {str(L): med[a][L] for L in LEVELS} for a in "FHB"},
               median_ci_pooled={a: {str(L): ci_median(H[a][L]) for L in LEVELS} for a in "FHB"},
               median_E7_seed={a: {str(L): e7[a][L] for L in LEVELS} for a in "FHB"},
               frac_hit_by_T={a: {str(L): float(np.isfinite(H[a][L]).mean()) for L in LEVELS} for a in "FHB"},
               rise={a: rise[a] for a in "FHB"}, rise_ci={a: [q(brise[a][0]), q(brise[a][1])] for a in "FHB"},
               rise_frac_of_target={a: [r / target for r in rise[a]] for a in "FHB"},
               diff_H_minus_F={"rise": [rise["H"][j] - rise["F"][j] for j in (0, 1)], "ci": [q(brise["H"][j] - brise["F"][j]) for j in (0, 1)]},
               diff_B_minus_H={"rise": [rise["B"][j] - rise["H"][j] for j in (0, 1)], "ci": [q(brise["B"][j] - brise["H"][j]) for j in (0, 1)]})
    rec["drivers"][X] = out
    for a in "FHB": hits_out[f"{X}_{a}"] = np.array([H[a][L] for L in LEVELS])
    print(f"{X}: lam {lam:.5f}, ln10/|lam| {target:.2f}, v {v:.4f}, n {n}; checks c1 c3 c4" + (" c2" if X == "G" else "") + " pass", flush=True)
    for a in "FHB":
        print(f"  {a}: medians {[round(med[a][L], 2) for L in LEVELS]} (E7 seed {[round(e7[a][L], 2) for L in LEVELS]}); rises "
              f"{[round(r, 2) for r in rise[a]]} CI {[[round(x, 2) for x in c] for c in out['rise_ci'][a]]}; of target "
              f"{[round(r, 3) for r in out['rise_frac_of_target'][a]]}; hit by T {[round(out['frac_hit_by_T'][a][str(L)], 3) for L in LEVELS]}")
    print(f"  H-F rises {[round(x, 2) for x in out['diff_H_minus_F']['rise']]} CI {[[round(x, 2) for x in c] for c in out['diff_H_minus_F']['ci']]};"
          f" B-H {[round(x, 2) for x in out['diff_B_minus_H']['rise']]} CI {[[round(x, 2) for x in c] for c in out['diff_B_minus_H']['ci']]}", flush=True)
os.makedirs(OUT, exist_ok=True); rec["meta"]["secs"] = round(time.time() - t0, 1)
json.dump(rec, open(f"{OUT}/S49_fpt3.json", "w"), indent=1, sort_keys=True); np.savez_compressed(f"{OUT}/S49_fpt3_hits.npz", **hits_out)
print("written", f"{OUT}/S49_fpt3.json", f"{OUT}/S49_fpt3_hits.npz", "secs", rec["meta"]["secs"])
