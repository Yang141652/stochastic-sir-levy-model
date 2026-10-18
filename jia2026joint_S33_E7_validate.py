#!/usr/bin/env python3
"""Session 33, WU-N2b2a: validate every E7 driver function before use (E2 criteria: |z| < 3 from batch standard errors).
T0 S28-2 targets; T1 TS6 through run() equals the reference simulate() bitwise; T2 closed forms equal thresholds();
T3 matching identities; T4 Monte Carlo acceptance of each sampler: log-increment mean, covariance, compensator
E exp(dJ_i) = exp(J_i dt), and the cumulants kappa3, kappa4 of the I log-increment (the shape, the variable under
test); radial-law kappa1..kappa4 and a Laplace-transform check for CP6 and TS1.
Acceptance (amended in Session 33 after run 1, F-S33-1): each block passes when every |z| is below the Bonferroni
threshold for a family-wise two-sided level 0.0027 (the 3-sigma convention) over the m statistics of the block; run 1
applied 3.0 to each of 66 correlated statistics (family-wise false alarm near 16%) and flagged the reference TS6 sampler
at |z| = 3.37 for E exp(dJ_I); a six-seed re-test pooled to |z| < 1 (E7_ts6_multiseed_retest.log). Identities whose
exact value is zero use an absolute tolerance scaled to the largest entry.
usage: python3 -B jia2026joint_S33_E7_validate.py   (from work/scripts); writes ../numerics/S33_WUN2b2a/E7_validation.json
"""
import hashlib, json, os, time
from statistics import NormalDist
zstar = lambda m: NormalDist().inv_cdf(1 - 0.0027 / (2 * m))
import numpy as np
from math import gamma as G, factorial
import SIRTS1_ref_thresholds as T
import jia2026joint_S33_E7_drivers as D

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "numerics", "S33_WUN2b2a")
os.makedirs(OUT, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
LAM, MU, EPS, G1 = 8.0, 0.75, 0.5, (-0.3, 0.5, 0.2)
PHI = np.array([[3, 2, 1], [2, 3, 2], [1, 2, 4]]) * 1e-3
LAW = dict(alpha=0.4, b=1.0, zeta=(0.01, 0.3, 0.01), n=6, om=0.5, c=(1, 5, 5, 5, 1, 1))
P = T.setup(**LAW); TS1_DIR = 1
KW = dict(Lam=LAM, mu=MU, eps=EPS, g1=G1, Phi=PHI, P=P, jumps=True)
B = D.build(LAM, MU, EPS, G1, PHI, P, TS1_DIR)
V = dict(meta=dict(script=os.path.basename(__file__), utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                   drivers_sha256=sha(os.path.join(HERE, "jia2026joint_S33_E7_drivers.py")),
                   ref_sha256={f: sha(os.path.join(HERE, f)) for f in ("SIRTS1_ref_thresholds.py", "SIRTS1_ref_simulation.py")},
                   ref_function_source_sha16=D.SEG, law=LAW, Phi=PHI.tolist(), Lam=LAM, mu=MU, eps=EPS, g1=G1, ts1_dir=TS1_DIR))
ok = []
def chk(name, cond, **info):
    ok.append(bool(cond)); V.setdefault("tests", {})[name] = dict(passed=bool(cond), **info); print(("PASS " if cond else "FAIL ") + name, info if not cond else "")

# T0 targets (S30 part 0 and part 4, read from the result file, never retyped)
s30 = json.load(open(os.path.join(HERE, "..", "numerics", "S30_WUN2a", "jia2026joint_S30_WUN2a_results.json")))
tg = s30["part0"]["s28_2_targets"]; r232 = [r for r in s30["part4"]["rows"] if abs(r["beta"] - 0.232) < 1e-12][0]
th232 = T.thresholds(0.232, **KW); th316 = T.thresholds(0.316, **KW)
chk("T0 S28-2 targets", abs(th232["a3"] - tg["a3"]) < 1e-12 and abs(r232["k1"] / th232["a2"] - tg["k1_a2"]) < 1e-12
    and abs(th232["R0"] - r232["R0"]) < 1e-12, R0=th232["R0"])

# T1 bitwise regression of run() with TS6 against the reference simulate()
x0 = tuple(r232["x0"])
D.SIM.rng = np.random.default_rng(33001)
avg_ref, lnI_ref = D.SIM.simulate(0.232, T=1.0, dt=0.005, npath=50, P=P, Lam=LAM, mu=MU, eps=EPS, g1=G1, Phi=PHI, x0=x0)
D.SIM.rng = np.random.default_rng(33001)
o = D.run(0.232, 1.0, 0.005, 50, x0, B["jumps"]["TS6"], B["mu_x"]["TS6"], LAM, PHI, keep_lnI=True)
chk("T1 TS6 run() == reference simulate() bitwise", np.array_equal(o["avg"], avg_ref) and np.array_equal(o["lnI"], lnI_ref),
    nstep=o["nstep"], seed=33001)

# T2 closed forms against thresholds()
S6 = {b: D.summary(B["laws"]["TS6"], b, LAM, B["a"], PHI) for b in (0.232, 0.316)}
chk("T2 TS6 summary == thresholds()", abs(S6[0.232]["R0"] - th232["R0"]) < 1e-12 and abs(S6[0.316]["R0"] - th316["R0"]) < 1e-12
    and abs(S6[0.232]["k1"] - r232["k1"]) < 1e-12 and abs(S6[0.232]["k3"] - r232["k3"]) < 1e-12
    and abs(S6[0.316]["Ibar"] - th316["Ibar"]) < 1e-12 and abs(S6[0.316]["Sbar"] - th316["Sbar"]) < 1e-12
    and abs(S6[0.316]["Rbar"] - th316["Rbar"]) < 1e-12, R0_316=S6[0.316]["R0"])

# T3 matching identities
L = B["laws"]; C6 = L["TS6"]["C"]
for X in D.DRIVERS:
    aX = B["mu_x"][X] - L[X]["J"]
    chk(f"T3 {X} mean rates a_i matched", np.allclose(aX, B["a"], rtol=0, atol=1e-12), a=aX.tolist())
    chk(f"T3 {X} I log-variance matched", abs(L[X]["C"][1, 1] / C6[1, 1] - 1) < 1e-12, C22=L[X]["C"][1, 1])
for X in ("G", "CP6"):
    chk(f"T3 {X} full log-covariance matched", np.allclose(L[X]["C"], C6, rtol=1e-12, atol=1e-12 * np.abs(C6).max()),
        maxabs=float(np.abs(L[X]["C"] - C6).max()))
chk("T3 CP6 log-mean matched", np.allclose(L["CP6"]["Lmean"], L["TS6"]["Lmean"], rtol=1e-12, atol=1e-12 * np.abs(L["TS6"]["Lmean"]).max()))

# T4 Monte Carlo acceptance, N = 40 batches x 10000 increments at dt = 0.005 (the run step)
DT, NB, NBS = 0.005, 40, 10000
def kstats(x):
    m = x.mean(); d = x - m; m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    return np.array([m, m2, m3, m4 - 3 * m2 ** 2])
def zs(est, exact):
    est = np.asarray(est); mu_ = est.mean(0); se = est.std(0, ddof=1) / np.sqrt(est.shape[0])
    return mu_, ((mu_ - np.asarray(exact)) / np.where(se > 0, se, np.inf))
def kap_I(X, k):          # integral of l_2^k against the driver's Levy measure (0 for G when k > 2)
    z2 = P["zeta"][1]
    if X == "G": return 0.0
    if X == "CP6":
        Pc = B["Pc"]; return float(np.sum(Pc["lam"] * (z2 * Pc["s"][:, 1]) ** k * factorial(k) * Pc["theta"] ** k))
    PX = P if X == "TS6" else B["P1"]
    return float(z2 ** k * np.sum(PX["w"] * PX["s"][:, 1] ** k) * G(k - PX["alpha"]) * PX["b"] ** (PX["alpha"] - k))
seed = 33010; V["T4_shape"] = {}
for X in D.DRIVERS:
    D.SIM.rng = np.random.default_rng(seed); seed += 1
    rows = []
    for _ in range(NB):
        dJ = B["jumps"][X](DT, NBS)
        cov = np.cov(dJ.T, bias=True)
        rows.append(np.concatenate([dJ.mean(0), cov[np.triu_indices(3)], np.exp(dJ).mean(0), kstats(dJ[:, 1])[2:]]))
    rows = np.array(rows); Lx = L[X]
    ex = np.concatenate([Lx["Lmean"] * DT, Lx["C"][np.triu_indices(3)] * DT, np.exp(Lx["J"] * DT),
                         [kap_I(X, 3) * DT, kap_I(X, 4) * DT]])
    m_, z = zs(rows, ex)
    names = ["mean_S", "mean_I", "mean_R"] + [f"cov_{i}{j}" for i, j in zip(*np.triu_indices(3))] + \
            ["Eexp_S", "Eexp_I", "Eexp_R", "kappa3_I", "kappa4_I"]
    sel = list(range(len(names)))            # every statistic, against the driver's own closed forms
    zmax = float(np.max(np.abs(z[sel])))
    chk(f"T4 {X} sampler (mean, cov, compensator, kappa3_I, kappa4_I)", zmax < zstar(len(sel)), zmax=zmax, zstar=zstar(len(sel)))
    V["tests"][f"T4 {X} sampler (mean, cov, compensator, kappa3_I, kappa4_I)"].update(
        z={names[k]: round(float(z[k]), 2) for k in sel}, zmax=zmax)
    V["T4_shape"][X] = dict(kappa3_I_per_time=kap_I(X, 3), kappa4_I_per_time=kap_I(X, 4),
                            mc_kappa3_I=float(m_[-2] / DT), mc_kappa4_I=float(m_[-1] / DT))

# radial laws: CP6 on direction 1 (weight 5) and TS1, kappa1..kappa4 and Laplace transform at s = 1
def radial(sample, exact_k, lt_exact, name):
    rows = []
    for _ in range(NB):
        r = sample(NBS); rows.append(np.concatenate([kstats(r), [np.exp(-r).mean()]]))
    m_, z = zs(np.array(rows), list(exact_k) + [lt_exact])
    chk(name, float(np.max(np.abs(z))) < zstar(len(z)), zmax=float(np.max(np.abs(z))), zstar=zstar(len(z)))
    V["tests"][name].update(z=[round(float(v), 2) for v in z])
Pc = B["Pc"]; j = 1; lj, th_ = Pc["lam"][j], Pc["theta"][j]
D.SIM.rng = np.random.default_rng(33020)
def cp_r(n):
    N = D.SIM.rng.poisson(lj * DT, n); r = np.zeros(n); m = N > 0; r[m] = D.SIM.rng.gamma(N[m], th_); return r
radial(cp_r, [DT * lj * factorial(k) * th_ ** k for k in (1, 2, 3, 4)], np.exp(DT * lj * (1 / (1 + th_) - 1)), "T4 CP6 radial kappa1..4 and LT")
P1 = B["P1"]; a, b, w1 = P1["alpha"], P1["b"], P1["w"][0]
D.SIM.rng = np.random.default_rng(33021)
radial(lambda n: D.SIM.ts_increment(a, w1, b, DT, n), [DT * w1 * G(k - a) * b ** (a - k) for k in (1, 2, 3, 4)],
       np.exp(DT * w1 * G(-a) * ((b + 1) ** a - b ** a)), "T4 TS1 radial kappa1..4 and LT")

# constants of record
V["constants"] = dict(a=B["a"].tolist(), mu0=B["mu0"].tolist(), ts1_weight=float(P1["w"][0]), ts1_direction=P1["s"][0].tolist(),
                      cp6_theta=Pc["theta"].tolist(), cp6_rate=Pc["lam"].tolist(),
                      laws={X: {k: np.asarray(v).tolist() for k, v in L[X].items()} for X in D.DRIVERS},
                      mu_x={X: B["mu_x"][X].tolist() for X in D.DRIVERS},
                      summary={X: {str(b_): D.summary(L[X], b_, LAM, B["a"], PHI) for b_ in (0.232, 0.316)} for X in D.DRIVERS})
V["all_pass"] = all(ok); V["n_tests"] = len(ok)
json.dump(V, open(os.path.join(OUT, "E7_validation.json"), "w"), indent=1)
print("all_pass", V["all_pass"], "of", len(ok))
