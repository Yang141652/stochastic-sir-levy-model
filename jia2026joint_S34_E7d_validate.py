#!/usr/bin/env python3
"""Session 34, WU-N2b2b1: validate the downward one-direction driver TS1d before use (ruling S34-2), extending
jia2026joint_S33_E7_validate.py: its T3 and T4 tests and its acceptance rule (S33-3, Bonferroni family-wise two-sided
0.0027 per block) applied to TS1d, plus the sign tests. TS1d is built by the unchanged S33 library, build(..., ts1_dir=3).
U0 the S33 constants reproduce (library and parameters as validated); U1 s_4 is the heaviest downward-I direction and the
exact I-mirror of s_2 with equal weight; U2 matching identities (T3); U3 closed-form sign test (kappa3_I reverses, kappa4_I
agrees, S and R laws equal TS1's); U4 penalty ordering Pen_2(TS1d) < Pen_2(G) < Pen_2(TS1); U5 Monte Carlo acceptance of
the sampler (T4: mean, covariance, compensator, kappa3_I, kappa4_I; radial kappa1..4 and Laplace transform) and every
sampled I-jump downward; U6 same seed gives TS1's increments with the I-component negated.
usage: python3 -B jia2026joint_S34_E7d_validate.py   (from work/scripts); writes ../numerics/S34_WUN2b2b1/E7d_validation.json
"""
import hashlib, json, os, time
from statistics import NormalDist
zstar = lambda m: NormalDist().inv_cdf(1 - 0.0027 / (2 * m))
import numpy as np
from math import gamma as G
import SIRTS1_ref_thresholds as T
import jia2026joint_S33_E7_drivers as D

HERE = os.path.dirname(os.path.abspath(__file__)); S33 = os.path.join(HERE, "..", "numerics", "S33_WUN2b2a")
OUT = os.path.join(HERE, "..", "numerics", "S34_WUN2b2b1"); os.makedirs(OUT, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
LIB = os.path.join(HERE, "jia2026joint_S33_E7_drivers.py"); V33P = os.path.join(S33, "E7_validation.json")
V33 = json.load(open(V33P))
assert V33["all_pass"] and V33["meta"]["drivers_sha256"] == sha(LIB), "S33 validation gate: library changed since validation"
LAM, MU, EPS, G1 = 8.0, 0.75, 0.5, (-0.3, 0.5, 0.2)
PHI = np.array([[3, 2, 1], [2, 3, 2], [1, 2, 4]]) * 1e-3
LAW = dict(alpha=0.4, b=1.0, zeta=(0.01, 0.3, 0.01), n=6, om=0.5, c=(1, 5, 5, 5, 1, 1))
m33 = V33["meta"]
assert m33["law"] == json.loads(json.dumps(LAW)) and np.allclose(m33["Phi"], PHI, rtol=0, atol=0) and \
    (m33["Lam"], m33["mu"], m33["eps"], list(m33["g1"])) == (LAM, MU, EPS, list(G1)), "parameters differ from the S33 set"
P = T.setup(**LAW); UP, DOWN = m33["ts1_dir"], 3
Bu = D.build(LAM, MU, EPS, G1, PHI, P, UP); Bd = D.build(LAM, MU, EPS, G1, PHI, P, DOWN)
Lu, Ld, L6, Lg = Bu["laws"]["TS1"], Bd["laws"]["TS1"], Bd["laws"]["TS6"], Bd["laws"]["G"]
summ = lambda L_, b: D.summary(L_, b, LAM, Bd["a"], PHI)
V = dict(meta=dict(script=os.path.basename(__file__), utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                   drivers_sha256=sha(LIB), s33_validation_sha256=sha(V33P),
                   ref_sha256={f: sha(os.path.join(HERE, f)) for f in ("SIRTS1_ref_thresholds.py", "SIRTS1_ref_simulation.py")},
                   ref_function_source_sha16=D.SEG, law=LAW, Phi=PHI.tolist(), Lam=LAM, mu=MU, eps=EPS, g1=G1,
                   ts1_dir=UP, ts1d_dir=DOWN))
ok = []
def chk(name, cond, **info):
    ok.append(bool(cond)); V.setdefault("tests", {})[name] = dict(passed=bool(cond), **info); print(("PASS " if cond else "FAIL ") + name, info if not cond else "")

c33 = V33["constants"]
chk("U0 S33 constants reproduce (a; R0 of G, CP6, TS1, TS6 at beta 0.232 and 0.316)",
    np.array_equal(Bd["a"], np.array(c33["a"])) and all(
        summ(Bu["laws"][X] if X == "TS1" else Bd["laws"][X], b)["R0"] == c33["summary"][X][str(b)]["R0"]
        for X in ("G", "CP6", "TS1", "TS6") for b in (0.232, 0.316)))
sI = P["s"][:, 1]; neg = [j for j in range(len(sI)) if sI[j] < -1e-12]
chk("U1 s_4 heaviest downward-I direction; exact I-mirror of s_2, equal weight",
    max(neg, key=lambda j: P["w"][j] * sI[j] ** 2) == DOWN and abs(sI[DOWN] + sI[UP]) < 1e-15
    and np.allclose(P["s"][DOWN][[0, 2]], P["s"][UP][[0, 2]], rtol=0, atol=1e-15) and P["w"][DOWN] == P["w"][UP],
    s4=P["s"][DOWN].tolist(), s2=P["s"][UP].tolist())
aX = Bd["mu_x"]["TS1"] - Ld["J"]
chk("U2 TS1d mean rates a_i matched", np.allclose(aX, Bd["a"], rtol=0, atol=1e-12), a=aX.tolist())
chk("U2 TS1d I log-variance matched", abs(Ld["C"][1, 1] / L6["C"][1, 1] - 1) < 1e-12, C22=float(Ld["C"][1, 1]))
def kap(PX, k):
    z2 = P["zeta"][1]; return float(z2 ** k * np.sum(PX["w"] * PX["s"][:, 1] ** k) * G(k - PX["alpha"]) * PX["b"] ** (PX["alpha"] - k))
k3u, k4u, k3d, k4d = kap(Bu["P1"], 3), kap(Bu["P1"], 4), kap(Bd["P1"], 3), kap(Bd["P1"], 4)
SR = np.ix_([0, 2], [0, 2])
chk("U3 kappa3_I reverses, kappa4_I agrees; S and R jump laws equal TS1's",
    k3d < 0 < k3u and abs(k3d + k3u) <= 1e-12 * abs(k3u) and abs(k4d - k4u) <= 1e-12 * abs(k4u)
    and np.allclose(Ld["C"][SR], Lu["C"][SR], rtol=1e-12, atol=0) and np.allclose(Ld["J"][[0, 2]], Lu["J"][[0, 2]], rtol=1e-12, atol=0),
    kappa3_I=[k3u, k3d], kappa4_I=[k4u, k4d])
Pd, Pg, Pu = Ld["Pen"][1], Lg["Pen"][1], Lu["Pen"][1]
S = {X: {b: summ(L_, b) for b in (0.232, 0.316)} for X, L_ in (("G", Lg), ("TS1", Lu), ("TS1d", Ld))}
chk("U4 Pen_2: TS1d < G < TS1, hence R0^TS1d > R0^G > R0^TS1 at both betas",
    Pd < Pg < Pu and all(S["TS1d"][b]["R0"] > S["G"][b]["R0"] > S["TS1"][b]["R0"] for b in (0.232, 0.316)),
    Pen2=[Pd, Pg, Pu], R0_232=[S[X][0.232]["R0"] for X in ("TS1d", "G", "TS1")])

DT, NB, NBS = 0.005, 40, 10000
def kstats(x):
    m = x.mean(); d = x - m; m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    return np.array([m, m2, m3, m4 - 3 * m2 ** 2])
def zs(est, exact):
    est = np.asarray(est); mu_ = est.mean(0); se = est.std(0, ddof=1) / np.sqrt(est.shape[0])
    return mu_, ((mu_ - np.asarray(exact)) / np.where(se > 0, se, np.inf))
D.SIM.rng = np.random.default_rng(34010); rows = []; down = True
for _ in range(NB):
    dJ = Bd["jumps"]["TS1"](DT, NBS); down &= bool((dJ[:, 1] <= 0).all())
    cov = np.cov(dJ.T, bias=True)
    rows.append(np.concatenate([dJ.mean(0), cov[np.triu_indices(3)], np.exp(dJ).mean(0), kstats(dJ[:, 1])[2:]]))
ex = np.concatenate([Ld["Lmean"] * DT, Ld["C"][np.triu_indices(3)] * DT, np.exp(Ld["J"] * DT), [k3d * DT, k4d * DT]])
m_, z = zs(np.array(rows), ex)
names = ["mean_S", "mean_I", "mean_R"] + [f"cov_{i}{j}" for i, j in zip(*np.triu_indices(3))] + ["Eexp_S", "Eexp_I", "Eexp_R", "kappa3_I", "kappa4_I"]
zmax = float(np.max(np.abs(z)))
chk("U5 TS1d sampler (mean, cov, compensator, kappa3_I, kappa4_I)", zmax < zstar(len(names)), zmax=zmax, zstar=zstar(len(names)))
V["tests"]["U5 TS1d sampler (mean, cov, compensator, kappa3_I, kappa4_I)"].update(z={n_: round(float(v), 2) for n_, v in zip(names, z)})
chk("U5 every sampled I-jump of TS1d is downward", down, n=NB * NBS)
P1 = Bd["P1"]; al, bb, w1 = P1["alpha"], P1["b"], P1["w"][0]
D.SIM.rng = np.random.default_rng(34021); rr = []
for _ in range(NB):
    r = D.SIM.ts_increment(al, w1, bb, DT, NBS); rr.append(np.concatenate([kstats(r), [np.exp(-r).mean()]]))
_, zr = zs(np.array(rr), [DT * w1 * G(k - al) * bb ** (al - k) for k in (1, 2, 3, 4)] + [np.exp(DT * w1 * G(-al) * ((bb + 1) ** al - bb ** al))])
chk("U5 TS1d radial kappa1..4 and LT", float(np.max(np.abs(zr))) < zstar(len(zr)), zmax=float(np.max(np.abs(zr))), zstar=zstar(len(zr)))
D.SIM.rng = np.random.default_rng(34030); ju = Bu["jumps"]["TS1"](DT, 1000)
D.SIM.rng = np.random.default_rng(34030); jd = Bd["jumps"]["TS1"](DT, 1000)
chk("U6 same seed: TS1d increments equal TS1's with the I-component negated",
    np.allclose(jd, ju * np.array([1.0, -1.0, 1.0]), rtol=1e-14, atol=0) and np.array_equal(jd[:, 1] < 0, ju[:, 1] > 0))
V["T4_shape"] = {"TS1d": dict(kappa3_I_per_time=k3d, kappa4_I_per_time=k4d, mc_kappa3_I=float(m_[-2] / DT), mc_kappa4_I=float(m_[-1] / DT))}
V["constants"] = dict(a=Bd["a"].tolist(), ts1d_weight=float(P1["w"][0]), ts1d_direction=P1["s"][0].tolist(),
                      laws={"TS1d": {k: np.asarray(v).tolist() for k, v in Ld.items()}}, mu_x={"TS1d": Bd["mu_x"]["TS1"].tolist()},
                      summary={"TS1d": {str(b_): summ(Ld, b_) for b_ in (0.232, 0.316)}})
V["all_pass"] = all(ok); V["n_tests"] = len(ok)
json.dump(V, open(os.path.join(OUT, "E7d_validation.json"), "w"), indent=1)
print("all_pass", V["all_pass"], "of", len(ok))
