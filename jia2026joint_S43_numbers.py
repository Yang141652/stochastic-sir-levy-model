"""Session 43 (row 41, CR-19; S40-4, S43-3): normalized local sensitivity indices U_p^f = (df/dp)(p/f) of R0, the infection clock
a2(1 - R0), and the clocks k1 and k3 of Theorem 4.1(a), at the baseline of S43-3 (i): the parameter set of Section 5.1, the TS6 law
(S28-2) and beta = 0.232. Every value comes from the validated library (jia2026joint_S33_E7_drivers over SIRTS1_ref_thresholds) with
every argument explicit (the SIRTS1_ref_*.py defaults are the pre-merge set, F-S28-4). Jump-law parameters move the mean rates a_i
through int eta_i dnu with mu and gamma_1 held fixed. Primary index: the closed-form chain-rule derivative where the four quantities
are explicit in the parameter (clock = a2 - beta Lam / a1 + Phi_22 / 2 + Pen_2; k_i = a_i + Phi_ii / 2 + Pen_i), else a fourth-order
central difference through the library (relative step 1e-3). Check: plain central differences at relative steps 1e-4 and 1e-5 must
agree with the primary index within |diff| <= 1e-6 (1 + |U|). The library first reproduces the printed a1, a2, a3, k1, k3, k1/a2,
R0 and the two crossovers of Section 5.2. Writes work/notes/S43/S43_numbers.json. usage (CCR root): python3 -B work/scripts/jia2026joint_S43_numbers.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, "work/scripts")
import SIRTS1_ref_thresholds as T
import jia2026joint_S33_E7_drivers as D
BASE = dict(beta=0.232, Lam=8.0, mu=0.75, eps=0.5, g11=-0.3, g12=0.5, g13=0.2, Phi11=3e-3, Phi22=3e-3, Phi33=4e-3,
            alpha=0.4, b=1.0, zeta1=0.01, zeta2=0.3, zeta3=0.01, om=0.5, cscale=1.0)
PHI_OFF = dict(Phi12=2e-3, Phi13=1e-3, Phi23=2e-3); C0 = (1, 5, 5, 5, 1, 1); N_DIR = 6
def quantities(q):
    """R0, infection clock a2(1 - R0), k1, k3 from the validated library; every argument explicit."""
    Phi = np.array([[q["Phi11"], PHI_OFF["Phi12"], PHI_OFF["Phi13"]], [PHI_OFF["Phi12"], q["Phi22"], PHI_OFF["Phi23"]],
                    [PHI_OFF["Phi13"], PHI_OFF["Phi23"], q["Phi33"]]])
    P = T.setup(alpha=q["alpha"], b=q["b"], zeta=(q["zeta1"], q["zeta2"], q["zeta3"]), n=N_DIR, om=q["om"], c=tuple(q["cscale"] * x for x in C0))
    a = D.rates(q["Lam"], q["mu"], q["eps"], (q["g11"], q["g12"], q["g13"]), Phi, P)[1]
    r = D.summary(D.law_ts(P), q["beta"], q["Lam"], a, Phi)
    return dict(R0=r["R0"], clock=-r["lam"], k1=r["k1"], k3=r["k3"]), a, P
F = ("R0", "clock", "k1", "k3")
f0, a, P0 = quantities(BASE)
# ---- the library reproduces what v1_12_0 prints (Section 5.1, Table 2, Section 5.2)
PHI0 = np.array([[3, 2, 1], [2, 3, 2], [1, 2, 4]]) * 1e-3
BU = D.build(8.0, 0.75, 0.5, (-0.3, 0.5, 0.2), PHI0, T.setup(alpha=0.4, b=1.0, zeta=(0.01, 0.3, 0.01), n=6, om=0.5, c=C0), 1)
assert np.allclose(BU["a"], a, rtol=0, atol=1e-14)
assert [round(x, 4) for x in a] == [1.1529, 1.4737, 0.4158] and round(f0["k1"], 4) == 1.1547 and round(f0["k3"], 4) == 0.4180
assert round(f0["k1"] / a[1], 3) == 0.784 and round(f0["R0"], 4) == 0.9039
assert abs(f0["clock"] - a[1] * (1 - f0["R0"])) < 1e-14
def beta_cross(k):   # a2 (1 - R0(beta)) = k; the clock is linear in beta with slope -Lam / a1
    return BASE["beta"] + (f0["clock"] - k) * a[0] / BASE["Lam"]
assert round(beta_cross(f0["k1"]), 3) == 0.086 and round(beta_cross(f0["k3"]), 3) == 0.192
print(f"check: library reproduces a = {[round(x, 4) for x in a]}, k1 {f0['k1']:.4f}, k3 {f0['k3']:.4f}, k1/a2 {f0['k1'] / a[1]:.3f}, "
      f"R0 {f0['R0']:.4f}, crossovers beta {beta_cross(f0['k1']):.3f} and {beta_cross(f0['k3']):.3f}; clock {f0['clock']:.4f}")
# ---- closed-form derivatives (chain rule through the closed forms the library evaluates)
a1, a2, a3 = a; bL = BASE["beta"] * BASE["Lam"]; R0 = f0["R0"]
CF = {"beta": dict(R0=BASE["Lam"] / (a1 * a2), clock=-BASE["Lam"] / a1, k1=0.0, k3=0.0),
      "Lam": dict(R0=BASE["beta"] / (a1 * a2), clock=-BASE["beta"] / a1, k1=0.0, k3=0.0),
      "mu": dict(R0=(-bL / a1 ** 2) / a2 - R0 / a2, clock=1 + bL / a1 ** 2, k1=1.0, k3=1.0),
      "eps": dict(R0=-R0 / a2, clock=1.0, k1=0.0, k3=0.0),
      "g11": dict(R0=(bL / a1 ** 2) / a2, clock=-bL / a1 ** 2, k1=-1.0, k3=0.0),
      "g12": dict(R0=R0 / a2, clock=-1.0, k1=0.0, k3=0.0),
      "g13": dict(R0=0.0, clock=0.0, k1=0.0, k3=-1.0),
      "Phi11": dict(R0=0.0, clock=0.0, k1=0.5, k3=0.0),
      "Phi22": dict(R0=-0.5 / a2, clock=0.5, k1=0.0, k3=0.0),
      "Phi33": dict(R0=0.0, clock=0.0, k1=0.0, k3=0.5)}
JUMP = ("alpha", "b", "zeta1", "zeta2", "zeta3", "om", "cscale")
def fd(p, rel, order):
    h = rel * abs(BASE[p]); g = lambda t: quantities(dict(BASE, **{p: BASE[p] + t}))[0]
    if order == 2:
        fp, fm = g(h), g(-h); return {k: (fp[k] - fm[k]) / (2 * h) for k in F}
    fp, fm, fp2, fm2 = g(h), g(-h), g(2 * h), g(-2 * h)
    return {k: (8 * (fp[k] - fm[k]) - (fp2[k] - fm2[k])) / (12 * h) for k in F}
U = {}; chk = {}; worst = 0.0
for p in list(CF) + list(JUMP):
    der = CF[p] if p in CF else fd(p, 1e-3, 4)
    U[p] = {k: der[k] * BASE[p] / f0[k] for k in F}
    for rel in (1e-4, 1e-5):
        c = fd(p, rel, 2); Uc = {k: c[k] * BASE[p] / f0[k] for k in F}
        for k in F:
            d = abs(Uc[k] - U[p][k]); worst = max(worst, d / (1 + abs(U[p][k])))
            assert d <= 1e-6 * (1 + abs(U[p][k])), (p, k, rel, U[p][k], Uc[k])
        chk[f"{p}@{rel:g}"] = Uc
# the off-diagonal entries of Phi enter none of the four quantities
for nm in PHI_OFF:
    PHI_OFF[nm] *= 1.5; f1 = quantities(BASE)[0]; PHI_OFF[nm] /= 1.5
    assert all(f1[k] == f0[k] for k in F), nm
# structural identities: U_beta^R0 = U_Lam^R0 = R0det / R0; U^clock = -a2 R0 U^R0 / clock for parameters that leave a2 fixed
R0det = bL / (a1 * a2); assert abs(U["beta"]["R0"] - R0det / R0) < 1e-12 and abs(U["Lam"]["R0"] - U["beta"]["R0"]) < 1e-12
for p in ("beta", "Lam", "g11", "Phi11", "Phi33"):
    assert abs(U[p]["clock"] - (-a2 * R0 * U[p]["R0"] / f0["clock"])) < 1e-9, p
# the identities Section S10 states: int ln(1 + eta_2) dnu = 0 under the TS6 weights, so zeta_2 leaves the infection clock unchanged;
# the clock and k1 depend on the jump law only through zeta_1 / b (U_b = -U_zeta1); k3 only through zeta_3 omega c / b
assert abs(D.law_ts(P0)["Lmean"][1]) < 1e-13 and abs(U["zeta2"]["clock"]) < 1e-8
for k in ("clock", "k1"): assert abs(U["b"][k] + U["zeta1"][k]) < 1e-8, k
assert all(abs(U[p]["k3"] - U["zeta3"]["k3"]) < 1e-8 for p in ("om", "cscale")) and abs(U["b"]["k3"] + U["zeta3"]["k3"]) < 1e-8
ZEROS = {(p, k) for p in U for k in F if abs(U[p][k]) < 1e-8}
assert ZEROS == {("beta", "k1"), ("beta", "k3"), ("Lam", "k1"), ("Lam", "k3"), ("eps", "k1"), ("eps", "k3"), ("g11", "k3"), ("g12", "k1"),
                 ("g12", "k3"), ("g13", "R0"), ("g13", "clock"), ("g13", "k1"), ("Phi11", "R0"), ("Phi11", "clock"), ("Phi11", "k3"),
                 ("Phi22", "k1"), ("Phi22", "k3"), ("Phi33", "R0"), ("Phi33", "clock"), ("Phi33", "k1"), ("zeta1", "k3"), ("zeta2", "clock"),
                 ("zeta2", "k1"), ("zeta2", "k3"), ("zeta3", "R0"), ("zeta3", "clock"), ("zeta3", "k1")}
print(f"check: every index matches plain central differences at relative steps 1e-4 and 1e-5 (worst scaled |diff| {worst:.1e} <= 1e-6); "
      f"off-diagonal Phi enters none of the four; U_beta^R0 = U_Lam^R0 = R0det/R0 = {R0det / R0:.4f}")
print(f"baseline: R0 {f0['R0']:.4f}, clock a2(1-R0) {f0['clock']:.4f}, k1 {f0['k1']:.4f}, k3 {f0['k3']:.4f}, R0det {R0det:.4f}")
for p in U: print(f"  {p:7s} " + "  ".join(f"{k} {U[p][k]:+.4f}" for k in F))
rec = dict(baseline=dict(params=BASE, Phi_offdiag=PHI_OFF, c0=list(C0), n=N_DIR), values=f0, a=list(a), R0det=R0det,
           crossovers=dict(k1=beta_cross(f0["k1"]), k3=beta_cross(f0["k3"])), indices=U, method=dict(
           closed_form=list(CF), fourth_order_cd=list(JUMP), primary_rel_step=1e-3, check_rel_steps=[1e-4, 1e-5],
           tolerance="|U_cd - U| <= 1e-6 (1 + |U|)", worst_scaled_diff=worst), checks_cd=chk)
os.makedirs("work/notes/S43", exist_ok=True)
json.dump(rec, open("work/notes/S43/S43_numbers.json", "w"), indent=1, sort_keys=True); print("written work/notes/S43/S43_numbers.json")
