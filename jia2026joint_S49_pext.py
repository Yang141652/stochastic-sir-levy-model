"""Session 49 (row 44 part 2d; S43-10, S48-3): a second base point for Table S1, P-ext (beta = 0.210, the regime of R0 near 0.8 named
in the E1 box), off the manuscript. The validated function quantities() is loaded by AST from jia2026joint_S43_numbers.py, with its
constants PHI_OFF, C0 and N_DIR read as literals from the same file, so that script's module-level code (which writes S43_numbers.json)
never runs. indices(base) repeats the S43 method with the base passed explicitly: closed-form chain-rule derivatives for the
deterministic, gamma_1 and Phi parameters, fourth-order central differences (relative step 1e-3) for the jump-law parameters, and the
check against plain central differences at relative steps 1e-4 and 1e-5 within 1e-6 (1 + |U|). Gate: at beta = 0.232 indices()
reproduces work/notes/S43/S43_numbers.json (values and every index) within 1e-12. Then beta = 0.210 with every other argument as in
Section 5.1. Writes work/notes/S49/S49_pext.json. usage (CCR root): python3 -B work/scripts/jia2026joint_S49_pext.py"""
import ast, json, os, sys
import numpy as np
sys.path.insert(0, "work/scripts")
import SIRTS1_ref_thresholds as T
import jia2026joint_S33_E7_drivers as D
SRC = "work/scripts/jia2026joint_S43_numbers.py"; src = open(SRC).read(); NS = dict(np=np, T=T, D=D); LIT = {}
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("BASE", "PHI_OFF"):
        LIT[node.targets[0].id] = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords}
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("C0", "N_DIR"):
        LIT[node.targets[0].id] = ast.literal_eval(node.value)
    if isinstance(node, ast.FunctionDef) and node.name == "quantities":
        QSRC = ast.get_source_segment(src, node); exec(compile(ast.Module(body=[node], type_ignores=[]), SRC, "exec"), NS)
NS.update(PHI_OFF=dict(LIT["PHI_OFF"]), C0=LIT["C0"], N_DIR=LIT["N_DIR"]); quantities = NS["quantities"]
BASE0 = LIT["BASE"]; F = ("R0", "clock", "k1", "k3"); JUMP = ("alpha", "b", "zeta1", "zeta2", "zeta3", "om", "cscale")


def indices(base):
    f0, a, P0 = quantities(base); a1, a2, a3 = a; bL = base["beta"] * base["Lam"]; R0 = f0["R0"]
    CF = {"beta": dict(R0=base["Lam"] / (a1 * a2), clock=-base["Lam"] / a1, k1=0.0, k3=0.0),
          "Lam": dict(R0=base["beta"] / (a1 * a2), clock=-base["beta"] / a1, k1=0.0, k3=0.0),
          "mu": dict(R0=(-bL / a1 ** 2) / a2 - R0 / a2, clock=1 + bL / a1 ** 2, k1=1.0, k3=1.0),
          "eps": dict(R0=-R0 / a2, clock=1.0, k1=0.0, k3=0.0),
          "g11": dict(R0=(bL / a1 ** 2) / a2, clock=-bL / a1 ** 2, k1=-1.0, k3=0.0),
          "g12": dict(R0=R0 / a2, clock=-1.0, k1=0.0, k3=0.0),
          "g13": dict(R0=0.0, clock=0.0, k1=0.0, k3=-1.0),
          "Phi11": dict(R0=0.0, clock=0.0, k1=0.5, k3=0.0),
          "Phi22": dict(R0=-0.5 / a2, clock=0.5, k1=0.0, k3=0.0),
          "Phi33": dict(R0=0.0, clock=0.0, k1=0.0, k3=0.5)}

    def fd(p, rel, order):
        h = rel * abs(base[p]); g = lambda t: quantities(dict(base, **{p: base[p] + t}))[0]
        if order == 2:
            fp, fm = g(h), g(-h); return {k: (fp[k] - fm[k]) / (2 * h) for k in F}
        fp, fm, fp2, fm2 = g(h), g(-h), g(2 * h), g(-2 * h)
        return {k: (8 * (fp[k] - fm[k]) - (fp2[k] - fm2[k])) / (12 * h) for k in F}
    U = {}; worst = 0.0
    for p in list(CF) + list(JUMP):
        der = CF[p] if p in CF else fd(p, 1e-3, 4); U[p] = {k: der[k] * base[p] / f0[k] for k in F}
        for rel in (1e-4, 1e-5):
            c = fd(p, rel, 2)
            for k in F:
                d = abs(c[k] * base[p] / f0[k] - U[p][k]); worst = max(worst, d / (1 + abs(U[p][k])))
                assert d <= 1e-6 * (1 + abs(U[p][k])), (p, k, rel)
    return f0, list(a), U, worst


assert BASE0["beta"] == 0.232
R = json.load(open("work/notes/S43/S43_numbers.json"))
f0, a0, U0, w0 = indices(dict(BASE0))
assert all(abs(f0[k] - R["values"][k]) <= 1e-12 for k in F)
gap = max(abs(U0[p][k] - R["indices"][p][k]) for p in U0 for k in F); assert gap <= 1e-12 and set(U0) == set(R["indices"]), gap
print(f"gate: at beta 0.232 indices() reproduces S43_numbers.json (values and {len(U0) * 4} indices; max |diff| {gap:.1e})")
BX = dict(BASE0, beta=0.210); f1, a1_, U1, w1 = indices(BX)
assert a1_ == a0, "the mean rates do not depend on beta"
cross = {k: BX["beta"] + (f1["clock"] - f1[k]) * a1_[0] / BX["Lam"] for k in ("k1", "k3")}
R0det = BX["beta"] * BX["Lam"] / (a1_[0] * a1_[1])
print(f"P-ext beta 0.210: R0 {f1['R0']:.4f}, R0det {R0det:.4f}, clock a2(1-R0) {f1['clock']:.4f}, k1 {f1['k1']:.4f}, k3 {f1['k3']:.4f}; "
      f"slowest of clock, k1, k3 for S: {'clock' if f1['clock'] < f1['k1'] else 'k1'}, for R: {'clock' if f1['clock'] < f1['k3'] else 'k3'}; "
      f"check worst scaled |diff| {w1:.1e}")
same = [p for p in U1 if all(abs(U1[p][k] - U0[p][k]) < 1e-12 for k in ("k1", "k3"))]
print(f"k1 and k3 columns equal at both base points for {len(same)} of {len(U1)} parameters")
for p in U1:
    print(f"  {p:7s} R0 {U0[p]['R0']:+.3f} -> {U1[p]['R0']:+.3f}   clock {U0[p]['clock']:+.3f} -> {U1[p]['clock']:+.3f}   "
          f"k1 {U1[p]['k1']:+.3f}   k3 {U1[p]['k3']:+.3f}")
rmax = max(abs(U1[p]["clock"]) for p in U1); pmax = max(U1, key=lambda p: abs(U1[p]["clock"]))
print(f"largest |clock index|: {pmax} {U1[pmax]['clock']:+.3f} at P-ext, {U0[pmax]['clock']:+.3f} at beta 0.232")
os.makedirs("work/notes/S49", exist_ok=True)
json.dump(dict(base=BX, values=f1, a=a1_, R0det=R0det, indices=U1, worst_scaled_diff=w1, gate_max_diff_vs_S43=gap,
               crossovers_beta=cross, quantities_source_sha16=__import__("hashlib").sha256(QSRC.encode()).hexdigest()[:16]),
          open("work/notes/S49/S49_pext.json", "w"), indent=1, sort_keys=True)
print("written work/notes/S49/S49_pext.json")
