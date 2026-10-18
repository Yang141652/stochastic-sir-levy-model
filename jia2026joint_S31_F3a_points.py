#!/usr/bin/env python3
"""Session 31: the P-bio points for F3a (the S28-2 point, ruling S28-2), written as the CSV that
jia2026joint_F3a_regime_map.py reads (columns label, k1_over_a2, R0). Values from `exact`, extracted unchanged
from jia2026joint_S30_WUN2a.py, on SIRTS1_ref_thresholds with every argument explicit; S28-2 targets asserted.
Rows: the ten beta of the S30 ladder (P-clock ladder; P-ext at beta 0.21; the band case at beta 0.232) and P-per.
usage: python3 -B jia2026joint_S31_F3a_points.py [csv]   (default ../data/jia2026joint_F3a_points_S28-2.csv)
"""
import ast, csv, os, sys, types
import numpy as np
import SIRTS1_ref_thresholds as T
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "jia2026joint_F3a_points_S28-2.csv")
LAM, EPS, MU, G1 = 8.0, 0.5, 0.75, (-0.3, 0.5, 0.2)
PHI = np.array([[3, 2, 1], [2, 3, 2], [1, 2, 4]]) * 1e-3
LAW = dict(alpha=0.4, b=1.0, zeta=(0.01, 0.3, 0.01), n=6, om=0.5, c=(1, 5, 5, 5, 1, 1))
P = T.setup(**LAW); KW = dict(Lam=LAM, mu=MU, eps=EPS, g1=G1, Phi=PHI, P=P); M = T.moments(P)
src = open(os.path.join(HERE, "jia2026joint_S30_WUN2a.py")).read()
node = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "exact"][0]
mod = types.ModuleType("S30"); mod.__dict__.update(dict(np=np, T=T, KW=KW, PHI=PHI, M=M, LAM=LAM))
exec(compile(ast.Module(body=[node], type_ignores=[]), "S30", "exec"), mod.__dict__)
e0 = mod.exact(0.2)
assert abs(e0["k1"] / e0["a2"] - 0.784) < 5e-4 and abs(e0["band"] - 0.189) < 5e-4 and abs(e0["a3"] - 0.416) < 5e-4
labels = {0.21: "P-ext", 0.232: "band", 0.316: "P-per"}
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["label", "k1_over_a2", "R0"])
    for b in (0.05, 0.065, 0.08, 0.095, 0.11, 0.14, 0.17, 0.20, 0.21, 0.232, 0.316):
        e = mod.exact(b); w.writerow([labels.get(b, ""), "%.6f" % (e["k1"] / e["a2"]), "%.6f" % e["R0"]])
print(OUT)
