#!/usr/bin/env python3
"""Session 33, WU-N2b2a: the E7 runs, four matched drivers in two regimes at the S28-2 point.
Gate: runs only if E7_validation.json has all_pass and its drivers_sha256 equals the current driver file (each driver
function validated before use). Regimes: band case beta 0.232 from the S30 band-case start x0 = (5.551, 0.5, 0.1),
horizon 80; P-per beta 0.316 from the TS6 stationary means, horizon 100, time averages over t > 50 (burn-in, S31-4).
dt 0.005 (the E3 choice), 1000 paths, seeds in params/E7_params.json. Writes E7_<regime>_<driver>.npz and .json.
usage: python3 -B jia2026joint_S33_E7_run.py [driver ...]   (from work/scripts; default: all four drivers)
"""
import hashlib, json, os, sys, time
import numpy as np
import SIRTS1_ref_thresholds as T
import jia2026joint_S33_E7_drivers as D

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "numerics", "S33_WUN2b2a")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
V = json.load(open(os.path.join(OUT, "E7_validation.json")))
assert V["all_pass"] and V["meta"]["drivers_sha256"] == sha(os.path.join(HERE, "jia2026joint_S33_E7_drivers.py")), "validation gate"
LAM, MU, EPS, G1 = 8.0, 0.75, 0.5, (-0.3, 0.5, 0.2)
PHI = np.array([[3, 2, 1], [2, 3, 2], [1, 2, 4]]) * 1e-3
LAW = dict(alpha=0.4, b=1.0, zeta=(0.01, 0.3, 0.01), n=6, om=0.5, c=(1, 5, 5, 5, 1, 1))
P = T.setup(**LAW)
assert V["meta"]["law"] == json.loads(json.dumps(LAW)) and np.allclose(V["meta"]["Phi"], PHI), "parameters differ from the validated set"
B = D.build(LAM, MU, EPS, G1, PHI, P, V["meta"]["ts1_dir"])
th316 = T.thresholds(0.316, Lam=LAM, mu=MU, eps=EPS, g1=G1, Phi=PHI, P=P, jumps=True)
REG = dict(band=dict(beta=0.232, x0=[5.551, 0.5, 0.1], T=80.0, burn=None, rec_dt=0.25, seed0=33101),
           per=dict(beta=0.316, x0=[th316["Sbar"], th316["Ibar"], th316["Rbar"]], T=100.0, burn=50.0, rec_dt=0.5, seed0=33201))
DT, NPATH, LEVELS = 0.005, 1000, (1e-2, 1e-3, 1e-4)
os.makedirs(os.path.join(OUT, "params"), exist_ok=True)
json.dump(dict(regimes=REG, dt=DT, npath=NPATH, levels=LEVELS, drivers=D.DRIVERS, Lam=LAM, mu=MU, eps=EPS, g1=G1,
               Phi=PHI.tolist(), law=LAW, ts1_dir=V["meta"]["ts1_dir"],
               seeds={r: {X: REG[r]["seed0"] + i for i, X in enumerate(D.DRIVERS)} for r in REG},
               validation_sha256=sha(os.path.join(OUT, "E7_validation.json"))),
          open(os.path.join(OUT, "params", "E7_params.json"), "w"), indent=1)
todo = sys.argv[1:] or list(D.DRIVERS)
for r, g in REG.items():
    for i, X in enumerate(D.DRIVERS):
        if X not in todo: continue
        seed = g["seed0"] + i; D.SIM.rng = np.random.default_rng(seed); t0 = time.time()
        o = D.run(g["beta"], g["T"], DT, NPATH, tuple(g["x0"]), B["jumps"][X], B["mu_x"][X], LAM, PHI,
                  burn=g["burn"], rec_dt=g["rec_dt"], levels=LEVELS)
        rt, ry = o["rec_t"], o["rec_lnI"]
        q = np.percentile(ry, [5, 25, 50, 75, 95], axis=1)
        np.savez_compressed(os.path.join(OUT, f"E7_{r}_{X}.npz"), rec_t=rt, q_lnI=q, frac_below=np.array([(ry < np.log(L)).mean(1) for L in LEVELS]),
                            lnImin=o["lnImin"], hit=np.array([o["hit"][L] for L in LEVELS]), avg=o["avg"], Yend=o["Yend"],
                            lnI_at=np.array([ry[np.argmin(np.abs(rt - tt))] for tt in (20.0, 40.0, 60.0, 80.0, 100.0) if tt <= g["T"] + 1e-9]),
                            pooled=ry[rt > (g["burn"] or 0) + 1e-9].ravel() if g["burn"] else np.array([]))
        info = dict(regime=r, driver=X, beta=g["beta"], seed=seed, T=g["T"], dt=DT, npath=NPATH, nstep=o["nstep"],
                    finite=bool(np.isfinite(o["Yend"]).all()), secs=round(time.time() - t0, 1),
                    utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        json.dump(info, open(os.path.join(OUT, f"E7_{r}_{X}.json"), "w"), indent=1); print(info, flush=True)
print("done", flush=True)
