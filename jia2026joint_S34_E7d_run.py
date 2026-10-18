#!/usr/bin/env python3
"""Session 34, WU-N2b2b1: the E7 runs for the downward driver TS1d (ruling S34-2) in the two E7 regimes, S28-2 point.
Gate: E7d_validation.json has all_pass, its drivers_sha256 equals the current library and its S33 validation hash equals
the current S33 E7_validation.json. Regimes, dt, paths and levels are read from the S33 E7_params.json, never retyped;
seeds are seed0 + 4 under the E7 rule (33105 band, 33205 P-per). Writes E7_<regime>_TS1d.npz and .json and
params/E7d_params.json in ../numerics/S34_WUN2b2b1.
usage: python3 -B jia2026joint_S34_E7d_run.py   (from work/scripts)
"""
import hashlib, json, os, time
import numpy as np
import SIRTS1_ref_thresholds as T
import jia2026joint_S33_E7_drivers as D

HERE = os.path.dirname(os.path.abspath(__file__)); S33 = os.path.join(HERE, "..", "numerics", "S33_WUN2b2a")
OUT = os.path.join(HERE, "..", "numerics", "S34_WUN2b2b1")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
Vd = json.load(open(os.path.join(OUT, "E7d_validation.json"))); md = Vd["meta"]
assert Vd["all_pass"] and md["drivers_sha256"] == sha(os.path.join(HERE, "jia2026joint_S33_E7_drivers.py")) and \
    md["s33_validation_sha256"] == sha(os.path.join(S33, "E7_validation.json")), "validation gate"
PR = json.load(open(os.path.join(S33, "params", "E7_params.json")))
LAM, MU, EPS, G1, PHI = md["Lam"], md["mu"], md["eps"], tuple(md["g1"]), np.array(md["Phi"])
assert PR["law"] == md["law"] and np.array_equal(np.array(PR["Phi"]), PHI) and (PR["Lam"], PR["mu"], PR["eps"], list(PR["g1"])) == (LAM, MU, EPS, list(G1))
P = T.setup(**md["law"]); B = D.build(LAM, MU, EPS, G1, PHI, P, md["ts1d_dir"])
assert np.allclose(B["mu_x"]["TS1"], Vd["constants"]["mu_x"]["TS1d"], rtol=0, atol=0), "TS1d drift constants differ from the validated set"
REG, DT, NPATH, LEVELS, IDX = PR["regimes"], PR["dt"], PR["npath"], tuple(PR["levels"]), 4
os.makedirs(os.path.join(OUT, "params"), exist_ok=True)
json.dump(dict(regimes=REG, dt=DT, npath=NPATH, levels=LEVELS, driver="TS1d", ts1d_dir=md["ts1d_dir"], seed_rule="seed0 + 4",
               seeds={r: REG[r]["seed0"] + IDX for r in REG}, Lam=LAM, mu=MU, eps=EPS, g1=list(G1), Phi=PHI.tolist(), law=md["law"],
               validation_sha256=sha(os.path.join(OUT, "E7d_validation.json")), s33_params_sha256=sha(os.path.join(S33, "params", "E7_params.json"))),
          open(os.path.join(OUT, "params", "E7d_params.json"), "w"), indent=1)
for r, g in REG.items():
    seed = g["seed0"] + IDX; D.SIM.rng = np.random.default_rng(seed); t0 = time.time()
    o = D.run(g["beta"], g["T"], DT, NPATH, tuple(g["x0"]), B["jumps"]["TS1"], B["mu_x"]["TS1"], LAM, PHI,
              burn=g["burn"], rec_dt=g["rec_dt"], levels=LEVELS)
    rt, ry = o["rec_t"], o["rec_lnI"]
    q = np.percentile(ry, [5, 25, 50, 75, 95], axis=1)
    np.savez_compressed(os.path.join(OUT, f"E7_{r}_TS1d.npz"), rec_t=rt, q_lnI=q, frac_below=np.array([(ry < np.log(L)).mean(1) for L in LEVELS]),
                        lnImin=o["lnImin"], hit=np.array([o["hit"][L] for L in LEVELS]), avg=o["avg"], Yend=o["Yend"],
                        lnI_at=np.array([ry[np.argmin(np.abs(rt - tt))] for tt in (20.0, 40.0, 60.0, 80.0, 100.0) if tt <= g["T"] + 1e-9]),
                        pooled=ry[rt > (g["burn"] or 0) + 1e-9].ravel() if g["burn"] else np.array([]))
    info = dict(regime=r, driver="TS1d", beta=g["beta"], seed=seed, T=g["T"], dt=DT, npath=NPATH, nstep=o["nstep"],
                finite=bool(np.isfinite(o["Yend"]).all()), secs=round(time.time() - t0, 1),
                utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    json.dump(info, open(os.path.join(OUT, f"E7_{r}_TS1d.json"), "w"), indent=1); print(info, flush=True)
print("done", flush=True)
