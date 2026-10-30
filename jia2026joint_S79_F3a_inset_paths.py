#!/usr/bin/env python3
"""Session 79, WU7c2 (R-4: R-F1): data for the Figure 2 (fig:regimes) insets, one path at each of two ladder points.

The functions are loaded unchanged by AST extraction (rule R-8): pos_stable, ts_increment and simulate from
SIRTS1_ref_simulation.py, run from SIRTS1_ref_coupling.py, and crn_run and exact from jia2026joint_S30_WUN2a.py,
the function validated bit for bit in Session 30 (Part V). Every argument is passed explicitly; the parameters
and the ladder points are those of Section 5.2 (S30 Part 4).
Part 0: the S28-2 targets and the exact rates of the two points are asserted against the stored Part 4 rows.
Part V: crn_run against simulate (V1) and run (V2), repeated as in Session 30 (bitwise).
Part R: the two stored Part 4 rows (1000 paths, dt 0.005 and 0.01, seeds 30201 and 30209) reproduced exactly.
Part P: one path at each point: dt 0.005, the Part 4 horizon and start, seeds 30501 and 30502, 8 paths per run;
path 0 is shown, with no selection. ln(S_hat - S) = ln S_hat + ln(1 - exp(-D)) from the log-difference D.
usage: python3 -B jia2026joint_S79_F3a_inset_paths.py [out.json]   (run from work/scripts; about a minute)
"""
import ast, hashlib, json, os, sys, types
import numpy as np
from math import gamma as G
import SIRTS1_ref_thresholds as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data', 'jia2026joint_F3a_inset_paths_S79.json')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load(path, names, modname, extra):
    src = open(path).read(); mod = types.ModuleType(modname); mod.__dict__.update(extra); seg = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), mod.__dict__)
            seg[node.name] = hashlib.sha256(ast.get_source_segment(src, node).encode()).hexdigest()[:16]
    assert set(seg) == set(names), (path, seg)
    return mod, seg


LAM, EPS, MU, G1 = 8.0, 0.5, 0.75, (-0.3, 0.5, 0.2)
PHI = np.array([[3, 2, 1], [2, 3, 2], [1, 2, 4]]) * 1e-3
LAW = dict(alpha=0.4, b=1.0, zeta=(0.01, 0.3, 0.01), n=6, om=0.5, c=(1, 5, 5, 5, 1, 1))
P = T.setup(**LAW)
KW = dict(Lam=LAM, mu=MU, eps=EPS, g1=G1, Phi=PHI, P=P)
M = T.moments(P)
SIM, segS = load(os.path.join(HERE, 'SIRTS1_ref_simulation.py'), ['pos_stable', 'ts_increment', 'simulate'],
                 'SIM', dict(np=np, G=G, rng=np.random.default_rng(0)))
CPL, segC = load(os.path.join(HERE, 'SIRTS1_ref_coupling.py'), ['run'], 'CPL',
                 dict(np=np, SIM=SIM, P=P, Phi=PHI, Lam=LAM, mu=MU, eps=EPS, g1=G1, rng=None))
W, segW = load(os.path.join(HERE, 'jia2026joint_S30_WUN2a.py'), ['crn_run', 'exact'], 'W',
               dict(np=np, SIM=SIM, T=T, KW=KW, PHI=PHI, M=M, LAM=LAM))
R = dict(meta=dict(script=os.path.basename(__file__), parameters=dict(Lam=LAM, eps=EPS, mu=MU, gamma1=G1,
         Phi=PHI.tolist(), law=LAW), source_sha256={f: sha(os.path.join(HERE, f)) for f in (
         'SIRTS1_ref_thresholds.py', 'SIRTS1_ref_simulation.py', 'SIRTS1_ref_coupling.py', 'jia2026joint_S30_WUN2a.py')},
         function_source_sha16={**segS, **segC, **segW}))
S30 = json.load(open(os.path.join(HERE, '..', 'numerics', 'S30_WUN2a', 'jia2026joint_S30_WUN2a_results.json')))
assert R['meta']['function_source_sha16']['crn_run'] == S30['meta']['ref_function_source_sha16'].get('crn_run', R['meta']['function_source_sha16']['crn_run'])
rows = {r['beta']: r for r in S30['part4']['rows']}

# Part 0
e0 = W.exact(0.2); a2 = e0['a2']
assert abs(e0['k1'] / a2 - 0.784) < 5e-4 and abs(e0['band'] - 0.189) < 5e-4
PTS = [('low', 0.05), ('P-ext', 0.21)]
for _, b in PTS:
    e, r = W.exact(b), rows[b]
    for k in ('R0', 'lam', 'k1', 'S_rate'):
        assert e[k] == r[k], (b, k, e[k], r[k])

# Part V (Session 30, repeated)
eP = W.exact(0.316); x0P = (eP['Sbar'], eP['Ibar'], eP['Rbar'])
SIM.rng = np.random.default_rng(30001)
avg_s, lnI_s = SIM.simulate(0.316, T=2.0, dt=0.0025, npath=200, P=P, Lam=LAM, mu=MU, eps=EPS, g1=G1, Phi=PHI, x0=x0P)
o = W.crn_run(0.316, 2.0, 0.0025, (1,), 200, x0P, 30001, P, LAM, MU, EPS, G1, PHI, couple=False, fit_from=1.0, keep=True)
v1 = bool(np.array_equal(o[1]['avg'], avg_s) and np.array_equal(o['paths']['lnI'], lnI_s))
eV = W.exact(0.08); x0V = (round(0.8 * eV['S_ext'], 3), 0.5, 0.1)
tV, _, gapV, _, _ = CPL.run(T=30.0, dt=0.005, npath=100, beta=0.08, x0=x0V, seed=30002)
o = W.crn_run(0.08, 30.0, 0.005, (1,), 100, x0V, 30002, P, LAM, MU, EPS, G1, PHI, couple=True, fit_from=10.0, keep=True)
v2 = bool(np.array_equal(o['paths']['gap'], gapV))
assert v1 and v2, (v1, v2)
R['partV'] = dict(V1_crn_run_vs_simulate_bitwise=v1, V2_crn_run_vs_run_gap_bitwise=v2)

# Part R: the stored Part 4 rows reproduced
R['partR'] = {}
for _, b in PTS:
    r = rows[b]
    o = W.crn_run(b, r['T'], 0.005, (1, 2), 1000, tuple(r['x0']), r['seed'], P, LAM, MU, EPS, G1, PHI,
                  couple=True, fit_from=r['T'] / 3, keep=False)
    got = {nm: float(o[1]['slopes'][:, c].mean()) for c, nm in enumerate('IRS')}
    want = {nm: r['levels']['0.005'][nm]['mean'] for nm in 'IRS'}
    assert got == want, (b, got, want)
    R['partR'][str(b)] = dict(seed=r['seed'], T=r['T'], mean_slopes=got, equal_to_stored=True)

# Part P: one path per point
R['paths'] = {}
for i, (lab, b) in enumerate(PTS):
    r, e = rows[b], W.exact(b); sd = 30501 + i; Tb = r['T']
    o = W.crn_run(b, Tb, 0.005, (1,), 8, tuple(r['x0']), sd, P, LAM, MU, EPS, G1, PHI, couple=True, fit_from=Tb / 3, keep=True)
    pk = o['paths']; n = pk['lnI'].shape[0]; t = np.arange(n) * 0.005
    lnI = pk['lnI'][:, 0]; D = pk['D'][:, 0]
    assert (D[1:] > 0).all()
    lngap = np.full(n, np.nan); lngap[1:] = pk['Yh'][1:, 0] + np.log(-np.expm1(-D[1:]))
    w = t >= Tb / 3
    fit = {nm: float(np.polyfit(t[w], y[w], 1)[0]) for nm, y in (('lnI', lnI), ('lngap', lngap))}
    mid = {nm: [float(t[w].mean()), float(y[w].mean())] for nm, y in (('lnI', lnI), ('lngap', lngap))}
    st = 100                                    # store every 0.5 time units
    R['paths'][lab] = dict(beta=b, seed=sd, npath=8, shown='path 0, no selection', T=Tb, dt=0.005, fit_window=[Tb / 3, Tb],
                           x0=r['x0'], R0=e['R0'], k1_over_a2=e['k1'] / e['a2'], exact=dict(lnI=e['lam'], lngap=e['S_rate']),
                           fit=fit, fit_rel_err={k: (fit[k] - v) / abs(v) for k, v in dict(lnI=e['lam'], lngap=e['S_rate']).items()},
                           crn_run_slopes_path0=dict(I=float(o[1]['slopes'][0, 0]), D=float(o[1]['slopes'][0, 2])),
                           window_mean_point=mid, t=t[::st].tolist(), lnI=lnI[::st].tolist(), lngap=lngap[::st].tolist())
json.dump(R, open(OUT, 'w'), indent=1)
for lab, p in R['paths'].items():
    print(lab, p['beta'], 'R0 %.4f' % p['R0'], 'exact', p['exact'], 'fit', {k: round(v, 4) for k, v in p['fit'].items()},
          'rel', {k: round(v, 4) for k, v in p['fit_rel_err'].items()})
print('V', R['partV'], 'R', {k: v['equal_to_stored'] for k, v in R['partR'].items()}, OUT)
