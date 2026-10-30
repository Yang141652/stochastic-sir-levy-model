#!/usr/bin/env python3
"""Session 30, WU-N2 part a: common-random-number dt ladder (E3, F-S29-2) and the S-return rate by
log-difference coupling (F-S29-3), at the S28-2 design point (D-34), the Section 5 baseline (ruling S29-2).

New function crn_run(). It draws the fine increments exactly as simulate() and run() do (one generator, the
same order: one Gaussian block, then ts_increment for each jump direction) and advances every requested level
m with the sum of m consecutive fine increments, so all levels are driven by the same noise. Brownian sums are
exact; a sum of independent exact increments of the tempered stable subordinator is an exact increment over
the longer step (infinite divisibility). With levels (1,) the function must reproduce simulate() and run()
bit for bit at the same seed (Part V).

For the coupled pair (S, S_hat) it also iterates the log-difference D = ln S_hat - ln S by the scheme's own
recursion
    D <- D + (-Lam exp(-ln S_hat) expm1(D) + beta I) h.
For the log-coordinate scheme this is exact algebra: the shared noise and the -m1 - Phi11/2 terms cancel. It
keeps relative precision as D -> 0, so the S rate can be fitted on long windows; the 1e-11 gap floor of
S29 Part 4b no longer applies. The per-path slopes of ln I and ln R come from the same paths.

Rule R-8 and F-S28-4: the reference functions are loaded unchanged by AST extraction (no module self-checks
run), every argument is passed explicitly, and the S28-2 targets are asserted first.
Parts: 0 assertions; V validation (V1 against simulate, V2 against run); 3 CRN dt ladder at P-per (E3);
4 beta ladder: pathwise rates of I, R and D at dt 0.005 and 0.01 (common random numbers).
usage: python3 -B jia2026joint_S30_WUN2a.py [outdir]   (run from work/scripts; about 8 minutes, one core)
"""
import ast, hashlib, json, os, sys, time, types
import numpy as np
from math import gamma as G
import SIRTS1_ref_thresholds as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'numerics', 'S30_WUN2a')
os.makedirs(OUT, exist_ok=True)
R = {'meta': {'script': os.path.basename(__file__), 'started_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}}
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
log = lambda s: print(time.strftime('%H:%M:%S ', time.gmtime()) + s, flush=True)


def load(path, names, modname, extra):
    src = open(path).read(); mod = types.ModuleType(modname); mod.__dict__.update(extra); seg = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), mod.__dict__)
            seg[node.name] = hashlib.sha256(ast.get_source_segment(src, node).encode()).hexdigest()[:16]
    assert set(seg) == set(names), (path, seg)
    return mod, seg


# ---------- parameters (every argument explicit; identical to the S29 driver) ----------
LAM, EPS, MU, G1 = 8.0, 0.5, 0.75, (-0.3, 0.5, 0.2)
PHI = np.array([[3, 2, 1], [2, 3, 2], [1, 2, 4]]) * 1e-3
LAW = dict(alpha=0.4, b=1.0, zeta=(0.01, 0.3, 0.01), n=6, om=0.5, c=(1, 5, 5, 5, 1, 1))
P = T.setup(**LAW)
KW = dict(Lam=LAM, mu=MU, eps=EPS, g1=G1, Phi=PHI, P=P)
SIM, segS = load(os.path.join(HERE, 'SIRTS1_ref_simulation.py'), ['pos_stable', 'ts_increment', 'simulate'],
                 'SIM', dict(np=np, G=G, rng=np.random.default_rng(0)))
CPL, segC = load(os.path.join(HERE, 'SIRTS1_ref_coupling.py'), ['run'], 'CPL',
                 dict(np=np, SIM=SIM, P=P, Phi=PHI, Lam=LAM, mu=MU, eps=EPS, g1=G1, rng=None))
R['meta']['ref_sha256'] = {f: sha(os.path.join(HERE, f)) for f in
                           ('SIRTS1_ref_thresholds.py', 'SIRTS1_ref_simulation.py', 'SIRTS1_ref_coupling.py')}
R['meta']['ref_function_source_sha16'] = {**segS, **segC}
R['meta']['parameters'] = dict(Lam=LAM, eps=EPS, mu=MU, gamma1=G1, Phi=PHI.tolist(), law=LAW)

# ---------- Part 0: assertions (as S29) ----------
Pb = T.setup(alpha=0.4, b=1.0, zeta=(0.1, 0.1, 0.1), n=6, om=0.5, c=(1, 5, 5, 5, 1, 1))
tb = T.thresholds(4.8, Lam=8.0, mu=5.3, eps=0.5, g1=(-0.3, 0.5, 0.2), Phi=PHI, P=Pb)
assert max(abs(tb['a1'] - 6.6023), abs(tb['a2'] - 6.2698), abs(tb['a3'] - 3.7391), abs(tb['R0'] - 0.9226)) < 5e-5
M = T.moments(P)


def exact(beta):
    th = T.thresholds(beta, **KW)
    k = [th['a%d' % (i + 1)] + 0.5 * PHI[i, i] + M[i]['pen'] for i in range(3)]
    nz = 0.5 * PHI[1, 1] + M[1]['pen']
    return dict(beta=beta, a1=th['a1'], a2=th['a2'], a3=th['a3'], k1=k[0], k2=k[1], k3=k[2], R0=th['R0'],
                R0det=beta * LAM / (th['a1'] * th['a2']), band=nz / th['a2'], jump_share=M[1]['pen'] / nz,
                lam=th['rate'], S_rate=max(-k[0], th['rate']), R_rate=max(-k[2], th['rate']),
                Sbar=th['Sbar'], Ibar=th['Ibar'], Rbar=th['Rbar'], S_ext=th['S_ext'])


e0 = exact(0.2)
a1, a2, nz = e0['a1'], e0['a2'], e0['band'] * e0['a2']
b_det1, b_R1 = a1 * a2 / LAM, a1 * (a2 + nz) / LAM
b_clock = a1 * (a2 - e0['k1'] + nz) / LAM
tgt = dict(k1_a2=(e0['k1'] / a2, 0.784), band=(e0['band'], 0.189), a3=(e0['a3'], 0.416), b_det1=(b_det1, 0.212),
           b_R1=(b_R1, 0.252), b_clock=(b_clock, 0.086))
for kname, (v, t) in tgt.items():
    assert abs(v - t) < 5e-4, (kname, v, t)
assert e0['jump_share'] > 0.985
beta_for = lambda r0: (r0 + nz / a2) * a1 * a2 / LAM
B_EXT, B_PER, B_BAND = round(beta_for(0.8), 3), round(beta_for(1.3), 3), round(0.5 * (b_det1 + b_R1), 3)
assert (B_EXT, B_PER, B_BAND) == (0.21, 0.316, 0.232), (B_EXT, B_PER, B_BAND)
LADDER = [0.05, 0.065, 0.08, 0.095, 0.11, 0.14, 0.17, 0.20, B_EXT, B_BAND]
b_clock_R = a1 * (a2 - e0['k3'] + nz) / LAM          # beta where lambda = -k3 (R crossover)
R['part0'] = dict(text_baseline=dict(a1=tb['a1'], a2=tb['a2'], a3=tb['a3'], R0=tb['R0']),
                  s28_2_targets={k: v[0] for k, v in tgt.items()}, jump_share=e0['jump_share'],
                  betas=dict(ext=B_EXT, per=B_PER, band=B_BAND, ladder=LADDER, clock_S=b_clock, clock_R=b_clock_R))
log('Part 0 assertions pass: S28-2 targets reproduce; beta_clock(S) %.4f, beta_clock(R) %.4f' % (b_clock, b_clock_R))


# ---------- new function ----------
def crn_run(beta, T_end, dt, levels, npath, x0, seed, P, Lam, mu, eps, g1, Phi, couple, fit_from, keep):
    """Levels m in `levels` advance with step m*dt driven by sums of the same fine increments.
    Returns {m: dict(avg, slopes, n_fit, d_nonpos)} and, if keep (levels == (1,)), the level-1 paths."""
    SIM.rng = np.random.default_rng(seed)
    Gm = np.linalg.cholesky(Phi); m1, m2, m3 = mu - g1[0], mu + eps + 1 - g1[1], mu - g1[2]
    nfine = int(T_end / dt); assert all(nfine % m == 0 for m in levels), (nfine, levels)
    st = {m: dict(Y=np.tile(np.log(x0), (npath, 1)), Yh=np.full(npath, np.log(x0[0])), D=np.zeros(npath),
                  aE=np.zeros((npath, 3)), aL=np.zeros((npath, 3)), acc=np.zeros((npath, 3)), n=0, st=0.0,
                  stt=0.0, sy=np.zeros((npath, 3)), sty=np.zeros((npath, 3)), nonpos=0) for m in levels}
    tmid = 0.5 * (fit_from + nfine * dt)
    if keep:
        assert tuple(levels) == (1,)
        pk = {k: np.empty((nfine + 1, npath)) for k in ('lnI', 'gap', 'D', 'Yh', 'Y0')}
        pk['lnI'][0] = st[1]['Y'][:, 1]; pk['gap'][0] = 0.0; pk['D'][0] = 0.0
        pk['Yh'][0] = st[1]['Yh']; pk['Y0'][0] = st[1]['Y'][:, 0]
    for k in range(nfine):
        dE = np.sqrt(dt) * SIM.rng.standard_normal((npath, 3)) @ Gm.T
        dL = np.zeros((npath, 3))
        for j in range(P['s'].shape[0]):
            dL += np.outer(SIM.ts_increment(P['alpha'], P['w'][j], P['b'], dt, npath), P['s'][j])
        for m in levels:
            s = st[m]; s['aE'] = s['aE'] + dE; s['aL'] = s['aL'] + dL
            if (k + 1) % m:
                continue
            h = m * dt; Y = s['Y']; aE, aL = s['aE'], s['aL']
            S, I, Rr = np.exp(Y[:, 0]), np.exp(Y[:, 1]), np.exp(Y[:, 2])
            drift = np.stack([Lam/S-m1-beta*I-Phi[0, 0]/2, beta*S-m2-Phi[1, 1]/2, I/Rr-m3-Phi[2, 2]/2], axis=1)
            if couple:
                Yh = s['Yh']
                s['D'] = s['D'] + (-Lam * np.exp(-Yh) * np.expm1(s['D']) + beta * I) * h
                s['Yh'] = Yh + (Lam/np.exp(Yh)-m1-Phi[0, 0]/2)*h + aE[:, 0] + P['zeta'][0]*aL[:, 0]
            s['Y'] = Y + drift*h + aE + P['zeta']*aL
            s['acc'] += np.exp(s['Y'])*h
            s['aE'] = np.zeros((npath, 3)); s['aL'] = np.zeros((npath, 3))
            if keep:
                pk['lnI'][k + 1] = s['Y'][:, 1]; pk['gap'][k + 1] = np.exp(s['Yh']) - np.exp(s['Y'][:, 0])
                pk['D'][k + 1] = s['D']; pk['Yh'][k + 1] = s['Yh']; pk['Y0'][k + 1] = s['Y'][:, 0]
            t = (k + 1) * dt
            if t >= fit_from:
                tc = t - tmid; s['nonpos'] += int((s['D'] <= 0).sum()) if couple else 0
                y = np.stack([s['Y'][:, 1], s['Y'][:, 2],
                              np.log(np.maximum(s['D'], 1e-300)) if couple else np.zeros(npath)], axis=1)
                s['n'] += 1; s['st'] += tc; s['stt'] += tc * tc; s['sy'] += y; s['sty'] += tc * y
    out = {}
    for m in levels:
        s = st[m]; n = s['n']
        sl = (s['sty'] - s['st'] * s['sy'] / n) / (s['stt'] - s['st'] ** 2 / n) if n > 2 else None
        out[m] = dict(avg=s['acc'] / T_end, slopes=sl, n_fit=n, d_nonpos=s['nonpos'])
    if keep:
        out['paths'] = pk
    return out


def ci(x):
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


# ---------- Part V: validation ----------
eP = exact(B_PER); x0P = (eP['Sbar'], eP['Ibar'], eP['Rbar'])
SIM.rng = np.random.default_rng(30001)
avg_s, lnI_s = SIM.simulate(B_PER, T=2.0, dt=0.0025, npath=200, P=P, Lam=LAM, mu=MU, eps=EPS, g1=G1, Phi=PHI, x0=x0P)
o = crn_run(B_PER, 2.0, 0.0025, (1,), 200, x0P, 30001, P, LAM, MU, EPS, G1, PHI, couple=False, fit_from=1.0, keep=True)
v1 = dict(beta=B_PER, T=2.0, dt=0.0025, npath=200, seed=30001, avg_bitwise=bool(np.array_equal(o[1]['avg'], avg_s)),
          lnI_bitwise=bool(np.array_equal(o['paths']['lnI'], lnI_s)),
          max_abs_avg=float(np.abs(o[1]['avg'] - avg_s).max()), max_abs_lnI=float(np.abs(o['paths']['lnI'] - lnI_s).max()))
assert v1['avg_bitwise'] and v1['lnI_bitwise'], v1
eV = exact(0.08); x0V = (round(0.8 * eV['S_ext'], 3), 0.5, 0.1)
tV, _, gapV, _, _ = CPL.run(T=30.0, dt=0.005, npath=100, beta=0.08, x0=x0V, seed=30002)
o = crn_run(0.08, 30.0, 0.005, (1,), 100, x0V, 30002, P, LAM, MU, EPS, G1, PHI, couple=True, fit_from=10.0, keep=True)
pk = o['paths']; Dd = pk['Yh'] - pk['Y0']; Dr = pk['D']
rel = lambda msk: float(np.abs(Dr[msk] / Dd[msk] - 1).max()) if msk.any() else None
grec = np.exp(pk['Yh']) * (-np.expm1(-Dr)); mg = gapV > 1e-9
v2 = dict(beta=0.08, T=30.0, dt=0.005, npath=100, seed=30002, x0=list(x0V),
          gap_bitwise=bool(np.array_equal(pk['gap'], gapV)), D_positive=bool((Dr[1:] > 0).all()),
          rel_D_diff_gt_1e4=rel(Dd > 1e-4), rel_D_diff_gt_1e6=rel(Dd > 1e-6),
          rel_gap_rec_gap_gt_1e9=float(np.abs(grec[mg] / gapV[mg] - 1).max()),
          frac_gap_lt_1e11_at_T=float((gapV[-1] < 1e-11).mean()), min_D_at_T=float(Dr[-1].min()),
          median_D_at_T=float(np.median(Dr[-1])))
assert v2['gap_bitwise'] and v2['D_positive'] and v2['rel_D_diff_gt_1e4'] < 1e-6, v2
R['partV'] = dict(V1_vs_simulate=v1, V2_vs_run=v2)
del o, pk, Dd, Dr, grec, gapV
log('Part V pass: V1 bitwise %s/%s; V2 gap bitwise %s, D>0 %s, rel(D) %.2e (diff>1e-4), %.2e (diff>1e-6)'
    % (v1['avg_bitwise'], v1['lnI_bitwise'], v2['gap_bitwise'], v2['D_positive'],
       v2['rel_D_diff_gt_1e4'], v2['rel_D_diff_gt_1e6'] or float('nan')))

# ---------- Part 3: CRN dt ladder at P-per (E3) ----------
t0 = time.time(); LEV3, DTF3, NP3, T3 = (1, 2, 4, 8, 16), 0.0025, 1000, 100.0
o = crn_run(B_PER, T3, DTF3, LEV3, NP3, x0P, 30101, P, LAM, MU, EPS, G1, PHI, couple=False, fit_from=1e9, keep=False)
ex3 = np.array(x0P); lv = []
for m in LEV3:
    a = o[m]['avg']; mean, se = a.mean(0), a.std(0, ddof=1) / np.sqrt(NP3)
    lv.append(dict(dt=m * DTF3, mean=mean.tolist(), se=se.tolist(), finite=bool(np.isfinite(a).all()),
                   z_vs_exact=((mean - ex3) / se).tolist()))
pairs = []
for q in range(len(LEV3) - 1):                       # coarse level LEV3[q+1] minus fine level LEV3[q]
    d = o[LEV3[q + 1]]['avg'] - o[LEV3[q]]['avg']; dm, ds = d.mean(0), d.std(0, ddof=1) / np.sqrt(NP3)
    pairs.append(dict(coarse=LEV3[q + 1] * DTF3, fine=LEV3[q] * DTF3, diff=dm.tolist(), se=ds.tolist(),
                      z=(dm / ds).tolist(), se_ratio_vs_independent=(ds / np.hypot(lv[q]['se'], lv[q + 1]['se'])).tolist()))
for q in range(1, len(pairs)):
    pairs[q]['order_ratio'] = (np.array(pairs[q]['diff']) / np.array(pairs[q - 1]['diff'])).tolist()
# bias(h) ~ m(2h) - m(h) under weak order one: pair q estimates the bias of its fine level
crit = [dict(dt=p['fine'], bias=p['diff'], bias_ci95=[abs(v) + 1.96 * s for v, s in zip(p['diff'], p['se'])],
             se_mc=lv[q]['se'], below=bool(all(abs(v) < s for v, s in zip(p['diff'], lv[q]['se']))),
             below_ci95=bool(all(abs(v) + 1.96 * e < s for v, e, s in zip(p['diff'], p['se'], lv[q]['se']))))
        for q, p in enumerate(pairs)]
ok = [c['dt'] for c in crit if c['below']]; ok95 = [c['dt'] for c in crit if c['below_ci95']]
R['part3'] = dict(beta=B_PER, T=T3, npath=NP3, dt_fine=DTF3, levels=[m * DTF3 for m in LEV3], seed=30101, x0=list(x0P),
                  exact_means=ex3.tolist(), levels_table=lv, pairs=pairs, criterion=crit,
                  largest_dt_bias_below_mc=max(ok) if ok else None, largest_dt_bias_ci95_below_mc=max(ok95) if ok95 else None,
                  secs=round(time.time() - t0, 1))
del o
log('Part 3 CRN ladder: largest dt with |bias| < MC s.e.: %s (95%% bound: %s); %.0f s'
    % (R['part3']['largest_dt_bias_below_mc'], R['part3']['largest_dt_bias_ci95_below_mc'], time.time() - t0))
json.dump(R, open(os.path.join(OUT, 'jia2026joint_S30_WUN2a_results.json'), 'w'), indent=1)

# ---------- Part 4: beta ladder, pathwise rates of I, R and D (CRN, dt 0.005 and 0.01) ----------
DT4, LEV4, NP4, rows = 0.005, (1, 2), 1000, []
for i, b in enumerate(LADDER):
    e = exact(b); lam, k1, k3 = e['lam'], e['k1'], e['k3']; sd = 30201 + i
    Tb = float(np.ceil(min(300.0, max(60.0, 15 / abs(lam), 12 / abs(lam + k1), 12 / abs(lam + k3)))))
    while int(Tb / DT4) % 2:
        Tb += 1.0
    x0 = (round(0.8 * e['S_ext'], 3), 0.5, 0.1); t0 = time.time()
    o = crn_run(b, Tb, DT4, LEV4, NP4, x0, sd, P, LAM, MU, EPS, G1, PHI, couple=True, fit_from=Tb / 3, keep=False)
    row = dict(beta=b, seed=sd, T=Tb, fit_window=[Tb / 3, Tb], x0=list(x0), R0=e['R0'], R0det=e['R0det'], lam=lam,
               k1=k1, k3=k3, S_rate=e['S_rate'], R_rate=e['R_rate'], levels={})
    for m in LEV4:
        sl = o[m]['slopes']; rr = {}
        for c, (name, ex) in enumerate((('I', lam), ('R', e['R_rate']), ('S', e['S_rate']))):
            mm, se = ci(sl[:, c]); rr[name] = dict(mean=mm, half95=1.96 * se, exact=ex, in_ci=bool(abs(mm - ex) <= 1.96 * se),
                                                   rel_err=(mm - ex) / abs(ex))
        rr['n_fit'] = o[m]['n_fit']; rr['D_nonpos'] = o[m]['d_nonpos']; row['levels'][str(m * DT4)] = rr
    dlv = o[2]['slopes'] - o[1]['slopes']
    row['dt_effect'] = {nm: dict(zip(('mean', 'se'), ci(dlv[:, c]))) for c, nm in enumerate('IRS')}
    row['secs'] = round(time.time() - t0, 1); rows.append(row); del o
    f = row['levels'][str(DT4)]
    log('beta %.3f T %g: I %.4f (%.4f) R %.4f (%.4f) S %.4f (%.4f) in CI %s/%s/%s; %.0f s' % (
        b, Tb, f['I']['mean'], lam, f['R']['mean'], e['R_rate'], f['S']['mean'], e['S_rate'],
        f['I']['in_ci'], f['R']['in_ci'], f['S']['in_ci'], row['secs']))
    R['part4'] = dict(dt_fine=DT4, levels=[m * DT4 for m in LEV4], npath=NP4,
                      method='per-path least-squares slope of ln I, ln R, ln D on [T/3, T]; D iterated by the log-difference recursion; '
                             'T = ceil(min(300, max(60, 15/|lambda|, 12/|lambda+k1|, 12/|lambda+k3|)))', rows=rows)
    json.dump(R, open(os.path.join(OUT, 'jia2026joint_S30_WUN2a_results.json'), 'w'), indent=1)

R['meta']['finished_utc'] = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
json.dump(R, open(os.path.join(OUT, 'jia2026joint_S30_WUN2a_results.json'), 'w'), indent=1)
json.dump(dict(V1=30001, V2=30002, part3=30101, part4={str(b): 30201 + i for i, b in enumerate(LADDER)},
               parameters=R['meta']['parameters']), open(os.path.join(OUT, 'jia2026joint_S30_WUN2a_params.json'), 'w'), indent=1)
log('done')
