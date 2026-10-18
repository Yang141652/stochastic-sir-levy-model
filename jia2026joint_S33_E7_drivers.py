#!/usr/bin/env python3
"""Session 33, WU-N2b2a (E7): comparison drivers at matched mean rates and matched log-scale covariance.

Matching convention (SI section C.8, cor:C-gauss and cor:C-order; ruling S33-2). Every driver X shares Lam, beta,
Phi, the mean rates a_i = mu_i - int eta_i dnu and the log-scale jump covariance C = int l l^T dnu per unit time,
where l = ln(1 + eta) is the log-scale jump. A driver enters the model only through its log-jump law: its drift
constant is mu_i^X = mu_i + (J_i^X - J_i^TS6) with J_i^X = int (e^{l_i} - 1) dnu^X, so a_i = mu_i^X - J_i^X is
the same for every driver. The drivers then differ only in the shape of the log-jump law (cumulants of order 3
and higher), and the threshold moves only through the penalty Pen_2^X = J_2^X - int l_2 dnu^X:
  R0^X = (beta Lam / a1 - Phi_22 / 2 - Pen_2^X) / a2.
Drivers:
  G    Brownian: Gaussian log increments, mean 0, covariance C dt (J_i = C_ii / 2 = Pen_i).
  CP6  compound Poisson on the six directions s_j; exponential radial sizes, mean theta = (1 - alpha) / (2 b),
       rate lam_j = kappa1_j / theta, so the first two radial cumulants match the TS law on every direction and
       the log-scale mean and covariance match exactly (l is linear in the radial size).
  TS1  the one-sided TS law on the single direction s_2 (angle 2 pi / 3, the heaviest upward-I direction), weight
       chosen so that int l_2^2 dnu matches TS6. S and R jump variances are not matched (order 1e-4; they enter
       k1 and k3 only). Every jump of I is upward.
  TS6  the S28-2 law: setup(alpha=0.4, b=1.0, zeta=(0.01, 0.3, 0.01), n=6, om=0.5, c=(1, 5, 5, 5, 1, 1)).
Every function takes every argument explicitly (F-S28-4). The TS sampler is the reference ts_increment, loaded by
source from SIRTS1_ref_simulation.py (whose module-level self-checks are not run); TS6 through run() reproduces
the reference simulate() bitwise (validation T1).
usage: imported by jia2026joint_S33_E7_validate.py, _run.py and _summary.py (run with python3 -B from work/scripts)
"""
import ast, hashlib, os, types
import numpy as np
from math import gamma as G
import SIRTS1_ref_thresholds as T

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVERS = ("G", "CP6", "TS1", "TS6")


def load(path, names, modname, extra):
    """Load named top-level functions from a reference file without running its module-level code (S29 pattern)."""
    src = open(path).read(); mod = types.ModuleType(modname); mod.__dict__.update(extra); seg = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), mod.__dict__)
            seg[node.name] = hashlib.sha256(ast.get_source_segment(src, node).encode()).hexdigest()[:16]
    assert set(seg) == set(names), (path, seg)
    return mod, seg


SIM, SEG = load(os.path.join(HERE, "SIRTS1_ref_simulation.py"), ["pos_stable", "ts_increment", "simulate"],
                "SIM", dict(np=np, G=G, rng=np.random.default_rng(0)))


def rates(Lam, mu, eps, g1, Phi, P):
    """Drift constants mu_i of the S28-2 law and mean rates a_i = mu_i - int eta_i dnu (reference thresholds())."""
    th = T.thresholds(1.0, Lam=Lam, mu=mu, eps=eps, g1=g1, Phi=Phi, P=P, jumps=True)
    return np.array([th["mu1"], th["mu2"], th["mu3"]]), np.array([th["a1"], th["a2"], th["a3"]])


def logcov_ts(P):
    """C_ik = zeta_i zeta_k sum_j w_j s_ji s_jk Gamma(2 - alpha) b^(alpha - 2): log-scale jump covariance per unit time."""
    a, b, s, w, z = P["alpha"], P["b"], P["s"], P["w"], P["zeta"]
    return np.outer(z, z) * np.einsum("j,ji,jk->ik", w * G(2 - a) * b ** (a - 2), s, s)


def law_ts(P):
    """Closed forms of a one-sided TS law (directions P['s'], weights P['w']) from the reference moments()."""
    M = T.moments(P)
    J = np.array([M[i]["int_eta"] for i in range(3)]); Lm = np.array([M[i]["int_log"] for i in range(3)])
    return dict(J=J, Lmean=Lm, C=logcov_ts(P), Pen=np.array([M[i]["pen"] for i in range(3)]))


def law_ts1(P, j):
    """All TS mass on direction s_j, weight matching int l_2^2 dnu of P. Returns (P1, closed forms)."""
    s1 = P["s"][j:j + 1]
    wbar = float(np.sum(P["w"] * P["s"][:, 1] ** 2) / s1[0, 1] ** 2)
    P1 = dict(alpha=P["alpha"], b=P["b"], zeta=P["zeta"], s=s1, w=np.array([wbar]))
    return P1, law_ts(P1)


def law_cp6(P):
    """Compound Poisson on the directions of P, exponential radial sizes matching kappa1 and kappa2 per direction."""
    a, b, s, w, z = P["alpha"], P["b"], P["s"], P["w"], P["zeta"]
    k1 = w * G(1 - a) * b ** (a - 1); k2 = w * G(2 - a) * b ** (a - 2)
    theta = k2 / (2 * k1); lam = k1 / theta
    c = z[None, :] * s * theta[:, None]                      # zeta_i s_ji theta_j
    assert np.all(c < 1), "exponential moment diverges"
    J = np.sum(lam[:, None] * (1 / (1 - c) - 1), axis=0)     # E e^{zeta_i s_ji R} = 1 / (1 - zeta_i s_ji theta)
    Lm = np.sum(lam[:, None] * c, axis=0)
    C = np.outer(z, z) * np.einsum("j,ji,jk->ik", lam * 2 * theta ** 2, s, s)
    return dict(theta=theta, lam=lam, s=s, zeta=z), dict(J=J, Lmean=Lm, C=C, Pen=J - Lm)


def law_g(C):
    """Brownian log increments with mean 0 and covariance C per unit time."""
    return dict(J=np.diag(C) / 2, Lmean=np.zeros(3), C=np.array(C, dtype=float), Pen=np.diag(C) / 2)


def build(Lam, mu, eps, g1, Phi, P, ts1_dir):
    """Closed forms, drift constants and samplers for the four drivers, every argument explicit."""
    mu0, a = rates(Lam, mu, eps, g1, Phi, P)
    L6 = law_ts(P); P1, L1 = law_ts1(P, ts1_dir); Pc, Lc = law_cp6(P); Lg = law_g(L6["C"])
    laws = dict(G=Lg, CP6=Lc, TS1=L1, TS6=L6)
    mu_x = {X: mu0 + (laws[X]["J"] - L6["J"]) for X in DRIVERS}
    Cg = np.linalg.cholesky(Lg["C"])

    def jump_ts6(dt, npath):                                 # the reference loop, same order of operations
        dL = np.zeros((npath, 3))
        for j in range(P["s"].shape[0]):
            dL += np.outer(SIM.ts_increment(P["alpha"], P["w"][j], P["b"], dt, npath), P["s"][j])
        return P["zeta"] * dL

    def jump_ts1(dt, npath):
        dL = np.outer(SIM.ts_increment(P1["alpha"], P1["w"][0], P1["b"], dt, npath), P1["s"][0])
        return P1["zeta"] * dL

    def jump_cp6(dt, npath):
        dL = np.zeros((npath, 3))
        for j in range(Pc["s"].shape[0]):
            N = SIM.rng.poisson(Pc["lam"][j] * dt, npath)
            Rr = np.zeros(npath); m = N > 0
            Rr[m] = SIM.rng.gamma(N[m], Pc["theta"][j])    # sum of N exponential sizes, exact
            dL += np.outer(Rr, Pc["s"][j])
        return Pc["zeta"] * dL

    def jump_g(dt, npath):
        return np.sqrt(dt) * SIM.rng.standard_normal((npath, 3)) @ Cg.T

    jumps = dict(G=jump_g, CP6=jump_cp6, TS1=jump_ts1, TS6=jump_ts6)
    return dict(a=a, mu0=mu0, laws=laws, mu_x=mu_x, jumps=jumps, P1=P1, Pc=Pc)


def summary(law, beta, Lam, a, Phi):
    """Threshold quantities of one driver: R0, lambda, k_i, and the stationary means when R0 > 1."""
    a1, a2, a3 = a; Pn = law["Pen"]
    R0 = (beta * Lam / a1 - 0.5 * Phi[1, 1] - Pn[1]) / a2
    k = a + 0.5 * np.diag(Phi) + Pn
    return dict(R0=float(R0), lam=float(a2 * (R0 - 1)), k1=float(k[0]), k2=float(k[1]), k3=float(k[2]),
                Sbar=float(k[1] / beta), Ibar=float(a1 * (R0 - 1) / beta), Rbar=float(a1 * (R0 - 1) / (beta * a3)))


def run(beta, T_end, dt, npath, x0, jump, mu_x, Lam, Phi, burn=None, rec_dt=None, levels=(), keep_lnI=False):
    """Log-coordinate scheme of the reference simulate(), with a driver's jump sampler and drift constants.
    Without burn, avg = acc / T_end as in the reference; with burn, the average over t > burn."""
    Gm = np.linalg.cholesky(Phi); m1, m2, m3 = mu_x
    Y = np.tile(np.log(x0), (npath, 1)); nstep = int(T_end / dt)
    acc = np.zeros((npath, 3)); nacc = 0
    lnImin = Y[:, 1].copy(); hit = {L: np.full(npath, np.inf) for L in levels}
    rec_every = None if rec_dt is None else int(round(rec_dt / dt))
    rec_t, rec_y = [0.0], [Y[:, 1].copy()]
    lnI = np.empty((nstep + 1, npath)) if keep_lnI else None
    if keep_lnI: lnI[0] = Y[:, 1]
    for k in range(nstep):
        dE = np.sqrt(dt) * SIM.rng.standard_normal((npath, 3)) @ Gm.T
        dJ = jump(dt, npath)
        S, I, R = np.exp(Y[:, 0]), np.exp(Y[:, 1]), np.exp(Y[:, 2])
        drift = np.stack([Lam / S - m1 - beta * I - Phi[0, 0] / 2, beta * S - m2 - Phi[1, 1] / 2,
                          I / R - m3 - Phi[2, 2] / 2], axis=1)
        Y = Y + drift * dt + dE + dJ
        t = (k + 1) * dt
        if burn is None or t > burn + 1e-9:
            acc += np.exp(Y) * dt; nacc += 1
        if keep_lnI: lnI[k + 1] = Y[:, 1]
        np.minimum(lnImin, Y[:, 1], out=lnImin)
        for L in levels:
            h = hit[L]; new = np.isinf(h) & (Y[:, 1] < np.log(L)); h[new] = t
        if rec_every and (k + 1) % rec_every == 0:
            rec_t.append(t); rec_y.append(Y[:, 1].copy())
    avg = acc / T_end if burn is None else acc / (nacc * dt)
    return dict(avg=avg, lnI=lnI, lnImin=lnImin, hit=hit, rec_t=np.array(rec_t), rec_lnI=np.array(rec_y),
                Yend=Y, nstep=nstep)
