# python_reference copy (Session 53, ruling S50-4): the module-level self-checks and the pre-merge parameter block they use (F-S28-4) of the validated source
# work/scripts/SIRTS1_ref_coupling.py now run only under the main guard at the end, so importing this module runs nothing.
# Every function body is byte-identical to the validated source; run this file as a script to see the self-checks.
"""
WU4 Steps 3-4 check (2026-09-22): three quantitative claims of the classification draft.
  Lemma C.3.3  synchronous coupling: |ln S^y - ln S^{y'}| decays at rate k_1
  Lemma C.3.4  E_pi[S_hat] = Lam/a_1,  E_pi[1/S_hat] = k_1/Lam
  Thm  C.4.1b  under extinction, ln(S_hat - S)/t -> max(-k_1, lambda)
Uses the exact log-coordinate scheme of SIRTS1_ref_simulation.py (paper parameters,
eta_i = exp(zeta_i u_i) - 1). Runtime about one minute.
"""
import numpy as np
import SIRTS1_ref_simulation as SIM          # runs its own sampler self-checks on import
from SIRTS1_ref_thresholds import setup, thresholds, moments

rng = SIM.rng

def run(T, dt, npath, beta, x0, x0p=None, seed=1):
    """Log-coordinate scheme for (S,I,R) and, with the same noise, the disease-free S_hat
    from S(0) and (optionally) a second disease-free S_hat' from x0p."""
    global rng; rng = np.random.default_rng(seed); SIM.rng = rng
    Gm = np.linalg.cholesky(Phi); m1, m2, m3 = mu-g1[0], mu+eps+1-g1[1], mu-g1[2]
    Y = np.tile(np.log(x0), (npath,1)); Yh = np.full(npath, np.log(x0[0]))
    Yh2 = np.full(npath, np.log(x0p)) if x0p else None
    n = int(T/dt); t = np.arange(n+1)*dt
    D = np.empty((n+1, npath)); gap = np.empty((n+1, npath)); accS = np.zeros(npath); accSi = np.zeros(npath)
    D[0] = 0 if Yh2 is None else np.abs(Yh-Yh2); gap[0] = 0.0
    for k in range(n):
        dE = np.sqrt(dt)*rng.standard_normal((npath,3))@Gm.T
        dL = np.zeros((npath,3))
        for j in range(P['s'].shape[0]):
            dL += np.outer(SIM.ts_increment(P['alpha'], P['w'][j], P['b'], dt, npath), P['s'][j])
        S, I, R = np.exp(Y[:,0]), np.exp(Y[:,1]), np.exp(Y[:,2])
        drift = np.stack([Lam/S-m1-beta*I-Phi[0,0]/2, beta*S-m2-Phi[1,1]/2, I/R-m3-Phi[2,2]/2], axis=1)
        Y = Y + drift*dt + dE + P['zeta']*dL
        Yh = Yh + (Lam/np.exp(Yh)-m1-Phi[0,0]/2)*dt + dE[:,0] + P['zeta'][0]*dL[:,0]
        if Yh2 is not None:
            Yh2 = Yh2 + (Lam/np.exp(Yh2)-m1-Phi[0,0]/2)*dt + dE[:,0] + P['zeta'][0]*dL[:,0]
            D[k+1] = np.abs(Yh-Yh2)
        gap[k+1] = np.exp(Yh) - np.exp(Y[:,0])
        accS += np.exp(Yh)*dt; accSi += np.exp(-Yh)*dt
    return t, D, gap, accS/T, accSi/T


if __name__ == "__main__":
    P = setup(); M = moments(P)
    Lam, mu, eps, g1 = 8.0, 5.3, 0.5, (-0.0032, 0.0039, -0.1615)
    Phi = np.array([[4,3.2,3],[3.2,4,3.84],[3,3.84,4.69]])*1e-2
    th = thresholds(4.0, P=P)                    # beta = 4: extinction regime in the paper
    a1, a2 = th['a1'], th['a2']
    k1 = a1 + 0.5*Phi[0,0] + M[0]['pen']         # pen = int (eta_1 - ln(1+eta_1)) dnu
    lam = th['rate']                             # a2 (R0 - 1)
    print(f"a1 = {a1:.4f}   k1 = {k1:.4f}   lambda(beta=4) = {lam:.4f}   max(-k1, lambda) = {max(-k1, lam):.4f}")

    # --- Lemma C.3.3: coupling rate ---
    t, D, _, _, _ = run(T=3.0, dt=0.005, npath=200, beta=4.0, x0=(0.5,0.4,0.04), x0p=5.0)
    lnD = np.log(D[1:]).mean(axis=1); tt = t[1:]
    sl = np.polyfit(tt[(tt>0.5)&(tt<2.5)], lnD[(tt>0.5)&(tt<2.5)], 1)[0]
    print(f"Coupling: slope of mean ln|D| on [0.5,2.5] = {sl:.3f}   vs  -k1 = {-k1:.3f}")

    # --- Lemma C.3.4: the two means of pi_- (long run, one path, after burn-in) ---
    t, _, _, mS, mSi = run(T=200.0, dt=0.01, npath=4, beta=4.0, x0=(1.6,0.4,0.04))
    print(f"<S_hat>_T  = {mS.mean():.4f} +- {mS.std()/2:.4f}   vs  Lam/a1 = {Lam/a1:.4f}")
    print(f"<1/S_hat>_T= {mSi.mean():.4f} +- {mSi.std()/2:.4f}   vs  k1/Lam = {k1/Lam:.4f}")

    # --- Theorem C.4.1(b): rate of S_hat - S under extinction (beta = 4) ---
    t, _, gap, _, _ = run(T=6.0, dt=0.005, npath=100, beta=4.0, x0=(1.6,0.4,0.04))
    g = np.log(np.maximum(gap[1:], 1e-300)).mean(axis=1); tt = t[1:]
    sel = (tt>2.0)&(tt<6.0)
    sl2 = np.polyfit(tt[sel], g[sel], 1)[0]
    print(f"Extinction: slope of mean ln(S_hat - S) on [2,6] = {sl2:.3f}   vs  max(-k1, lambda) = {max(-k1, lam):.3f}")
