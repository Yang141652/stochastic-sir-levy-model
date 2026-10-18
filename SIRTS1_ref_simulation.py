# python_reference copy (Session 53, ruling S50-4): the module-level self-checks of the validated source
# work/scripts/SIRTS1_ref_simulation.py now run only under the main guard at the end, so importing this module runs nothing.
# Every function body is byte-identical to the validated source; run this file as a script to see the self-checks.
"""
Reference simulator for SIRTS1 (review companion, 2026-09-22).

1. Exact one-sided tempered-stable increments: positive alpha-stable draw
   (Kanter / Chambers-Mallows-Stuck) followed by exponential rejection e^{-bX}.
2. Log-coordinate scheme: with eta_i(u) = exp(zeta_i u_i) - 1 the jump part of
   d ln X_i is exactly zeta_i dL_i, so only drift and Gaussian parts are discretized.
3. Self-checks against the paper's theorems (Theorem 2 time averages at beta=6;
   Theorem 1 exact extinction rate a2(R0-1) at beta=4).

Usage: python3 SIRTS1_ref_simulation.py   (needs SIRTS1_ref_thresholds.py alongside;
several minutes on one core). Port to R, then reproduce the printed checks.
"""
import numpy as np
from math import gamma as G
from SIRTS1_ref_thresholds import setup, thresholds

rng = np.random.default_rng(20260922)

def pos_stable(alpha, size):
    """Kanter/CMS: E[exp(-lam*S)] = exp(-lam**alpha), 0<alpha<1."""
    U = rng.uniform(0, np.pi, size); E = rng.exponential(1.0, size)
    return (np.sin(alpha*U)/np.sin(U)**(1/alpha)) * (np.sin((1-alpha)*U)/E)**((1-alpha)/alpha)

def ts_increment(alpha, w, b, dt, size):
    """Exact increment over dt of the one-sided TS subordinator with Levy measure
    w r^(-1-alpha) e^(-b r) dr: positive stable scaled to Laplace exponent dt*w*|Gamma(-alpha)|*th^alpha,
    then exponential rejection with acceptance prob exp(-b X)."""
    scale = (dt*w*abs(G(-alpha)))**(1/alpha)
    out = np.empty(size); todo = np.arange(size)
    while todo.size:
        X = scale*pos_stable(alpha, todo.size)
        acc = rng.uniform(size=todo.size) < np.exp(-b*X)
        out[todo[acc]] = X[acc]; todo = todo[~acc]
    return out

# 3) Log-coordinate scheme; jump part exact because ln(1+eta_i(u)) = zeta_i u_i
def simulate(beta, T, dt, npath, P, Lam=8.0, mu=5.3, eps=0.5, g1=(-0.0032,0.0039,-0.1615), Phi=None, x0=(1.6,0.4,0.04)):
    Phi = np.array([[4,3.2,3],[3.2,4,3.84],[3,3.84,4.69]])*1e-2 if Phi is None else Phi
    Gm = np.linalg.cholesky(Phi)                       # Phi = Gm Gm^T
    m1, m2, m3 = mu-g1[0], mu+eps+1-g1[1], mu-g1[2]
    Y = np.tile(np.log(x0), (npath,1)); nstep = int(T/dt)
    lnI = np.empty((nstep+1, npath)); lnI[0] = Y[:,1]; acc = np.zeros((npath,3))
    for k in range(nstep):
        dE = np.sqrt(dt)*rng.standard_normal((npath,3))@Gm.T
        dL = np.zeros((npath,3))
        for j in range(P['s'].shape[0]):
            dL += np.outer(ts_increment(P['alpha'], P['w'][j], P['b'], dt, npath), P['s'][j])
        S, I, R = np.exp(Y[:,0]), np.exp(Y[:,1]), np.exp(Y[:,2])
        drift = np.stack([Lam/S-m1-beta*I-Phi[0,0]/2, beta*S-m2-Phi[1,1]/2, I/R-m3-Phi[2,2]/2], axis=1)
        Y = Y + drift*dt + dE + P['zeta']*dL
        acc += np.exp(Y)*dt; lnI[k+1] = Y[:,1]
    return acc/T, lnI


if __name__ == "__main__":
    # 1) Laplace-transform check of the positive-stable sampler
    S = pos_stable(0.4, 400000)
    for lam in (0.5, 1.0, 2.0):
        print(f"LT check lam={lam}: MC {np.mean(np.exp(-lam*S)):.4f} vs exact {np.exp(-lam**0.4):.4f}")
    # 2) Moment check of the TS increment (one direction, text parameters)
    a, b, w, dt = 0.4, 1.5, 1.4436*1.5**(-0.4), 0.01
    Y = ts_increment(a, w, b, dt, 400000)
    print(f"TS mean  MC {Y.mean():.6f} vs exact {dt*w*G(1-a)*b**(a-1):.6f}")
    print(f"TS var   MC {Y.var():.6f} vs exact {dt*w*G(2-a)*b**(a-2):.6f}")

    P = setup()
    # Theorem 2 check (persistence, beta = 6): time averages vs exact limits
    T6 = thresholds(6.0, P=P)
    avg, _ = simulate(6.0, T=400, dt=0.005, npath=40, P=P)
    m, se = avg.mean(0), avg.std(0)/np.sqrt(avg.shape[0])
    print(f"\nThm 2 (beta=6, R0={T6['R0']:.4f}): <S>,<I>,<R> sim = {m.round(4)} +/- {se.round(4)}")
    print(f"                          exact = {np.round([T6['Sbar'],T6['Ibar'],T6['Rbar']],4)}")
    # Theorem 1 check (extinction, beta = 4.0): slope of ln I vs exact rate a2(R0-1)
    T4 = thresholds(4.0, P=P)
    _, lnI = simulate(4.0, T=60, dt=0.005, npath=40, P=P)
    t = np.arange(lnI.shape[0])*0.005; sel = t > 20
    slope = np.polyfit(t[sel], lnI[sel].mean(1), 1)[0]
    print(f"Thm 1 (beta=4.0, R0={T4['R0']:.4f}): slope of ln I sim = {slope:.4f}; exact a2(R0-1) = {T4['rate']:.4f}")
