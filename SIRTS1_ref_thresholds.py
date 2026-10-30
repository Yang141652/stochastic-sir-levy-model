"""
Reference computation of the threshold quantities in SIRTS1.tex (Section 5)
from the parameter values stated in the text. Closed forms follow from the
discrete spectral measure sigma = sum_j w_j delta_{s_j} and, for 0<alpha<1,
  int_0^inf (e^{-k r} - 1) r^{-1-alpha} e^{-b r} dr = Gamma(-alpha)[(b+k)^alpha - b^alpha],  b+k>0.
With eta_i(u) = exp(zeta_i u_i) - 1, ln(1+eta_i(u)) = zeta_i u_i exactly.
"""
import numpy as np
from math import gamma as G

def setup(alpha=0.4, b=1.5, zeta=(0.03,0.03,0.03), n=6, om=0.5,
          c=(1.4436,1.4719,1.5002,1.5285,1.5568,1.5851)):
    th = 2*np.pi*np.arange(1,n+1)/n
    s = np.stack([np.sqrt(1-om**2)*np.cos(th), np.sqrt(1-om**2)*np.sin(th), om*np.ones(n)], axis=1)
    w = np.array(c)*b**(-alpha)          # w_j = c_j b(s_j)^(-alpha), as stated in the text
    return dict(alpha=alpha, b=b, zeta=np.array(zeta), s=s, w=w)

def I_exp(k, P):
    """sum_j w_j int (e^{-k_j r}-1) r^{-1-a} e^{-b r} dr  for vector k_j (requires b+k_j>0)."""
    a, b, w = P['alpha'], P['b'], P['w']
    k = np.asarray(k)
    assert np.all(b + k > 0), "tempering too weak: integral diverges"
    return np.sum(w*G(-a)*((b+k)**a - b**a))

def moments(P):
    a, b, s, w, z = P['alpha'], P['b'], P['s'], P['w'], P['zeta']
    out = {}
    for i in range(3):
        k = -z[i]*s[:,i]                                   # eta_i = e^{z s_i r} - 1
        m1 = I_exp(k, P)                                   # int eta_i dnu
        m2 = I_exp(2*k, P) - 2*I_exp(k, P)                  # int eta_i^2 dnu
        lin = np.sum(w*z[i]*s[:,i]*G(1-a)*b**(a-1))         # int zeta u_i dnu = int ln(1+eta_i) dnu
        out[i] = dict(int_eta=m1, int_eta2=m2, int_log=lin, pen=m1-lin)  # pen = int (eta - ln(1+eta)) dnu
    mj = np.max(z*s, axis=1)                                # zeta*max_i s_ji (equal zetas)
    out['tau2'] = I_exp(-2*mj, P) if np.all(mj>0) else np.nan
    return out

def thresholds(beta, Lam=8.0, mu=5.3, eps=0.5, g1=(-0.0032,0.0039,-0.1615), Phi=None, P=None, jumps=True):
    Phi = np.array([[4,3.2,3],[3.2,4,3.84],[3,3.84,4.69]])*1e-2 if Phi is None else Phi
    g1 = np.array(g1) if jumps else np.zeros(3)
    mu1, mu2, mu3 = mu-g1[0], mu+eps+1-g1[1], mu-g1[2]
    if jumps:
        M = moments(P)
        a1, a2, a3 = mu1-M[0]['int_eta'], mu2-M[1]['int_eta'], mu3-M[2]['int_eta']
        pen2, tau2 = M[1]['pen'], M['tau2']
    else:
        a1, a2, a3, pen2, tau2 = mu1, mu2, mu3, 0.0, 0.0
    R0 = beta*Lam/(a1*a2) - (0.5*Phi[1,1] + pen2)/a2
    q = min(mu1, mu2-1, mu3) - 0.5*np.max(np.abs(Phi)) - 0.5*tau2
    Sbar = Lam/a1 - a2*(R0-1)/beta
    Ibar = a1*(R0-1)/beta
    Rbar = Ibar/a3
    return dict(mu1=mu1, mu2=mu2, mu3=mu3, a1=a1, a2=a2, a3=a3, pen2=pen2, tau2=tau2,
                q=q, R0=R0, rate=a2*(R0-1), Sbar=Sbar, Ibar=Ibar, Rbar=Rbar, S_ext=Lam/a1)

def A_H5(p, qq, r, beta, P, Lam=8.0, mu=5.3, eps=0.5, g1=(-0.0032,0.0039,-0.1615), Phi=None):
    Phi = np.array([[4,3.2,3],[3.2,4,3.84],[3,3.84,4.69]])*1e-2 if Phi is None else Phi
    g1 = np.array(g1); mu1, mu2, mu3 = mu-g1[0], mu+eps+1-g1[1], mu-g1[2]
    v = np.array([p,qq,r])
    diff = v@np.array([mu1,mu2,mu3]) + 0.5*v@Phi@v + 0.5*(v*np.diag(Phi)).sum()
    k = P['s']@(v*P['zeta'])            # (1+eta1)^-p (1+eta2)^-q (1+eta3)^-r = exp(-r * k_j)
    return diff + I_exp(k, P)

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    P = setup()
    print("Direction matrix A (rows = components):\n", P['s'].T)
    Phi = np.array([[4,3.2,3],[3.2,4,3.84],[3,3.84,4.69]])*1e-2
    print("eig(Phi) =", np.linalg.eigvalsh(Phi))
    M = moments(P)
    for i in range(3):
        print(f"comp {i+1}: int eta = {M[i]['int_eta']:.6f}, int eta^2 = {M[i]['int_eta2']:.6f}, "
              f"int ln(1+eta) = {M[i]['int_log']:.6f}, int(eta - ln(1+eta)) = {M[i]['pen']:.3e}")
    print(f"tau(2) = {M['tau2']:.6f}")
    print("\n--- Baseline (beta = 4.8), text parameter set ---")
    Rdet = 4.8*8/(5.3*6.8)
    T = thresholds(4.8, P=P); Tw = thresholds(4.8, P=P, jumps=False)
    print(f"deterministic R0 = {Rdet:.4f}")
    for k in ['mu1','mu2','mu3','a1','a2','a3','tau2','q','R0','rate','S_ext']:
        print(f"  {k:6s} = {T[k]:.4f}")
    print(f"  white-noise-only R0 = {Tw['R0']:.4f}")
    print("  REPORTED in text: q=3.8924, a1=5.8778, a2=7.8361, a3=3.9633, R0=0.9593, <S>->53.2213")
    print(f"  Check of reported values: 38.4/(5.8778*7.8361) = {38.4/(5.8778*7.8361):.4f} (upper bound on R0); 8/5.8778 = {8/5.8778:.4f}")
    print("\n--- Strong diffusion (3*Phi) ---")
    T3 = thresholds(4.8, P=P, Phi=3*Phi); T3w = thresholds(4.8, P=P, Phi=3*Phi, jumps=False)
    print(f"  R0 (WN+TS) = {T3['R0']:.4f}; R0 (WN only) = {T3w['R0']:.4f}; REPORTED 0.9351")
    print("\n--- Persistence case beta = 6 ---")
    T6 = thresholds(6.0, P=P)
    print(f"  R0 = {T6['R0']:.4f} (REPORTED 1.5617); Sbar={T6['Sbar']:.4f}, Ibar={T6['Ibar']:.4f}, Rbar={T6['Rbar']:.4f}")
    print("  REPORTED: 39.5973, 3.9319, 5.5069")
    A = A_H5(0.2,0.2,0.05,6.0,P); rhs = 2*(1-0.2)*np.sqrt(0.2*0.2*6*8)
    print(f"  (H5) with (p,q,r,delta)=(0.2,0.2,0.05,0.2): A = {A:.4f}, RHS = {rhs:.4f}  -> holds? {A<rhs}")
    print("  REPORTED: A = 0.0082 < 0.0315")
    # search feasible (H5) region for beta = 6 and 4.8
    for beta in (4.8, 6.0):
        best = None
        for p in np.linspace(0.01, 1.0, 100):
            for dq in np.linspace(0, 1.0, 51):
                qq = p + dq
                for r in (1e-4, 1e-3, 1e-2):
                    A = A_H5(p, qq, r, beta, P)
                    dmax = 1 - A/(2*np.sqrt(p*qq*beta*8))   # largest delta for which (H5) holds
                    if best is None or dmax > best[0]:
                        best = (dmax, p, qq, r, A)
        print(f"  beta={beta}: largest admissible delta = {best[0]:.4f} at p={best[1]:.3f}, q={best[2]:.3f}, r={best[3]}")
    print("\n--- alpha and b sweeps (beta = 4.8) ---")
    for a in (0.1, 0.4, 0.9):
        Pa = setup(alpha=a); print(f"  alpha={a}: R0 = {thresholds(4.8, P=Pa)['R0']:.4f}")
    print("  REPORTED: alpha=0.1 -> 0.9065, alpha=0.9 -> 0.9285")
    for bb in (0.55, 1.0, 1.5, 2.0):
        Pb = setup(b=bb); print(f"  b={bb}: R0 = {thresholds(4.8, P=Pb)['R0']:.4f}")
    print("  REPORTED: b=1 -> 0.9491, b=2 -> 1.0059")
