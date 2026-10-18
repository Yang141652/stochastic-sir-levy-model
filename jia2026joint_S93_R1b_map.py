r"""Session 93, R1b: closed-form robustness map. Read-only; no simulation. Checks (1) the map identities symbolically,
(2) that the closed forms reproduce the numbers printed in main v1_15_0 (Sections 5.1 to 5.4, Table 1), and
(3) prints the table of rho(alpha, x) = P/(v/2) - 1 for one-signed jumps on one direction under the exponential form.
Usage (CCR root): python3 -B work/scripts/jia2026joint_S93_R1b_map.py"""
import math
import sympy as sp
# (1) identities. Q_i = sigma_ii/2 + P_i (penalty of class i); R0det = beta*Lam/(a1*a2).
b, L, a1, a2, a3, Q1, Q2, Q3 = sp.symbols("beta Lambda a1 a2 a3 Q1 Q2 Q3", positive=True)
k1, k2, k3 = a1 + Q1, a2 + Q2, a3 + Q3
R0 = (b*L/a1 - Q2)/a2; R0d = b*L/(a1*a2); lam = b*L/a1 - k2
ok = []
ok.append(sp.simplify(lam - a2*(R0 - 1)) == 0)                       # eq:lambda
ok.append(sp.simplify(-lam - (a2*(1 - R0d) + Q2)) == 0)               # infection clock = deterministic rate + Q2
ok.append(sp.simplify((R0d - R0) - Q2/a2) == 0)                       # band width in R0 (and R0det)
bS = sp.solve(sp.Eq(-lam, k1), b)[0]; bR = sp.solve(sp.Eq(-lam, k3), b)[0]
bc = sp.solve(sp.Eq(R0, 1), b)[0]; bd = sp.solve(sp.Eq(R0d, 1), b)[0]
ok.append(sp.simplify(bc - bd - a1*Q2/L) == 0)                        # band width in beta
ok.append(sp.simplify(bS - a1*(k2 - k1)/L) == 0 and sp.simplify(bR - a1*(k2 - k3)/L) == 0)
# S-return regime: infection clock slower iff -lam < k1 iff a2*(1-R0d) - a1 < Q1 - Q2
ok.append(sp.simplify((-lam - k1) - ((a2*(1 - R0d) - a1) - (Q1 - Q2))) == 0)
# inside the band (R0d > 1, R0 < 1): 0 < -lam < Q2
ok.append(sp.simplify(Q2 - (-lam) - a2*(R0d - 1)) == 0)
print("identities:", ok, "all" if all(ok) else "FAIL")
# (2) the paper's numbers
al, bt = 0.4, 1.0; c = [1, 5, 5, 5, 1, 1]; w = [cj*bt**(-al) for cj in c]
A = [[0.433, -0.433, -0.866, -0.433, 0.433, 0.866], [0.75, 0.75, 0, -0.75, -0.75, 0], [0.5]*6]
z = [0.01, 0.3, 0.01]; G = math.gamma(-al)
mf = lambda q: G*((bt - q)**al - bt**al)
Pf = lambda q: G*((bt - q)**al - bt**al + al*q*bt**(al - 1))
sig = [3e-3, 3e-3, 4e-3]; mu, eps, g1 = 0.75, 0.5, (-0.3, 0.5, 0.2)
mus = [mu - g1[0], mu + eps + 1 - g1[1], mu - g1[2]]; Lam = 8.0
a, P = [], []
for i in range(3):
    q = [z[i]*A[i][j] for j in range(6)]
    a.append(mus[i] - sum(w[j]*mf(q[j]) for j in range(6))); P.append(sum(w[j]*Pf(q[j]) for j in range(6)))
Q = [0.5*sig[i] + P[i] for i in range(3)]; k = [a[i] + Q[i] for i in range(3)]
print("a", [round(x, 4) for x in a], "k", [round(x, 4) for x in k], "P2", round(P[1], 4), "Q2", round(Q[1], 4),
      "Q2/a2", round(Q[1]/a[1], 4), "jump share", round(P[1]/Q[1], 4))
print("beta_det", round(a[0]*a[1]/Lam, 4), "beta_c", round(a[0]*k[1]/Lam, 4), "beta_S", round(a[0]*(k[1]-k[0])/Lam, 4),
      "beta_R", round(a[0]*(k[1]-k[2])/Lam, 4), "band width in beta", round(a[0]*Q[1]/Lam, 4))
be = 0.232; R0n = (be*Lam/a[0] - Q[1])/a[1]; lamn = be*Lam/a[0] - k[1]
print("beta 0.232: R0", round(R0n, 4), "R0det", round(be*Lam/(a[0]*a[1]), 4), "|lambda|", round(-lamn, 4),
      "< Q2", -lamn < Q[1], "1/|lambda|", round(-1/lamn, 2), "1/Q2", round(1/Q[1], 2), "1-k1/a2", round(1 - k[0]/a[1], 4))
v1 = 12*(0.225**2)*math.gamma(2 - al)
for name, wt, q in (("TS1", 12, 0.225), ("TS1d", 12, -0.225)):
    p = wt*Pf(q); print(name, "P2", round(p, 4), "shift", round((0.5*v1 - p)/a[1], 4))
print("matched log-variance C22", round(v1, 4))
# (3) rho(alpha, x) for one-signed jumps on one direction, x = zeta2*s2/b, |x| < 1/2 under (T)
def rho(al, x):
    g = (1 - x)**al - 1 + al*x
    return 2*math.gamma(-al)*g/(x*x*math.gamma(2 - al)) - 1
xs = [-0.45, -0.3, -0.15, 0.15, 0.3, 0.45]; als = [0.1, 0.25, 0.4, 0.55, 0.7, 0.9]
print("rho(alpha,x): rows alpha", als, "cols x", xs)
for al_ in als:
    print(al_, [round(100*rho(al_, x), 1) for x in xs], "balanced", [round(100*0.5*(rho(al_, x) + rho(al_, -x)), 2) for x in (0.15, 0.3, 0.45)])
print("paper point alpha 0.4, x 0.225:", round(rho(0.4, 0.225), 4), "first order (2-alpha)x/3:", round(1.6*0.225/3, 4))
