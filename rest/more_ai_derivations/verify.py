import numpy as np

# ---- model in the user's notation ----
# dN = N(c(1-l)I + dR - m) + DN lap N
# dI = K - r I - c N I + DI lap I
# dR = c l N I - r R - d N R + DR lap R,  DR = p DI, d = gamma c

def steady(c, gam, r, K, l, m):
    d = gam*c
    # quadratic in N:  m c d N^2 + (m r (c+d) - K c d) N + m r^2 - K c r (1-l) = 0
    A2 = m*c*d
    B2 = m*r*(c+d) - K*c*d
    C2 = m*r**2 - K*c*r*(1-l)
    disc = B2**2 - 4*A2*C2
    if disc < 0:
        return []
    roots = [(-B2+np.sqrt(disc))/(2*A2), (-B2-np.sqrt(disc))/(2*A2)]
    return sorted([x for x in roots if x > 0])

def coeffs(N, c, gam, r, K, l, m, DN, DI, p):
    d = gam*c
    u = r + c*N
    v = r + d*N
    DR = p*DI
    lI2 = DI/u
    lR2 = DR/v
    A = N*d*c*l*K*r/(u*v**2)
    I1 = N*c**2*(1-l)*K/u**2
    I2 = N**2*d*c**2*l*K/(u**2*v)
    return A, I1, I2, lI2, lR2

def lam_analytic(k, N, c, gam, r, K, l, m, DN, DI, p):
    A, I1, I2, lI2, lR2 = coeffs(N, c, gam, r, K, l, m, DN, DI, p)
    return (-DN*k**2 + A/(1+lR2*k**2) - I1/(1+lI2*k**2)
            - I2/((1+lI2*k**2)*(1+lR2*k**2)))

def lam_direct(k, N, c, gam, r, K, l, m, DN, DI, p):
    """Solve the QSS 2x2 system for rho,sigma numerically, then lambda."""
    d = gam*c; DR = p*DI
    u = r + c*N; v = r + d*N
    Istar = K/u
    Rstar = c*l*K*N/(u*v)
    # 0 = -(u + DI k^2) rho - c Istar nu
    # 0 = c l (Istar nu + N rho) - (v + DR k^2) sigma - d Rstar nu
    nu = 1.0
    rho = -c*Istar*nu/(u + DI*k**2)
    sigma = (c*l*(Istar*nu + N*rho) - d*Rstar*nu)/(v + DR*k**2)
    return N*(c*(1-l)*rho + d*sigma) - DN*k**2

def Gfun(N, c, gam, r, K, l, m):
    d = gam*c
    u = r + c*N; v = r + d*N
    return c*(1-l)*K/u + d*c*l*K*N/(u*v) - m

rng = np.random.default_rng(0)
maxerr = 0.0
for _ in range(400):
    c = 10**rng.uniform(-1,1); gam = 10**rng.uniform(-1,1)
    r = 10**rng.uniform(-1,1); K = 10**rng.uniform(-1,1.5)
    l = rng.uniform(0.05,0.95); m = 10**rng.uniform(-1.5,0.5)
    DN = 10**rng.uniform(-4,-1); DI = 10**rng.uniform(-2,1); p = 10**rng.uniform(-1,1)
    Ns = steady(c,gam,r,K,l,m)
    for N in Ns:
        assert abs(Gfun(N,c,gam,r,K,l,m)) < 1e-8*max(1,m), "G(N*)!=0"
        for k in [0.0, 0.3, 1.0, 3.0, 17.0]:
            a = lam_analytic(k,N,c,gam,r,K,l,m,DN,DI,p)
            b = lam_direct(k,N,c,gam,r,K,l,m,DN,DI,p)
            maxerr = max(maxerr, abs(a-b)/max(1e-12,abs(b)))
print("1) dispersion relation vs direct QSS solve, max rel err:", maxerr)

# 2) lambda(0) = N G'(N)
err = 0
for _ in range(200):
    c = 10**rng.uniform(-1,1); gam = 10**rng.uniform(-1,1)
    r = 10**rng.uniform(-1,1); K = 10**rng.uniform(-1,1.5)
    l = rng.uniform(0.05,0.95); m = 10**rng.uniform(-1.5,0.5)
    for N in steady(c,gam,r,K,l,m):
        A,I1,I2,_,_ = coeffs(N,c,gam,r,K,l,m,0,1,1)
        h = N*1e-6
        Gp = (Gfun(N+h,c,gam,r,K,l,m)-Gfun(N-h,c,gam,r,K,l,m))/(2*h)
        err = max(err, abs((A-I1-I2) - N*Gp)/max(1e-12,abs(N*Gp)))
print("2) lambda(0) = N G'(N), max rel err:", err)

# 3) sign condition  A lI^2 > I1 lR^2  <=>  instability at large k (DN=0)
bad = 0; tested = 0
for _ in range(600):
    c = 10**rng.uniform(-1,1); gam = 10**rng.uniform(-1,1)
    r = 10**rng.uniform(-1,1); K = 10**rng.uniform(-1,1.5)
    l = rng.uniform(0.05,0.95); m = 10**rng.uniform(-1.5,0.5)
    DI = 10**rng.uniform(-2,1); p = 10**rng.uniform(-1,1)
    for N in steady(c,gam,r,K,l,m):
        if Gfun(N+1e-9,c,gam,r,K,l,m) > Gfun(N-1e-9,c,gam,r,K,l,m):
            continue  # unstable branch, skip
        A,I1,I2,lI2,lR2 = coeffs(N,c,gam,r,K,l,m,0,DI,p)
        pred = (A*lI2 > I1*lR2)
        ks = np.logspace(-3,4,2000)
        obs = np.max(lam_analytic(ks,N,c,gam,r,K,l,m,0,DI,p)) > 0
        tested += 1
        if pred != obs: bad += 1
print("3) sign condition vs numerics (DN=0):", bad, "mismatches out of", tested)

# 4) beta_c and V_c
def beta_of_V(V,gam,l): return V*(V+gam-1)/(gam*(V-l))
bad4 = 0; tested4 = 0
for _ in range(600):
    c = 10**rng.uniform(-1,1); gam = 10**rng.uniform(-1,1)
    r = 10**rng.uniform(-1,1); K = 10**rng.uniform(-1,1.5)
    l = rng.uniform(0.05,0.95); m = 10**rng.uniform(-1.5,0.5)
    DI = 1.0; p = 10**rng.uniform(-1,1)
    beta = K*c/(m*r)
    Ns = steady(c,gam,r,K,l,m)
    if not Ns: continue
    N = Ns[-1]  # upper branch
    if Gfun(N+1e-9,c,gam,r,K,l,m) > Gfun(N-1e-9,c,gam,r,K,l,m): continue
    V = 1 + gam*c*N/r
    assert abs(beta_of_V(V,gam,l)-beta)/beta < 1e-6, "beta(V) wrong"
    Vc = (gam/p)*(l/(1-l))
    A,I1,I2,lI2,lR2 = coeffs(N,c,gam,r,K,l,m,0,DI,p)
    obs = (A*lI2 > I1*lR2)
    tested4 += 1
    if (V < Vc) != obs: bad4 += 1
    if Vc > l:
        bc = (gam*l + p*(gam-1)*(1-l))/(p*(1-l)*(gam-p*(1-l))) if (gam-p*(1-l))>0 else None
        if bc is not None and abs(beta_of_V(Vc,gam,l)-bc)/abs(bc) > 1e-8:
            print("   beta_c closed form mismatch!", beta_of_V(Vc,gam,l), bc)
print("4) V<Vc equivalence:", bad4, "mismatches out of", tested4, "; beta_c closed form checked")

# 5) V_fold, beta_min piecewise, ordering
bad5 = 0
for _ in range(4000):
    gam = 10**rng.uniform(-1,1); l = rng.uniform(0.02,0.98)
    # numeric beta_min = smallest beta admitting a positive stable root
    betas = np.logspace(-2,3,20000)
    ok = []
    for b in betas[::40]:
        c,r,m = 1.0,1.0,1.0; K = b*m*r/c
        Ns = steady(c,gam,r,K,l,m)
        good = [N for N in Ns if Gfun(N+1e-9,c,gam,r,K,l,m) < Gfun(N-1e-9,c,gam,r,K,l,m)]
        if good: ok.append((b,max(good)))
    if not ok: continue
    bmin_num = ok[0][0]
    if l*(1+gam) > 1:
        bmin = (np.sqrt(l)+np.sqrt(max(l+gam-1,0)))**2/gam
        Vfold = np.sqrt(l)*(np.sqrt(l)+np.sqrt(max(l+gam-1,0)))
        # ordering check
        bminus = (2*l+gam-1-2*np.sqrt(l*(l+gam-1)))/gam
        if not (bminus < (1+gam)/gam < bmin <= 1/(1-l)+1e-12): bad5 += 1
    else:
        bmin = 1/(1-l)
    if abs(bmin_num-bmin)/bmin > 0.02: bad5 += 1
print("5) beta_min piecewise + ordering:", bad5, "mismatches")

# 6) V_fold formula
bad6 = 0
for _ in range(500):
    gam = 10**rng.uniform(-1,1); l = rng.uniform(0.02,0.98)
    if l*(1+gam) <= 1: continue
    bmin = (np.sqrt(l)+np.sqrt(l+gam-1))**2/gam
    Vf = (1+bmin*gam-gam)/2
    Vf2 = np.sqrt(l)*(np.sqrt(l)+np.sqrt(l+gam-1))
    if abs(Vf-Vf2) > 1e-9*max(1,Vf): bad6 += 1
print("6) V_fold = sqrt(l)(sqrt(l)+sqrt(l+gam-1)):", bad6, "mismatches")

# 7) monotonicity of beta(V) on upper branch: dbeta/dV > 0 for V > V_fold
bad7 = 0
for _ in range(500):
    gam = 10**rng.uniform(-1,1); l = rng.uniform(0.02,0.98)
    Vmin = np.sqrt(l)*(np.sqrt(l)+np.sqrt(l+gam-1)) if l*(1+gam)>1 else 1.0
    Vs = Vmin*np.linspace(1.0001, 50, 500)
    dbdV = (Vs**2 - 2*l*Vs - (gam-1)*l)
    if np.any(dbdV <= 0): bad7 += 1
print("7) beta increasing on upper branch:", bad7, "failures")

# 8) DN=0 numerator is linear in s -> at most one crossing; DN>0 -> single band
bad8 = 0
for _ in range(400):
    c = 10**rng.uniform(-1,1); gam = 10**rng.uniform(-1,1)
    r = 10**rng.uniform(-1,1); K = 10**rng.uniform(-1,1.5)
    l = rng.uniform(0.05,0.95); m = 10**rng.uniform(-1.5,0.5)
    DN = 10**rng.uniform(-5,-1); DI = 10**rng.uniform(-2,1); p = 10**rng.uniform(-1,1)
    for N in steady(c,gam,r,K,l,m):
        if Gfun(N+1e-9,c,gam,r,K,l,m) > Gfun(N-1e-9,c,gam,r,K,l,m): continue
        ks = np.logspace(-4,5,60000)
        L = lam_analytic(ks,N,c,gam,r,K,l,m,DN,DI,p)
        sgn = np.sign(L); ncross = np.sum(sgn[1:]*sgn[:-1] < 0)
        if ncross > 2: bad8 += 1
print("8) number of sign changes never exceeds 2 (DN>0):", bad8, "violations")
