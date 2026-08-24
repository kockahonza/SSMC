import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 200,
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False,
})

def steady_upper(c, gam, r, K, l, m):
    d = gam*c
    A2 = m*c*d; B2 = m*r*(c+d) - K*c*d; C2 = m*r**2 - K*c*r*(1-l)
    disc = B2**2 - 4*A2*C2
    if disc < 0: return None
    rts = sorted([x for x in [(-B2+np.sqrt(disc))/(2*A2), (-B2-np.sqrt(disc))/(2*A2)] if x > 0])
    return rts[-1] if rts else None

def lam(k, N, c, gam, r, K, l, m, DN, DI, p):
    d = gam*c; u = r+c*N; v = r+d*N; DR = p*DI
    lI2 = DI/u; lR2 = DR/v
    A = N*d*c*l*K*r/(u*v**2); I1 = N*c**2*(1-l)*K/u**2; I2 = N**2*d*c**2*l*K/(u**2*v)
    return -DN*k**2 + A/(1+lR2*k**2) - I1/(1+lI2*k**2) - I2/((1+lI2*k**2)*(1+lR2*k**2))

# ---------------- Figure 1: dispersion relations ----------------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
c, r, m, gam, l = 1.0, 1.0, 1.0, 1.0, 0.6
DI = 1.0
ks = np.logspace(-2, 2.5, 4000)

ax = axes[0]
for p, col in [(0.4, "#1b4965"), (0.9, "#5fa8d3"), (1.4, "#bfa14a")]:
    beta = 2.45; K = beta*m*r/c
    N = steady_upper(c, gam, r, K, l, m)
    ax.plot(ks, lam(ks, N, c, gam, r, K, l, m, 0.0, DI, p), color=col, lw=1.4,
            label=rf"$p={p}$")
ax.axhline(0, color="0.4", lw=0.6)
ax.set_xscale("log"); ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$\lambda(k)$")
ax.set_title(r"$D_N=0$,  $\beta=2.45$", fontsize=9)
ax.set_ylim(-0.02, 0.05)
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax = axes[1]
beta = 2.45; K = beta*m*r/c; p = 0.4
N = steady_upper(c, gam, r, K, l, m)
for DN, col in [(0.0, "#1b4965"), (1e-5, "#5fa8d3"), (5e-5, "#bfa14a")]:
    ax.plot(ks, lam(ks, N, c, gam, r, K, l, m, DN, DI, p), color=col, lw=1.4,
            label=(r"$D_N=0$" if DN==0 else rf"$D_N=%s$" % {1e-5:r"10^{-5}",5e-5:r"5\times10^{-5}"}[DN]))
ax.axhline(0, color="0.4", lw=0.6)
ax.set_xscale("log"); ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$\lambda(k)$")
ax.set_title(r"effect of $D_N$  ($p=0.4$)", fontsize=9)
ax.set_ylim(-0.02, 0.05)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("fig_dispersion.pdf", bbox_inches="tight")

# ---------------- Figure 2: (l, p) sign-condition boundary ----------------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))

ax = axes[0]
ls = np.linspace(0.02, 0.97, 800)
for gam, col in [(0.6, "#bfa14a"), (1.0, "#1b4965"), (2.0, "#5fa8d3")]:
    Vmin = np.where(ls*(1+gam) > 1,
                    np.sqrt(ls)*(np.sqrt(ls)+np.sqrt(np.maximum(ls+gam-1, 0))), 1.0)
    pb = gam*ls/((1-ls)*Vmin)
    ax.plot(ls, pb, color=col, lw=1.5, label=rf"$\gamma={gam}$")
    lk = 1/(1+gam)
    if 0.02 < lk < 0.97:
        Vk = 1.0
        ax.plot([lk], [gam*lk/((1-lk)*Vk)], "o", color=col, ms=3.5, mec="w", mew=0.6)
    # naive (regime-A) formula extended, for contrast
    ax.plot(ls, gam*ls/(1-ls), color=col, lw=0.8, ls=":", alpha=0.75)
ax.set_yscale("log"); ax.set_ylim(0.05, 40)
ax.set_xlabel(r"$l$"); ax.set_ylabel(r"$p$")
ax.set_title(r"sign condition: pattern possible below the curve", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax = axes[1]
gam = 1.0; l = 0.6
ps = np.linspace(0.05, 1.30, 600)
bmin = (np.sqrt(l)+np.sqrt(l+gam-1))**2/gam
bc = (gam*l + ps*(gam-1)*(1-l))/(ps*(1-l)*(gam-ps*(1-l)))
ok = bc > bmin
ax.fill_between(ps[ok], bmin, bc[ok], color="#5fa8d3", alpha=0.35, lw=0)
ax.plot(ps[ok], bc[ok], color="#1b4965", lw=1.5, label=r"$\beta_c(p)$")
ax.axhline(bmin, color="#bfa14a", lw=1.5, ls="--", label=r"$\beta_{\min}$")
ax.axhline(1/(1-l), color="0.55", lw=0.8, ls=":", label=r"$1/(1-l)$")
ax.set_xlabel(r"$p$"); ax.set_ylabel(r"$\beta$")
ax.set_ylim(2.2, 4.2); ax.set_xlim(0.05, 1.30)
ax.set_title(r"pattern window, $\gamma=1$, $l=0.6$", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("fig_regions.pdf", bbox_inches="tight")
print("figures written")
