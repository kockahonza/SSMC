# Numbers quoted in spike_interactions.tex, measured off the arrested state in
# cluster_env/runs/mm_pdes_vs_p/run4_arrest_check/state1.jld2 (p=1, DN=1e-6,
# 29 peaks). Reads the state only - no SSMCMain, no solving.
using JLD2, Printf, LinearAlgebra

const STATE = joinpath(@__DIR__, "..", "..", "cluster_env", "runs",
    "mm_pdes_vs_p", "run4_arrest_check", "state1.jld2")

d = jldopen(STATE)
u, L, sN, p = d["u_star"], d["L"], d["sN"], d["p"]
close(d)

K, l, m, c, dd, r = 15.0, 0.999, 1.0, 1.0, 1.0, 1.0
DN, DI = 1e-6, 1.0
DR = p
dx = L / sN
N, I, R = u[1, :], u[2, :], u[3, :]
mu = @. (1 - l) * c * I + dd * R - m       # local per-capita growth rate

wrap(i) = mod1(i, sN)
@printf("L=%.6f  sN=%d  dx=%.6g  p=%g  DN=%g\n", L, sN, dx, p, DN)

# ---- peaks and the one-cell window we work in --------------------------------
thr = minimum(N) + 0.1 * (maximum(N) - minimum(N))
starts = [i for i in 1:sN if N[i] > thr && N[wrap(i-1)] <= thr]
k = length(starts)
spacing = L / k
@printf("peaks=%d  spacing=%.6f  cell=%d gridpoints\n", k, spacing, round(Int, spacing/dx))

ctrs = map(starts) do s
    idx = Int[]; i = s
    while N[wrap(i)] > thr; push!(idx, i); i += 1; end
    idx[argmax([N[wrap(j)] for j in idx])]
end
ctr = ctrs[1]
half = fld(sN, 2k)
win = [wrap(ctr + j) for j in -half:half]      # one full period around a spike
xi = collect(-half:half) .* dx                  # position relative to the core

# ---- spike: mass, amplitude, widths -----------------------------------------
S = sum(N[win]) * dx
A = N[ctr]
x0_mom = sqrt(sum(N[win] .* xi .^ 2) * dx / S)        # 2nd moment; = x0 for a gaussian
x0_ma = S / A / sqrt(2pi)                             # mass/amplitude; = x0 for a gaussian
@printf("\nspike: A=%.6g  S=%.6g  x0(2nd moment)=%.6g  x0(mass/amp)=%.6g\n", A, S, x0_mom, x0_ma)
@printf("       sqrt(DN/m)=%.3g  -> x0/sqrt(DN/m)=%.1f    gridpoints per x0=%.1f\n",
    sqrt(DN/m), x0_mom/sqrt(DN/m), x0_mom/dx)

# ---- the well mu(x) ----------------------------------------------------------
mu0 = mu[ctr]
@printf("\nwell: mu at core=%.6g   mu at mid-gap=%.6g\n", mu0, mu[win[end]])
@printf("      I core=%.6g mid-gap=%.6g   R core=%.6g mid-gap=%.6g\n",
    I[ctr], I[win[end]], R[ctr], R[win[end]])

# outer slopes just outside the spike, at 4*x0 from the core
off = max(2, round(Int, 4 * x0_mom / dx))
dI = (I[wrap(ctr+off+1)] - I[wrap(ctr+off-1)]) / (2dx)
dR = (R[wrap(ctr+off+1)] - R[wrap(ctr+off-1)]) / (2dx)
alpha = -((1 - l) * c * dI + dd * dR)     # mu ~ mu0 - alpha|xi| on the inner scale
@printf("      at %.4f from core: I_x=%+.5g  R_x=%+.5g  ->  alpha=%.5g\n", off*dx, dI, dR, alpha)
@printf("      (I part %+.4g, R part %+.4g of mu_x)\n", (1-l)*c*dI, dd*dR)

# ---- flux jump conditions ----------------------------------------------------
# integrate the reaction terms over a window round the spike and compare with
# D times the jump in the gradient across it. The window is only ~0.1*ell, so
# the linear terms (K - rI) and -rR are NOT negligible over it and have to be
# kept - dropping them costs 16%.
core = [wrap(ctr + j) for j in -off:off]
uptake_I = c * sum(N[core] .* I[core]) * dx
lin_I = sum(@. K - r * I[core]) * dx
jump_I = DI * ((I[wrap(ctr+off+1)] - I[wrap(ctr+off)]) / dx - (I[wrap(ctr-off)] - I[wrap(ctr-off-1)]) / dx)
src_R = sum(@. l * c * N[core] * I[core] - dd * N[core] * R[core]) * dx
lin_R = sum(@. -r * R[core]) * dx
jump_R = -DR * ((R[wrap(ctr+off+1)] - R[wrap(ctr+off)]) / dx - (R[wrap(ctr-off)] - R[wrap(ctr-off-1)]) / dx)
@printf("\njumps over a window of +-%.4f:\n", off*dx)
@printf("   I:  uptake=%.5g  linear=%+.5g  net=%.5g   DI*[I_x]=%.5g   ratio=%.4f\n",
    uptake_I, -lin_I, uptake_I - lin_I, jump_I, jump_I / (uptake_I - lin_I))
@printf("   R:  source=%.5g  linear=%+.5g  net=%.5g   -DR*[R_x]=%.5g   ratio=%.4f\n",
    src_R, lin_R, src_R + lin_R, jump_R, jump_R / (src_R + lin_R))
sigI = -(uptake_I - lin_I)     # signed point-source strengths for the outer problem
sigR = src_R + lin_R

# ---- the virial identity -----------------------------------------------------
# DN N'' + mu N = 0  ==>  DN * int(N_x^2) = int(mu N^2), exactly, for any spike
# steady state. Shape independent, so it is a clean check on the whole picture.
Nx = [(N[wrap(i+1)] - N[wrap(i-1)]) / (2dx) for i in win]
lhs = DN * sum(Nx .^ 2) * dx
rhs = sum(mu[win] .* N[win] .^ 2) * dx
@printf("\nvirial:  DN*int(N_x^2)=%.6g   int(mu N^2)=%.6g   ratio=%.5f\n", lhs, rhs, lhs/rhs)
@printf("         gaussian corollary mu0*x0^2 = DN:  %.5g vs %.5g  (ratio %.3f)\n",
    mu0 * x0_mom^2, DN, mu0 * x0_mom^2 / DN)

# ---- inner-scale predictions -------------------------------------------------
s_airy = (DN / alpha)^(1/3)
E_airy = 1.018793 * (DN * alpha^2)^(1/3)      # even ground state of -psi''+|z|psi
@printf("\ntriangular-well (sharp-source) estimates:\n")
@printf("      s=(DN/alpha)^(1/3)=%.5g   vs measured x0=%.5g  (ratio %.2f)\n", s_airy, x0_mom, x0_mom/s_airy)
@printf("      E0=1.0188*(DN alpha^2)^(1/3)=%.5g  vs measured mu0=%.5g  (ratio %.2f)\n", E_airy, mu0, mu0/E_airy)

# harmonic alternative, using the curvature of the smooth (neighbour) part
kap = 2 * DN / x0_mom^4
@printf("harmonic reading: kappa implied by x0 = 2DN/x0^4 = %.4g\n", kap)
@printf("      then mu0 = sqrt(DN kappa/2) = %.5g   vs measured %.5g  (ratio %.2f)\n",
    sqrt(DN*kap/2), mu0, mu0/sqrt(DN*kap/2))

# ---- outer field shape: is it the cosh Green's function? ---------------------
lI, lR = sqrt(DI/r), sqrt(DR/r)
mid = fld(spacing/dx, 2) |> Int
gap = [wrap(ctr + j) for j in off:(round(Int, spacing/dx) - off)]
xg = (collect(off:(round(Int, spacing/dx) - off))) .* dx
# fit I - K/r = a*cosh((x - spacing/2)/lI) and R = b*cosh((x - spacing/2)/lR)
fI = @. cosh((xg - spacing/2)/lI)
fR = @. cosh((xg - spacing/2)/lR)
aI = (fI \ (I[gap] .- K/r))
bR = (fR \ R[gap])
rI = norm(I[gap] .- K/r .- aI .* fI) / norm(I[gap] .- K/r)
rR = norm(R[gap] .- bR .* fR) / norm(R[gap])
@printf("\nouter fit over the gap (lI=%.3f, lR=%.3f):\n", lI, lR)
@printf("      I - K/r = %.5g*cosh((x-d/2)/lI)   rel resid %.2e\n", aI, rI)
@printf("      R       = %.5g*cosh((x-d/2)/lR)   rel resid %.2e\n", bR, rR)

# ---- mobility ----------------------------------------------------------------
@printf("\nmobility x0^2 = %.4g   (sqrt(DN)-like, not DN-like: sqrt(DN)=%.3g)\n", x0_mom^2, sqrt(DN))
@printf("spike-to-spike scatter: masses %s\n",
    string(round.([sum(N[[wrap(cc+j) for j in -half:half]])*dx for cc in ctrs[1:min(5,end)]], digits=6)))

# ---- frozen-strength pair force ---------------------------------------------
# Treat each spike as a point sink of I of strength sigI and a point source of R
# of strength sigR, with the strengths held FIXED as positions move. On the ring
# the field of one neighbour is sig*g(x), g(x) = (l/2D)cosh((L/2-|x|)/l)/sinh(L/2l).
g(x, ell, D) = (ell / (2D)) * cosh((L/2 - abs(x)) / ell) / sinh(L / (2ell))
gp(x, ell, D) = -(1 / (2D)) * sinh((L/2 - abs(x)) / ell) / sinh(L / (2ell)) * sign(x)
cI, cR = (1 - l) * c, dd
@printf("\nfrozen-strength pair force (sigI=%.4g, sigR=%.4g):\n", sigI, sigR)
@printf("   mu from one neighbour at distance s:  %.4g*g_I(s) + %.4g*g_R(s)\n", cI*sigI, cR*sigR)
@printf("   at s=d=%.4f:  I part %+.4g   R part %+.4g   (R/I = %.1f)\n",
    spacing, cI*sigI*g(spacing,lI,DI), cR*sigR*g(spacing,lR,DR),
    abs(cR*sigR*g(spacing,lR,DR) / (cI*sigI*g(spacing,lI,DI))))
# net gradient at spike j when it is displaced by delta towards its right
# neighbour: right neighbour is then at -(d-delta), left one at +(d+delta)
for delta in (0.01, 0.1) .* spacing
    G = (cI*sigI + cR*sigR) * (gp(-(spacing - delta), lI, DI) + gp(spacing + delta, lI, DI))
    v = x0_mom^2 * G
    @printf("   displaced by %.5f (%3.0f%% of d):  G=%+.4g   v=x0^2*G=%+.4g   rate v/delta=%.4g (1/e time %.4g)\n",
        delta, 100delta/spacing, G, v, v/delta, delta/abs(v))
end

# ---- where the two channels cross over, as a function of p -------------------
# On the line g(s) ~ (ell/2D)exp(-s/ell), so with lR=sqrt(p), DR=p the R channel
# is sigR*exp(-s/sqrt(p))/(2sqrt(p)) and the I channel (1-l)*sigI*exp(-s)/2.
# They balance at s*, inside which R (attraction) wins and outside which I
# (repulsion) wins. Uses the p=1 strengths throughout, so this is the shape of
# the p dependence and not a quantitative prediction at other p.
ratio = sigR / ((1 - l) * abs(sigI))
@printf("\ncrossover distance s* (sigR/((1-l)|sigI|) = %.1f, p=1 strengths):\n", ratio)
for pp in (0.01, 0.03, 0.1, 0.3, 1.0)
    sp = sqrt(pp)
    if sp >= 1
        @printf("   p=%-6g  no crossover (R wins at every distance)\n", pp)
    else
        @printf("   p=%-6g  lR=%.3f  s*=%.4f\n", pp, sp, sp * log(ratio / sp) / (1 - sp))
    end
end
