################################################################################
# Physics/base, constructing M1 and M
################################################################################
function make_M1!(M1, p::AbstractMMiCRMParams, ss)
    Ns, Nr = get_Ns(p)

    Nss = @view ss[1:Ns]
    Rss = @view ss[Ns+1:Ns+Nr]

    for i in 1:Ns
        # 0 everywhere
        for j in 1:Ns
            M1[i, j] = 0.0
        end
        # add things to the diagonal - this is T
        for a in 1:Nr
            M1[i, i] += p.g[i] * (1 - p.l[i, a]) * p.w[a] * p.c[i, a] * Rss[a]
        end
        M1[i, i] -= p.g[i] * p.m[i]
    end
    for i in 1:Ns
        for a in 1:Nr
            # this is U
            M1[i, Ns+a] = p.g[i] * (1 - p.l[i, a]) * p.w[a] * p.c[i, a] * Nss[i]
            # this is V
            M1[Ns+a, i] = 0.0
            for b in 1:Nr
                M1[Ns+a, i] += p.D[i, a, b] * p.l[i, b] * (p.w[b] / p.w[a]) * p.c[i, b] * Rss[b]
            end
            M1[Ns+a, i] -= p.c[i, a] * Rss[a]
        end
    end
    for a in 1:Nr
        for b in 1:Nr
            # makes sure everything is initialized
            M1[Ns+a, Ns+b] = 0.0
            # this does the complex part of W
            for i in 1:Ns
                M1[Ns+a, Ns+b] += p.D[i, a, b] * p.l[i, b] * (p.w[b] / p.w[a]) * p.c[i, b] * Nss[i]
            end
        end
        # and add the diagonal bit of W
        for i in 1:Ns
            M1[Ns+a, Ns+a] -= p.c[i, a] * Nss[i]
        end
        M1[Ns+a, Ns+a] -= p.r[a]
    end
end
function make_M1(p::AbstractMMiCRMParams{F}, args...) where {F}
    Ns, Nr = get_Ns(p)
    M1 = Matrix{F}(undef, Ns + Nr, Ns + Nr)
    make_M1!(M1, p, args...)
    M1
end
export make_M1!, make_M1

function make_M(M1, k, Ds)
    M1 + Diagonal(-(k^2) .* Ds)
end
function make_M(p::AbstractMMiCRMParams, k, ss, Ds)
    make_M(make_M1(p, ss), k, Ds)
end
export make_M

################################################################################
# Util bits
################################################################################
function eigen_sortby_reverse(l::Complex)
    (-real(l), imag(l))
end
function eigen_sortby_reverse(l::Real)
    -l
end
export eigen_sortby_reverse

################################################################################
# Directly solving for a set of ks
################################################################################
function do_linstab_for_ks(ks, p::AbstractMMiCRMParams{F}, Ds, ss; kwargs...) where {F}
    lambda_func = linstab_make_lambda_func(p, ss, Ds; kwargs...)
    lambdas = Matrix{Complex{F}}(undef, length(ks), length(ss))
    for (i, k) in enumerate(ks)
        lambdas[i, :] .= lambda_func(k)
    end
    lambdas
end
function do_linstab_for_ks(ks, p::ODEProblem, Ds, ss=nothing; kwargs...)
    if isnothing(ss)
        ssprob = SteadyStateProblem(p)
        sssol = solve(ssprob, DynamicSS())
        if sssol.retcode != ReturnCode.Success
            @error "the steady state solver did not succeed"
        end
        ss = sssol.u
    end
    do_linstab_for_ks(ks, p.p, Ds, ss; kwargs...)
end
export do_linstab_for_ks

function linstab_make_lambda_func(p::AbstractMMiCRMParams, ss, Ds=nothing; kwargs...)
    if !isnothing(Ds)
        let M1 = make_M1(p, ss)
            function (k)
                eigvals(M1 + Diagonal(-(k^2) .* Ds); sortby=eigen_sortby_reverse, kwargs...)
            end
        end
    else
        let M1 = make_M1(p, ss)
            function (k, Ds)
                eigvals(M1 + Diagonal(-(k^2) .* Ds); sortby=eigen_sortby_reverse, kwargs...)
            end
        end
    end
end
function linstab_make_full_func(p::AbstractMMiCRMParams, ss, Ds=nothing; kwargs...)
    if !isnothing(Ds)
        let M1 = make_M1(p, ss)
            function (k)
                eigen(M1 + Diagonal(-(k^2) .* Ds); sortby=eigen_sortby_reverse, kwargs...)
            end
        end
    else
        let M1 = make_M1(p, ss)
            function (k, Ds)
                eigen(M1 + Diagonal(-(k^2) .* Ds); sortby=eigen_sortby_reverse, kwargs...)
            end
        end
    end
end
export linstab_make_lambda_func, linstab_make_full_func

function fast_linstab_evals!(M1::Matrix{F}, k::F, Ds::Vector{F}) where {F}
    for i in 1:length(Ds)
        M1[i, i] -= k^2 * Ds[i]
    end
    eigvals!(M1)
end
function fast_linstab_evals!(xx, M1, k, Ds)
    xx .= M1
    fast_linstab_evals!(xx, k, Ds)
end
export fast_linstab_evals!

################################################################################
# The K polynomial functions, aka finding modes by solving for 0 evals
################################################################################
# include("linstab_Kpolynomial.jl")

################################################################################
# Simple counters
################################################################################
"""Returns the number of non-decaying modes"""
function find_number_nondec_modes(MorM1; threshold=eps(eltype(MorM1)))
    e = eigen(MorM1)
    count(x -> real(x) > -threshold, e.values)
end
function find_number_nondec_modes(args...; kwargs...)
    M = make_M(args...)
    find_number_nondec_modes(M; kwargs...)
end
export find_number_nondec_modes

"""Returns the number of non-decaying modes"""
function find_number_growing_modes(MorM1; threshold=eps(eltype(MorM1)))
    e = eigen(MorM1)
    count(x -> real(x) > threshold, e.values)
end
function find_number_growing_modes(args...; kwargs...)
    M = make_M(args...)
    find_number_growing_modes(M; kwargs...)
end
export find_number_growing_modes
