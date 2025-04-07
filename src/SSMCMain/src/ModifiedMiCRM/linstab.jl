function do_linstab_for_ks(ks, p::MMiCRMParams{Ns,Nr,F}, Ds, ss; kwargs...) where {Ns,Nr,F}
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

function eigen_sortby_reverse(l::Complex)
    (-real(l), imag(l))
end
function eigen_sortby_reverse(l::Real)
    -l
end
export eigen_sortby_reverse

function linstab_make_lambda_func(p::MMiCRMParams{Ns,Nr}, ss, Ds=nothing; kwargs...) where {Ns,Nr}
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
function linstab_make_full_func(p::MMiCRMParams{Ns,Nr}, ss, Ds=nothing; kwargs...) where {Ns,Nr}
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

function make_M1!(M1, p::MMiCRMParams{Ns,Nr}, ss) where {Ns,Nr}
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
function make_M1(p::MMiCRMParams{Ns,Nr,F}, args...) where {Ns,Nr,F}
    M1 = Matrix{F}(undef, Ns + Nr, Ns + Nr)
    make_M1!(M1, p, args...)
    M1
end
export make_M1!, make_M1

function make_M(M1, k, Ds)
    M1 + Diagonal(-(k^2) .* Ds)
end
function make_M(p::MMiCRMParams, k, ss, Ds)
    make_M(make_M1(p, ss), k, Ds)
end
export make_M

"""Returns the number of non-decaying modes"""
function find_number_nondec_modes(M; threshold=eps(eltype(M)))
    e = eigen(M)
    count(x -> real(x) > -threshold, e.values)
end
function find_number_nondec_modes(args...; kwargs...)
    M = make_M(args...)
    find_number_nondec_modes(M; kwargs...)
end
export find_number_nondec_modes

"""Returns the non-decaying modes"""
function find_nondec_modes(M; threshold=eps(eltype(M)))
    e = eigen(M)

    Ctype = Complex{eltype(M)}
    lambdas = Ctype[]
    modes = Vector{Ctype}[]

    for i in 1:length(e.values)
        if real(e.values[i]) > -threshold
            push!(lambdas, e.values[i])
            push!(modes, e.vectors[:, i])
        end
    end

    lambdas, modes
end
function find_nondec_modes(args...; kwargs...)
    M = make_M(args...)
    find_nondec_modes(M; kwargs...)
end
export find_nondec_modes

"""Returns the number of non-decaying modes"""
function find_number_growing_modes(M; threshold=eps(eltype(M)))
    e = eigen(M)
    count(x -> real(x) > threshold, e.values)
end
function find_number_growing_modes(args...; kwargs...)
    M = make_M(args...)
    find_number_growing_modes(M; kwargs...)
end
export find_number_growing_modes
