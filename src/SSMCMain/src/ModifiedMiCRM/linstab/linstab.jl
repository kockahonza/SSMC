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
    M1
end
function make_M1(p::AbstractMMiCRMParams{F}, ss) where {F}
    Ns, Nr = get_Ns(p)
    M1 = Matrix{F}(undef, Ns + Nr, Ns + Nr)
    make_M1!(M1, p, ss)
    M1
end
export make_M1!, make_M1

function M1_to_M!(M1, Ds, k)
    for i in 1:length(Ds)
        M1[i, i] -= k^2 * Ds[i]
    end
end
function M1_to_M(M1, Ds, k)
    M = copy(M1)
    M1_to_M!(M, Ds, k)
    M
end
export M1_to_M!, M1_to_M

function make_M!(M, p::AbstractMMiCRMParams, Ds, ss, k)
    make_M1!(M, p, ss)
    M1_to_M!(M, Ds, k)
    M
end
make_M!(M, sp::AbstractSMMiCRMParams, ss, k) = make_M!(M, sp, get_Ds(sp), ss, k)
export make_M!

function make_M(p::AbstractMMiCRMParams{F}, Ds, ss, k) where {F}
    Ns, Nr = get_Ns(p)
    M = Matrix{F}(undef, Ns + Nr, Ns + Nr)
    make_M!(M, p, Ds, ss, k)
    M
end
make_M(sp::AbstractSMMiCRMParams, ss, k) = make_M(sp, get_Ds(sp), ss, k)
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
function linstab_make_k_func(p::AbstractMMiCRMParams, Ds, ss;
    returnobj=:evals
)
    func = if returnobj == :evals
        M -> eigvals!(M; sortby=eigen_sortby_reverse)
    elseif returnobj == :full
        M -> eigen!(M; sortby=eigen_sortby_reverse)
    elseif returnobj == :maxeval
        M -> eigvals!(M; sortby=eigen_sortby_reverse)[1]
    else
        throw(ArgumentError(@sprintf "unrecognized returnobj %s" string(returnobj)))
    end

    let M1 = make_M1(p, ss), M = copy(M1), Ds = Ds, func = func
        function (k)
            M .= M1
            M1_to_M!(M, Ds, k)
            func(M)
            # eigvals!(M; sortby=eigen_sortby_reverse)
        end
    end
end
linstab_make_k_func(sp::AbstractSMMiCRMParams, ss) = linstab_make_k_func(sp, get_Ds(sp), ss)
export linstab_make_k_func

################################################################################
# The K polynomial functions, aka finding modes by solving for 0 evals
################################################################################
include("linstab_Kpolynomial.jl")
