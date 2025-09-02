################################################################################
# Physics/base, constructing M1 and M
################################################################################
function make_M1!(M1, p::AbstractMMiCRMParams, ss)
    Ns, Nr = get_Ns(p)

    for i in 1:Ns
        # 0 everywhere
        for j in 1:Ns
            M1[i, j] = 0.0
        end
        # add things to the diagonal - this is T
        for a in 1:Nr
            M1[i, i] += p.g[i] * (1 - p.l[i, a]) * p.w[a] * p.c[i, a] * ss[Ns+a]
        end
        M1[i, i] -= p.g[i] * p.m[i]
    end
    for i in 1:Ns
        for a in 1:Nr
            # this is U
            M1[i, Ns+a] = p.g[i] * (1 - p.l[i, a]) * p.w[a] * p.c[i, a] * ss[i]
            # this is V
            M1[Ns+a, i] = 0.0
            for b in 1:Nr
                M1[Ns+a, i] += p.D[i, a, b] * p.l[i, b] * (p.w[b] / p.w[a]) * p.c[i, b] * ss[Ns+b]
            end
            M1[Ns+a, i] -= p.c[i, a] * ss[Ns+a]
        end
    end
    for a in 1:Nr
        for b in 1:Nr
            # makes sure everything is initialized
            M1[Ns+a, Ns+b] = 0.0
            # this does the complex part of W
            for i in 1:Ns
                M1[Ns+a, Ns+b] += p.D[i, a, b] * p.l[i, b] * (p.w[b] / p.w[a]) * p.c[i, b] * ss[i]
            end
        end
        # and add the diagonal bit of W
        for i in 1:Ns
            M1[Ns+a, Ns+a] -= p.c[i, a] * ss[i]
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
# Doing linstab
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

"""
Optimized function that will test if a system has an instability by scanning
a range of preselected ks.
"""
struct LinstabScanTester{F}
    ks::Vector{F}
    M1::Matrix{F}
    M::Matrix{F}
    threshold::F
    function LinstabScanTester(ks, N, threshold=2 * eps())
        M1 = Matrix{eltype(ks)}(undef, N, N)
        M = Matrix{eltype(ks)}(undef, N, N)
        new{eltype(ks)}(ks, M1, M, threshold)
    end
end
function LinstabScanTester(ks, p::AbstractMMiCRMParams, threshold=2 * eps())
    LinstabScanTester(ks, sum(get_Ns(p)), threshold)
end
function (lst::LinstabScanTester)(sp::AbstractSMMiCRMParams, ss)
    make_M1!(lst.M1, sp, ss)
    for k in lst.ks
        lst.M .= lst.M1
        M1_to_M!(lst.M, get_Ds(sp), k)
        evals = eigvals!(lst.M)
        if any(l -> real(l) > lst.threshold, evals)
            return true
        end
    end
    return false
end
function copy(lst::LinstabScanTester)
    LinstabScanTester(lst.ks, size(lst.M)[1], lst.threshold)
end
export LinstabScanTester

struct LinstabScanTester2{F}
    zerothr::F
    peakthr::F
    ks::Vector{F}
    M1::Matrix{F}
    M::Matrix{F}
    mrls::Vector{F}
    function LinstabScanTester2(kmax, Nks, N;
        zerothr=1000 * eps(),      # values +- this are considered 0 in linstab analysis
        peakthr=zerothr,         # values above this can be considered a peak
    )
        ks = range(0, kmax; length=Nks)[2:end]
        new{typeof(kmax)}(zerothr, peakthr,
            ks,
            Matrix{typeof(kmax)}(undef, N, N), Matrix{typeof(kmax)}(undef, N, N), Vector{typeof(kmax)}(undef, length(ks))
        )
    end
end
function LinstabScanTester2(ps::AbstractMMiCRMParams, args...; kwargs...)
    LinstabScanTester2(args..., sum(get_Ns(ps)); kwargs...)
end
function (lst::LinstabScanTester2)(sp::AbstractSMMiCRMParams, ss)
    # handle the k=0 case
    make_M1!(lst.M1, sp, ss)
    k0mrl = maximum(real, eigvals!(lst.M1))

    # calculate mrls
    make_M1!(lst.M1, sp, ss)
    for (ki, k) in enumerate(lst.ks)
        lst.M .= lst.M1
        M1_to_M!(lst.M, get_Ds(sp), k)
        evals = eigvals!(lst.M)
        lst.mrls[ki] = maximum(real, evals)
    end

    # evaluate the mrl results
    maxmrl, maxi = findmax(lst.mrls)

    maxmrl_positive = maxmrl > lst.peakthr

    is_separated = false
    for intermediate_mrl in lst.mrls[1:maxi]
        if intermediate_mrl < -lst.zerothr
            is_separated = true
            break
        end
    end

    if k0mrl < -lst.zerothr # this is the ideal case
        if !maxmrl_positive
            code = 1
        else
            code = 2
        end
    elseif k0mrl < lst.zerothr # this can happen when there are interchangeable species, or when close to numerical issues
        if !maxmrl_positive
            code = 11
        else
            if is_separated # k0 is sketchy but we have a separated positive peak
                code = 12
            else # the largest peak is connected to a positive k0 - clearly messy
                code = 13
            end
        end
    else # something is definitely off here, however still do the same analysis for extra info
        if !maxmrl_positive
            code = 21
        else
            if is_separated
                code = 22
            else
                code = 23
            end
        end
    end

    code, k0mrl, maxmrl, maxmrl_positive && is_separated
end
export LinstabScanTester2

################################################################################
# The K polynomial functions, aka finding modes by solving for 0 evals
################################################################################
include("linstab_Kpolynomial.jl")
