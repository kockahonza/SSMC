using Distributions, QuickTypes

@qstruct_fp QTT2(Ns::Int, Nr::Int, m, r, Ds, Dr, num_influx_resources, K, num_used_resources, num_byproducts, c, l)
function QTT2(Ns, Nr;
    m=1.0, r=1.0, Ds=1.0, Dr=1.0,
    num_influx_resources=nothing, Kp=nothing, sparsity_influx=nothing, K=1.0,
    num_used_resources=nothing, sparsity_resources=nothing,
    num_byproducts=nothing, sparsity_byproducts=nothing,
    c=1.0, l=0.5,
    usenthreads=nothing
)
    if isa(m, Number)
        m = Dirac(m)
    elseif isa(m, Tuple) && length(m) == 2
        m = Normal(m[1], m[2])
    end
    if isa(r, Number)
        r = Dirac(r)
    elseif isa(r, Tuple) && length(r) == 2
        r = Normal(r[1], r[2])
    end
    if isa(Ds, Number)
        Ds = Dirac(Ds)
    elseif isa(Ds, Tuple) && length(Ds) == 2
        Ds = Normal(Ds[1], Ds[2])
    end
    if isa(Dr, Number)
        Dr = Dirac(Dr)
    elseif isa(Dr, Tuple) && length(Dr) == 2
        Dr = Normal(Dr[1], Dr[2])
    end
    if isa(K, Number)
        K = Dirac(K)
    elseif isa(K, Tuple) && length(K) == 2
        K = Normal(K[1], K[2])
    end
    if isa(c, Number)
        c = Dirac(c)
    elseif isa(c, Tuple) && length(c) == 2
        c = Normal(c[1], c[2])
    end
    if isa(l, Number)
        l = Dirac(l)
    elseif isa(l, Tuple) && length(l) == 2
        l = Normal(l[1], l[2])
    end

    if isnothing(num_influx_resources) && isnothing(Kp) && isnothing(sparsity_influx)
        num_influx_resources = Dirac(Nr)
    elseif isnothing(num_influx_resources)
        if isnothing(sparsity_influx)
            num_influx_resources = Binomial(Nr, Kp)
        else
            num_influx_resources = Dirac(round(Int, sparsity_influx * Nr))
        end
    elseif !isnothing(Kp) && !isnothing(sparsity_influx)
        @warn "RSGJans1 has been passed multiple kwargs for influx resources some are being ignored"
    end

    if isnothing(num_used_resources) && isnothing(sparsity_resources)
        num_used_resources = Nr
    elseif isnothing(num_used_resources)
        num_used_resources = round(Int, sparsity_resources * Nr)
    end
    if isa(num_used_resources, Number)
        num_used_resources = Dirac(num_used_resources)
    end

    if isnothing(num_byproducts) && isnothing(sparsity_byproducts)
        num_byproducts = Nr
    elseif isnothing(num_byproducts)
        num_byproducts = round(Int, sparsity_byproducts * Nr)
    end
    if isa(num_byproducts, Number)
        num_byproducts = Dirac(num_byproducts)
    end

    QTT2(Ns, Nr, m, r, Ds, Dr, num_influx_resources, K, num_used_resources, num_byproducts, c, l)
end




struct QTT{Dm,Dr,DDs,DDr,Di,DK,Ds1,Ds2,Dc,Dl}
    Ns::Int # number of strains
    Nr::Int # number of resources

    m::Dm   # upkeep energy rate distribution
    r::Dr   # resource dilution rate distribution
    Ds::DDs # strain diffusion constant distribution
    Dr::DDr # resource diffusion constant distribution

    num_influx_resources::Di # a discrete distribution for generating the number of resources with non-zero K
    K::DK       # distribution of Ks for those resources which are being added

    num_used_resources::Ds1 # discrete distribution of the number of resources each strain eats
    num_byproducts::Ds2     # discrete distribution of the number of byproducts for each consumption process

    c::Dc # distribution for those consumption rates which are not zero
    l::Dl # leakage distribution

    usenthreads::Union{Nothing,Int}
end
#     function RSGJans1(Ns, Nr;
#         m=1.0, r=1.0, Ds=1.0, Dr=1.0,
#         num_influx_resources=nothing, Kp=nothing, sparsity_influx=nothing, K=1.0,
#         num_used_resources=nothing, sparsity_resources=nothing,
#         num_byproducts=nothing, sparsity_byproducts=nothing,
#         c=1.0, l=0.5,
#         usenthreads=nothing
#     )
#         if isa(m, Number)
#             m = Dirac(m)
#         elseif isa(m, Tuple) && length(m) == 2
#             m = Normal(m[1], m[2])
#         end
#         if isa(r, Number)
#             r = Dirac(r)
#         elseif isa(r, Tuple) && length(r) == 2
#             r = Normal(r[1], r[2])
#         end
#         if isa(Ds, Number)
#             Ds = Dirac(Ds)
#         elseif isa(Ds, Tuple) && length(Ds) == 2
#             Ds = Normal(Ds[1], Ds[2])
#         end
#         if isa(Dr, Number)
#             Dr = Dirac(Dr)
#         elseif isa(Dr, Tuple) && length(Dr) == 2
#             Dr = Normal(Dr[1], Dr[2])
#         end
#         if isa(K, Number)
#             K = Dirac(K)
#         elseif isa(K, Tuple) && length(K) == 2
#             K = Normal(K[1], K[2])
#         end
#         if isa(c, Number)
#             c = Dirac(c)
#         elseif isa(c, Tuple) && length(c) == 2
#             c = Normal(c[1], c[2])
#         end
#         if isa(l, Number)
#             l = Dirac(l)
#         elseif isa(l, Tuple) && length(l) == 2
#             l = Normal(l[1], l[2])
#         end
#
#         if isnothing(num_influx_resources) && isnothing(Kp) && isnothing(sparsity_influx)
#             num_influx_resources = Dirac(Nr)
#         elseif isnothing(num_influx_resources)
#             if isnothing(sparsity_influx)
#                 num_influx_resources = Binomial(Nr, Kp)
#             else
#                 num_influx_resources = Dirac(round(Int, sparsity_influx * Nr))
#             end
#         elseif !isnothing(Kp) && !isnothing(sparsity_influx)
#             @warn "RSGJans1 has been passed multiple kwargs for influx resources some are being ignored"
#         end
#
#         if isnothing(num_used_resources) && isnothing(sparsity_resources)
#             num_used_resources = Nr
#         elseif isnothing(num_used_resources)
#             num_used_resources = round(Int, sparsity_resources * Nr)
#         end
#         if isa(num_used_resources, Number)
#             num_used_resources = Dirac(num_used_resources)
#         end
#
#         if isnothing(num_byproducts) && isnothing(sparsity_byproducts)
#             num_byproducts = Nr
#         elseif isnothing(num_byproducts)
#             num_byproducts = round(Int, sparsity_byproducts * Nr)
#         end
#         if isa(num_byproducts, Number)
#             num_byproducts = Dirac(num_byproducts)
#         end
#
#         new{
#             typeof(m),typeof(r),typeof(Ds),typeof(Dr),typeof(num_influx_resources),typeof(K),
#             typeof(num_used_resources),typeof(num_byproducts),typeof(c),typeof(l)
#         }(Ns, Nr, m, r, Ds, Dr, num_influx_resources, K, num_used_resources, num_byproducts, c, l, usenthreads)
#     end
