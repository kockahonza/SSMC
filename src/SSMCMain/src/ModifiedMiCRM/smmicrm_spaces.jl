struct CartesianSpace{D,BCs,F} <: AbstractSpace
    dx::SVector{D,F}
    function CartesianSpace(dx, bcs)
        for bc in bcs
            if !isa(bc, SingleAxisBC)
                throw(ArgumentError(@sprintf "invalid boundary condition passed got a bc value %s which is not a subtype of SingleAxisBC" string(bc)))
            end
        end
        if length(dx) != length(bcs)
            throw(ArgumentError(@sprintf "the number of dimensions inferred via dx and the number of bcs do not match, they are %d and %d" length(dx) length(bcs)))
        end
        new{length(dx),typeof(Tuple(bcs)),eltype(dx)}(dx)
    end
end
ndims(_::CartesianSpace{D}) where {D} = D
export CartesianSpace

function make_cartesianspace_smart(D; dx=nothing, bcs=nothing)
    dx = smart_sval(dx, nothing, D)
    bcs = smart_sval(bcs, Periodic(), D)
    CartesianSpace(dx, bcs)
end
export make_cartesianspace_smart

# implement the actual diffusion
function _add_diffusion_1d_common!(du, u, diffusion_constants, ssize, dx)
    @inbounds for i in is
        du[:, i] .+= diffusion_constants .*
                     (u[:, mod1(i + 1, ssize)] .- 2 .* u[:, i] .+ u[:, mod1(i - 1, ssize)]) ./
                     (cs1.dx[1]^2)
    end
end

function add_diffusion!(
    du::AbstractArray{F1,2}, u::AbstractArray{F2,2},
    diffusion_constants,
    cs1::CartesianSpace{1,Tuple{Periodic}},
    usenthreads=nothing
) where {F1,F2}
    ssize = size(u)[2]
    if isnothing(usenthreads)
        @inbounds for i in 1:ssize
            du[:, i] .+= diffusion_constants .*
                         (u[:, mod1(i + 1, ssize)] .- 2 .* u[:, i] .+ u[:, mod1(i - 1, ssize)]) ./
                         (cs1.dx[1]^2)
        end
    else
        ichunks = chunks(1:ssize, usenthreads)
        @sync for (is, _) in ichunks
            @spawn begin
                @inbounds for i in is
                    du[:, i] .+= diffusion_constants .*
                                 (u[:, mod1(i + 1, ssize)] .- 2 .* u[:, i] .+ u[:, mod1(i - 1, ssize)]) ./
                                 (cs1.dx[1]^2)
                end
            end
        end
    end
end
