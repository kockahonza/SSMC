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
    function CartesianSpace{D,BCs}(dx) where {D,BCs}
        fts = fieldtypes(BCs)
        if !((BCs <: Tuple) && all(bct -> bct <: SingleAxisBC, fts))
            throw(ArgumentError(@sprintf "invalid BCs of %s" string(BCs)))
        end
        if length(fts) != D
            throw(ArgumentError(@sprintf "incorrect number of BCs, should be %s but passed BCs are %s" string(D) string(BCs)))
        end
        if length(dx) != D
            throw(ArgumentError(@sprintf "incorrect number of dx s, should be %s but passed dx is %s" string(D) string(dx)))
        end
        new{D,BCs,eltype(dx)}(dx)
    end
    CartesianSpace{BCs}(dx) where {BCs} = CartesianSpace{length(dx),BCs}(dx)
end
ndims(_::CartesianSpace{D}) where {D} = D
copy(cs::CartesianSpace) = cs
export CartesianSpace

function make_cartesianspace_smart(D; dx=nothing, bcs=nothing)
    dx = smart_sval(dx, nothing, D)
    bcs = smart_sval(bcs, Periodic(), D)
    CartesianSpace(dx, bcs)
end
export make_cartesianspace_smart

function change_cartesianspace_dx(cs::CartesianSpace{D,BCs}, dx) where {D,BCs}
    if length(dx) != D
        throw(ArgumentError(@sprintf "incorrect number of dx s, should be %s but passed dx is %s" string(D) string(dx)))
    end
    CartesianSpace{BCs}(dx)
end
export change_cartesianspace_dx

function change_cartesianspace_bcs(cs::CartesianSpace{D}, bcs) where {D}
    if length(bcs) != D
        throw(ArgumentError(@sprintf "incorrect number of BCs, should be %s but passed BCs are %s" string(D) string(bcs)))
    end
    CartesianSpace{typeof(Tuple(bcs))}(cs.dx)
end
export change_cartesianspace_bcs

################################################################################
# implement the actual diffusion
################################################################################
# 1D
function _ad_1d_bulk!(du, u, Ds, dx, is)
    @inbounds for i in is
        for ui in axes(u, 1)
            du[ui, i] += Ds[ui] * (
                u[ui, i-1] - 2 * u[ui, i] + u[ui, i+1]
            ) / (dx^2)
        end
    end
end

function add_diffusion!(
    du::AbstractArray{F1,2}, u::AbstractArray{F2,2},
    Ds,
    cs1::CartesianSpace{1,Tuple{BC}},
    usenthreads=nothing
) where {BC,F1,F2}
    ssize = size(u)[2]

    # handle edges
    if BC == Periodic
        @inbounds for ui in axes(u, 1)
            du[ui, 1] += Ds[ui] * (
                u[ui, ssize] - 2 * u[ui, 1] + u[ui, 2]
            ) / (cs1.dx[1]^2)
            du[ui, ssize] += Ds[ui] * (
                u[ui, ssize-1] - 2 * u[ui, ssize] + u[ui, 1]
            ) / (cs1.dx[1]^2)
        end
    elseif BC == Closed
        @inbounds for ui in axes(u, 1)
            du[ui, 1] += Ds[ui] * (
                -u[ui, 1] + u[ui, 2]
            ) / (cs1.dx[1]^2)
            du[ui, ssize] += Ds[ui] * (
                u[ui, ssize-1] - u[ui, ssize-1]
            ) / (cs1.dx[1]^2)
        end
    else
        throw(ErrorException("Unsupported BCS"))
    end

    bulk_is = 2:(ssize-1)

    if isnothing(usenthreads)
        _ad_1d_bulk!(du, u, Ds, cs1.dx[1], bulk_is)
    else
        ichunks = chunks(bulk_is; n=usenthreads)
        @sync for is in ichunks
            @spawn begin
                _ad_1d_bulk!(du, u, Ds, cs1.dx[1], is)
            end
        end
    end
end

# 2D
function _ad_2d_bulk!(du, u, Ds, dx, xis, yis)
    @inbounds for xi in xis
        @inbounds for yi in yis
            for ui in axes(u, 1)
                du[ui, xi, yi] += Ds[ui] * (
                    u[ui, xi-1, yi] - 2 * u[ui, xi, yi] + u[ui, xi+1, yi]
                ) / (dx^2)
            end
        end
    end
end

function _ad_2d_bulk!(du, u, Ds, dx, cis)
    @inbounds for ci in cis
        for ui in axes(u, 1)
            du[ui, ci] += Ds[ui] * (
                u[ui, ci[1]-1, ci[2]] - 2 * u[ui, ci] + u[ui, ci[1]+1, ci[2]]
            ) / (dx[1]^2)
            du[ui, ci] += Ds[ui] * (
                u[ui, ci[1], ci[2]-1] - 2 * u[ui, ci] + u[ui, ci[1], ci[2]+1]
            ) / (dx[2]^2)
        end
    end
end

function add_diffusion!(
    du::AbstractArray{F1,3}, u::AbstractArray{F2,3},
    Ds,
    cs1::CartesianSpace{2,Tuple{BCX,BCY}},
    usenthreads=nothing
) where {BCX,BCY,F1,F2}
    xsize = size(u)[2]
    ysize = size(u)[3]

    # handle edges
    if BCX == Periodic
        @inbounds for yi in 2:(ysize-1)
            for ui in axes(u, 1)
                # the x=1 face
                du[ui, 1, yi] += Ds[ui] * (
                    u[ui, xsize, yi] - 2 * u[ui, 1, yi] + u[ui, 2, yi]
                ) / (cs1.dx[1]^2)
                du[ui, 1, yi] += Ds[ui] * (
                    u[ui, 1, yi-1] - 2 * u[ui, 1, yi] + u[ui, 1, yi+1]
                ) / (cs1.dx[2]^2)
                # the x=end face
                du[ui, xsize, yi] += Ds[ui] * (
                    u[ui, xsize-1, yi] - 2 * u[ui, xsize, yi] + u[ui, 1, yi]
                ) / (cs1.dx[1]^2)
                du[ui, xsize, yi] += Ds[ui] * (
                    u[ui, xsize, yi-1] - 2 * u[ui, xsize, yi] + u[ui, xsize, yi+1]
                ) / (cs1.dx[2]^2)
            end
        end
        @inbounds for xi in 2:(xsize-1)
            for ui in axes(u, 1)
                # the y=1 face
                du[ui, xi, 1] += Ds[ui] * (
                    u[ui, xi-1, 1] - 2 * u[ui, xi, 1] + u[ui, xi+1, 1]
                ) / (cs1.dx[1]^2)
                du[ui, xi, 1] += Ds[ui] * (
                    u[ui, xi, ysize] - 2 * u[ui, xi, 1] + u[ui, xi, 2]
                ) / (cs1.dx[2]^2)
                # the y=end face
                du[ui, xi, ysize] += Ds[ui] * (
                    u[ui, xi-1, ysize] - 2 * u[ui, xi, ysize] + u[ui, xi+1, ysize]
                ) / (cs1.dx[1]^2)
                du[ui, xi, ysize] += Ds[ui] * (
                    u[ui, xi, ysize-1] - 2 * u[ui, xi, ysize] + u[ui, xi, 1]
                ) / (cs1.dx[2]^2)
            end
        end
        # corners
        for ui in axes(u, 1)
            du[ui, 1, 1] += Ds[ui] * (
                u[ui, xsize, 1] - 2 * u[ui, 1, 1] + u[ui, 2, 1]
            ) / (cs1.dx[1]^2)
            du[ui, 1, 1] += Ds[ui] * (
                u[ui, 1, ysize] - 2 * u[ui, 1, 1] + u[ui, 1, 2]
            ) / (cs1.dx[2]^2)

            du[ui, 1, ysize] += Ds[ui] * (
                u[ui, xsize, ysize] - 2 * u[ui, 1, ysize] + u[ui, 2, ysize]
            ) / (cs1.dx[1]^2)
            du[ui, 1, ysize] += Ds[ui] * (
                u[ui, 1, ysize-1] - 2 * u[ui, 1, ysize] + u[ui, 1, 1]
            ) / (cs1.dx[2]^2)

            du[ui, xsize, 1] += Ds[ui] * (
                u[ui, xsize-1, 1] - 2 * u[ui, xsize, 1] + u[ui, 1, 1]
            ) / (cs1.dx[1]^2)
            du[ui, xsize, 1] += Ds[ui] * (
                u[ui, xsize, ysize] - 2 * u[ui, xsize, 1] + u[ui, xsize, 2]
            ) / (cs1.dx[2]^2)

            du[ui, xsize, ysize] += Ds[ui] * (
                u[ui, xsize-1, ysize] - 2 * u[ui, xsize, ysize] + u[ui, 1, ysize]
            ) / (cs1.dx[1]^2)
            du[ui, xsize, ysize] += Ds[ui] * (
                u[ui, xsize, ysize-1] - 2 * u[ui, xsize, ysize] + u[ui, xsize, 1]
            ) / (cs1.dx[2]^2)
        end
    else
        throw(ErrorException("Unsupported BCS"))
    end
    if !(BCX == BCY == Periodic)
        throw(ErrorException("Unsupported BCS"))
    end

    bulk_cis = CartesianIndices((2:(xsize-1), 2:(ysize-1)))

    if isnothing(usenthreads)
        _ad_2d_bulk!(du, u, Ds, cs1.dx[1], bulk_cis)
    else
        cichunks = chunks(bulk_cis; n=usenthreads)
        @sync for cis in cichunks
            @spawn begin
                _ad_2d_bulk!(du, u, Ds, cs1.dx[1], cis)
            end
        end
    end
end

