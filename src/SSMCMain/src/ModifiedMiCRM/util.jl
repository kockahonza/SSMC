function print_spatial_solution_stats(sol)
    println(sol.retcode)
    @printf "nf is %d\n" sol.destats.nf
    @printf "solution has %d saved states\n" length(sol.u)
    max_resid = maximum(abs, uninplace(sol.prob.f)(sol.u[end], sol.prob.p))
    @printf "max resid is %g\n" max_resid

    mm = minimum(minimum, sol.u)
    if mm < 0
        @warn (@sprintf "reaching negative values, minimum is %g" mm)
    end
end
export print_spatial_solution_stats

function get_space_axes(space_size, dx=1.0; collect=true)
    dx = smart_val(dx, nothing, length(space_size))
    xx = tuple(((dx/2):dx:((s*dx-dx/2)) for (s, dx) in zip(space_size, dx))...)
    if collect
        map(Base.collect, xx)
    else
        xx
    end
end
get_space_axes(sample_array, args...; kwargs...) = get_space_axes(size(sample_array), args...; kwargs...)
export get_space_axes

function get_u_axes(u, args...; kwargs...)
    get_space_axes(size(u)[2:end], args...; kwargs...)
end
export get_u_axes

function resample_cartesian_u(u, space::CartesianSpace, Ns...)
    if length(Ns) == 1
        Ns = Ns[1]
    else
        Ns = collect(Ns)
    end
    Ns = smart_val(Ns, nothing, ndims(u) - 1)

    ranges = get_u_axes(u, space.dx)
    new_ranges = [LinRange(or[1], or[end], N) for (or, N) in zip(ranges, Ns)]
    new_dx = [r[2] - r[1] for r in new_ranges]
    new_space = change_cartesianspace_dx(space, new_dx)

    ru = similar(u, size(u)[1], Ns...)

    for i in axes(u, 1)
        old_ui = selectdim(u, 1, i)
        ii = linear_interpolation(ranges, old_ui)
        selectdim(ru, 1, i) .= ii(new_ranges...)
    end

    ru, new_space
end
export resample_cartesian_u

function clamp_ss!(ss, val=eps())
    for i in eachindex(ss)
        x = ss[i]
        if x < 0.0
            ss[i] = val
            if x < -100 * eps()
                @warn (@sprintf "clamping %g to 0 which is more than %g" x 100 * eps())
            end
        end
    end
    ss
end
function clamp_ss(ss, args...)
    ss = copy(ss)
    clamp_ss!(ss, args...)
end
export clamp_ss!, clamp_ss

function base10_lognormal(b10m, b10std)
    LogNormal(b10m * log(10), b10std * log(10))
end
export base10_lognormal

function num_survivors_in_space(u, Ns, threshold=1e5 * eps(eltype(u)))
    num = 0
    for i in 1:Ns
        us = @view u[i, :]
        for v in us
            if v > threshold
                num += 1
                break
            end
        end
    end
    num
end
export num_survivors_in_space

"""
    reduce_extinct(ps, hss, threshold)

Drop every strain whose homogeneous steady state abundance is below `threshold`,
then drop every resource that none of the surviving strains either takes up or
produces. Returns `(; params, hss, keep_s, keep_r)` for the reduced system.

Zeroing a dead strain in the initial condition is not enough to keep it out: the
solver reintroduces it at roundoff level and any strain with a positive invasion
fitness grows that back to O(1) over a long run. Taking it out of the state
vector is the only hard guarantee.

Resource `a` counts as used by strain `i` if `i` takes it up (`c[i,a] != 0`) or
leaks into it from something it does take up (`D[i,a,b] * l[i,b] * c[i,b] != 0`
for some `b`, matching the production term in `mmicrmfunc!`). The second test
already implies `b` itself is kept, since `c[i,b] != 0` is exactly the uptake
test, so a single pass suffices - no iterating to a fixed point.

Note this also drops an influx resource (`K[a] != 0`) that no survivor touches.
Such a resource is decoupled from the rest of the system and just sits at
`K[a]/r[a]`, so removing it does not change any other field's dynamics, but it
does mean `keep_r` is needed to map reduced resource indices back to the
original ones.

`hss` is the full `[strains; resources]` state vector; the returned one is that
same state restricted to the kept indices, so it remains a steady state of the
reduced system.
"""
function reduce_extinct(ps::BMMiCRMParams, final_state, threshold)
    Ns, Nr = get_Ns(ps)
    if length(final_state) != Ns + Nr
        throw(ArgumentError(@sprintf(
            "final_state has length %d, expected %d for a %d strain, %d resource system",
            length(final_state), Ns + Nr, Ns, Nr)))
    end

    keep_s = [i for i in 1:Ns if final_state[i] >= threshold]
    uses(i, a) = !iszero(ps.c[i, a]) || any(1:Nr) do b
        !iszero(ps.D[i, a, b]) && !iszero(ps.l[i, b]) && !iszero(ps.c[i, b])
    end
    keep_r = [a for a in 1:Nr if any(i -> uses(i, a), keep_s)]

    rps = BMMiCRMParams(
        ps.g[keep_s], ps.w[keep_r], ps.m[keep_s],
        ps.K[keep_r], ps.r[keep_r],
        ps.l[keep_s, keep_r], ps.c[keep_s, keep_r], ps.D[keep_s, keep_r, keep_r],
        ps.usenthreads,
    )

    (; params=rps, final_state=[final_state[keep_s]; final_state[Ns .+ keep_r]], keep_s, keep_r)
end
export reduce_extinct
