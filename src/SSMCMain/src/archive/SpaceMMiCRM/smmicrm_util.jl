function print_spatial_solution_stats(sol)
    if SciMLBase.successful_retcode(sol.retcode)
        println(sol.retcode)
    else
        @error (@sprintf "Non successful retcode of %s" string(sol.retcode))
    end
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
