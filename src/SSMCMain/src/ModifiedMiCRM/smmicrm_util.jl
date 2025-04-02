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
