"""Symbolic solver using MathLink"""
module MLSolver

using Printf

using DifferentialEquations
using Symbolics, SymbolicsMathLink

function symbolic_solve_ode_ml(p::ODEProblem; raw=true)
    n = length(p.u0)
    su = [Symbolics.variable(Symbol(@sprintf "u%d" i)) for i in 1:n]
    du = if isinplace(p)
        x = similar(su)
        p.f(x, su, p.p, 0)
        x
    else
        p.f(su, p.p, 0)
    end
    ws = wcall("Solve", du .~ 0, su)
    if raw
        ws
    else
        map.(x -> Symbolics.unwrap.(x[2]), ws)
    end
end
export symbolic_solve_ode_ml

end # module MLSolver
