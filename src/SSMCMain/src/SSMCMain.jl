module SSMCMain

using Reexport
@reexport using Printf
@reexport using SciMLBase, DifferentialEquations, NonlinearSolve, Optimization
@reexport using StaticArrays
@reexport using LinearAlgebra
@reexport using StatsBase
@reexport using Distributions
@reexport using Makie, LaTeXStrings
@reexport using PrettyTables
@reexport using Geppetto
@reexport using NamedArrays
@reexport using DimensionalData
@reexport using EnumX

using REPL.TerminalMenus
using Logging

import Base: show

projectdir(args...) = abspath(joinpath(pkgdir(SSMCMain), "../../", args...))
scriptsdir(args...) = joinpath(projectdir(), "scripts", args...)
datadir(args...) = joinpath(projectdir(), "data", args...)
export projectdir, scriptsdir, datadir

################################################################################
# Other modules/files
################################################################################
include("util.jl")

# Main stuff!
include("ModifiedMiCRM/ModifiedMiCRM.jl")

# Old stuff, just keeping for old times
include("archive/BasicMiCRM.jl")

end # module SSMCMain
