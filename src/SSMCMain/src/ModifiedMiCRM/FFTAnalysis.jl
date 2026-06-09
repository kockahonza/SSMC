module FFTAnalysis

using Reexport
@reexport using ..ModifiedMiCRM

using FFTW

import ..MinimalModelSemisymbolic

################################################################################
# Random bits
################################################################################
function get_total_biomass_1d(u::AbstractMatrix, Ns)
    sN = size(u, 2)
    biom = Vector{eltype(u)}(undef, sN)
    for x in 1:sN
        biom[x] = 0.
        for i in 1:Ns
            biom[x] += u[i, x]
        end
    end
    biom
end
export get_total_biomass_1d

function get_dominant_lengthscale(ys, dx)
    sN = length(ys)
    f = fft(ys)

    halfi = div(sN, 2, RoundUp)
    sP = abs2.(f[2:halfi]) .* (dx / sN)

    Pmax, iPmax = findmax(sP)

    freqs = fftfreq(sN, 1. / dx)
    1 / freqs[iPmax+1]
end
export get_dominant_lengthscale

function get_dominant_lengthscale2(ys, dx)
    sN = length(ys)
    f = fft(ys)

    halfi = div(sN, 2, RoundUp)
    sP = abs2.(f[2:halfi]) .* (dx / sN)
    nsP = sP ./ sum(sP)
    freqs = fftfreq(sN, 1. / dx)[2:halfi]
    lengths = 1 ./ freqs

    sum(nsP .* lengths)
end
export get_dominant_lengthscale2

function get_sP(ys::AbstractVector, dx)
    sN = length(ys)
    f = fft(ys)

    halfi = div(sN, 2, RoundUp)
    sP = abs2.(f[2:halfi]) .* (dx / sN)
end
export get_sP

function get_sp(ys, dx)
    sp = get_sP(ys, dx)
    @. sp = sqrt(sp)
    sp
end
export get_sp

function get_fftfactor1(ys, dx)
    sP = get_sP(ys, dx)
    maximum(sP) / mean(sP)
end
export get_fftfactor1

function get_maxreldiff(u::AbstractMatrix, ode_ss::AbstractVector)
    maximum(abs, (u .- ode_ss) ./ ode_ss)
end
export get_maxreldiff

function get_maxdiff(u::AbstractMatrix, ode_ss::AbstractVector)
    maximum(abs, u .- ode_ss)
end
export get_maxdiff

################################################################################
# Useful plots
################################################################################
function make_fft_dl_plot!(where, s, N, dx)
    ax1 = Axis(where;
        xscale=log10,
        yticklabelcolor=Makie.wong_colors()[1],
        ylabelcolor=Makie.wong_colors()[1],
        ylabel="Dominant lengthscale",
    )
    ax2 = Axis(where;
        xscale=log10,
        yaxisposition=:right,
        yticklabelcolor=Makie.wong_colors()[2],
        ylabelcolor=Makie.wong_colors()[2],
        ylabel="FFT dominance metric",
    )
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax2)
    
    Ls = map(s.u) do u
        get_dominant_lengthscale(get_total_biomass_1d(u, N), dx)
    end;
    lines!(ax1, s.t, Ls;
        color=Makie.wong_colors()[1],
    )

    @show extrema(Ls) extrema(s.t)

    fft_metric = map(s.u) do u
        sP = get_sP(get_total_biomass_1d(u, N), dx)
        maximum(sP) / mean(sP)
    end;
    lines!(ax2, s.t, fft_metric;
        color=Makie.wong_colors()[2],
    )
    
    xlims!(ax1, s.t[3]/2, s.t[end]*2)
    
    ax1, ax2
end
function make_fft_dl_plot(s, N, dx)
    fig = Figure()
    make_fft_dl_plot!(fig[1, 1], s, N, dx)
    fig
end
export make_fft_dl_plot!, make_fft_dl_plot

function make_fft_dl_plot2!(where, s, N, dx)
    gl = GridLayout(where)
    ax1 = Axis(gl[1,1];
        xscale=log10,
        ylabel="Dominant lengthscale",
    )
    ax2 = Axis(gl[2,1];
        xscale=log10,
        ylabel="FFT dominance metric",
    )
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1; grid=false)
    
    Ls = map(s.u) do u
        get_dominant_lengthscale(get_total_biomass_1d(u, N), dx)
    end;
    lines!(ax1, s.t, Ls;
    )

    @show extrema(Ls) extrema(s.t)

    fft_metric = map(s.u) do u
        sP = get_sP(get_total_biomass_1d(u, N), dx)
        maximum(sP) / mean(sP)
    end;
    lines!(ax2, s.t, fft_metric;
    )
    
    xlims!(ax1, s.t[3]/2, s.t[end]*2)

    rowgap!(gl, 6.)
    
    ax1, ax2
end
function make_fft_dl_plot2(s, N, dx)
    fig = Figure()
    make_fft_dl_plot2!(fig[1, 1], s, N, dx)
    fig
end
export make_fft_dl_plot2!, make_fft_dl_plot2

function make_dl_maxdiff_plot!(where, s, ode_ss, N, dx)
    gl = GridLayout(where)
    ax1 = Axis(gl[1,1];
        xscale=log10,
        ylabel="Dominant lengthscale",
    )
    ax2 = Axis(gl[2,1];
        xscale=log10,
        ylabel="Max difference from ODE levels",
    )
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1; grid=false)
    
    Ls = map(s.u) do u
        get_dominant_lengthscale(get_total_biomass_1d(u, N), dx)
    end;
    lines!(ax1, s.t, Ls;
    )

    @show extrema(Ls) extrema(s.t)

    maxdiffs = get_maxdiff.(s.u, Ref(ode_ss))
    lines!(ax2, s.t, maxdiffs;
    )
    
    xlims!(ax1, s.t[3]/2, s.t[end]*2)

    rowgap!(gl, 6.)
    
    ax1, ax2
end
function make_dl_maxdiff_plot(s, ode_ss, N, dx)
    fig = Figure()
    make_dl_maxdiff_plot!(fig[1, 1], s, ode_ss, N, dx)
    fig
end
export make_dl_maxdiff_plot!, make_dl_maxdiff_plot


################################################################################
# FFT based solver exit callbacks
################################################################################
function make_fft_callback1(Ns, sN, dx, exit_ratio)
    halfi = div(sN, 2, RoundUp)
    f = Vector{ComplexF64}(undef, sN)
    sP = Vector{Float64}(undef, halfi-1)
    
    cc = let Ns=Ns, f=f, sP=sP, halfi=halfi
        function(u, t, i)
            for x in 1:sN
                f[x] = 0.
                for i in 1:Ns
                    f[x] += u[i,x]
                end
            end
            fft!(f)
            
            for i in 1:(halfi-1)
                sP[i] = abs2(f[i+1]) * (dx / sN)
            end
        
            Pmax = maximum(sP)
            Pmean = mean(sP)
            
            (Pmax / Pmean) > exit_ratio
        end
    end

    DiscreteCallback(cc, i->terminate!(i))
end
export make_fft_callback1

function make_fft_callback2(Ns, sN, dx, exit_ratio)
    halfi = div(sN, 2, RoundUp)
    f = Vector{ComplexF64}(undef, sN)
    sP = Vector{Float64}(undef, halfi-1)
    
    cc = let Ns=Ns, f=f, sP=sP, halfi=halfi
        function(u, t, i)
            for x in 1:sN
                f[x] = 0.
                for i in 1:Ns
                    f[x] += u[i,x]
                end
            end
            fft!(f)
            
            for i in 1:(halfi-1)
                sP[i] = abs2(f[i+1]) * (dx / sN)
            end
        
            Pmax = maximum(sP)
            Pmean = mean(sP)
            
            (Pmax / Pmean) - exit_ratio
        end
    end

    ContinuousCallback(cc, i->terminate!(i), i->terminate!(i, ReturnCode.Failure))
end
export make_fft_callback2

function make_perturb_callback(sN, ode_ss, exit_level)
    reldiffs = Matrix{Float64}(undef, length(ode_ss), sN)
    
    cc = let ode_ss=ode_ss, reldiffs=reldiffs
        function(u, t, i)
            @. reldiffs = (u - ode_ss) / ode_ss
            maximum(abs, reldiffs) - exit_level
        end
    end

    ContinuousCallback(cc, i->terminate!(i), i->terminate!(i, ReturnCode.Failure))
end
export make_perturb_callback

function make_perturb_callback2(sN, ode_ss, exit_level)
    diffs = Matrix{Float64}(undef, length(ode_ss), sN)
    
    cc = let ode_ss=ode_ss, diffs=diffs
        function(u, t, i)
            @. diffs = u - ode_ss
            maximum(abs, diffs) - exit_level
        end
    end

    ContinuousCallback(cc, i->terminate!(i), i->terminate!(i, ReturnCode.Failure))
end
export make_perturb_callback2

end
