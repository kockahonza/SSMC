using SSMCMain, SSMCMain.ModifiedMiCRM.RandomSystems

# using LinearAlgebra
# using Random
# using Base.Iterators
# using Base.Threads

using Logging

using OhMyThreads
using Distributions
using JLD2
using Geppetto
import NOMAD

# Params for the optimizer

function calc_score(cm, top=[2], bottom=[1, 2, 101])
    sum(x -> get(cm, x, 0.0), top) / sum(x -> get(cm, x, 0.0), bottom)
end

function do_run_unimodal_c(
    N,
    me, mev,
    Dse, Dsev, Dre, Drev,
    la, lb, ce, cev, sr, sb,
    K;
    num_repeats=100,
    kmax=50,
    Nks=1000,
    disable_log=true,
    # analyzing result codes
    min_good_data_ratio=0.9,
    bad_data_val=-1.0,
    score_func=calc_score,
    timelimit=nothing
)
    rsg = RSGJans1(N, N;
        m=base10_lognormal(me, mev),
        Ds=base10_lognormal(Dse, Dsev),
        Dr=base10_lognormal(Dre, Drev),
        # use a unimodal c dist
        l=Beta(la, lb),
        c=base10_lognormal(ce, cev),
        sparsity_resources=sr,
        sparsity_byproducts=sb,
        # fix a single influx resource
        num_influx_resources=Dirac(1),
        K=Dirac(K),
    )

    logger_to_use = disable_log ? NullLogger() : current_logger
    result_codes = with_logger(logger_to_use) do
        example_do_rg_run2(rsg, num_repeats, kmax, Nks; timelimit)
    end

    cm = countmap(result_codes)

    good_codes = sum(x -> get(cm, x, 0.0), [1, 2, 101])
    good_ratio = good_codes / num_repeats

    score = score_func(cm)

    if good_ratio < min_good_data_ratio
        @warn "Getting less than $(min_good_data_ratio*100)% good runs!! cm is $cm"
        if !isnothing(bad_data_val)
            score = bad_data_val
        end
    end

    score, cm
end

function main1(;
    N=10,
    num_starts=5,
    num_prescreens=50,
    num_repeats=1000,
    max_nomad_time=60 * 60 * 1,
    max_single_solver_time=5,
    # granularity=1e-3,
)
    BLAS.set_num_threads(1)

    function do_run(u0)
        do_run_unimodal_c(N, u0..., 1.0;
            num_repeats,
            timelimit=max_single_solver_time,
        )
    end

    function gen_u0()
        [
            rand(Uniform(-10.0, 10.0)), 0.0, # m
            -12.0, 0.0, 0.0, 0.0, # diffs 
            10 * rand(), 10 * rand(), # l
            rand(Uniform(-10.0, 10.0)), 0.0, # c
            rand(), rand(), # sparsities
        ]
    end

    trajectories = []

    for i in 1:num_starts
        traj = []

        # Prescreening
        @info "Prescreening $i"
        flush(stdout)
        if !isnothing(num_prescreens)
            found_u0 = false
            for _ in 1:num_prescreens
                u0 = gen_u0()
                score, cm = do_run(u0)
                if score > 0.0
                    found_u0 = true
                    break
                end
                push!(traj, (copy(u0), score, cm))
            end
            if !found_u0
                @warn "Could not find a satisfactory u0 during prescreening"
            end
        else
            u0 = gen_u0()
        end

        # NOMAD run6
        @info "Starting NOMAD run $i"
        flush(stdout)
        function nomad_func(u)
            score, cm = do_run(u0)

            push!(traj, (copy(u), score, cm))

            (true, true, -score) # - as we want to maximize not minimize
        end

        max1 = 10.0
        max2 = 100.0
        max3 = 100.0

        np = NOMAD.NomadProblem(12, 1, ["OBJ"], nomad_func;
            lower_bound=[
                -max1, 0.0,
                -max2, 0.0, -max2, 0.0,
                0.0, 0.0,
                -max1, 0.0,
                0.0, 0.0
            ],
            upper_bound=[
                max1, Inf,
                max2, Inf, max2, Inf,
                max3, max3,
                max1, Inf,
                1.0, 1.0
            ],
            # granularity=fill(granularity, 12)
        )
        np.options.max_time = max_nomad_time
        np.options.display_all_eval = true
        np.options.display_stats = ["OBJ", "SOL", "BBE", "BBO"]

        GC.gc()
        @time s = NOMAD.solve(np, u0)

        push!(trajectories, traj)
        @info "Finished run $i"
        flush(stdout)
    end

    save_object("N$(N)_unic_" * randname() * ".jld2", trajectories)
    trajectories
end
