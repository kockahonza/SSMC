include("base.jl")
st = load("state1.jld2")
# the failure modes that matter most for "is this really arrested" go first;
# `tight` last and capped, since rtol=1e-12 crawls and the /tmp four-variant
# test already cleared the tolerances at sN=2500.
run_restart(st, "long_T";       T=1e12)
run_restart(st, "perturb_1e-6"; perturb=1e-6)
run_restart(st, "perturb_1e-3"; perturb=1e-3)
run_restart(st, "nopos";        positivity=false)
run_restart(st, "fine_long";    sN=20000, T=1e12, maxtime=2*60*60)
run_restart(st, "tight";        abstol=1e-12, reltol=1e-12, maxtime=45*60)
