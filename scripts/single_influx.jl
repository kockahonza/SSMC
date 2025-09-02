using SSMCMain, SSMCMain.ModifiedMiCRM
using SSMCMain.ModifiedMiCRM.RandomSystems

function get_rsg_unimodalc(Ns, Nr=Ns;
    me=0.0, mev=0.0,
    Dse=-12.0, Dsev=0.0, Dre=0.0, Drev=0.0,
    la=2.0, lb=2.0,
    cinflux=nothing, linflux=nothing,
    ce=0.0, cev=0.0,
    sr=0.5, sb=0.5,
    K=1.0,
)
    RSGJans1(Ns, Nr;
        m=base10_lognormal(me, mev),
        Ds=base10_lognormal(Dse, Dsev),
        Dr=base10_lognormal(Dre, Drev),
        # use a unimodal c dist
        l=Beta(la, lb),
        c=base10_lognormal(ce, cev),
        cinflux, linflux,
        sparsity_resources=sr,
        sparsity_byproducts=sb,
        # fix a single influx resource
        num_influx_resources=Dirac(1),
        K=Dirac(K),
    )
end
