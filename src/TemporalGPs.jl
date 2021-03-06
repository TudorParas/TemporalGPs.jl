module TemporalGPs

    using BlockDiagonals, FillArrays, Kronecker, LinearAlgebra, Random,
        StaticArrays, Stheno, Strided, Zygote, ZygoteRules

    using FillArrays: AbstractFill
    using Kronecker: KroneckerProduct

    import Stheno: mean, cov, pairwise, logpdf, AV, AM

    export to_sde, SArrayStorage, ArrayStorage, RegularSpacing

    # Various bits-and-bobs. Often commiting some type piracy.
    include(joinpath("util", "zygote_rules.jl"))
    include(joinpath("util", "gaussian.jl"))
    include(joinpath("util", "mul.jl"))
    include(joinpath("util", "storage_types.jl"))
    include(joinpath("util", "regular_data.jl"))

    # Linear-Gaussian State Space Models.
    include(joinpath("models", "gauss_markov.jl"))
    include(joinpath("models", "lgssm.jl"))

    include(joinpath("models", "immutable_inference.jl"))
    include(joinpath("models", "immutable_inference_pullbacks.jl"))

    include(joinpath("models", "mutable_inference.jl"))
    include(joinpath("models", "mutable_inference_pullbacks.jl"))

    include(joinpath("models", "scalar_lgssm.jl"))

    # Converting GPs to Linear-Gaussian SSMs.
    include(joinpath("gp", "to_gauss_markov.jl"))
    include(joinpath("gp", "lti_sde.jl"))

    # Converting space-time GPs to Linear-Gaussian SSMs.
    include(joinpath("space_time", "rectilinear_grid.jl"))
    include(joinpath("space_time", "separable_kernel.jl"))
    include(joinpath("space_time", "to_gauss_markov.jl"))
end # module
