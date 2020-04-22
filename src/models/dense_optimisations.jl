#
# decorrelate with dense arrays. Handles a lot of pre-allocating.
#

function decorrelate(model::LGSSM, ys::Vector{<:Vector{T}}, f=copy_first) where {T<:Real}
    @assert length(model) == length(ys)

    # Pre-allocate for intermediates.
    α = Vector{T}(undef, length(first(ys)))
    mf = copy(model.gmm.x0.m)
    Pf = copy(model.gmm.x0.P)
    cached_step_decorrelate! = build_step_decorrelate!(model[1], model.gmm.x0, first(ys))

    # Process first latent.
    lml, α, x = cached_step_decorrelate!(α, mf, Pf, model[1], model.gmm.x0, first(ys))
    v = f(α, x)
    vs = Vector{typeof(v)}(undef, length(model))
    vs[1] = v

    # Process remaining latents.
    @inbounds for t in 2:length(model)
        lml_, α, x = cached_step_decorrelate!(α, mf, Pf, model[t], x, ys[t])
        lml += lml_
        vs[t] = f(α, x)
    end
    return lml, vs
end


#
# step_decorrelate
#

function step_decorrelate(
    model::LGSSM,
    x::Gaussian{Vector{T}, Matrix{T}},
    y::Vector{T}
) where {T<:Real}
    α = Vector{T}(undef, length(y))
    mf = Vector{T}(undef, dim(x))
    Pf = Matrix{T}(undef, dim(x), dim(x))
    return step_decorrelate!(α, mf, Pf, model, x, y)
end

function step_decorrelate!(α, mf, Pf, model, x::Gaussian, y::AV{<:Real})
    cached_step_decorrelate! = build_step_decorrelate!(model, x, y)
    return cached_step_decorrelate!(α, mf, Pf, model, x, y)
end

function build_step_decorrelate!(
    model,
    x::Gaussian{Vector{T}, Matrix{T}},
    y::Vector{T},
) where {T<:Real}

    mp = Vector{T}(undef, dim(x))
    Pp = Matrix{T}(undef, dim(x), dim(x))
    cached_predict! = build_predict!(x.m, Symmetric(x.P), model.A, model.a, model.Q)

    function cached_step_decorrelate!(α, mf, Pf, model, x, y)
        mp, Pp = cached_predict!(mp, Pp, x.m, Symmetric(x.P), model.A, model.a, model.Q)
        mf_, Pf_, lml, α_ = update_decorrelate(mp, Pp, model.H, model.h, model.Σ, y)
        α .= α_
        mf .= mf_
        Pf .= Pf_
        return lml, α, Gaussian(mf, Pf)
    end

    return cached_step_decorrelate!
end


