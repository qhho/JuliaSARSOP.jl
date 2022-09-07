struct AlphaVec{A} <: AbstractVector{Float64}
    alpha::Vector{Float64}
    action::A
    witnesses::Vector{Int}
    value_at_witnesses::Vector{Float64} #lower bound value
end

@inline Base.length(v::AlphaVec) = length(v.alpha)

@inline Base.size(v::AlphaVec) = size(v.alpha)

@inline Base.getindex(v::AlphaVec, i) = v.alpha[i]

@inline Base.setindex!(v::AlphaVec, x, i) = setindex!(v.alpha, x, i)
