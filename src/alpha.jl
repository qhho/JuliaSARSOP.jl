struct AlphaVec{A}
    alpha::Vector{Float64}
    action::A
    witnesses::Vector{Int}
    value_at_witnesses::Vector{Float64}
end