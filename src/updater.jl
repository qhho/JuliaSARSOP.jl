function is_terminal_belief(b, terminal_s_idxs::Vector{Int})
    return sum(b[terminal_s_idxs]) ≈ 1.0
end

function predictor(pomdp::ModifiedSparseTabular, b::SparseVector, a::Int)
    return predictor!(similar(b), pomdp, b, a)
end

function predictor!(cache, pomdp::ModifiedSparseTabular, b::SparseVector, a::Int)
    return mul!(cache, pomdp.T[a], b)
end

function corrector(pomdp::ModifiedSparseTabular, pred::AbstractVector, a, o::Int)
    return pred .* @view(pomdp.O[a][:,o])
end
