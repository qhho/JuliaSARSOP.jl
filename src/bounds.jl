function initUpperBound!(tree::SARSOPTree, b_idx::Int)
    b = tree.b[b_idx]
    tmp = 0.0
    for i in 1:length(values) 
        tmp += values[i] * b[i]
    end
    push!(tree.V_upper, tmp)
end

function getUpperBoundSimple(values::Vector{Float64}, b::Vector{Float64})
        return sum(values .* b)
end

function sawtoothUB!(tree::SARSOPTree, b_idx::Int)
    b = tree.b[b_idx]
    α_corner = tree.Vs_upper
    V_corner = sum(b .* α_corner)
    V_upper = tree.V_upper

    upperVvec = Float64[]
    for (bint, vint) in zip(tree.b, V_upper)
        ϕ = minimum(b[s]/bint[s] for s in 1:length(b))
        push!(upperVvec, V_corner + ϕ * (vint - (sum(bint .* α_corner))))
    end
    tree.V_upper[b_idx] = minimum(upperVvec)
end

# Get upper bound value for each belief in tree
function updateUpperBound!(tree::SARSOPTree, b::Int, ba_idx::Int, o_idx::Int, b_parent::Int)
    # Qa_upper::Vector{Vector{Pair{A, Float64}}}
    oldV = tree.V_upper[b]
    newV = maximum(x -> x.second, tree.Qa_upper[b])
    ΔV = newV - oldV
    ΔQ = tree._discount * tree.poba[ba_idx][o_idx] * ΔV

    obs = tree.Qa_upper[b_parent].first
    Q = tree.Qa_upper[b_parent].second
    tree.Qa_upper[b_parent] = Pair(obs, Q + ΔQ)
end

function updateUpperBounds!(tree::SARSOPTree)

end

function updateLowerBounds!(tree::SARSOPTree)

end