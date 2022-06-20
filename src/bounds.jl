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

function upper_value(tree::SARSOPTree, b::Vector{Float64})
    # TODO: don't include pruned beliefs in interior beliefs 
    α_corner = tree.Vs_upper
    V_corner = dot(b, α_corner)
    V_upper = tree.V_upper

    upperVvec = Float64[]
    for (bint, vint) in zip(tree.b, V_upper)
        ϕ = minimum(b[s]/bint[s] for s in 1:length(b))
        push!(upperVvec, V_corner + ϕ * (vint - (dot(bint, α_corner))))
    end
    return minimum(upperVvec)
end

function init_lower_value!(tree::SARSOPTree, pomdp::POMDP)
    A = tree.actions
    S = tree.states
    r = StateActionReward(pomdp)
    γ = discount(pomdp)
    b = tree.b[1] # root belief

    α_init = 1 / (1 - γ) * maximum(minimum(r(s, a) for s in S) for a in A)
    Γ = [fill(α_init, length(S)) for a in A]

    MAX_VAL = dot(tree.Γ[1], b)
    MAX_ALPHA = Γ[1]
    ACTION = A[1]
    for (idx,α) in enumerate(Γ[2:end])
        new_val = dot(α, b)
        if new_val > MAX_VAL
            MAX_VAL = new_val
            MAX_ALPHA = α
            ACTION = A[idx]
        end
    end

    tree.Γ = [AlphaVec(MAX_ALPHA, ACTION, [tree.b[1]], [MAX_VAL])]
end

function lower_value(tree::SARSOPTree, b::Vector{Float64})
    MAX_VAL = dot(tree.Γ[1].alpha, b)
    for α in tree.Γ[2:end].alpha
        new_val = dot(α, b)
        if new_val > MAX_VAL
            MAX_VAL = new_val
        end
    end
    return MAX_VAL
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