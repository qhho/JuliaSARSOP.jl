function init_root_value(tree::SARSOPTree, b::Vector{Float64})
    corner_values = tree.Vs_upper
    value = 0.0
    for i in 1:length(corner_values)
        value += corner_values[i] * b[i]
    end
    return value
end

function min_ratio(v1, v2)
    min_ratio = Inf
    for (a,b) ∈ zip(v1, v2)
        ratio = a/b
        ratio < min_ratio && (min_ratio = ratio)
    end
    return min_ratio
end

function upper_value(tree::SARSOPTree, b::Vector{Float64})
    # TODO: don't include pruned beliefs in interior beliefs
    α_corner = tree.Vs_upper
    V_corner = dot(b, α_corner)
    V_upper = tree.V_upper

    v̂_min = Inf
    for (bint, vint) in zip(tree.b, V_upper)
        ϕ = min_ratio(b, bint)
        v̂ = V_corner + ϕ * (vint - (dot(bint, α_corner)))
        v̂ < v̂_min && (v̂_min = v̂)
    end
    return v̂_min
end

function init_lower_value!(tree::SARSOPTree, pomdp::POMDP)
    A = tree.actions
    S = tree.states
    r = StateActionReward(pomdp)
    γ = discount(pomdp)
    b = tree.b[1] # root belief

    α_init = 1 / (1 - γ) * maximum(minimum(r(s, a) for s in S) for a in A)
    Γ = [fill(α_init, length(S)) for a in A]

    MAX_VAL = -Inf #dot(tree.Γ[1], b)
    MAX_ALPHA = Γ[1]
    ACTION = A[1]
    if length(Γ) > 1
        for (idx,α) in enumerate(Γ[2:end])
            new_val = dot(α, b)
            if new_val > MAX_VAL
                MAX_VAL = new_val
                MAX_ALPHA = α
                ACTION = A[idx]
            end
        end
    end

    push!(tree.Γ, AlphaVec(MAX_ALPHA, ACTION, [1], [MAX_VAL]))
end

function lower_value(tree::SARSOPTree, b::Vector{Float64})
    MAX_VAL = -Inf
    alphas = tree.Γ
    #MAX_VAL = dot(tree.Γ[1].alpha, b)
    for alphavec in tree.Γ
        α = alphavec.alpha
        new_val = dot(α, b)
        if new_val > MAX_VAL
            MAX_VAL = new_val
        end
    end
    return MAX_VAL
end

# Get upper bound value for each belief in tree
function updateUpperBound!(tree::SARSOPTree, b::Int, ba_idx::Int, o_idx::Int, b_parent::Int)
    #check b pruned
    if b_parent > 0
        oldV = tree.V_upper[b]
        newV = maximum(x -> x.second, tree.Qa_upper[b])
        tree.V_upper[b] = newV

        ΔV = newV - oldV
        ΔQ = tree._discount * tree.poba[ba_idx][o_idx] * ΔV
        obs = tree.Qa_upper[b_parent].first
        Q = tree.Qa_upper[b_parent].second
        tree.Qa_upper[b_parent] = Pair(obs, Q + ΔQ)
    end
end

function updateUpperBounds!(tree::SARSOPTree)
    for b_sampled in tree.sampled
        ba_idx, o_idx, b_parent = tree.b_parent[b_sampled]
        updateUpperBound!(tree, b_sampled, ba_idx, o_idx, b_parent)
    end
end

function updateLowerBounds!(tree::SARSOPTree)
    γ = discount(tree)
    for b_sampled in tree.sampled
        ba_idx, _, b_parent = tree.b_parent[b_sampled]
        tree.Qa_lower[parent].second = belief_reward(tree, tree.b[b_parent], tree.ba_actions[ba_idx]) + γ*dot(tree.poba[ba_idx], tree.V_lower[tree.ba_children[ba_idx]]) #R(b,a) + γ E[V[b']]
    end
end

root_diff(tree) = tree.V_upper[1] - tree.V_lower[1]
