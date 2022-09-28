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
    α_corner = tree.Vs_upper
    V_corner = dot(b, α_corner)
    V_upper = tree.V_upper
    v̂_min = Inf
    for b_idx ∈ tree.real
        tree.b_pruned[b_idx] && continue
        vint = V_upper[b_idx]
        bint = tree.b[b_idx]
        ϕ = min_ratio(b, bint)
        v̂ = V_corner + ϕ * (vint - dot(bint, α_corner))
        v̂ < v̂_min && (v̂_min = v̂)
    end

    return v̂_min
end

function init_lower_value!(tree::SARSOPTree, pomdp::POMDP)
    S = states(tree)
    A = actions(tree)
    r = StateActionReward(pomdp)
    γ = discount(pomdp)
    b = tree.b[1] # root belief

    Qa_lower = [zeros(length(S)) for _ in eachindex(A)]
    for (a_idx, a) in enumerate(A)
        for (s_idx, s) in enumerate(S)
            Qa = 1 / (1 - γ) * minimum(r(s′, a) for (p, s′) in transition(pomdp, s, a))
            Qa_lower[a_idx][s_idx] = Qa
        end
    end

    for _ in 1:100000
        for (a_idx, a) in enumerate(A)
            for (s_idx, s) in enumerate(S)
                Qas = reward(pomdp, s, a)
                for (p, s′) in transition(pomdp, s, a)
                    Qas += γ*p*Qa_lower[a_idx][stateindex(pomdp, s′)]
                end
                Qa_lower[a_idx][s_idx] = Qas
            end
        end
    end
    for (idx,α) in enumerate(Qa_lower)
        new_val = dot(α, b)
        push!(tree.Γ, AlphaVec(α, A[idx], [1], [new_val]))
    end
end

function lower_value(tree::SARSOPTree, b::Vector{Float64})
    MAX_VAL = -Inf
    for α in tree.Γ
        new_val = dot(α, b)
        if new_val > MAX_VAL
            MAX_VAL = new_val
        end
    end
    return MAX_VAL
end

function update_upper_bound!(tree::SARSOPTree, b_idx::Int)
    b = tree.b[b_idx]
    b_children = tree.b_children[b_idx]
    for (a_idx,a) in enumerate(tree.actions)
        Rba = belief_reward(tree, b, a)
        Q̄ = Rba
        ba_idx = last(b_children[a_idx])
        for (o_idx, o) in enumerate(tree.observations)
            _, V̄, _ = update(tree, b_idx, a, o)
            po = tree.poba[ba_idx][o_idx]
            Q̄ += tree._discount*po*V̄
        end
        tree.Qa_upper[b_idx][a_idx] = a => Q̄
    end
    tree.V_upper[b_idx] = maximum(last, tree.Qa_upper[b_idx])
end

function update_upper_bounds!(tree::SARSOPTree)
    for b_sampled in reverse(tree.sampled)
        update_upper_bound!(tree, b_sampled)
    end
end

root_diff(tree) = tree.V_upper[1] - tree.V_lower[1]
