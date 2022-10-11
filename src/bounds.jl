function init_root_value(tree::SARSOPTree, b::Vector{Float64})
    corner_values = tree.Vs_upper
    value = 0.0
    for i in 1:length(corner_values)
        value += corner_values[i] * b[i]
    end
    return value
end

function safe_div(a,b)
    return if iszero(a) && iszero(b)
        Inf
    else
        a/b
    end
end

sparse_min_ratio(b_cache, v1, v2) = minimum(b_cache .= safe_div.(v1,v2))

function min_ratio(v1, v2)
    min_ratio = Inf
    I,V = v2.nzind, v2.nzval
    for (_i,i) ∈ enumerate(I)
        ratio = v1[i] / V[_i]
        ratio < min_ratio && (min_ratio = ratio)
    end
    # for (a,b) ∈ zip(v1, v2)
    #     ratio = a/b
    #     ratio < min_ratio && (min_ratio = ratio)
    # end
    return min_ratio
end

function upper_value(tree::SARSOPTree, b::AbstractVector)
    b_cache = Vector{Float64}(undef, length(b))
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

function lower_value(tree::SARSOPTree, b::AbstractVector)
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
    for a in tree.actions
        Rba = belief_reward(tree, b, a)
        Q̄ = Rba
        ba_idx = b_children[a_idx]
        for o in observations(tree)
            _, V̄, _ = update(tree, b_idx, a, o)
            po = tree.poba[ba_idx][o]
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
