function sample!(sol, tree)
    empty!(tree.sampled)
    L = tree.V_lower[1]
    U = L + sol.epsilon
    sample_points(sol, tree, 1, L, U, 1)
end

function sample_points(sol::SARSOPSolver, tree::SARSOPTree, b_idx::Int, L, U, t)
    V̲, V̄ = tree.V_lower[b_idx], tree.V_upper[b_idx]
    ϵ = sol.epsilon
    γ = discount(tree)

    V̂ = V̄ #TODO: BAD, binning method
    # @show tree.b[b_idx]
    # @show V̂
    # @show L
    # @show U
    # @show V̲
    # @show t
    # @show V̲ + ϵ*γ^(-t)
    # println()

    # For default TigerPOMDP, V̂ and L are always constant, so V̂ > L ∀ b ∈ ℬ
    if V̂ ≤ L || V̄ ≤ max(U, V̲ + ϵ*γ^(-t)) || t > sol.max_steps
        return
    else # b_idx, ba_idx, o_idx
        fill_belief!(tree, b_idx)
        Q̲, Q̄, ap_idx = max_r_and_q(tree, b_idx)
        a′, ba_idx = tree.b_children[b_idx][ap_idx]
        Rba′ = belief_reward(tree, tree.b[b_idx], a′)

        L′ = max(L, Q̲)
        U′ = max(U, Q̲ + γ^(-t)*ϵ)

        # @show L′
        # @show U′

        op_idx = best_obs(tree, b_idx, ba_idx)

        Lt, Ut = get_LtUt(tree, ba_idx, Rba′, L′, U′, op_idx)

        bp_idx = tree.ba_children[ba_idx][op_idx].second
        push!(tree.sampled, b_idx)

        sample_points(sol, tree, bp_idx, Lt, Ut, t+1)
    end
end

function belief_reward(tree, b, a)
    Rba = 0.0
    for (i,s) in enumerate(states(tree))
        Rba += b[i]*reward(tree.pomdp, s, a)
    end
    return Rba
end

# TODO: check pruning
function max_r_and_q(tree::SARSOPTree, b_idx::Int)
    Q̲ = -Inf
    Q̄ = -Inf
    ap_idx = 0
    for (i,(a, ba_idx)) in enumerate(tree.b_children[b_idx])
        Q̄′ = tree.Qa_upper[b_idx][i].second
        Q̲′ = tree.Qa_lower[b_idx][i].second
        if Q̲′ > Q̲
            Q̲ = Q̲′
            Q̄ = Q̄′
            ap_idx = i
        end
    end
    return Q̲, Q̄, ap_idx
end


function best_obs(tree::SARSOPTree, b_idx, ba_idx)
    S = states(tree)
    O = observations(tree)

    best_o_idx = 0
    best_o = first(O)
    best_gap = -Inf

    for (o_idx,o) in enumerate(O)
        poba = tree.poba[ba_idx][o_idx]
        bp_idx = tree.ba_children[ba_idx][o_idx].second
        gap = poba*(tree.V_upper[bp_idx] - tree.V_lower[bp_idx])

        if gap > best_gap
            best_gap = gap
            best_o_idx = o_idx
        end
    end

    return best_o_idx
end

obs_prob(tree::SARSOPTree, ba_idx::Int, o_idx::Int) = tree.poba[ba_idx][o_idx]

function get_LtUt(tree, ba_idx, Rba, L′, U′, op_idx)
    γ = tree._discount
    Lt = (L′ - Rba)/γ
    Ut = (U′ - Rba)/γ

    for (o_idx, o) in enumerate(observations(tree))
        if op_idx != o_idx
            bp_idx = tree.ba_children[ba_idx][o_idx].second
            V̲ = tree.V_lower[bp_idx]
            V̄ = tree.V_upper[bp_idx]
            poba = obs_prob(tree, ba_idx, o_idx)
            Lt -= poba*V̲
            Ut -= poba*V̄
        end
    end
    poba = obs_prob(tree, ba_idx, op_idx)
    return Lt / poba, Ut / poba
end
