function sample!(sol, tree)
    empty!(tree.b_touched)
    L = tree.V_lower[1]
    U = L + sol.epsilon
    sample_points(sol, tree, 1, L, U, 1)
end

function sample_points(sol::SARSOPSolver, tree::SARSOPTree, b_idx::Int, L, U, t)
    V̲, V̄ = tree.V_lower[b_idx], tree.V_upper[b_idx]
    ϵ = sol.epsilon
    γ = discount(tree.pomdp)

    push!(tree.b_touched, b_idx)

    if V̂ ≤ L && V̄ ≤ max(U, V̲ + ϵ*γ^(-t)) # TODO: V̂ not defined
        return
    else
        fill_belief!(tree, b_idx)
        Rba′, Q̲, Q̄, ap_idx = max_r_and_q(tree, b_idx)

        L′ = max(L, Q̲)
        U′ = max(U, Q̲ + γ^(-t)*ϵ)

        op_idx= best_obs(tree, b_idx, ap_idx)

        Lt, Ut = get_LtUt(tree, b_idx, ap_idx, Rba, L′, U′, op_idx)

        bp_idx = tree.ba_children[op_idx].second

        sample_points(sol, tree, bp_idx, Lt, Ut, t+1)
    end
end

function max_r_and_q(tree::SARSOPTree, b_idx::Int)
    Q̲ = -Inf
    Q̄ = -Inf
    Rba′ = 0.0
    ap_idx = 0
    for (a,ba_idx) in tree.b_children[b_idx]
        Rba, q_lower, q_upper = q_bounds(tree, b_idx, ba_idx)
        q_lower > Q̲ && (Q̲ = q_lower)
        if q_upper > Q̄
            Q̄ = q_upper
            ap_idx = ba_idx
            Rba′ = Rba
        end
    end
    return Rba′, Q̲, Q̄, ap_idx
end

function q_bounds(tree::SARSOPTree, b_idx::Int, ba_idx::Int)
    b = tree.b[b_idx]
    a = tree.a[a_idx]

    S = tree.states
    O = tree.obs

    Rba = 0.0
    for (i,s) in enumerate(S)
        Rba += b[i]*reward(pomdp, s, a)
    end

    EV̲b′ = 0.0
    EV̄b′ = 0.0

    for (o_idx, o) in enumerate(O)
        poba = obs_prob(tree, ba_idx, o_idx)
        bp_idx = ba_child(tree, ba_idx, o)
        V̲ = tree.V_lower[bp_idx]
        V̄ = tree.V_upper[bp_idx]
        EV̲b′ += poba*V̲
        EV̄b′ += poba*V̄
    end

    return Rba, Rba + γ*EV̲b′, Rba + γ*EV̄b′
end


function best_obs(tree::SARSOPTree, b_idx, ba_idx)
    b = tree.b[b_idx]

    S = tree.states
    O = tree.obs

    best_o_idx = 0
    best_o = first(O)
    best_gap = -Inf

    for (o_idx,o) in O
        poba = obs_prob(tree, ba_idx, o_idx)
        bp_idx = ba_child(tree, ba_idx, o)

        gap = poba*(tree.V_upper[bp_idx] - tree.V_lower[bp_idx])

        if gap > best_gap
            best_gap = gap
            best_o_idx = o_idx
        end
    end

    return best_o_idx
end

obs_prob(tree::SARSOPTree, ba_idx::Int, o_idx::Int) = tree.poba[ba_idx][o_idx]

function get_LtUt(tree, b_idx, ba_idx, Rba, L′, U′, op_idx)
    Lt = (L′ - Rba)/γ
    Ut = (U′ - Rba)/γ
    b = tree.b[b_idx]

    for (o_idx, o) in enumerate(tree.obs)
        if op_idx != o_idx
            bp_idx = ba_child(tree, ba_idx, o)
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
