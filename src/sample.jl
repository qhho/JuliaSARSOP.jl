function sample!(sol, tree)
    empty!(tree.b_touched)
    L = tree.V_lower[1]
    U = L + sol.epsilon
    sample_points(sol, tree, 1, L, U, 1)
end

function sample_points(sol::SARSOPSolver, tree::SARSOPTree, b_idx::Int, L, U, t)
    V̲, V̄ = tree.V_lower[b_idx], tree.V_upper[b_idx]
    ϵ = sol.epsilon
    γ = tree._discount

    push!(tree.b_touched, b_idx)

    if V̂ ≤ L && V̄ ≤ max(U, V̲ + ϵ*γ^(-t))
        return
    else
        Rba′, Q̲, Q̄, ap_idx = max_r_and_q(tree, b_idx)

        L′ = max(L, Q̲)
        U′ = max(U, Q̲ + γ^(-t)*ϵ)

        op_idx, op = best_obs(tree, b_idx, ap_idx)

        Lt, Ut = get_LtUt(tree, b_idx, ap_idx, Rba, L′, U′, op_idx, op)

        bp_idx = update(tree, b_idx, a, o)

        sample_points(sol, tree, bp_idx, Lt, Ut, t+1)
    end
end

function max_r_and_q(tree::SARSOPTree, b_idx::Int)
    Q̲ = -Inf
    Q̄ = -Inf
    Rba′ = 0.0
    local ap_idx::Int
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
        poba = obs_prob(o, b, a)
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
    a = tree.a[a_idx]

    S = tree.states
    O = tree.obs

    best_o_idx = 0
    best_o = first(O)
    best_gap = -Inf

    for (o_idx,o) in O
        poba = obs_prob(o, b, a)
        bp_idx = ba_child(tree, ba_idx, o)

        gap = poba*(tree.V_upper[bp_idx] - tree.V_lower[bp_idx])

        if gap > best_gap
            best_gap = gap
            best_o = o_idx
        end
    end

    return best_o_idx, best_o
end

function obs_prob(o, b, a)
    # can we just do `observation(pomdp, s, a)` without including s'?
    poba = 0.0
    for (s_idx, s) in enumerate(ordered_states(pomdp))
        poba += b[s_idx]*pdf(observation(pomdp, s, a), o)
    end
    return poba
end

function get_LtUt(tree, b_idx, ba_idx, Rba, L′, U′, op_idx, op)
    Lt = (L′ - Rba)/γ
    Ut = (U′ - Rba)/γ
    b = tree.b[b_idx]

    for (o_idx, o) in enumerate(tree.obs)
        if op_idx != o_idx
            bp_idx = ba_child(tree, ba_idx, o)
            V̲ = tree.V_lower[bp_idx]
            V̄ = tree.V_upper[bp_idx]
            poba = obs_prob(o,b,a)
            Lt -= poba*V̲
            Ut -= poba*V̄
        end
    end
    poba = obs_prob(op, b, a)
    return Lt / poba, Ut / poba
end
