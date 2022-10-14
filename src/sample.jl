function sample!(sol, tree)
    empty!(tree.sampled)
    L = tree.V_lower[1]
    U = L + sol.epsilon*root_diff(tree)
    sample_points(sol, tree, 1, L, U, 0, sol.epsilon*root_diff(tree))
end

function sample_points(sol::SARSOPSolver, tree::SARSOPTree, b_idx::Int, L, U, t, ϵ)
    tree.b_pruned[b_idx] = false
    if !tree.is_real[b_idx]
        tree.is_real[b_idx] = true
        push!(tree.real, b_idx)
    end

    tree.is_terminal[b_idx] && return

    fill_belief!(tree, b_idx)
    V̲, V̄ = tree.V_lower[b_idx], tree.V_upper[b_idx]
    γ = discount(tree)

    V̂ = bin_value(tree,b_idx,V̄) #V̄ #TODO: BAD, binning method

    if V̂ ≤ V̲ + sol.kappa*ϵ*γ^(-t) || (V̂ ≤ L && V̄ ≤ max(U, V̲ + ϵ*γ^(-t)))
        return
    else
        Q̲, Q̄, ap_idx = max_r_and_q(tree, b_idx)
        a′, ba_idx = tree.b_children[b_idx][ap_idx] #line 10
        tree.ba_pruned[ba_idx] = false

        Rba′ = belief_reward(tree, tree.b[b_idx], a′)

        L′ = max(L, Q̲)
        U′ = max(U, Q̲ + γ^(-t)*ϵ)

        op_idx = best_obs(tree, b_idx, ba_idx, ϵ, t+1)
        Lt, Ut = get_LtUt(tree, ba_idx, Rba′, L′, U′, op_idx)

        bp_idx = tree.ba_children[ba_idx][op_idx].second
        push!(tree.sampled, b_idx)
        sample_points(sol, tree, bp_idx, Lt, Ut, t+1, ϵ)
    end
end

function belief_reward(tree, b, a)
    Rba = 0.0
    for (i,s) in enumerate(states(tree))
        Rba += b[i]*reward(tree.pomdp, s, a)
    end
    return Rba
end

function max_r_and_q(tree::SARSOPTree, b_idx::Int)
    Q̲ = -Inf
    Q̄ = -Inf
    ap_idx = 0
    for (i,(a, ba_idx)) in enumerate(tree.b_children[b_idx])
        Q̄′ = tree.Qa_upper[b_idx][i].second
        Q̲′ = tree.Qa_lower[b_idx][i].second
        if Q̲′ > Q̲
            Q̲ = Q̲′
        end
        if Q̄′ > Q̄
            Q̄ = Q̄′
            ap_idx = i
        end
    end
    return Q̲, Q̄, ap_idx
end

function best_obs(tree::SARSOPTree, b_idx, ba_idx, ϵ, t)
    S = states(tree)
    O = observations(tree)
    γ = discount(tree)

    best_o_idx = 0
    best_o = first(O)
    best_gap = -Inf


    for (o_idx,o) in enumerate(O)
        poba = tree.poba[ba_idx][o_idx]
        bp_idx = tree.ba_children[ba_idx][o_idx].second
        gap = poba*(tree.V_upper[bp_idx] - tree.V_lower[bp_idx] - ϵ*γ^(-(t)))
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

function bin_value(tree::SARSOPTree, ba_idx::Int, value::Float64)
    bel = tree.b[ba_idx]
    ent = -bel'*log.(bel)
    ent_idx = argmin(abs.(tree.bin_ents .-ent))
    map = argmax(bel)
    bin = tree.bins[map][ent_idx]
    bin[1] = (bin[1]*bin[2]+value)/(bin[2]+1)
    bin[2] += 1
    push!(tree.bel_bins,ba_idx=>(map,ent_idx))
    return bin[1]
end

function initialize_bins(states,threshold)
    sl = length(states)
    el = -(1/sl)'*log(1/sl)*sl
    bin_ents = collect(0:threshold:el)
    bins = Vector{Vector{Float64}}[]
    for _ in 1:sl
        inner = Vector{Float64}[]
        for _ in 1:length(bin_ents)
            push!(inner,[0.0,0.0])
        end
        push!(bins,inner)
    end
    return (bins,bin_ents)
end
