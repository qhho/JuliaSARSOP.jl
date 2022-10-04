struct SARSOPTree
    pomdp::SparseTabularPOMDP

    b::Vector{Vector{Float64}} # b_idx => belief vector
    b_children::Vector{Vector{Int}} # b_idx => [a_1=>ba_idx1, a_2=>ba_idx2, ...]
    Vs_upper::Vector{Float64}
    V_upper::Vector{Float64}
    V_lower::Vector{Float64}
    Qa_upper::Vector{Vector{Float64}}
    Qa_lower::Vector{Vector{Float64}}

    ba_children::Vector{Vector{Int}} # (ba_idx, o) => bp_idx # deleted nodes have (o, 0) pairs for first element
    ba_action::Vector{Int}
    poba::Vector{Vector{Float64}} # ba_idx => o_idx

    _discount::Float64
    is_terminal::BitVector
    terminal_s_idxs::Vector{Int}

    #do we need both b_pruned and ba_pruned? b_pruned might be enough
    sampled::Vector{Int} # b_idx
    b_pruned::BitVector
    ba_pruned::BitVector
    real::Vector{Int} # b_idx
    is_real::BitVector

    Γ::Vector{AlphaVec{Int}}
end

function SARSOPTree(solver, pomdp::POMDP)
    sparse_pomdp = SparseTabularPOMDP(pomdp)
    ordered_s = ordered_states(pomdp)

    upper_policy = solve(solver.init_upper, sparse_pomdp)
    corner_values = map(maximum, zip(upper_policy.alphas...))
    terminals = filter(i->isterminal(pomdp, ordered_s[i]), states(sparse_pomdp))

    tree = SARSOPTree(
        sparse_pomdp,

        Vector{Float64}[],
        Vector{Int}[],
        corner_values, #upper_policy.util,
        Float64[],
        Float64[],
        Vector{Float64}[],
        Vector{Float64}[],
        Vector{Int}[],
        Int[],
        Vector{Float64}[],
        discount(pomdp),
        BitVector(),
        terminals,
        Int[],
        BitVector(),
        BitVector(),
        Vector{Int}(),
        BitVector(),
        AlphaVec{Int}[]
    )
    return insert_root!(solver, tree, _initialize_belief(pomdp, initialstate(pomdp)))
end

POMDPs.states(tree::SARSOPTree) = ordered_states(tree)
POMDPTools.ordered_states(tree::SARSOPTree) = states(tree.pomdp)
POMDPs.actions(tree::SARSOPTree) = ordered_actions(tree)
POMDPTools.ordered_actions(tree::SARSOPTree) = actions(tree.pomdp)
POMDPs.observations(tree::SARSOPTree) = ordered_observations(tree)
POMDPTools.ordered_observations(tree::SARSOPTree) = observations(tree.pomdp)
POMDPs.discount(tree) = discount(tree.pomdp)

# type piracy - should be defined already
# POMDPs.stateindex(pomdp::SparseTabularPOMDP, s::Int) = s
# POMDPs.actionindex(pomdp::SparseTabularPOMDP, a::Int) = a
# POMDPs.obsindex(pomdp::SparseTabularPOMDP, o::Int) = o

function _initialize_belief(pomdp::POMDP, dist::Any)
    ns = length(states(pomdp))
    b = zeros(ns)
    for s in support(dist)
        sidx = stateindex(pomdp, s)
        b[sidx] = pdf(dist, s)
    end
    return b
end

function insert_root!(solver, tree::SARSOPTree, b)
    pomdp = tree.pomdp

    Γ_lower = solve(solver.init_lower, pomdp)
    for (α,a) ∈ alphapairs(Γ_lower)
        new_val = dot(α, b)
        push!(tree.Γ, AlphaVec(α, a, [1], [new_val]))
    end

    push!(tree.b, b)
    push!(tree.b_children, Int[])
    push!(tree.V_upper, init_root_value(tree, b))
    push!(tree.real, 1)
    push!(tree.is_real, true)
    push!(tree.V_lower, lower_value(tree, b))
    push!(tree.Qa_upper, Float64[])
    push!(tree.Qa_lower, Float64[])
    push!(tree.b_pruned, false)
    push!(tree.is_terminal, is_terminal_belief(b, tree.terminal_s_idxs))
    fill_belief!(tree, 1)
    return tree
end

function update_and_push(tree::SARSOPTree, b_idx::Int, a, o)
    b = tree.b[b_idx]
    ba_idx = b_child(tree, b_idx, a)
    bp_idx = -1
    V_upper = Inf
    V_lower = -Inf
    if ba_idx === -1
        ba_idx = add_action!(tree, b_idx, a)
        b′ = update_and_shift(tree, b, a, o)
        bp_idx, V_upper, V_lower = add_belief!(tree, b′, ba_idx, o)
    else
        bp_idx = ba_child(tree, ba_idx, o)
        if bp_idx === -1
            b′ = update_and_shift(tree, b, a, o)
            bp_idx, V_upper, V_lower = add_belief!(tree, b′, ba_idx, o)
        end
    end
    return bp_idx, V_upper, V_lower
end

function update_and_push_null(tree::SARSOPTree, b_idx::Int, a, o)
    b = tree.b[b_idx]
    ba_idx = b_child(tree, b_idx, a)
    bp_idx = -1
    V_upper = 0.
    V_lower = 0.
    b′ = zero(b)
    if ba_idx === -1
        ba_idx = add_action!(tree, b_idx, a)
        bp_idx, V_upper, V_lower = add_belief_null!(tree, b′, ba_idx, o)
    else
        bp_idx = ba_child(tree, ba_idx, o)
        if bp_idx === -1
            bp_idx, V_upper, V_lower = add_belief_null!(tree, b′, ba_idx, o)
        end
    end
    return bp_idx, V_upper, V_lower
end

function update(tree::SARSOPTree, b_idx::Int, a, o)
    b = tree.b[b_idx]
    ba_idx = b_child(tree, b_idx, a)
    bp_idx = ba_child(tree, ba_idx, o)
    V_upper = upper_value(tree, tree.b[bp_idx])
    V_lower = lower_value(tree, tree.b[bp_idx])
    tree.V_upper[bp_idx] = V_upper
    tree.V_lower[bp_idx] = V_lower
    return bp_idx, V_upper, V_lower
end

function b_child(tree::SARSOPTree, b_idx::Int, a::Int)
    return if checkbounds(Bool,tree.b_children[b_idx], a)
        tree.b_children[b_idx][a]
    else
        -1
    end
end

# should be unnecessary
function ba_child(tree::SARSOPTree, ba_idx::Int, o::Int)
    return if checkbounds(Bool,tree.ba_children[ba_idx], o)
        tree.ba_children[ba_idx][o]
    else
        -1
    end
end

function add_belief!(tree::SARSOPTree, b, ba_idx::Int, o::Int)
    push!(tree.b, b)
    b_idx = length(tree.b)
    push!(tree.ba_children[ba_idx], b_idx)
    push!(tree.b_children, Int[])
    push!(tree.is_real, false)
    push!(tree.Qa_upper, Float64[])
    push!(tree.Qa_lower, Float64[])
    V_upper = upper_value(tree, b)
    V_lower = lower_value(tree, b)
    push!(tree.is_terminal, is_terminal_belief(b, tree.terminal_s_idxs))
    push!(tree.V_upper, V_upper)
    push!(tree.V_lower, V_lower)
    push!(tree.b_pruned, true)
    return b_idx, V_upper, V_lower
end

function add_belief_null!(tree::SARSOPTree, b, ba_idx::Int, o::Int)
    push!(tree.b, b)
    b_idx = length(tree.b)
    push!(tree.ba_children[ba_idx], b_idx)
    push!(tree.b_children, Int[])
    push!(tree.is_real, false)
    push!(tree.Qa_upper, Float64[])
    push!(tree.Qa_lower, Float64[])
    V_upper = 0.
    V_lower = 0.
    push!(tree.is_terminal, true)
    push!(tree.V_upper, V_upper)
    push!(tree.V_lower, V_lower)
    push!(tree.b_pruned, true)
    return b_idx, V_upper, V_lower
end

function add_action!(tree::SARSOPTree, b_idx::Int, a::Int)
    ba_idx = length(tree.ba_children) + 1
    push!(tree.b_children[b_idx], ba_idx)
    push!(tree.ba_children, Int[])
    push!(tree.ba_pruned, true)
    push!(tree.ba_action, a)
    return ba_idx
end

# actually can we just stick this in the belief updater to reduce repeated comp??
function obs_prob(tree::SARSOPTree, b::Vector, a, o)
    pomdp = tree.pomdp
    poba = 0.0
    for (s_idx, s) in enumerate(states(tree))
        T = transition(pomdp, s, a)
        for (sp_idx, sp) in enumerate(states(tree))
            Tsasp = pdf(T, sp)
            !iszero(Tsasp) && (poba += pdf(observation(pomdp, s, a, sp),o)*Tsasp*b[s_idx])
        end
    end
    return poba
end

function fill_belief!(tree::SARSOPTree, b_idx::Int)
    if isempty(tree.b_children[b_idx])
        fill_unpopulated!(tree, b_idx)
    else
        fill_populated!(tree, b_idx)
    end
end

"""
Fill p(o|b,a), V̲(τ(bao)), V̄(τ(bao)) ∀ o,a if not already filled
"""
function fill_populated!(tree::SARSOPTree, b_idx::Int)
    γ = discount(tree)
    ACT = actions(tree)
    OBS = observations(tree)
    b = tree.b[b_idx]
    Qa_upper = tree.Qa_upper[b_idx]
    Qa_lower = tree.Qa_lower[b_idx]
    for (a_idx,a) in enumerate(ACT)
        ba_idx = last(tree.b_children[b_idx][a_idx])
        Rba = belief_reward(tree, b, a)
        Q̄ = Rba
        Q̲ = Rba

        for (o_idx, o) in enumerate(OBS)
            # bp_idx = last(tree.ba_children[ba_idx][o_idx])
            # V̄ = tree.V_upper[bp_idx]
            # V̲ = tree.V_lower[bp_idx]
            bp_idx, V̄, V̲ = update(tree, b_idx, a, o)
            b′ = tree.b[bp_idx]
            po = obs_prob(tree, b, a, o)
            Q̄ += γ*po*V̄
            Q̲ += γ*po*V̲
        end

        Qa_upper[a_idx] = Q̄
        Qa_lower[a_idx] = Q̲
    end

    tree.V_lower[b_idx] = lower_value(tree, tree.b[b_idx])
    tree.V_upper[b_idx] = maximum(last, tree.Qa_upper[b_idx])
end

function fill_unpopulated!(tree::SARSOPTree, b_idx::Int)
    γ = discount(tree)
    ACT = actions(tree)
    OBS = observations(tree)
    N_OBS = length(OBS)
    b = tree.b[b_idx]
    n_b = length(tree.b)
    n_ba = length(tree.ba_children)

    b_children = Vector{Int}(undef, length(ACT))
    Qa_upper = Vector{Float64}(undef, length(ACT))
    Qa_lower = Vector{Float64}(undef, length(ACT))
    for (a_idx,a) in enumerate(ACT)
        b_children[a_idx] = n_ba + a_idx

        # TODO: If all observations are always expanded, there's little to no need for a `o => bp_idx` pair
        #       Just need nested vector mapping ba_idx => o_idx => bp_idx
        ba_children = Vector{Int}(undef, N_OBS)
        poba = Vector{Float64}(undef, N_OBS)

        Rba = belief_reward(tree, b, a)
        Q̄ = Rba
        Q̲ = Rba

        for (o_idx, o) in enumerate(OBS)
            po = obs_prob(tree, b, a, o)

            bp_idx, V̄, V̲ = if !iszero(po)
                update_and_push(tree, b_idx, a, o)
            else
                update_and_push_null(tree, b_idx, a, o)
            end
            b′ = tree.b[bp_idx]
            ba_children[o_idx] = bp_idx
            poba[o_idx] = po
            Q̄ += γ*po*V̄
            Q̲ += γ*po*V̲
        end
        push!(tree.poba, poba)

        Qa_upper[a_idx] = Q̄
        Qa_lower[a_idx] = Q̲
    end
    tree.b_children[b_idx] = b_children
    tree.Qa_upper[b_idx] = Qa_upper
    tree.Qa_lower[b_idx] = Qa_lower

    tree.V_lower[b_idx] = lower_value(tree, tree.b[b_idx])
    tree.V_upper[b_idx] = maximum(last, tree.Qa_upper[b_idx])
end
