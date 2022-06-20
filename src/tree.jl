struct SARSOPTree{S,A,O,P<:POMDP}
    states::Vector{S}
    b::Vector{Vector{Float64}}
    b_children::Vector{Vector{Pair{A,Int}}}
    Vs_upper::Vector{Float64}
    V_upper::Vector{Float64}
    V_lower::Vector{Float64}
    Qa_upper::Vector{Vector{Pair{A, Float64}}}
    Qa_lower::Vector{Vector{Pair{A, Float64}}}

    obs::Vector{O}
    actions::Vector{A}

    ba_children::Vector{Vector{Pair{O,Int}}} # (ba_idx, o) => bp_idx # deleted nodes have (o, 0) pairs for first element
    ba_action::Vector{A}
    poba::Vector{Vector{Float64}} # ba_idx => o_idx

    _discount::Float64

    not_terminals::Vector{Int}
    terminals::Vector{Int}

    #do we need both b_pruned and ba_pruned? b_pruned might be enough
    b_touched::Vector{Int}
    b_pruned::BitVector
    ba_pruned::BitVector

    pomdp::P
end

function SARSOPTree(pomdp::POMDP{S,A,O}) where {S,A,O}
    mdp_solver = ValueIterationSolver()
    upper_policy = solve(mdp_solver, UnderlyingMDP(pomdp))

    not_terminals = Int[stateindex(pomdp, s) for s in states(pomdp) if !isterminal(pomdp, s)]
    terminals = Int[stateindex(pomdp, s) for s in states(pomdp) if isterminal(pomdp, s)]
    obs = ordered_observations(pomdp)

    return SARSOPTree(
        ordered_states(pomdp),
        Vector{Float64}[],
        Vector{Pair{A,Int}}[],
        upper_policy.util,
        Float64[],
        Float64[],
        Vector{Pair{A, Float64}}[],
        Vector{Pair{A, Float64}}[],
        obs,
        ordered_actions(pomdp),
        Vector{Pair{O,Int}}[],
        ordered_actions(pomdp),
        Vector{Float64}[],
        discount(pomdp),
        not_terminals,
        terminals,
        Int[],
        BitVector(undef, 0),
        BitVector(undef, 0),
        pomdp
    )
end

# TODO: gimme non-placeholder bounds pls
upper_value(::SARSOPTree, b::Vector{Float64}) = +Inf
lower_value(::SARSOPTree, b::Vector{Float64}) = -Inf

function insert_root!(tree::SARSOPTree, pomdp::POMDP)
    b0 = initialstate(pomdp)
    b = initialize_belief(bu, b0).b
    push!(tree.b, b)
    push!(tree.b_children, Pair{A, Float64}[])
    push!(tree.V_upper, upper_value(tree, b))
    push!(tree.V_upper, lower_value(tree, b))
    push!(tree.b_pruned, false)
end

function BeliefUpdaters.update(tree::SARSOPTree, b_idx::Int, a, o)
    b = tree.b[b_idx]
    ba_idx = b_child(tree, b_idx, a)
    bp_idx = -1
    if ba_idx === -1
        add_action!(tree, b_idx, a)
        b′ = update(tree, b, a, o)
        bp_idx = add_belief!(tree, b′, ba_idx, o)
    else
        bp_idx = ba_child(tree, ba_idx, o)
        if bp_idx === -1
            b′ = update(tree, b, a, o)
            bp_idx = add_belief!(tree, b′, ba_idx, o)
        end
    end
    return bp_idx
end

function b_child(tree::SARSOPTree{S,A,O}, b_idx::Int, a::A) where {S,A,O}
    children = tree.b_children[b_idx]
    for (a′, ba_idx) in children
        a′ == a && return ba_idx
    end
    return -1 # child not found
end

function ba_child(tree::SARSOPTree{S,A,O}, ba_idx::Int, o::O) where {S,A,O}
    children = tree.ba_children[ba_idx]
    for (o′, bp_idx) in children
        o′ == o && return bp_idx
    end
    return -1 # child not found
end

function add_belief!(tree::SARSOPTree{S,A,O}, b, ba_idx::Int, o::O) where {S,A,O}
    push!(tree.b, b)
    b_idx = length(tree.b)
    push!(tree.ba_children[ba_idx], o=>b_idx)
    push!(tree.b_children, Pair{A, Float64}[])
    push!(tree.V_upper, upper_value(tree, b))
    push!(tree.V_upper, lower_value(tree, b))
    push!(tree.b_pruned, false)
    return b_idx
end

function add_action!(tree::SARSOPTree{S,A,O}, b_idx::Int, a::A) where {S,A,O}
    ba_idx = length(tree.ba_children) + 1
    push!(tree.b_children[b_idx], a=>ba_idx)
    push!(tree.ba_children, Pair{O, Float64}[])
    push!(tree.ba_pruned, false)
    return ba_idx
end

# actually can we just stick this in the belief updater to reduce repeated comp??
function obs_prob(tree::SARSOPTree, b::Vector, a, o, bp::Vector)
    pomdp = tree.pomdp
    pobabp = 0.0
    for (s_idx, s) in enumerate(tree.states), (sp_idx, sp) in enumerate(tree.states)
        pobabp += b[s_idx]*b[sp_idx]*pdf(observation(pomdp, s, a, sp), o)
    end
    return pobabp
end

"""
Fill p(o|b,a), V̲(τ(bao)), V̄(τ(bao)) ∀ o,a if not already filled
"""
function fill_belief!(tree::SARSOPTree{S,A,O}, b_idx::Int) where {S,A,O}
    isempty!(tree.b_children[b_idx]) && return
    ACT = ordered_actions(tree.pomdp)
    OBS = tree.obs
    N_OBS = length(OBS)
    b = tree.b[b_idx]
    n_b = length(tree.b)
    n_ba = length(tree.ba_children)

    b_children = Vector{Pair{A, Int}}(undef, length(ACT))
    Qa_upper = Vector{Pair{A, Float64}}(undef, length(ACT))
    Qa_lower = Vector{Pair{A, Float64}}(undef, length(ACT))
    for (a_idx,a) in enumerate(ACT)
        b_children[a_idx] = n_ba + a_idx

        # TODO: If all observations are always expanded, there's little to no need for a `o => bp_idx` pair
        #       Just need nested vector mapping ba_idx => o_idx => bp_idx
        ba_children = Vector{Pair{O, Int}}(undef, N_OBS)
        poba = Vector{Float64}(undef, N_OBS)

        Rba = belief_reward(tree, b, a)
        Q̄ = Rba
        Q̲ = Rba

        for (o_idx, o) in enumerate(OBS)
            bp_idx = n_b + o_idx + N_OBS*(a_idx-1)
            b′ = update(tree, b, a, o)
            po = obs_prob(tree.pomdp, b, a, o, b′)
            ba_children[o_idx] = (o => bp_idx)
            poba[o_idx] = po
            push!(tree.b, b′)
            V̄ = upper_value(tree, b′)
            V̲ = lower_value(tree, b′)
            Q̄ += γ*po*V̄
            Q̲ += γ*po*V̲
            push!(tree.V_upper, V̄)
            push!(tree.V_lower, V̲)
            push!(tree.b_pruned, false)
        end
        push!(tree.ba_pruned, false)
        push!(tree.ba_children, ba_children)
        push!(tree.poba, poba)

        Qa_upper[a_idx] = a => Q̄
        Qa_lower[a_idx] = a => Q̲
    end
    push!(tree.b_children, b_children)
    push!(tree.Qa_upper, Qa_upper)
    push!(tree.Qa_lower, Qa_lower)
    nothing
end
