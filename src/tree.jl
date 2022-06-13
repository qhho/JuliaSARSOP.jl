struct SARSOPTree{S,A,O,UPD}
    states::Vector{S}
    b::Vector{Vector{Float64}}
    b_children::Vector{Vector{Pair{A,Int}}}
    V_upper::Vector{Float64}
    V_lower::Vector{Float64}

    obs::Vector{O}

    ba_children::Vector{Vector{Pair{O,Int}}} # (ba_idx, o) => bp_idx
    ba_action::Vector{A}

    _discount::Float64

    not_terminals::Vector{Int}
    terminals::Vector{Int}

    updater::UPD
    cache::SARSOPCache

    function SARSOPTree{S,A,O}(pomdp::POMDP) where {S,A,O}
        solver = ValueIterationSolver()
        upper_policy = solve(solver, UnderlyingMDP(pomdp))
        upper_values = upper_policy.util
        not_terminals = [stateindex(pomdp, s) for s in states(pomdp) if !isterminal(pomdp, s)]
        terminals = [stateindex(pomdp, s) for s in states(pomdp) if isterminal(pomdp, s)]
        obs = ordered_observations(pomdp)

        return new(
            ordered_states(pomdp),
            Vector{Float64}[],
            Vector{Pair{A,Int}}[],
            upper_values,
            Float64[],
            obs,
            Vector{Pair{O,Int}}[],
            ordered_actions(pomdp),
            discount(pomdp),
            not_terminals,
            terminals,
            SARSOPCache(length(obs))
        )
    end
end

SARSOPTree(pomdp::POMDP{S,A,O}) where {S,A,O} = SARSOPTree{S,A,O}(pomdp)

function BeliefUpdaters.update(tree::SARSOPTree, b_idx, a, o)
    b = tree.b[b_idx]
    ba_idx = b_child(tree, b_idx, a)
    bp_idx = -1
    if ba_idx === -1
        add_action!(tree, b_idx, a)
        b′ = update(tree.updater, b, a, o)
        bp_idx = add_belief!(tree, b′, ba_idx, o)
    else
        bp_idx = ba_child(tree, ba_idx, o)
        if bp_idx === -1
            b′ = update(tree.updater, b, a, o)
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

function add_belief!(tree::SARSOPTree, b, ba_idx, o) # TODO: instantiate value bounds
    push!(tree.b, b)
    b_idx = length(tree.b)
    push!(tree.ba_children[ba_idx], o=>b_idx)
    return b_idx
end

function add_action!(tree::SARSOPTree, b_idx, a)
    ba_idx = length(tree.ba_children) + 1
    push!(tree.b_children[b_idx], a=>ba_idx)
    return ba_idx
end
