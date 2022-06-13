struct SARSOPTree{S,A,O}
    states::Vector{S}
    b::Vector{Vector{Float64}}
    children::Vector{Vector{Int}} # to children *ba nodes*
    V_upper::Vector{Float64}
    V_lower::Vector{Float64}

    obs::Vector{O}

    ba_children::Vector{Vector{Int}} # to b children 
    ba_action::Vector{A}

    _discount::Float64

    not_terminals::Vector{Int}
    terminals::Vector{Int}

    function SARSOPTree{S,A,O}(pomdp::POMDP) where {S,A,O}
        solver = ValueIterationSolver()
        upper_policy = solve(solver, UnderlyingMDP(pomdp))
        upper_values = upper_policy.util
        not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
        terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]

        return new(
            ordered_states(pomdp),
            Vector{Float64}[],
            Vector{Int}[],
            upper_values,
            Float64[],
            ordered_observations(pomdp),
            Vector{Int}[],
            ordered_actions(pomdp),
            discount(pomdp),
            not_terminals,
            terminals
        )
    end
end