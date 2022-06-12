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

    function SARSOPTree{S,A,O}(pomdp::POMDP) where {S,A,O}
        return new(
            ordered_states(pomdp),
            Vector{Float64}[],
            Vector{Int}[],
            Float64[],
            Float64[],
            ordered_observations(pomdp),
            Vector{Int}[],
            ordered_actions(pomdp),
            discount(pomdp)
        )
    end
end