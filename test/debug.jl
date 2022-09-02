using Revise
using JuliaSARSOP
using POMDPModels
using Test
using POMDPs

pomdp = TigerPOMDP()
solver = SARSOPSolver()
tree = SARSOPTree(pomdp)
# @testset "Tree Sampling" begin
#     pomdp = TigerPOMDP()
#     tree = SARSOPTree(pomdp)
#     solver = SARSOPSolver()
#     sample!(solver, tree)

# end
A = actiontype(pomdp)
O = obstype(pomdp)
b_idx = 1
#test insert_root!
begin
    pomdp = tree.pomdp
    b0 = initialstate(pomdp)
    b = initialize_belief(DiscreteUpdater(pomdp), b0).b
    push!(tree.b, b)
    push!(tree.b_children, Pair{actiontype(pomdp), Int}[])
    push!(tree.b_parent, (-1, -1, -1))
    push!(tree.V_upper, JuliaSARSOP.init_root_value(tree, b))
    JuliaSARSOP.init_lower_value!(tree, pomdp)
    push!(tree.V_lower, JuliaSARSOP.lower_value(tree, b))
    push!(tree.b_pruned, false)
end

#test fill_belief
begin
    γ = discount(tree.pomdp)
    ACT = tree.actions
    OBS = tree.obs
    N_OBS = length(OBS)
    b = tree.b[b_idx]
    n_b = length(tree.b)
    n_ba = length(tree.ba_children)
    b_children = Vector{Pair{A, Int}}(undef, length(ACT))
    Qa_upper = Vector{Pair{A, Float64}}(undef, length(ACT))
    Qa_lower = Vector{Pair{A, Float64}}(undef, length(ACT))
end

begin
    for (a_idx,a) in enumerate(ACT)
        b_children[a_idx] = a => (n_ba + a_idx)

        # TODO: If all observations are always expanded, there's little to no need for a `o => bp_idx` pair
        #       Just need nested vector mapping ba_idx => o_idx => bp_idx
        ba_children = Vector{Pair{O, Int}}(undef, N_OBS)
        poba = Vector{Float64}(undef, N_OBS)

        Rba = JuliaSARSOP.belief_reward(tree, b, a)
        Q̄ = Rba
        Q̲ = Rba

        for (o_idx, o) in enumerate(OBS)
            # bp_idx = n_b + o_idx + N_OBS*(a_idx-1)
            bp_idx, _, _ = JuliaSARSOP.update_and_push(tree, b_idx, a, o)
            b′ = tree.b[bp_idx]
            @show tree.b_children
            po = JuliaSARSOP.obs_prob(tree, b, a, o, b′)
            ba_children[o_idx] = (o => bp_idx)
            poba[o_idx] = po
            V̄ = JuliaSARSOP.upper_value(tree, b′)
            V̲ = JuliaSARSOP.lower_value(tree, b′)
            Q̄ += γ*po*V̄
            Q̲ += γ*po*V̲
            # push!(tree.V_upper, V̄)
            # push!(tree.V_lower, V̲)
            # push!(tree.b_pruned, false)
        end
        push!(tree.ba_pruned, false)
        push!(tree.ba_children, ba_children)
        push!(tree.poba, poba)

        Qa_upper[a_idx] = a => Q̄
        Qa_lower[a_idx] = a => Q̲
    end
    push!(tree.b_children, b_children)
    tree.b_children[b_idx] = b_children
    push!(tree.Qa_upper, Qa_upper)
    push!(tree.Qa_lower, Qa_lower)
end


Γnew = JuliaSARSOP.AlphaVec{A}[]
JuliaSARSOP.sample!(solver, tree)
