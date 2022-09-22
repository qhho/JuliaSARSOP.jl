using Revise
using JuliaSARSOP
const JSOP = JuliaSARSOP # convenience alias
using POMDPModels
using POMDPTools
using Test
using POMDPs


@testset "Basic Functionality" begin
    pomdp = TigerPOMDP()
    solver = SARSOPSolver()
    @test solver isa SARSOPSolver
    @test SARSOPTree(pomdp) isa SARSOPTree
    # @test policy = solve(solver, pomdp)
end

include("sample.jl")

include("updater.jl")

include("tree.jl")


begin
    pomdp = TigerPOMDP();
    solver = SARSOPSolver(max_steps = 1000);
    tree = SARSOPTree(pomdp);
    for _ ∈ 1:100
        JSOP.sample!(solver, tree)
        JSOP.backup!(tree)
        JSOP.updateUpperBounds!(tree)
        JSOP.prune!(solver, tree)
        # @show tree.sampled
    end
    @show tree.Γ
    @show tree.V_lower[1]
    @show tree.V_upper[1]
    # @show tree.V_upper

    solver_noprune = SARSOPSolver(max_steps = 10);
    tree_noprune = SARSOPTree(pomdp);
    for _ ∈ 1:1000
        JSOP.sample!(solver, tree_noprune)
        JSOP.backup!(tree_noprune)
        JSOP.prune!(solver, tree_noprune)
        # @show tree.sampled
    end
    @show tree_noprune.Γ
    @show tree_noprune.V_lower[1]
    @show tree_noprune.V_upper[1]
    # @show tree_noprune.V_upper

end


begin
    pomdp = TigerPOMDP();
    solver = SARSOPSolver(max_steps = 100);
    tree = SARSOPTree(pomdp);
    for _ ∈ 1:100
        JSOP.sample!(solver, tree)
        JSOP.backup!(tree)
        JSOP.updateUpperBounds!(tree)
        JSOP.prune!(solver, tree)
    end
    @show tree.Γ
    @show tree.V_lower[1]
    @show tree.V_upper[1]
end

begin
    solver_noprune = SARSOPSolver(max_steps = 10);
    tree_noprune = SARSOPTree(pomdp);
    for _ ∈ 1:1000
        JSOP.sample!(solver, tree_noprune)
        JSOP.backup!(tree_noprune)
        JSOP.prune!(solver, tree_noprune)
    end
    @show tree_noprune.Γ
    @show tree_noprune.V_lower[1]
    @show tree_noprune.V_upper[1]
end