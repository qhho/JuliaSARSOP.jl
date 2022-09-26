using Revise
using JuliaSARSOP
const JSOP = JuliaSARSOP # convenience alias
using POMDPModels
using POMDPTools
using Test
using POMDPs
import SARSOP


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

@testset "Tiger POMDP" begin
    pomdp = TigerPOMDP();
    solver = SARSOPSolver(max_steps = 100, epsilon = 0.5);
    Γ = solve(solver, pomdp)
    @show Γ
end


begin
    pomdp = TigerPOMDP();
    solver = SARSOPSolver(max_steps = 100, epsilon = 0.5);
    tree = SARSOPTree(pomdp);
    for _ ∈ 1:50
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
    pomdp = BabyPOMDP();
    solver = SARSOPSolver(max_steps = 1000, epsilon = 0.1, delta = 0.1);
    tree = SARSOPTree(pomdp);
    # push!(tree.sampled, 1);
    # JSOP.backup!(tree);
    # JSOP.updateUpperBounds!(tree);
    for _ ∈ 1:20
        JSOP.sample!(solver, tree)
        JSOP.backup!(tree)
        JSOP.updateUpperBounds!(tree)
        JSOP.prune!(solver, tree)
    #     @show reverse(tree.sampled)[2:end]
    end
    @show tree.Γ
    @show tree.V_upper[1]
    @show tree.V_lower[1]
end

begin
    pomdp = BabyPOMDP();
    solver = SARSOP.SARSOPSolver();
    policy = solve(solver, pomdp);
    @show policy
end

begin
    pomdp = TigerPOMDP();
    solver = SARSOP.SARSOPSolver();
    policy = solve(solver, pomdp);
    @show policy
end