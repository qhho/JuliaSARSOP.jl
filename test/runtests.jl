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


begin
    pomdp = TigerPOMDP();
    solver = SARSOPSolver(max_steps = 500);
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
    pomdp = BabyPOMDP();
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
    pomdp = BabyPOMDP();
    solver = SARSOP.SARSOPSolver();
    policy = solve(solver, pomdp);
    @show policy
end