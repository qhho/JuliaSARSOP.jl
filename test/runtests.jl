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

# @testset "Tiger POMDP" begin
#     pomdp = TigerPOMDP();
#     solver = SARSOPSolver(max_steps = 100, epsilon = 0.5);
#     Γ = solve(solver, pomdp)
#     @show Γ
# end


@testset "Tiger POMDP" begin
    pomdp = TigerPOMDP();
    solver = SARSOPSolver(epsilon = 0.5, precision = 1e-3);
    tree = SARSOPTree(pomdp);
    Γ = solve(solver, pomdp)
    iterations = 0
    while JSOP.root_diff(tree) > solver.precision
        iterations += 1
        JSOP.sample!(solver, tree)
        JSOP.backup!(tree)
        JSOP.updateUpperBounds!(tree)
        JSOP.prune!(solver, tree)
    end
    @test tree.V_lower[1] - 19.37 < 0.001
    @test JSOP.root_diff(tree) < solver.precision

    solverCPP = SARSOP.SARSOPSolver(trial_improvement_factor = 0.5, precision = 1e-3, verbose = false);
    policyCPP = solve(solverCPP, pomdp);
    @test value(policyCPP, initialstate(pomdp)) - tree.V_lower[1] < solver.precision
    @test value(policyCPP, initialstate(pomdp)) - value(Γ, initialstate(pomdp)) < solver.precision
end

@testset "Baby POMDP" begin
    pomdp = BabyPOMDP();
    solver = SARSOPSolver(epsilon = 0.1, delta = 0.1);
    tree = SARSOPTree(pomdp);
    Γ = solve(solver, pomdp)
    iterations = 0
    while JSOP.root_diff(tree) > solver.precision
        iterations += 1
        JSOP.sample!(solver, tree)
        JSOP.backup!(tree)
        JSOP.updateUpperBounds!(tree)
        JSOP.prune!(solver, tree)
    end
    @test tree.V_lower[1] - 19.37 < 0.001
    @test JSOP.root_diff(tree) < solver.precision

    solverCPP = SARSOP.SARSOPSolver(trial_improvement_factor = 0.5, precision = 1e-3, verbose = false);
    policyCPP = solve(solverCPP, pomdp);
    @test value(policyCPP, initialstate(pomdp)) - tree.V_lower[1] < solver.precision
    @test value(policyCPP, initialstate(pomdp)) - value(Γ, initialstate(pomdp)) < solver.precision
end