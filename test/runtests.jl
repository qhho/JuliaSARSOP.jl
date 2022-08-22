using JuliaSARSOP
using POMDPModels
using Test
using POMDPs

@testset "Basic Functionality" begin
    pomdp = TigerPOMDP()
    solver = SARSOPSolver()
    @test solver isa SARSOPSolver
    @test SARSOPTree(pomdp) isa SARSOPTree
    # @test policy = solve(solver, pomdp)
end

@testset "sample" begin
    pomdp = TigerPOMDP()
    solver = SARSOPSolver(max_steps = 10)
    tree = SARSOPTree(pomdp)
    JuliaSARSOP.sample!(solver, tree)
end
