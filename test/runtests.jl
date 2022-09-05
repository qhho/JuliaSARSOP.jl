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
