using JuliaSARSOP
using POMDPModels
using Test

@testset "Basic Functionality" begin
    pomdp = TigerPOMDP()
    @test SARSOPSolver() isa SARSOPSolver
    @test SARSOPTree(pomdp) isa SARSOPTree
end
