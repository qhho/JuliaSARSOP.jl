@testset "updater" begin
    pomdp = TigerPOMDP()
    tree = SARSOPTree(pomdp)
    updater = DiscreteUpdater(pomdp)

    b0 = initialstate(pomdp)
    b0_vec = initialize_belief(updater, b0).b
    for a ∈ actions(pomdp), o ∈ observations(pomdp)
        @test all(update(updater, b0, a, o).b .≈ JSOP.update(tree, b0_vec, a, o))
    end
end
