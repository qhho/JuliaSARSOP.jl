@testset "tree" begin
    @testset "poba" begin
        pomdp = TigerPOMDP()
        bu = DiscreteUpdater(pomdp)
        tree = SARSOPTree(pomdp)
        O = ordered_observations(pomdp)
        b0 = initialstate(pomdp)
        b0_vec = initialize_belief(DiscreteUpdater(pomdp), b0).b
        for a ∈ actions(pomdp)
            prob_sum = 0.0
            for o ∈ observations(pomdp)
                poba = JSOP.obs_prob(tree, b0_vec, a, o)
                @test 0. ≤ poba ≤ 1.
                prob_sum += poba
            end
            # probability of getting *any* observation at all must be 1
            @test prob_sum ≈ 1.0 # sum of conditional probabilities must equal 1
        end
    end
end
