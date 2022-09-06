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

    function get_LpUp(tree, ba_idx, Rba, Lt, Ut, op_idx)
        γ = discount(tree)
        Lp,Up = Rba,Rba

        for (o_idx,o) ∈ enumerate(observations(tree))
            if o_idx == op_idx
                Lp += γ*JSOP.obs_prob(tree, ba_idx, op_idx)*Lt
                Up += γ*JSOP.obs_prob(tree, ba_idx, op_idx)*Ut
            else
                bp_idx = tree.ba_children[ba_idx][o_idx].second
                Lp += γ*JSOP.obs_prob(tree, ba_idx, op_idx)*tree.V_lower[bp_idx]
                Up += γ*JSOP.obs_prob(tree, ba_idx, op_idx)*tree.V_upper[bp_idx]
            end
        end
        return Lp, Up
    end

    @testset "LtUt" begin
        pomdp = TigerPOMDP()
        tree = SARSOPTree(pomdp)
        sol = SARSOPSolver(epsilon=1.)

        t = 1
        b_idx = 1
        V̲, V̄ = tree.V_lower[b_idx], tree.V_upper[b_idx]
        ϵ = sol.epsilon
        γ = discount(tree)
        L = tree.V_lower[1]
        U = L + sol.epsilon


        JSOP.fill_belief!(tree, b_idx)
        Q̲, Q̄, ap_idx = JSOP.max_r_and_q(tree, b_idx)
        a′, ba_idx = tree.b_children[b_idx][ap_idx]
        Rba′ = JSOP.belief_reward(tree, tree.b[b_idx], a′)
        L′ = max(L, Q̲)
        U′ = max(U, Q̲ + γ^(-t)*ϵ)
        op_idx = JSOP.best_obs(tree, b_idx, ba_idx)
        Lt, Ut = JSOP.get_LtUt(tree, ba_idx, Rba′, L′, U′, op_idx)
        Lp, Up = get_LpUp(tree, ba_idx, Rba′, Lt, Ut, op_idx)
        @test Lp ≈ L′
        @test Up ≈ U′
    end
end
