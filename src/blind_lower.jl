Base.@kwdef struct BlindLowerBound <: Solver
    max_iter::Int               = typemax(Int)
    max_time::Float64           = 1.
    bel_res::Float64            = 1e-3
    α_tmp::Vector{Float64}      = Float64[]
    residuals::Vector{Float64}  = Float64[]
end

function update!(pomdp::POMDP, M::BlindLowerBound, Γ, S, A, O)
    residuals = M.residuals
    γ = discount(pomdp)

    for (a_idx, a) in enumerate(A)
        α_a = M.α_tmp
        for (s_idx, s) in enumerate(S)
            Qas = reward(pomdp, s, a)
            for (s′, p) in weighted_iterator(transition(pomdp, s, a))
                Qas += γ*p*Γ[a_idx][stateindex(pomdp, s′)]
            end
            α_a[s_idx] = Qas
        end
        res = bel_res(Γ[a_idx], α_a)
        residuals[a_idx] = res
        copyto!(Γ[a_idx], α_a)
    end
    return Γ
end

function worst_state_alphas(pomdp::POMDP, S, A)
    γ = discount(pomdp)
    Γ = [zeros(length(S)) for _ in eachindex(A)]
    for (a_idx, a) in enumerate(A)
        for (s_idx, s) in enumerate(S)
            Qa = 1 / (1 - γ) * minimum(reward(pomdp, s′, a) for (s′, p) in weighted_iterator(transition(pomdp, s, a)))
            Γ[a_idx][s_idx] = Qa
        end
    end
    return Γ
end

function POMDPs.solve(sol::BlindLowerBound, pomdp::POMDP)
    t0 = time()
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)

    Γ = worst_state_alphas(pomdp, S, A)
    resize!(sol.α_tmp, length(S))
    residuals = resize!(sol.residuals, length(A))

    iter = 0
    res_criterion = <(sol.bel_res)
    while iter < sol.max_iter && time() - t0 < sol.max_time
        update!(pomdp, sol, Γ, S, A, O)
        iter += 1
        all(res_criterion,residuals) && break
    end

    return AlphaVectorPolicy(pomdp, Γ, A)
end
