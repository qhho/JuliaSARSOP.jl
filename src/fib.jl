#=
NOTE: With default utility initialization not already an upper bound on value,
if too few iterations are run, the calculated upper value may not actually be
an upper bound on the value.

Same applies to using QMDP as an upper bound, but provided that the upper bound is
used (relative to lower bound) to guide search as well as determining convergence
the detriment to the final policy may be minor.
=#
struct FastInformedBound
    max_iter::Int
    max_time::Float64
    init_value::Float64
    α_tmp::Vector{Float64}
    FastInformedBound(n; init_value=0., max_time=Inf) = new(n, max_time, init_value, Float64[])
end

function update!(𝒫::POMDP, M::FastInformedBound, Γ, 𝒮, 𝒜, 𝒪)
    γ = discount(𝒫)

    for (a_idx, a) ∈ enumerate(𝒜)
        α_a = M.α_tmp
        for (s_idx, s) ∈ enumerate(𝒮)
            T = transition(𝒫, s, a)
            tmp = 0.0
            for o ∈ 𝒪
                Vmax = -Inf
                for α′ ∈ Γ
                    Vb′ = 0.0
                    for (sp_idx, sp) ∈ enumerate(𝒮)
                        Oprob = pdf(observation(𝒫, s, a, sp), o)
                        Tprob = pdf(T, sp)
                        @inbounds Vb′ += Oprob*Tprob*α′[sp_idx]
                    end
                    Vb′ > Vmax && (Vmax = Vb′)
                end
                tmp += Vmax
            end
            α_a[s_idx] = reward(𝒫, s, a) + γ*tmp
        end
        copyto!(Γ[a_idx], α_a)
    end
    return Γ
end

function POMDPs.solve(sol::FastInformedBound, pomdp::POMDP)
    t0 = time()
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)
    γ = discount(pomdp)

    init_value = sol.init_value
    Γ = if isfinite(sol.init_value)
        [fill(sol.init_value, length(S)) for a in A]
    else
        r_max = maximum(reward(pomdp, s, a) for a ∈ actions(pomdp), s ∈ states(pomdp))
        V̄ = r_max/(1-γ)
        [fill(V̄, length(S)) for a in A]
    end
    resize!(sol.α_tmp, length(S))

    iter = 0
    while iter < sol.max_iter && time() - t0 < sol.max_time
        update!(pomdp, sol, Γ, S, A, O)
        iter += 1
    end

    # return AlphaVectorPolicy(pomdp, Γ, A)
    return Γ
end
