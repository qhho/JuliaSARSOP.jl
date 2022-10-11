function belief_norm(b::Vector{Float64}, b′::Vector{Float64}, terminals, not_terminals)
    #TODO: fix this nonsense
    if sum(b′[not_terminals]) != 0.
        if !isempty(terminals)
            b′[not_terminals] = b′[not_terminals] / (sum(b′[not_terminals]) / (1. - sum(b[terminals]) - sum(b′[terminals])))
            b′[terminals] += b[terminals]
        else
            b′[not_terminals] /= sum(b′[not_terminals])
        end
    else
        b′[terminals] += b[terminals]
        b′[terminals] /= sum(b′[terminals])
    end
    return b′
end

function max_alpha_val(Γ, b)
    max_α = first(Γ)
    max_val = -Inf
    for α ∈ Γ
        val = dot(α, b)
        if val > max_val
            max_α = α
            max_val = val
        end
    end
    return max_α.alpha
end

function backup_a!(α, pomdp::SparseTabularPOMDP, a, Γao)
    γ = discount(pomdp)
    R = @view pomdp.R[:,a]
    T = pomdp.T[a]
    Z = pomdp.O[a]
    Γa = @view Γao[:,:,a]
    @tullio α[s] = T[sp,s]*Z[sp,o]*Γa[sp,o]
    @. α = R + γ*α
end

function backup!(tree, b_idx)
    Γ = tree.Γ
    b = tree.b[b_idx]
    pomdp = tree.pomdp
    γ = discount(tree)
    S = states(tree)
    A = actions(tree)
    O = observations(tree)

    # TODO: can easily cache, but we have bigger fish to fry atm
    Γao = Array{Float64,3}(undef, length(S), length(O), length(A))

    for a ∈ A
        ba_idx = tree.b_children[b_idx][a]
        for o ∈ O
            bp_idx = tree.ba_children[ba_idx][o]
            bp = tree.b[bp_idx]
            Γao[:,o,a] .= max_alpha_val(Γ, bp)
        end
    end

    V = -Inf
    α_a = zeros(Float64, length(S))
    best_α = zeros(Float64, length(S))
    best_action = first(A)

    for a ∈ A
        α_a = backup_a!(α_a, pomdp, a, Γao)
        Qba = dot(α_a, b)
        tree.Qa_lower[b_idx][a] = Qba
        if Qba > V
            V = Qba
            best_α .= α_a
            best_action = a
        end
    end

    α = AlphaVec(best_α, best_action, [b_idx], [V])
    push!(Γ, α)
    tree.V_lower[b_idx] = V
end

function backup!(tree)
    for i ∈ reverse(eachindex(tree.sampled))
        backup!(tree, tree.sampled[i])
    end
end
