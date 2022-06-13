function belief_norm(b::Vector{Float64}, b′::Vector{Float64}, terminals, not_terminals)
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

function backup_belief(tree::SARSOPTree, Γ::Vector{AlphaVec}, node::Int)
    b = tree.b[node]
    S = tree.states
    A = tree.ba_action
    O = tree.obs
    pomdp = tree.pomdp
    γ = tree._discount
    Γa = Vector{Vector{Float64}}(undef, length(A))

    terminals = tree.terminals
    not_terminals = tree.not_terminals

    for a in A
        Γao = Vector{Vector{Float64}}(undef, length(O))
        trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b[is] for sp in S, is in not_terminals], dims=2), dims=2)
        for o in O
            obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
            b′ = obs_probs .* trans_probs
            if sum(b′) > 0.
                b′ = belief_norm(b, b′, terminals, not_terminals)
            else
                b′ = zeros(length(S))
            end

            # extract optimal alpha vector at resulting belief
            Γao[obsindex(pomdp, o)] = _argmax(α -> α ⋅ b′, Γ)
        end

        for s in S
            if isterminal(pomdp, s)
                Γa[actionindex(pomdp, a)] = r(s,a)
            else
                tmp = 0.0
                for (i, o) in enumerate(O)
                    for (j, sp) in enumerate(S)
                        tmp += pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) * Γao[i][j]
                    end
                end
                Γa[actionindex(pomdp, a)] =  r(s, a) + γ*tmp
            end
        end
    end

    val,idx = findmax(map(αa -> αa ⋅ b, Γa))
    alphavec =  AlphaVec(Γa[idx],A[idx],node,val)

    return alphavec
end

function tree_backup!(tree::SARSOPTree, Γ::Vector{AlphaVec})
    for node in tree.b_touched
        push!(Γ,backup_belief(tree, Γ, tree.b[node]))
    end
end
