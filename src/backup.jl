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
        val = dot(α.alpha, b)
        if val > max_val
            max_α = α
            max_val = val
        end
    end
    return max_α.alpha
end

function backup_belief(tree::SARSOPTree, node::Int)
    b = tree.b[node]
    S = states(tree)
    A = actions(tree)
    O = observations(tree)
    pomdp = tree.pomdp
    γ = tree._discount
    Γ = tree.Γ
    Γa = Vector{Vector{Float64}}(undef, length(A))

    terminals = tree.terminals
    not_terminals = tree.not_terminals

    for a in A
        Γao = Vector{Vector{Float64}}(undef, length(O))
        trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b[is] for sp in S, is in not_terminals], dims=2), dims=2)
        for (o_idx,o) in enumerate(O)
            obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
            b′ = obs_probs .* trans_probs
            if sum(b′) > 0.
                b′ = belief_norm(b, b′, terminals, not_terminals)
            else
                b′ = zeros(length(S))
            end

            # extract optimal alpha vector at resulting belief
            Γao[o_idx] = max_alpha_val(Γ, b′)
            # Γao[o_idx] = argmax(α -> α.alpha ⋅ b′, Γ).alpha
        end

        Γs = Vector{Float64}(undef, length(S))
        for s in S
            if isterminal(pomdp, s)
                Γs[stateindex(pomdp, s)] = 0.0
            else
                tmp = 0.0
                for (i, o) in enumerate(O)
                    for (j, sp) in enumerate(S)
                        tmp += pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) * Γao[i][j]
                    end
                end
                Γs[stateindex(pomdp, s)] =  reward(pomdp, s, a) + γ*tmp
            end
        end
        Γa[actionindex(pomdp, a)] = Γs
    end

    # val,idx = findmax(map(αa -> αa ⋅ b, Γa))

    idx_max = 0
    v_max = -Inf
    for i ∈ eachindex(Γa)
        Qba = dot(Γa[i], b)
        ba_idx = tree.b_children[node][i].second
        tree.Qa_lower[node][i] = tree.Qa_lower[node][i].first => Qba
        if Qba > v_max
            v_max = Qba
            idx_max = i
        end
    end

    tree.V_lower[node] = v_max

    alphavec = AlphaVec(Γa[idx_max],A[idx_max],[node],[v_max])
    return alphavec
end

function tree_backup!(Γnew::Vector{<:AlphaVec}, tree::SARSOPTree)
    resize!(Γnew,length(tree.sampled))
    for (i,node) in enumerate(tree.sampled)
        Γnew[i] = backup_belief(tree, node)
    end
    push!(tree.Γ,Γnew...)
end
