function prune!(solver::SARSOPSolver, tree::SARSOPTree)
    # prune from B points that are provably suboptimal
    # For node b in Tree,
    prune!(tree)
    prune_alpha!(tree.Γ, Γold, δ)
end

function pruneSubTreeBa!(tree::SARSOPTree, ba_idx::Int)
    for (o,b_idx) in tree.ba_children[ba_idx]
        pruneSubTreeB!(tree, b_idx)
    end
    tree.ba_pruned[ba_idx] = true
end

function pruneSubTreeB!(tree::SARSOPTree, b_idx::Int)
    for (a, ba_idx) in tree.b_children[b_idx]
        pruneSubTreeBa!(tree, ba_idx)
    end
    tree.b_pruned[b_idx] = true
end

function prune!(tree::SARSOPTree)
    # For a node b, if upper bound Q(b,a) < lower bound Q(b, a'), prune a
    for b_idx in tree.sampled
        if tree.b_pruned[b_idx]
            break
        else
            # this `Vector{<:Pair}` shit is really annoying please GOD change it
            Qa_upper = tree.Qa_upper[b_idx]::Vector{<:Pair}
            Qa_lower = tree.Qa_lower[b_idx]::Vector{<:Pair}
            b_children = tree.b_children[b_idx]
            ba = tree.b_children[b_idx]
            max_lower_bound = maximum(last, Qa_lower)
            for (idx, (a, Qba)) ∈ enumerate(Qa_upper)
                ba_idx = last(b_children[idx])
                all_ba_pruned = true
                if !tree.ba_pruned[ba_idx] && last(Qa_upper[idx]) < max_lower_bound
                    pruneSubTreeBa!(tree, ba_idx)
                else
                    all_ba_pruned = false
                end
                all_ba_pruned && (tree.b_pruned[b_idx] = true)
            end
        end
    end
end

function prune_alpha!(Γnew::Vector{<:AlphaVec}, Γold::Vector{<:AlphaVec}, δ::Float64)
    # prune alpha based on witness nodes with delta dominance
    #(this is different from SARSOP paper description, but similar to HSVI and SARSOP implementation in APPL)
    # currently inefficient with pushing to non-fixed size vector
    Γfinal = copy(Γnew)
    for alphavec_old in Γold
        for alphavec_new in Γnew
            to_del = Int[]
            Γnewα = alphavec_new.alpha
            for (witness_idx, (witness, value_at_witness)) in enumerate(zip(alphavec_old.witnesses, alphavec_old.value_at_witnesses))
                if tree.b_pruned[witness]
                    push!(to_del, witness_idx)
                else
                    b = tree.b[witness]
                    val = 0.0
                    for (idx, v) in enumerate(Γnewα)
                        val += v * b[idx]
                    end
                    alpha_dist = Γnewα - alpha_vec_old.alpha
                    deltaValue = (val - value_at_witness)*(val - value_at_witness)/(alpha_dist*alpha_dist)
                    if (deltaValue > δ^2)
                        push!(Γnewα.witnesses, witness)
                        push!(Γnewα.value_at_witnesses, val)
                        push!(to_del, witness_idx)
                    end
                end
            end
            deleteat!(alphavec_old.witnesses, to_del)
            deleteat!(alphavec_old.value_at_witnesses, to_del)
        end
        if !isempty(alphavec_old.witnesses)
            push!(Γfinal, alphavec_old)
        end
    end
    resize!(Γnew, length(Γfinal))
    copyto!(Γnew, Γfinal)
end
