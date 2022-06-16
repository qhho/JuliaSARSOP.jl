function prune(tree::SARSOPTree, Γnew::Vector{AlphaVec}, Γold::Vector{AlphaVec})
    # prune from B points that are provably suboptimal
    # For node b in Tree, 
    pruneTree!(tree::SARSOPTree)
    Γnew = pruneAlpha(Γnew, Γold, δ)
end

function pruneTree!(tree::SARSOPTree)
    # For a node b, if upper bound Q(b,a) < lower bound Q(b, a'), prune a
    # current method doesn't care about pruned subtrees, how do we check if subtree is pruned?
    for b_idx in tree.b_touched
        if tree.b_pruned[b_idx]
            break
        else
            Qa_upper = Qa_upper[b_idx]
            Qa_lower = Qa_lower[b_idx]
            ba = tree.b_children[b_idx]
            for (idx, Qvals) in enumerate(zip(Qa_lower, Qa_upper))
                if (!tree.ba_pruned[ba[idx].second]!= 0)
                    for i in idx+1:length(Qa_upper)
                        if (Qa_upper[i].second < Qvals[1].second)
                            for b_child_pruned_idx in tree.ba_children[ba[i].second]
                                tree.b_pruned[b_child_pruned_idx] = true
                            end
                            tree.ba_pruned[ba[idx].second] = true
                        end
                    end
                end
            end
        end
    end
end

function pruneAlpha(Γnew::Vector{AlphaVec}, Γold::Vector{AlphaVec}, δ::Float64)
    # prune alpha based on witness nodes with delta dominance 
    #(this is different from SARSOP paper description, but similar to HSVI and SARSOP implementation in APPL)
    # currently inefficient with pushing to non-fixed size vector
    Γfinal = copy(Γnew)
    for alphavec_new in Γnew
        Γnewfinal
        Γnewα = alphavec_new.alpha
        for alphavec_old in Γold
            witness = alphavec_old.witness
            val = 0.0
            for (idx, v) in enumerate(Γnewα) 
                val += v * witness[idx]
            end
            alpha_dist = Γnewα - alpha_vec_old.alpha
            deltaValue = sqrt((val - alphavec_old.value_at_witness)*(val - alphavec_old.value_at_witness)/(alpha_dist*alpha_dist))
            if (deltaValue < δ)
                push!(Γfinal, alphavec_old)
            end
        end
    end

    return Γfinal
end