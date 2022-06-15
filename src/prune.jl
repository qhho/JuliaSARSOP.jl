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
        Qa_upper = Qa_upper[b_idx]
        Qa_lower = Qa_lower[b_idx]
        ba = tree.b_children[b_idx]
        for (idx, Qvals) in enumerate(zip(Qa_lower, Qa_upper))
            if (tree.ba_children[ba[idx].second][1].second != 0) #assume deleted nodes are 0
                for i in idx+1:length(Qa_upper)
                    if (Qa_upper[i].second < Qvals[1].second)
                        for b_child_pruned_idx in tree.ba_children[ba[i].second]
                            pushfirst!(tree.b_children[b_child_pruned_idx], Pair(tree.ba_action[ba[i].second], 0)) # set first element of root of subtree to 0
                        end
                        pushfirst!(tree.ba_children[ba[i].second], Pair(tree.obs[1], 0)) # hacky method to set first element to deleted node
                        
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