function prune!(solver::SARSOPSolver, tree::SARSOPTree)
    # prune from B points that are provably suboptimal
    # For node b in Tree,
    prune!(tree)
    prune_alpha!(tree, solver.delta)
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

function prune_alpha!(tree::SARSOPTree, δ)
    Γ = tree.Γ
    B_valid = tree.b[map(!,tree.b_pruned)]
    V = [
        dot(α, b) + δ*norm(α,2)
        for α ∈ Γ, b ∈ B_valid
    ]
    pruned = falses(length(Γ))

    # checking if α_i dominates α_j
    for (i,α_i) ∈ enumerate(Γ)
        pruned[i] && continue
        for (j,α_j) ∈ enumerate(Γ)
            if i == j || pruned[j]
                continue
            else
                pruned[j] = all(V[i,:] > V[j,:])
            end
        end
    end
    deleteat!(Γ, pruned)
end

# function prune_alpha!(tree::SARSOPTree, δ::Float64)
#     # prune alpha based on witness nodes with delta dominance
#     #(this is different from SARSOP paper description, but similar to HSVI and SARSOP implementation in APPL)
#     # currently inefficient with pushing to non-fixed size vector
#     n_new = length(tree.sampled)
#     n_old = length(tree.Γ) - n_new
#     Γold = @view tree.Γ[1:n_old]
#     Γnew = @view tree.Γ[n_old+1:end]
#
#     Γfinal = copy(Γnew)
#     alpha_idxs_to_delete = Int[]
#     for (α_old_idx, α_old) in enumerate(Γold)
#         for α_new in Γnew
#             to_del = Int[]
#             for (witness_idx, (witness, value_at_witness)) in enumerate(zip(α_old.witnesses, α_old.value_at_witnesses))
#                 if tree.b_pruned[witness]
#                     push!(to_del, witness_idx)
#                 else
#                     b = tree.b[witness]
#                     val = dot(α_new, b)
#                     sq_alpha_dist = sum(abs2, α_new .- α_old) # Γnewα - α_old.alpha
#                     δV = val - value_at_witness
#                     deltaValue = sign(δV)*abs2(δV)/sqrt(sq_alpha_dist)
#                     if deltaValue > δ^2
#                         push!(α_new.witnesses, witness)
#                         push!(α_new.value_at_witnesses, val)
#                         push!(to_del, witness_idx)
#                         tree.Qa_lower[witness][actionindex(tree.pomdp, α_new.action)] = α_new.action => val
#                         tree.V_lower[witness] = val
#                     end
#                 end
#             end
#             deleteat!(α_old.witnesses, to_del)
#             deleteat!(α_old.value_at_witnesses, to_del)
#         end
#         push!(alpha_idxs_to_delete, α_old_idx)
#         # if !isempty(α_old.witnesses) # keep alpha vec if it still has witnesses
#         #     push!(Γfinal, α_old)
#         # end
#     end
#     deleteat!(tree.Γ, alpha_idxs_to_delete)
#     # resize!(Γnew, length(Γfinal))
#     # copyto!(Γnew, Γfinal)
# end
