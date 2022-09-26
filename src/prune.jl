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
            Qa_upper = tree.Qa_upper[b_idx]#::Vector{<:Pair}
            Qa_lower = tree.Qa_lower[b_idx]#::Vector{<:Pair}
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

function belief_space_domination(α1, α2, B, δ)
    a1_dominant = true
    a2_dominant = true
    for b ∈ B
        !a1_dominant && !a2_dominant && return (false, false)
        δV = intersection_distance(α1, α2, b)
        δV ≤ δ && (a1_dominant = false)
        δV ≥ -δ && (a2_dominant = false)
    end
    return a1_dominant, a2_dominant
end

function intersection_distance(α1, α2, b)
    s = 0.0
    dot_sum = 0.0
    @inbounds for i ∈ eachindex(α1, α2, b)
        diff = α1[i] - α2[i]
        s += abs2(diff)
        dot_sum += diff*b[i]
    end
    return dot_sum / sqrt(s)
end

function prune_alpha!(tree::SARSOPTree, δ)
    Γ = tree.Γ
    B_valid = tree.b[map(!,tree.b_pruned)]
    pruned = falses(length(Γ))

    # checking if α_i dominates α_j
    for (i,α_i) ∈ enumerate(Γ)
        pruned[i] && continue
        for (j,α_j) ∈ enumerate(Γ)
            (j ≤ i || pruned[j]) && continue
            a1_dominant,a2_dominant = belief_space_domination(α_i, α_j, B_valid, δ)
            #=
            NOTE: α1 and α2 shouldn't technically be able to mutually dominate
            i.e. a1_dominant and a2_dominant should never both be true.
            But this does happen when α1 == α2 because intersection_distance returns NaN.
            Current impl prunes α2 without doing an equality check, removing
            the duplicate α. Could do equality check to short-circuit
            belief_space_domination which would speed things up if we have
            a lot of duplicates, but the equality check can slow things down
            if α's are sufficiently diverse.
            =#
            if a1_dominant
                pruned[j] = true
                # b_idx = Γ[j].witnesses[1]
                # Qba = dot(α_i, tree.b[b_idx])
                # tree.Qa_lower[b_idx][actionindex(tree.pomdp, Γ[i].action)] = Γ[i].action => Qba
            elseif a2_dominant
                pruned[i] = true
                # b_idx = Γ[i].witnesses[1]
                # Qba = dot(α_j, tree.b[b_idx])
                # tree.Qa_lower[b_idx][actionindex(tree.pomdp, Γ[j].action)]  =  Γ[j].action => Qba
                break
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
