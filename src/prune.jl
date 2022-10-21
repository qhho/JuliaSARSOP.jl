function prune!(solver::SARSOPSolver, tree::SARSOPTree)
    # prune from B points that are provably suboptimal
    # For node b in Tree,
    prune!(tree)
    prune_alpha!(tree, solver.delta)
end

function pruneSubTreeBa!(tree::SARSOPTree, ba_idx::Int)
    for (o,b_idx) in tree.ba_children[ba_idx]
        pruneSubTreeB!(tree, b_idx)
        unbin_value(tree, b_idx)
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
            elseif a2_dominant
                pruned[i] = true
                break
            end
        end
    end
    deleteat!(Γ, pruned)
end

function unbin_value(tree::SARSOPTree, b_idx::Int)
    value = tree.V_upper[b_idx]
    bindex = tree.bel_bins[[x.first == b_idx for x in tree.bel_bins]]
    bin = tree.bins[bindex[1]][bindex[2]]
    bin[1] = (bin[1]*bin[2]-value)/(bin[2]-1)
    bin[2] -= 1
    pop!(tree.bel_bins,tree.bel_bins[b_idx])
end

function unbin_values(tree::SARSOPTree, pruned::Vector{Int})
    for ba_idx in pruned
        unbin_value(tree,ba_idx)
    end
end
