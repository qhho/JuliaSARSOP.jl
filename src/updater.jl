function update(tree::SARSOPTree{S,A,O}, b::Vector{Float64}, a::A, o::O) where {S,A,O}
    bp = zero(b)
    return _update!(bp, tree, b, a, o)
end

# what if observation o has 0 probability in transition ba???
# maybe it's okay to have new belief as all zeros for the case that
# https://github.com/JuliaPOMDP/BeliefUpdaters.jl/blob/afa7d80e47631340cb210548a0e2dc8a73886e2d/src/discrete.jl#L111
# inplace in case we want to cache beliefs prior to search
function _update!(bp, tree::SARSOPTree{S,A,O}, b::Vector{Float64}, a::A, o::O) where {S,A,O}
    bp_sum = 0.0
    pomdp = tree.pomdp

    for (s_idx, s) in enumerate(tree.states)
        if b[s_idx] > 0.0
            td = transition(pomdp, s, a)
            for (sp, tp) in weighted_iterator(td)
                sp_idx = stateindex(pomdp, sp)
                op = obs_weight(pomdp, s, a, sp, o)
                w = op * tp * b[s_idx]
                bp[sp_idx] += w
                bp_sum += w
            end
        end
    end


    if iszero(bp_sum)
        error("""
              Failed discrete belief update: new probabilities sum to zero.
              b = $b
              a = $a
              o = $o
              Failed discrete belief update: new probabilities sum to zero.
              """)
    end

    return bp ./= bp_sum
end

function update_and_shift(tree::SARSOPTree, b::Vector{Float64}, a, o)
    bp = update(tree, b, a, o)
    return shift_to_nonterminal!(bp, tree.terminal_s_idxs)
end

function shift_to_nonterminal!(b, terminal_s_idxs::Vector{Int})
    b[terminal_s_idxs] .= zero(eltype(b))
    b_sum = sum(b)
    return b_sum > 0. ? b ./= b_sum : b
end

function is_terminal_belief(b, terminal_s_idxs::Vector{Int})
    return sum(b[terminal_s_idxs]) ≈ 1.0
end
