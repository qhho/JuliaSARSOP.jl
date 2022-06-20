function update(tree::SARSOPTree{S,A,O}, b::Vector{Float64}, a::A, o::O) where {S,A,O}
    bp = zero(b)
    return _update!(bp, tree, b, a, o)
end

# https://github.com/JuliaPOMDP/BeliefUpdaters.jl/blob/afa7d80e47631340cb210548a0e2dc8a73886e2d/src/discrete.jl#L111
# inplace in case we want to cache beliefs prior to search
function _update!(bp, tree::SARSOPTree, b::Vector{Float64}, a::A, o::O) where {S,A,O}
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
