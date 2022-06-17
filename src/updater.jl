function BeliefUpdaters.update(tree::SARSOPTree{S,A,O}, b::Vector{Float64}, a::A, o::O)
    bp = zero(b)
    return _update!(bp, tree.pomdp, b, a, o)
end

# https://github.com/JuliaPOMDP/BeliefUpdaters.jl/blob/afa7d80e47631340cb210548a0e2dc8a73886e2d/src/discrete.jl#L111
# inplace in case we want to cache beliefs prior to search
function _update!(bp, pomdp::POMDP{S,A,O}, b::Vector{Float64}, a::A, o::O) where {S,A,O}
    S = tree.states
    bp_sum = 0.0

    for (s_idx, s) in enumerate(S)

        if pdf(b, s) > 0.0
            td = transition(pomdp, s, a)

            for (sp_idx, tp) in weighted_iterator(td)
                spi = stateindex(pomdp, sp)
                op = obs_weight(pomdp, s, a, sp, o)

                w = op * tp * b.b[s_idx]
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
