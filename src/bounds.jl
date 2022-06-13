function initUpperBound!(tree::SARSOPTree, values::Vector{Float64})
    b0 = tree.b[1]
    tmp = 0.0
    for i in 1:length(values) 
        tmp += values[i] * b0[i]
    end
    push!(tree.V_upper, tmp)
end

function getUpperBoundSimple(values::Vector{Float64}, b::Vector{Float64})
        return sum(values .* b)
end

# Get upper bound value for each belief in tree
function updateUpperBound!(tree::SARSOPTree)
    for b in tree.b
        # DO sawtooth updating 
        nothing
    end
end