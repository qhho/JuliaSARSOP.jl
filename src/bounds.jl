function initUpperBound!(tree::SARSOPTree, b_idx::Int)
    b = b[b_idx]
    tmp = 0.0
    for i in 1:length(values) 
        tmp += values[i] * b[i]
    end
    push!(tree.V_upper, tmp)
end

function getUpperBoundSimple(values::Vector{Float64}, b::Vector{Float64})
        return sum(values .* b)
end

# Get upper bound value for each belief in tree
function updateUpperBound!(tree::SARSOPTree)
    for b in tree.b
        
        nothing
    end
    #tree.V_upper[b_idx] = tmp
end