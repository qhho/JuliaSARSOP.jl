function initUpperBound!(tree::SARSOPTree, b_idx::Int)
    b = tree.b[b_idx]
    tmp = 0.0
    for i in 1:length(values) 
        tmp += values[i] * b[i]
    end
    push!(tree.V_upper, tmp)
end

function getUpperBoundSimple(values::Vector{Float64}, b::Vector{Float64})
        return sum(values .* b)
end

function sawtoothUB!(tree::SARSOPTree, b_idx::Int)
    b = tree.b[b_idx]
    α_corner = tree.Vs_upper
    V_corner = b .* α_corner
    V_upper = tree.V_upper

    upperVvec = Float64[]
    for (bint, vint) in zip(tree.b, V_upper)  
        ϕ = minimum(b[s]/bint[s] for s in 1:length(b))
        push!(upperVvec, V_corner + ϕ * (vint - (bint .* α_corner)))
    end
    tree.V_upper[b_idx] = minimum(upperVvec)
end

# Get upper bound value for each belief in tree
function updateUpperBound!(tree::SARSOPTree)
    for b in tree.b
        # DO sawtooth updating on remaining belief nodes
        nothing
    end
    #tree.V_upper[b_idx] = tmp
end