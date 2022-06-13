function getUpperBound(values::Vector{Float64}, b::Vector{Float64})
    tmp = 0.0
    for i in 1:length(values) 
        tmp += values[i] * b[i]
    end
    return tmp
end

function getUpperBoundSimple(values::Vector{Float64}, b::Vector{Float64})
        return sum(values .* b)
end