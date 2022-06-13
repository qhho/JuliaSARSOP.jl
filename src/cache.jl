struct SARSOPCache
    poba::Vector{Float64}
end

SARSOPCache(l::Int) = SARSOPCache(Vector{Float64}(undef, l))
