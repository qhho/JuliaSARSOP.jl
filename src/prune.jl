function prune(tree::SARSOPTree, Γ::AlphaVectorPolicy)
    # prune from B points that are provably suboptimal
    # For node b in Tree, 

    pruneTree!(tree::SARSOPTree)
    pruneAlpha!(Γ, δ)
end

function pruneTree!(tree::SARSOPTree)



end

function pruneAlpha!(Γ::AlphaVectorPolicy, δ::Float64)
    # prune alpha based on witness nodes with delta dominance 
    #(this is different from SARSOP paper description, but similar to HSVI and SARSOP implementation in APPL)


end