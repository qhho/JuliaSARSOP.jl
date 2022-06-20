Base.@kwdef struct SARSOPSolver <: Solver
    epsilon::Float64    = 1e-3
    kappa::Float64      = 0.5
    delta::Float64      = 0.1
    max_time::Float64   = Inf
    verbose::Bool       = true
end

function POMDPs.solve(solver::SARSOPSolver{S,A}, pomdp::POMDP) where {S,A}
    tree = SARSOPTree(pomdp)

    initUpperBound!(tree, 1)
    
    start_time = time()
    Γnew = AlphaVec{A}[]
    while time()-start_time < solver.max_time
        sample!(solver, tree)
        tree_backup!(Γnew, tree)
        pruneAlpha!(Γnew, tree.Γ, solver.delta)
        updateBounds!(tree, Γnew)
        pruneTree!(tree)
    end

    return AlphaVectorPolicy(pomdp, Γ, acts)
end
