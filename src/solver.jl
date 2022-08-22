Base.@kwdef struct SARSOPSolver <: Solver
    epsilon::Float64    = 1e-3
    kappa::Float64      = 0.5
    delta::Float64      = 0.1
    max_time::Float64   = 2.0
    max_steps::Int      = 10
    verbose::Bool       = true
end

function POMDPs.solve(solver::SARSOPSolver, pomdp::POMDP{S,A}) where {S,A}
    tree = SARSOPTree(pomdp)
    
    start_time = time()
    Γnew = AlphaVec{A}[]
    while time()-start_time < solver.max_time
        @info "Running Sample"
        sample!(solver, tree)
        @info "Running Backup"
        tree_backup!(Γnew, tree)
        @info "Running Bounds Update"
        updateUpperBounds!(tree)
        updateLowerBounds!(tree)
        @info "Running Pruning"
        pruneTree!(tree)
        pruneAlpha!(Γnew, tree.Γ, solver.delta)
        # updateLowerBounds!(tree)
    end

    return AlphaVectorPolicy(pomdp, Γ, acts)
end
