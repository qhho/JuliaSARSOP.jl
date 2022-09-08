Base.@kwdef struct SARSOPSolver <: Solver
    epsilon::Float64    = 0.5
    precision::Float64  = 1e-3
    kappa::Float64      = 0.5
    delta::Float64      = 0.1
    max_time::Float64   = 2.0
    max_steps::Int      = 100
    verbose::Bool       = true
end

function POMDPs.solve(solver::SARSOPSolver, pomdp::POMDP{S,A}) where {S,A}
    tree = SARSOPTree(pomdp)

    start_time = time()
    while time()-start_time < solver.max_time && root_diff(tree) > precision
        @info "Running Sample"
        sample!(solver, tree)
        @info "Running Backup"
        backup!(tree)
        @info "Running Bounds Update"
        updateUpperBounds!(tree)
        @info "Running Pruning"
        prune!(solver, tree)
    end

    return AlphaVectorPolicy(pomdp, Γ, acts)
end
