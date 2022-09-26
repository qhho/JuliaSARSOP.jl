Base.@kwdef struct SARSOPSolver <: Solver
    epsilon::Float64    = 0.5
    precision::Float64  = 1e-3
    kappa::Float64      = 0.5
    delta::Float64      = 1e-1
    max_time::Float64   = 1.0
    max_steps::Int      = 100
    verbose::Bool       = true
end

function POMDPs.solve(solver::SARSOPSolver, pomdp::POMDP{S,A}) where {S,A}
    tree = SARSOPTree(pomdp)

    start_time = time()
    iterations = 0
    while time()-start_time < solver.max_time && root_diff(tree) > solver.precision
        # @info "Running Sample"
        sample!(solver, tree)
        # @info "Running Backup"
        backup!(tree)
        # @info "Running Bounds Update"
        updateUpperBounds!(tree)
        # @info "Running Pruning"
        prune!(solver, tree)
        iterations += 1
    end
    return AlphaVectorPolicy(pomdp, getproperty.(tree.Γ, :alpha), getproperty.(tree.Γ, :action))
end
