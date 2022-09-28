Base.@kwdef struct SARSOPSolver <: Solver
    epsilon::Float64    = 0.5
    precision::Float64  = 1e-3
    kappa::Float64      = 0.5
    delta::Float64      = 1e-1
    max_time::Float64   = 1.0
    max_steps::Int      = typemax(Int)
    verbose::Bool       = true
end

function POMDPs.solve(solver::SARSOPSolver, pomdp::POMDP{S,A}) where {S,A}
    tree = SARSOPTree(pomdp)

    t0 = time()
    iterations = 0
    while time()-t0 < solver.max_time && root_diff(tree) > solver.precision
        sample!(solver, tree)
        backup!(tree)
        # update_upper_bounds!(tree)
        prune!(solver, tree)
        iterations += 1
    end
    return AlphaVectorPolicy(
        pomdp,
        getproperty.(tree.Γ, :alpha),
        getproperty.(tree.Γ, :action)
    )
end
