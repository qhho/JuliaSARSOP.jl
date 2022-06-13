Base.@kwdef struct SARSOPSolver <: Solver
    epsilon::Float64    = 1e-3
    kappa::Float64      = 0.5
    delta::Float64      = 0.1
    max_time::Float64   = Inf
    verbose::Bool       = true
end

struct AlphaVec{A}
    alpha::Vector{Float64}
    action::A
    witness::Int
    value_at_witness::Float64
end

function POMDPs.solve(solver::SARSOPSolver, pomdp::POMDP)
    tree = SARSOPTree(pomdp)

    mdp_solver = ValueIterationSolver()
    upper_policy = solve(mdp_solver, UnderlyingMDP(pomdp))
    initUpperBound!(tree, upper_policy.util)
    
    start_time = time()
    while time()-start_time < solver.max_time
        sample!(solver, tree)
        backup!(tree, alphavecs, b)
        prune!(tree, alphavecs)
    end

    return AlphaVectorPolicy(pomdp, Γ, acts)
end
