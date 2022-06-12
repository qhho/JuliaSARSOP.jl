struct SARSOPSolver <: Solver
    epsilon::Float64
    kappa::Float64
    max_time::Float64
    verbose::Bool
end

struct AlphaVec{A}
    alpha::Vector{Float64}
    action::A
end

function solve(solver::SARSOPSolver, pomdp::POMDP)

    start_time = time()
    while time()-start_time < solver.max_time
        sample!(solver, tree)
        backup!(tree, alphavecs, b)
        prune!(tree, alphavecs)
    end

    return AlphaVectorPolicy(pomdp, Γ, acts)
end