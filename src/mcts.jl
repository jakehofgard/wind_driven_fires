using Plots
using POMDPs
using POMDPModelTools
using POMCPOW
using POMDPPolicies
using POMDPSimulators
using ParticleFilters
using Random
using LinearAlgebra
using StatsBase
using Distances
using Combinatorics
include("FireWorld.jl")
using .FirePOMDP
include("updater.jl")
include("observations.jl")

GRID_SIZE = 4
MAX_ACT = 1

pomdp = FireWorld(grid_size = GRID_SIZE, tprob=0.7)

function a_default(belief, ex)
    @warn ex
    obs = rand(belief).burning
    a = Int64[]
    total = 0
    for i in 1:length(obs)
        if obs[i] == 1
            push!(a, i)
            total += 1
            if total == MAX_ACT
                break
            end
        end
    end
    while total < MAX_ACT
        push!(a, 1)
        total += 1
    end
    return a
end

solver = POMCPOWSolver(
    rng=MersenneTwister(264), 
    default_action = a_default, 
    tree_queries = 1000,
    max_time = 60
)

planner = solve(solver, pomdp);
up = HistoryUpdater();

s0 = rand(initialstate(pomdp))
b0 = initialize_belief(up, initialstate(pomdp))

# Stepthrough entire simulation
using Profile

for (s,a,r,sp,o) in stepthrough(pomdp, planner, up, b0, s0, "s, a, r, sp, o")
    # println("in state $s")
    println("took action $a")
    # println("received observation $o and reward $r")
end
