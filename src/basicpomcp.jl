#=
basicpomcp:
- Julia version: 
- Author: aditi
- Date: 2023-03-18
=#

using Plots
using POMDPs
using DataFrames
using POMDPModelTools
using BasicPOMCP
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

GRID_SIZES = [4, 6, 8, 10, 12]
results = DataFrame([[],[],[]], ["size", "n_steps", "reward"])

grid_size = 4
max_act = ceil(0.25 * grid_size)
pomdp = FireWorld(grid_size = grid_size, tprob=0.8)
COSTS = sortperm(pomdp.costs)

function default_act(belief)
    obs = rand(belief).burning
    a = Int64[]
    total = 0
    for i in 1:length(COSTS)
        if obs[COSTS[i]] == 1
            push!(a, i)
            total += 1
            if total == max_act
                break
            end
        end
    end
    while total < max_act
        push!(a, 1)
        total += 1
    end
    return a
end

function default_for_solver(belief, ex)
    # @warn ex
    return default_act(belief)
end

# Tune hyperparameters here!
solver = POMCPSolver(
    rng=MersenneTwister(264),
    default_action = default_for_solver,
    max_depth = 20, # Tune
    tree_queries = 1000, # Tune
    c = 5.0 # Tune
)

planner = solve(solver, pomdp);
up = HistoryUpdater();

s0 = rand(initialstate(pomdp))
b0 = initialize_belief(up, initialstate(pomdp))

# The following experiment tests a randomly generated policy against POMCPOW

rng = MersenneTwister()
default_policy = FunctionPolicy(default_act)

default = Sim(pomdp, default_policy, up, b0, s0)
pomcp = Sim(pomdp, planner, up, b0, s0)

reward_default = discounted_reward(simulate(default))
reward_pomcp = discounted_reward(simulate(pomcp))


run([default, pomcp]) do sim, hist
    reward = discounted_reward(hist)
    println("got discounted reward of $reward")
    push!(results, [grid_size, n_steps(hist), discounted_reward(hist)])
    return (n_steps=n_steps(hist), reward=discounted_reward(hist))
end

results