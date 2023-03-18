using Plots
using POMDPs
using POMDPModelTools
using ARDESPOT
using POMDPPolicies
using POMDPSimulators
using ParticleFilters
using Random
using LinearAlgebra
using StatsBase
using Distances
using Combinatorics
using .FirePOMDP
include("updater.jl")
include("observations.jl")
include("FireWorld.jl")

GRID_SIZE = 4
MAX_ACT = ceil(0.25 * GRID_SIZE)

pomdp = FireWorld(grid_size = GRID_SIZE, tprob=1.0)

up = HistoryUpdater();

s0 = rand(initialstate(pomdp))
b0 = initialize_belief(up, initialstate(pomdp))

# In line with DESPOT paper, upper bound is R_max/(1 - γ), lower bound is determined by random policy
rng = MersenneTwister()
random_policy = RandomPolicy(rng, pomdp, up)
lower = DefaultPolicyLB(random_policy)
upper = 0.0

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

# Tune hyperparameters here!
solver = DESPOTSolver(
    bounds=IndependentBounds(lower, upper),
    max_trials=10, # tune
    default_action=a_default
)

planner = solve(solver, pomdp);

# Stepthrough entire simulation
using Profile

for (s,a,r,sp,o) in stepthrough(pomdp, planner, up, b0, s0, "s, a, r, sp, o")
    # println("in state $s")
    println("took action $a")
    println("received reward $r")
    # println("received observation $o and reward $r")
end

# The following experiment tests a randomly generated policy against POMCPOW

rng = MersenneTwister(256)
default_policy = RandomPolicy(rng, pomdp, up)

default = Sim(pomdp, default_policy, up, b0, s0)
pomcpow = Sim(pomdp, planner, up, b0, s0)

run([default, pomcpow]) do sim, hist
    return (n_steps=n_steps(hist), reward=discounted_reward(hist))
end