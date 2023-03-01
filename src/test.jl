using POMDPs
using POMDPModelTools
using POMCPOW
using POMDPPolicies
using POMDPSimulators
using ParticleFilters
using Random
using LinearAlgebra
using StatsBase
using Combinatorics
using .FirePOMDP

GRID_SIZE = 12
MAX_ACT = 1

pomdp = FireWorld(grid_size = GRID_SIZE) 

a_default = sortperm(pomdp.costs)[1:MAX_ACT]
solver = POMCPOWSolver(rng=MersenneTwister(264), default_action = a_default, tree_queries = 1000, max_time = 60);

planner = solve(solver, pomdp);
up = HistoryUpdater();

s0 = rand(MersenneTwister(264), initialstate(pomdp))
b0 = initialize_belief(up, initialstate(pomdp))

for (s,a,r,sp,o) in stepthrough(pomdp, planner, up, b0, s0, "s, a, r, sp, o")
    println("in state $s")
    println("took action $a")
    println("received observation $o and reward $r")
end

pomdp.wind