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

GRID_SIZE = 8
MAX_ACT = ceil(0.25 * GRID_SIZE)

pomdp = FireWorld(grid_size = GRID_SIZE, tprob=1.0)

COSTS = sortperm(pomdp.costs)

function default_act(belief)
    obs = rand(belief).burning
    a = Int64[]
    total = 0
    for i in 1:length(COSTS)
        if obs[COSTS[i]] == 1
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

function default_for_solver(belief, ex)
#     @warn ex
    return default_act(belief)
end


# Tune hyperparameters here!
solver = POMCPOWSolver(
    rng=MersenneTwister(264), 
    default_action = default_for_solver,
    max_depth = 10, # Tune
    tree_queries = 1000,
    criterion = MaxUCB(10.0), # Tune
    max_time = 60 # See the effect of decreasing this
)

planner = solve(solver, pomdp);
up = HistoryUpdater();

s0 = rand(initialstate(pomdp))
b0 = initialize_belief(up, initialstate(pomdp))

# Stepthrough entire simulation
using Profile

function compute_belief_dist(s::FireState, b::SparseCat{Array{FireState,1},Array{Float64,1}})
    dist = 0
    for i in 1:length(b.vals)
        o, p = b.vals[i], b.probs[i]
        dist += euclidean(s.burning, o.burning) * p + euclidean(s.fuels, o.fuels) * p / 5
    end
    return dist
end

function draw_grid(s)
    size = Int(sqrt(length(s.burning)))
    i = 0
    for x in 1:size
        for y in 1:size
            i += 1
            if s.burning[i] == 1
                print("B")
            else
                print(" ")
            end
        end
        println()
    end
end

Profile.clear()
for (s,a,r,sp,o,b) in stepthrough(pomdp, planner, up, b0, s0, "s, a, r, sp, o, b")
    # println("in state $s")
    dist = compute_belief_dist(s, b)
    println("distance: $dist, $(length(b.vals))")
    draw_grid(s)
    # println("received reward $r")
    # println("received observation $o and reward $r")
end

# The following experiment tests a randomly generated policy against POMCPOW

rng = MersenneTwister()
default_policy = FunctionPolicy(default_act)

default = Sim(pomdp, default_policy, up, b0, s0)
pomcpow = Sim(pomdp, planner, up, b0, s0)

run([default, pomcpow]) do sim, hist
    reward = discounted_reward(hist)
    println("got discounted reward of $reward")
    return (n_steps=n_steps(hist), reward=discounted_reward(hist))
end
