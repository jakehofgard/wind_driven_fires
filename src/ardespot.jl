using Plots
using POMDPs
using POMDPModelTools
using ARDESPOT
using POMDPPolicies
using POMDPSimulators
using ParticleFilters
using Random
using DataFrames
using CSV
using LinearAlgebra
using StatsBase
using Distances
using Combinatorics
include("FireWorld.jl")
using .FirePOMDP
include("updater.jl")
include("observations.jl")

GRID_SIZES = [4]
results = DataFrame([[],[],[],[],[]], ["size", "n_default", "n_online", "reward_default", "reward_online"])
NUM_ITERS = 50

pomdp = FireWorld(grid_size=4, tprob=0.8)

for iter in 1:NUM_ITERS
    for grid_size in GRID_SIZES
        println("Testing grid size ", grid_size)
        max_act = ceil(0.25 * grid_size)

        global pomdp = FireWorld(grid_size=grid_size, tprob=0.8)

        up = HistoryUpdater();

        s0 = rand(initialstate(pomdp))
        b0 = initialize_belief(up, initialstate(pomdp))

        # In line with DESPOT paper, upper bound is R_max/(1 - γ), lower bound is determined by random policy
        lower = -250.0
        upper = 0.0

        costs = sortperm(pomdp.costs)

        function default_act(belief)
            obs = rand(belief).burning
            a = Int64[]
            total = 0
            for i in 1:length(costs)
                if obs[costs[i]] == 1
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
        #     @warn ex
            return default_act(belief)
        end

        # Tune hyperparameters here!
        solver = DESPOTSolver(
            bounds=IndependentBounds(lower, upper),
            max_trials=10, # tune
            default_action=default_for_solver
        )

        planner = solve(solver, pomdp);
        print("solved")

        # The following experiment tests a randomly generated policy against POMCPOW

        rng = MersenneTwister()
        default_policy = FunctionPolicy(default_act)

        default = Sim(pomdp, default_policy, up, b0, s0)
        online = Sim(pomdp, planner, up, b0, s0)

        sim_default = simulate(default)
        reward_default = discounted_reward(sim_default)
        steps_default = n_steps(sim_default)
        print("simulated default, reward is $reward_default")

        sim_online = simulate(online)
        reward_online = discounted_reward(sim_online)
        steps_online = n_steps(sim_online)
        print("simulated online, reward is $reward_online")

        push!(results, [grid_size, steps_default, steps_online, reward_default, reward_online])
        CSV.write("../experiments/ardespot_3.csv", results)
    end
    println("Finished iter ", iter)
end
