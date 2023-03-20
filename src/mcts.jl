using Plots
using POMDPs
using POMDPModelTools
using POMCPOW
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

GRID_SIZES = [10]
results = DataFrame([[],[],[],[],[]], ["size", "n_default", "n_online", "reward_default", "reward_online"])
NUM_ITERS = 5

pomdp = FireWorld(grid_size = 4, tprob=0.8)
steps_default, steps_online, reward_default, reward_online = 1, 1, 0, 0

for iter in 1:NUM_ITERS
    @time begin
    for grid_size in GRID_SIZES
        println("Testing grid size ", grid_size)
        max_act = ceil(0.25 * grid_size)

        pomdp = FireWorld(grid_size = grid_size, tprob=0.8)
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
            # @warn ex
            return default_act(belief)
        end

        # Tune hyperparameters here!
        solver = POMCPOWSolver(
            rng=MersenneTwister(), 
            default_action = default_for_solver,
            max_depth = 10, # Tune
            tree_queries = 100,
            criterion = MaxUCB(0.5), # Tune
            max_time = 2 # See the effect of decreasing this
        )

        planner = solve(solver, pomdp);
        up = HistoryUpdater();

        s0 = rand(initialstate(pomdp))
        b0 = initialize_belief(up, initialstate(pomdp))

        # The following experiment tests a randomly generated policy against POMCPOW

        rng = MersenneTwister()
        default_policy = FunctionPolicy(default_act)

        default = Sim(pomdp, default_policy, up, b0, s0)
        online = Sim(pomdp, planner, up, b0, s0)

        try
            sim_default = simulate(default)
            reward_default = discounted_reward(sim_default)
            steps_default = n_steps(sim_default)

            sim_online = simulate(online)
            reward_online = discounted_reward(sim_online)
            steps_online = n_steps(sim_online)
        catch TaskFailedException
            println("Threading error! Continuing.")
            continue
        end

        push!(results, [grid_size, steps_default, steps_online, reward_default, reward_online])
    end
    end
    println("Finished iter ", iter)
    CSV.write("src/experiments/pomcpow_c=0.5_n=10_fast.csv", results)
end
