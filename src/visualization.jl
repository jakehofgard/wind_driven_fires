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
using .FirePOMDP

GRID_SIZE = 12
MAX_ACT = 1

pomdp = FireWorld(grid_size = GRID_SIZE, tprob = 0.8)

initial_state = pomdp.state
wind = pomdp.state.wind
burning = reshape(pomdp.state.burning, (GRID_SIZE, GRID_SIZE))
burn_probs = reshape(pomdp.state.burn_probs, (GRID_SIZE, GRID_SIZE))
fuels = reshape(pomdp.state.fuels, (GRID_SIZE, GRID_SIZE))

sample_actions = actions(pomdp)[pomdp.state.burning]

transition_model = transition(pomdp, initial_state, rand(sample_actions))
sample_update = transition_model.vals[10]
new_burning = reshape(sample_update.burning, (GRID_SIZE, GRID_SIZE))

heatmap(1:GRID_SIZE, 1:1:GRID_SIZE, burning)
heatmap(1:GRID_SIZE, 1:1:GRID_SIZE, new_burning)



cost_example = heatmap(
    1:GRID_SIZE, 
    1:1:GRID_SIZE, 
    reshape(pomdp.costs, (GRID_SIZE, GRID_SIZE)), 
    c=colors
)

savefig(cost_example, "src/figures/example_cost_map.png")