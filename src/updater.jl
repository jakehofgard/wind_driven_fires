import POMDPs
using POMDPModelTools

mutable struct HistoryUpdater <: POMDPs.Updater end

lk = ReentrantLock()

# update(up, pomdp, b, a, o)
# function POMDPs.update(up::HistoryUpdater, pomdp::FireWorld, b, a::Array{Int64,1}, o::FireObs)
function POMDPs.update(up::HistoryUpdater, pomdp::FireWorld, b::SparseCat{Array{FireState,1},Array{Float64,1}}, a::Array{Int64,1}, o::FireObs)
    # particle filter without rejection
    fn = 1.0/pomdp.bprob_fn
    fp = 1.0/pomdp.bprob_fp

    states = FireState[]
    probabilities = Array{Float64,1}(undef,0)
    
    next_states = FireState[]
    weights = Array{Float64,1}(undef,0)
    
    belief_particles = ParticleCollection(b.vals)
    rng = MersenneTwister(264)
    Threads.@threads for i in 1:n_particles(belief_particles)
        s_i = rand(belief_particles)
        sp_gen = rand(transition(pomdp, s_i, a))
        w_i = compute_weight(fn, fp, o, sp_gen)
        lock(lk) do
            push!(next_states, sp_gen)
            push!(weights, w_i)
        end
    end
    if sum(weights) == 0 # all next states do not have observation o
        println("All zero. No observation o.")
        weights = ones(length(weights)) * (1/length(weights))
    end
    for_sampling = SparseCat(next_states, normalize!(weights,1))
    
    # resample
    for i in 1:n_particles(belief_particles)
        sp_sampled = sample(next_states, Weights(weights))
        in_sample, idx = in_dist_states(states, sp_sampled)
        if !in_sample # new state
            push!(states, sp_sampled)
            push!(probabilities, pdf(for_sampling, sp_sampled))
        else # state already represented
            probabilities[idx] += probabilities[idx]
        end
    end
    return SparseCat(states, normalize!(probabilities,1))
end

POMDPs.update(up::HistoryUpdater, b::SparseCat{Array{FireState,1},Array{Float64,1}}, a::Array{Int64,1}, o::FireObs) = update(up, pomdp, b, a, o)
POMDPs.initialize_belief(updater::HistoryUpdater, belief::Any) = belief

function compute_weight(fn, fp, obs::FireObs, sp_gen::FireState)
    burning_o = obs.burning
    burning_sp = sp_gen.burning
    prob1 = fn/(1 - fn)
    prob2 = fp/(1 - fp)

    prob = 1.0

    for i in 1:length(burning_o)
        if burning_o[i] == 1 && burning_sp[i] == 0
            prob *= prob2
        elseif burning_o[i] == 0 && burning_sp[i] == 1
            prob *= prob1
        end
    end

    return prob
end

# returns probability of observing o
function in_dist_obs(obs_dist::SparseCat{Array{FireObs,1},Array{Float64,1}}, o::FireObs)
    in_dist = false
    prob = 0.0
    for obs in obs_dist.vals
        burning = obs.burning
        if burning == o.burning
            in_dist = true
            prob = pdf(obs_dist, obs)
        end
    end
    return prob
end

# returns boolean whether a state is in distribution and its index
function in_dist_states(s_dist::Array{FireState,1}, s::FireState)
    in_dist = false
    idx = 0
    for i in 1:length(s_dist)
        state = s_dist[i]
        if isequal(state, s)  
            in_dist = true
            idx = i
        end
    end
    return in_dist, idx
end