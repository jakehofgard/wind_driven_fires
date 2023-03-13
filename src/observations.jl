# Observations
# observation(pomdp, a, sp)
function POMDPs.observation(pomdp::FireWorld, a::Array{Int64,1}, sp::FireState)
    fn = pomdp.bprob_fn
    fp = pomdp.bprob_fp

    neighbors = FireObs[]
    probabilities = Array{Float64,1}(undef,0)

    NUM_SAMPLES = 10
    prob = 1 / NUM_SAMPLES
    for i in 1:NUM_SAMPLES
        burn_obs = deepcopy(sp.burning)
        for j in 1:length(burn_obs)
            if sp.burning[j] == 1
                if rand() < fn
                    burn_obs[j] = 0
                end
            else
                if rand() < fp
                    burn_obs[j] = 1
                end
            end
        end
        for ai in a
            burn_obs[ai] = sp.burning[ai]
        end

        push!(neighbors, FireObs(burn_obs))
        push!(probabilities, prob)
    end

    return SparseCat(neighbors, probabilities)
end