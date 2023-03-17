using Distributions

# Observations
# observation(pomdp, a, sp)
function POMDPs.observation(pomdp::FireWorld, a::Array{Int64,1}, sp::FireState)
    fn = pomdp.bprob_fn
    fp = pomdp.bprob_fp

    ImplicitDistribution(sp, a) do sp, a, rng
        burn_obs = deepcopy(sp.burning)
        prob = 1.0
        for j in 1:length(burn_obs)
            if sp.burning[j] == 1
                if rand(rng, 1:fn) == 1
                    prob *= 1/fn
                    burn_obs[j] = 0
                else
                    prob *= (1 - 1/fn)
                end
            else
                if rand(rng, 1:fp) == 1
                    prob *= 1/fp
                    burn_obs[j] = 1
                else
                    prob *= (1 - 1/fp)
                end
            end
        end
        for ai in a
            burn_obs[ai] = sp.burning[ai]
        end
        return FireObs(burn_obs, prob)
    end
end

function Distributions.pdf(dist::ImplicitDistribution, o::FireObs)
    return o.prob
end
