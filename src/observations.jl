# Observations
# observation(pomdp, a, sp)
function POMDPs.observation(pomdp::FireWorld, a::Array{Int64,1}, sp::FireState)
    fn = pomdp.bprob_fn
    fp = pomdp.bprob_fp

    ImplicitDistribution(sp, a) do sp, a, rng
        burn_obs = deepcopy(sp.burning)
        for j in 1:length(burn_obs)
            if sp.burning[j] == 1
                if rand(rng, 1:fn) == 1
                    burn_obs[j] = 0
                end
            else
                if rand(rng, 1:fp) == 1
                    burn_obs[j] = 1
                end
            end
        end
        for ai in a
            burn_obs[ai] = sp.burning[ai]
        end
    end
end