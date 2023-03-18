# import POMDPs
# Action space

function POMDPs.actions(pomdp::FireWorld)
    total_size = pomdp.map_size[1] * pomdp.map_size[2]
    x = Array[1:total_size]
    actions = []
    max_act = Int(ceil(0.25 * pomdp.grid_size))
    for i in 1:max_act
        push!(actions, collect(combinations(x[1], i)))
    end
    return collect(Iterators.flatten(actions))
end

# action_space = actions(mdp);
# POMDPs.n_actions(pomdp::FireWorld) = length(actions(pomdp))
POMDPs.actionindex(pomdp::FireWorld, a::Array{Int64,1}) = findall(x->x==a, actions(pomdp))[1]

