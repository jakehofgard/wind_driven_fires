import POMDPs
# State space; not complete and no need to initiate

# Initial state distribution
# function POMDPs.initialstate_distribution(pomdp::FireWorld)

function POMDPs.initialstate(pomdp::FireWorld)
    return SparseCat([pomdp.state], [1.0])
end