using Distances

const NEIGHBOR_DIST = 1 # how big a neighborhood, in terms of Euclidean distance 

# # not used now
# const WIND_DIR_DICT = Dict(:north=>1, 
#                         :north_east=>2, 
#                         :east=>3, 
#                         :south_east=>4, 
#                         :south=>5, 
#                         :south_west=>6, 
#                         :west=>7, 
#                         :north_west=>8)

function POMDPs.transition(pomdp::FireWorld, state::FireState, action::Array{Int64,1})
    total_size = pomdp.grid_size * pomdp.grid_size
    # 1. update transition of state
    sp_trans_dist = transition_of_states(pomdp, state, action)
    
    neighbors = FireState[] 
    probabilities = Float64[]
    
    for sp_trans in sp_trans_dist.vals
        # 2. update fuels - other cells that were burning but no action was taking to put it out
        prob = pdf(sp_trans_dist, sp_trans)

        sp_burn = sp_trans.burning
        sp_burn_probs = sp_trans.burn_probs
        sp_fuels = sp_trans.fuels
        sp_wind = sp_trans.wind

        # get indices of cells where no put out action was applied to
        all_cells = Array[1:total_size]
        no_action_cells = all_cells[1][minus(action, all_cells[1])]

        # update fuel levels of cells that were burning yet no action was applied to
        sp_fuels_new = deepcopy(sp_fuels)
        for cell in no_action_cells
            if sp_burn[cell]
                new_fuel = sp_fuels[cell]-1
                if new_fuel < 1
                    sp_fuels_new[cell] = 0
                    sp_burn[cell] = false
                else
                    sp_fuels_new[cell] = new_fuel
                end
            end
        end
        # want to keep the burning indicator even if fuel of a cell is 0 for step 3 below
        # as that cell may spread the fire to other cells despite itself being burned to fuel exhaustion

        sp_new = FireState(sp_burn, sp_burn_probs, sp_fuels_new, sp_wind)

        # 3. update fire spread
        # P_xy = fire_spread(pomdp, sp_new)
        sp = update_burn(pomdp, sp_new)
        
        push!(neighbors, sp)
        push!(probabilities, prob)
    end
    return SparseCat(neighbors, normalize!(probabilities, 1))
end

# Transition helper
function transition_of_states(pomdp::FireWorld, state::FireState, action::Array{Int64,1})
    tprob = pomdp.tprob
    burning = state.burning
    burn_probs = state.burn_probs
    fuels = state.fuels
    wind = state.wind

    # Generate wind transitions
    wind_neighbors = get_wind_neighbors(wind)
    num_winds = length(wind_neighbors)

    neighbors = FireState[]
    probabilities = Array{Float64,1}(undef,0)
    
    # action is an array of indices that we will apply fire fighting actions to
    # those cells may or may not be burning
    # 1. find cell indices of the action set where cell is burning now
    act_burn = burning[action] .* action 
    act_on_burning_cells = act_burn[findall(x->x>0, act_burn)]
    num_act_n_burning = length(act_on_burning_cells)

    # check if any cell is burning now
    if num_act_n_burning > 0 
        # 1.(a) decrement fuels in those cells
        fuels_new = [i in act_on_burning_cells ? max(0, fuels[i]-1) : fuels[i] for i in 1:length(fuels) ]
        # 1.(b) account for varying fire fighting outcomes
        # some cells may be put out successfully and some may not be
        # for i in space([false, true], num_act_n_burning) # tuple is faster
        for i in space((false, true), num_act_n_burning) # different permutations of fire fighting outcomes
            burn = collect(i)
            burning_new = update_fullburn(burning, burn, act_on_burning_cells)
            prob = (1-tprob)^sum(burn)*(tprob)^(num_act_n_burning - sum(burn)) # basically binomial
            for wind in wind_neighbors
                push!(neighbors, FireState(burning_new, burn_probs, fuels_new, wind))
                push!(probabilities, prob / num_winds)
            end
        end
    else # none of the cells we apply action to is burning to begin with
        for wind in wind_neighbors
            push!(neighbors, FireState(burning, burn_probs, fuels, wind))
            push!(probabilities, 1.0 / num_winds)
        end
    end
    return SparseCat(neighbors, probabilities)
end


# update burning or not for cells that were burning
function update_fullburn(burning::BitArray{1}, burn_perm::Array{Bool,1}, burn_indices::Array{Int64,1})
    burns = deepcopy(burning)
    for i in 1:length(burn_perm)
        burn_new = burn_perm[i]
        idx = burn_indices[i]
        burns[idx] = burn_new
    end
    return burns
end

# FIRE SPREAD
# returns new FireState with different probability burn map

direction_cart = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
wind_map = [direction_cart[[8,1,2]], direction_cart[1:3], direction_cart[2:4], direction_cart[3:5], direction_cart[4:6], direction_cart[5:7], direction_cart[6:8], direction_cart[[7,8,1]]]


function update_burn(pomdp::FireWorld, s::FireState)
    burn = s.burning
    burn_probs = s.burn_probs
    fuels = s.fuels
    wind = s.wind
    
    total_size = length(burn)

    burn_result = zeros(total_size)
    burn_result_prob = 0
    wind_strength, wind_acc, wind_dir = s.wind
    neighbors = wind_map[wind_dir]
    i = 1
    to_norm = wind_acc * DEFAULT_FUEL
    wind_weights = [0.5, 1.0, 0.5]
    for x in 1:pomdp.grid_size
        for y in 1:pomdp.grid_size
            if burn[i] == 0
                not_spread = 1
                for (n, ww) in zip(neighbors, wind_weights)
                    x0, y0 = x + n[1], y + n[2]
                    idx0 = (x0 - 1) * pomdp.grid_size + y0
                    if 1 ≤ x0 ≤ pomdp.grid_size && 1 ≤ y0 ≤ pomdp.grid_size && fuels[idx0] > 0
                        fuel_level = fuels[idx0]
                        lambda = 1 - exp(-wind_strength*(wind_acc)*ww*fuel_level/to_norm)
                        not_spread *= 1 - lambda * burn[idx0]
                    end
                end
                if rand() < 1 - not_spread
                    burn_result[i] = true
                end
            elseif fuels[i] == 0
                burn_result[i] = 0
            end
            i += 1
        end
    end

    return FireState(burn_result, burn_probs, fuels, wind)
end


function fire_spread(pomdp::FireWorld, s::FireState)
    total_size = pomdp.grid_size * pomdp.grid_size
# indices conversion
    cartesian = CartesianIndices((1:pomdp.grid_size, 1:pomdp.grid_size))
    linear = LinearIndices((1:pomdp.grid_size, 1:pomdp.grid_size))
#     "Lambda won't be input; 
#     this needs to take input of distance, wind, and constant lambda_b (terrain-specifc)
#     to update lambda
#     Distance: can calculate from grid; the further, the harder to spread
#    # 2020/05/18 update: restrict to neighboring cells only
#     Wind: Rating of how strong wind is - want to get at direction and speed (relative to direction of two cells)
#     i.e. need 8 directions (for cell and wind)
#     Lambda_b: say, fuel level at the cell"
    wind_strength, wind_acc, wind_dir = s.wind
    longest_dist = euclidean([1,1], [pomdp.grid_size, pomdp.grid_size])
    to_norm = wind_acc * DEFAULT_FUEL
    P_xy = zeros((total_size, total_size))
    burning = s.burning
    fuels = s.fuels
    for i in 1:total_size
        fuel_level = fuels[i]
        cart_i = cartesian[i]
        for j in 1:total_size
            if i == j
                P_xy[i,j] = 1
            else
                cart_j = cartesian[j]
                rel_pos = relative_direction(cart_j, cart_i)
                wind_factor = find_wind_dir_factor(wind_dir, rel_pos)
                distance_ij = sqrt((cart_i[1]-cart_j[1])^2 + (cart_i[2] - cart_j[2])^2)
                if distance_ij > NEIGHBOR_DIST
                    distance_ij = 0
                end
                rel_dist = distance_ij/longest_dist
                lambda_ij = wind_strength*(wind_acc*rel_dist)*wind_factor*fuel_level/to_norm
                P_xy[i,j]= 1 - exp(-lambda_ij)
            end
        end
    end
    lambda_ij_weak = 1 - exp(-wind_strength*(wind_acc*rel_dist)*0.5*fuel_level/to_norm)
    lambda_ij_strong = 1 - exp(-wind_strength*(wind_acc*rel_dist)*1.0*fuel_level/to_norm)
    idx = 1
    for i in 1:pomdp.grid_size
        for j in 1:pomdp.grid_size
            neigbor_idx = idx + 1
            idx += 1
        end
    end
    return P_xy
end


function find_wind_dir_factor(wind_dir::Int, cell_dir::Int)
    alignment = abs(wind_dir - cell_dir)
    if alignment == 0
        return 1
    elseif alignment == 1 || alignment == 7
        return 0.5
    else
        return 0 # the rest are not considered
    end
end

function relative_direction(cart_i::CartesianIndex{2}, cart_j::CartesianIndex{2})
    x, y = cart_j[1] - cart_i[1], cart_j[2] - cart_i[2]
    angle = atan(x, y) + pi
    rel_dir = mod(round(Int, 4 / pi * angle) - 3, 1:8)
    return rel_dir
end

function update_wind(wind)
    updates = [-1, 0, 1]
    strength, acc, dir = wind
    strength_update, dir_update = rand(updates), rand(updates)
    strength = (strength + strength_update <= 10 & strength + strength_update >= 0) ? strength + strength_update : strength - strength_update
    dir = (dir + dir_update <= 8 & dir + dir_update >= 1) ? dir + dir_update : dir - dir_update
    return [strength, acc, dir]
end

function check_feasible_wind(wind)
    strength, _, dir = wind
    return (strength >= 1 && strength <= 10 && dir >= 1 && dir <= 8)
end

function get_wind_neighbors(wind)
    strength, acc, dir = wind
    strengths = [strength - 1, strength, strength + 1]
    dirs = [mod(dir - 1, 1:8), dir, mod(dir + 1, 1:8)]
    neighbors = []
    for pair in Iterators.product(strengths, dirs) |> collect
        candidate = [pair[1], acc, pair[2]]
        if check_feasible_wind(candidate)
            push!(neighbors, candidate)
        end
    end
    return neighbors
end

# to get cells no actions applied to
minus(indx, x) = setdiff(1:length(x), indx)

# get permutations
space(x, n) = vec(collect(Iterators.product([x for i = 1:n]...)))
