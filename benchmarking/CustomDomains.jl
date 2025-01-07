using POMDPs
using POMDPModels, POMDPLinter
using POMDPTools

export generate_test_domain
export generate_random_domain

struct GWCoords
    x::Int
    y::Int

    function GWCoords(x::Int, y::Int)
        new(x, y)
    end
end

Base.@kwdef struct CustomDomain <: MDP{GWCoords, Symbol}
    size::Tuple{Int64, Int64}
    grid::Matrix
    t_prob::Float64                = 0.7
    discount::Float64              = 1
    rewards::Dict{String, Float64} = Dict("step" => -1.0, "E" => 50.0, "D" => -50.0)
end

# init domain like:
# filepath = "C:/repos/jukia_solvers/PolicyIteration.jl/examples/maze-7-A1.txt"
# domain_size, grid_matrix = generate_test_domain(filepath)
# mdp = CustomDomain(size = domain_size, grid = grid_matrix)
function generate_test_domain(filepath::String)
    
    df = open(filepath, "r")

    size_string = readline(df)
    lb, ub = split(size_string, " ")
    size = (parse(Int64, lb), parse(Int64, ub))

    grid = fill("def", size)

    i = 1
    for line in readlines(df)
        row = split(line, "")
        row = String.(row)
        grid[i, :] = row
        i += 1
    end
    close(df)

    return size, grid
end

# init domain like:
# domain_size, grid_matrix = generate_random_domain(size, type)
# mdp = CustomDomain(size = domain_size, grid = grid_matrix)
# TYPES:
# gap:      tunel:      empty:
# #######   #######     ####
# #     #   #     #     #  #
# ### ###   #     #     # E#
# #    E#   ##### #     ####
# #######   #     #
#           # #####
#           #     #
#           #    E#
#           #######
function generate_random_domain(size::Tuple{Int64, Int64}, type::String)
    rows, cols = size

    grid = fill("def", size)

    middle = ceil(rows / 2)
    
    for row in 1:rows 
        # make top and bottom wall
        if row == 1 || row == rows
            wall = repeat("#", cols)
            wall = split(wall, "")
            grid[row, :] = wall
        # sort out the middle for different types than empty
        elseif row == middle && type == "gap"
            for i in 1:cols
                if i == ceil(cols/2)
                    grid[row, i] = " "
                else
                    grid[row, i] = "#"
                end
            end
        elseif row == middle - 1 && type == "tunnel"
            for i in 1:cols
                if i == cols - 1
                    grid[row, i] = " "
                else
                    grid[row, i] = "#"
                end
            end
        elseif row == middle + 1 && type == "tunnel"
            for i in 1:cols
                if i == 2
                    grid[row, i] = " "
                else
                    grid[row, i] = "#"
                end
            end
        # add the rest in between (#    #)
        else
            for col in 1:cols
                if col == 1 || col == cols
                    grid[row, col] = "#"
                elseif row == rows - 1 
                    grid[row, cols - 1] = "E"
                else
                    grid[row, col] = " "
                end
            end
        end
    end
    return size, grid
end


# STATES

function POMDPs.states(mdp::CustomDomain)
    state_vec = Vector{GWCoords}()
    
    for x in 1:mdp.size[1], y in 1:mdp.size[2]
        if mdp.grid[x, y] != "#"
            pos = GWCoords(x, y)
            push!(state_vec, pos)
        end
    end    
    return state_vec
end

function POMDPs.stateindex(mdp::CustomDomain, s::GWCoords)
    all_states = POMDPs.states(mdp)
    for index in eachindex(all_states) # linear indexing
        if all_states[index] == s
            return index
        end
    end
    return -1
end

function POMDPs.initialstate(mdp::CustomDomain)
    state_vec = POMDPs.states(mdp)
    init_state_prob = zeros(Int64, length(state_vec))
    init_state_prob[mdp.size[2] + 2] = 1
    return init_state_prob
end

# ACTIONS

function POMDPs.actions(mdp::CustomDomain)
    return (:up, :down, :left, :right)
end

const aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4)

function POMDPs.actionindex(mdp::CustomDomain, a::Symbol)
    return aind[a]
end

# TRANSITIONS

function POMDPs.isterminal(mdp::CustomDomain, s::GWCoords)
    if mdp.grid[s.x, s.y] == "E"
        return true
    else
        return false
    end
end

function POMDPs.transition(mdp::CustomDomain, s::GWCoords, a::Symbol)
    if isterminal(mdp, s) || !in_bounds(mdp, s)
        return Deterministic(s)
    end

    possible_dest_nr = length(POMDPs.actions(mdp)) + 1
    destinations = Vector{GWCoords}(undef, possible_dest_nr)
    probabilities = zeros(possible_dest_nr)


    for (i, action) in enumerate(actions(mdp)) # up, down, left, right
        if action == a # probability of transitioning to the desired cell
            prob = mdp.t_prob 
        else # probability of transitioning to another cell
            # prob = (1.0 - mdp.tprob)/(length(actions(mdp)) - 1) 
            prob = (1.0 - mdp.t_prob) / 3
        end
        # println(action, " ", prob)

        dest = next_state(mdp, s, action)
        destinations[1] = s
        destinations[i+1] = dest

        if !in_bounds(mdp, dest) # hit a wall and come back to s
            probabilities[1] += prob
        else
            probabilities[i+1] += prob
        end
    end

    return SparseCat(destinations, probabilities)
end

function in_bounds(mdp::CustomDomain, s::GWCoords) 
    if 1 < s.x < mdp.size[1] && 1 < s.y < mdp.size[2]
        return mdp.grid[s.x, s.y] != "#"
    else
        return false
    end
end

function next_state(mdp::CustomDomain, s::GWCoords, a::Symbol)
    x = s.x
    y = s.y

    if a == :up
        x -= 1
    elseif a == :down
        x += 1
    elseif a == :left
        y -= 1
    elseif a == :right
        y += 1
    end

    # check if in boundaries
    new_state = GWCoords(x, y)
    if !in_bounds(mdp, new_state)
        return s
    else
        return new_state
    end
end

# REWARDS

function POMDPs.reward(mdp::CustomDomain, s::GWCoords, a::Symbol)
    ns = next_state(mdp, s, a)

    if mdp.grid[ns.x, ns.y] == "D"
        return mdp.rewards["D"]
    elseif mdp.grid[ns.x, ns.y] == "E"
        return mdp.rewards["E"]
    else
        return mdp.rewards["step"]
    end
end

function POMDPs.reward(mdp::CustomDomain, s::GWCoords, a::Symbol, sp::GWCoords)
    if mdp.grid[sp.x, sp.y] == "D"
        return mdp.rewards["D"]
    elseif mdp.grid[sp.x, sp.y] == "E"
        return mdp.rewards["E"]
    else
        return mdp.rewards["step"]
    end
end


# DISCOUNT

function POMDPs.discount(mdp::CustomDomain)
    return mdp.discount
end

# CONVERSIONS? - not sure if needed

function POMDPs.convert_a(::Type{V}, a::Symbol, m::SimpleGridWorld) where {V<:AbstractArray}
    convert(V, [aind[a]])
end
function POMDPs.convert_a(::Type{Symbol}, vec::V, m::SimpleGridWorld) where {V<:AbstractArray}
    actions(m)[convert(Int, first(vec))]
end