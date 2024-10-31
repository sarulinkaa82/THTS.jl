using Revise
using POMDPs, POMDPTools
using POMDPModels


mutable struct THTSTree{S,A}
    d_node_ids::Dict{S, Int}
    c_node_ids::Dict{Tuple{S, A}, Int}

    # decision nodes data
    states::Vector{S}
    d_visits::Vector{Int}
    d_values::Vector{Float64}
    d_children::Vector{Vector{Int}}

    # chance nodes data
    state_actions::Vector{Tuple{S, A}}
    c_visits::Vector{Int}
    c_qvalues::Vector{Float64}
    c_children::Vector{Dict{Int, Float64}}

    # Constructor to initialize empty dictionaries and arrays
    function THTSTree()
        new(
            Dict{S, Int}(),
            Dict{Tuple{S, A}, Int}(),
            S[],
            Int[],
            Float64[],
            Vector{Vector{Int}}(),
            Tuple{S, A}[],
            Int[],
            Float64[],
            Vector{Dict{Int, Float64}}()
        )
    end

end


function add_decision_node(tree::THTSTree{S, A}, state::S)
    if !haskey(tree.d_node_ids, state)
        state_id = length(tree.d_node_ids) + 1
        tree.d_node_ids[state] = state_id
        push!(tree.states, state)
        push!(tree.d_visits, 0)
        push!(tree.d_values, 0)
        push!(tree.d_children, Int[])
    end
end

function add_chance_node(tree::THTSTree{S, A}, state::S, action::A)
    state_action = (state, action)
    if !haskey(tree.c_node_ids, state_action)
        action_id = length(tree.c_node_ids) + 1
        tree.c_node_ids[state_action] = action_id
        push!(tree.state_actions, state_action)
        push!(tree.c_visits, 0)
        push!(tree.c_values, 0)
        push!(tree.c_children, Dict{Int, Float64}())
    end
end

function get_decision_id(tree::Tree{S, A}, state::S)
    if !haskey(tree.d_node_ids, state)
        add_decision_node(tree, state)
    end

    return tree.d_node_ids[state]
end


function get_chance_id(tree::Tree{S, A}, state::S, action::A)
    state_action = (state, action)
    if !haskey(tree.c_node_ids, state_action)
        add_chance_node(tree, state, action)
    end

    return tree.c_node_ids[state_action]
end


function add_transition(tree::Tree{S, A}, state::S, action::A, next_state::S, probability::Float64)
    state_id = get_decision_id(tree, state)
    next_state_id = get_decision_id(tree, next_state)
    action_id = get_chance_id(tree, state, action)

    # link chance node as a child to decision node
    if !(action_id in tree.d_children[state_id])
        push!(tree.d_children[state_id], action_id)
    end

    # link next_state decision node to chance node
    tree.c_children[action_id][next_state_id] = probability
end



##################### THTS #####################

"""
Selects best action according to chosen criteria\n
Returns chance_id::Int of a chance node
"""
function select_chance_node(tree::THTSTree{S, A}, node_id::Int)
    nothing
end

"""
Selects outcoming next state base on the probability of the outcomes
Return decision_id::Int of the next decision node
"""
function select_outcome(outcomes::Dict{S, Float64})
    nothing
end

"""
Backpropagates the result into the chance node
"""
function backpropagate_c(tree::THTSTree{S, A}, node_id::Int, result::Float64)
    nothing
end

"""
Backpropagates the result into the decision node
"""
function backpropagate_d(tree::THTSTree{S, A}, node_id::Int, result::Float64)
    nothing
end


"""
Runs simulation from the decision node into a terminal state\n
Returns result::Float64
"""
function simulate(tree::THTSTree{S, A}, mdp::MDP, node_id::Int)
    nothing
end

"""
Returns next_states::Dict{S, Float64} dictionary containing next states and probabilities of transitioning into them
"""
function get_next_states(tree::THTSTree{S, A}, node_id::Int)
    nothing
end

"""
Chooses greedy action for a state of the decision node\n
Returns action::A
"""
function greedy_action(tree::THTSTree{S, A}, node_id)
    nothing
end


function visit_d_node(tree::THTSTree{S, A}, node_id::Int) end
function visit_c_node(tree::THTSTree{S, A}, node_id::Int) end

function visit_d_node(tree::THTSTree{S, A}, mdp::MDP, node_id::Int)
    if tree.d_visits[node_id] == 0
        # add expand the children
        res = simulate(tree, mdp, node_id)
        backpropagate_d(tree, node_id, res)
        return res
    end

    chance_node_id = select_chance_node(tree, node_id)
    res = visit_c_node(tree, mdp, chance_node_id)
    backpropagate_d(tree, node_id, res)
    return res
end

function visit_c_node(tree::THTSTree{S, A}, mdp::MDP, node_id::Int)
    if tree.c_visits == 0
        # add expand next d nodes
    end

    outcomes = get_next_states(tree, node_id)

    # select outcome based on probability
    next_state_id = select_outcome(outcomes)

    res = visit_d_node(tree, next_state_id)
    backpropagate_c(tree, node_id, res)
    return res
end

function base_thts(mdp::MDP, initial_state::S, max_iters::Int; tree::THTSTree)
    tree = THTSTree()
    add_decision_node(tree, initial_state)
    root_id = get_decision_id(tree, initial_state) # should be 0 tho

    for iter in 1:max_iters
        visit_d_node(mdp, tree, root_id)
    end

    return greedy_action(tree, root_id)
end
