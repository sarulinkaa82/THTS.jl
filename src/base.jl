
using POMDPs
using POMDPLinter
using POMDPTools
using StatsBase




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
    function THTSTree{S, A}() where  {S, A}
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


function add_decision_node(tree::THTSTree{S, A}, state::S) where {S, A}
    if !haskey(tree.d_node_ids, state)
        state_id = length(tree.d_node_ids) + 1
        tree.d_node_ids[state] = state_id
        push!(tree.states, state)
        push!(tree.d_visits, 0)
        push!(tree.d_values, 0)
        push!(tree.d_children, Int[])
    end
end

function add_chance_node(tree::THTSTree{S, A}, state::S, action::A) where {S, A}
    state_action = (state, action)
    if !haskey(tree.c_node_ids, state_action)
        action_id = length(tree.c_node_ids) + 1
        tree.c_node_ids[state_action] = action_id
        push!(tree.state_actions, state_action)
        push!(tree.c_visits, 0)
        push!(tree.c_qvalues, 0)
        push!(tree.c_children, Dict{Int, Float64}())
    end
end

function get_decision_id(tree::THTSTree{S, A}, state::S) where {S, A}
    if !haskey(tree.d_node_ids, state)
        add_decision_node(tree, state)
    end

    return tree.d_node_ids[state]
end


function get_chance_id(tree::THTSTree{S, A}, state::S, action::A) where {S, A}
    state_action = (state, action)
    if !haskey(tree.c_node_ids, state_action)
        add_chance_node(tree, state, action)
    end

    return tree.c_node_ids[state_action]
end


# function add_transition(tree::THTSTree{S, A}, state::S, action::A, next_state::S, probability::Float64) where {S, A}
#     state_id = get_decision_id(tree, state)
#     next_state_id = get_decision_id(tree, next_state)
#     action_id = get_chance_id(tree, state, action)

#     # link chance node as a child to decision node
#     if !(action_id in tree.d_children[state_id])
#         push!(tree.d_children[state_id], action_id)
#     end

#     # link next_state decision node to chance node
#     tree.c_children[action_id][next_state_id] = probability
# end



##################### THTS #####################

function init_heuristic(mdp::MDP)
    return 0
end

"""
Selects best action according to chosen criteria\n
Returns chance_id::Int of a chance node
"""
function select_chance_node(tree::THTSTree{S, A}, node_id::Int, exploration_bias::Float64) where {S, A}

    children_ids = tree.d_children[node_id]
    best_val = -Inf
    best_child = nothing

    Ck_nd = tree.d_visits[node_id]

    for child_id in children_ids
        Ck_nc = tree.c_visits[child_id]
        Qk_nc = tree.c_qvalues[child_id]
        if Ck_nc == 0
            ucb1_val = Inf
        else
            ucb1_val = exploration_bias * sqrt(log(Ck_nd)/Ck_nc) + Qk_nc
        end

        if ucb1_val > best_val
            best_val = ucb1_val
            best_child = child_id
        end
    end

    return best_child
end

"""
Selects outcoming next state base on the probability of the outcomes
Returns decision_id::Int of the next decision node
"""
function select_outcome(outcomes::Dict{S, Float64}) where {S, A}
    next_nodes = Vector{Int}()
    weights = Vector{Float64}()
    for (decision_node_id, prob) in outcomes
        push!(next_nodes, decision_node_id)
        push!(weights, Float64(prob))
    end

    w = Weights(weights)
    next_state_id = sample(next_nodes, w)
    # println(next_state_id)
    return next_state_id
end

"""
Backpropagates the result into the chance node
"""
function backpropagate_c(tree::THTSTree{S, A}, node_id::Int, result::Float64) where {S, A}
    tree.c_visits[node_id] += 1
end

"""
Backpropagates the result into the decision node
"""
function backpropagate_d(tree::THTSTree{S, A}, node_id::Int, result::Float64) where {S, A}
    tree.d_visits[node_id] += 1
end


"""
Uses Max-Monte-Carlo backup to backup a decision node.
"""
function MaxUCT_backpropagate_d(tree::THTSTree{S, A}, mdp::MDP, node_id::Int, result::Float64) where {S, A}
    if isterminal(mdp, tree.states[node_id])
        new_v = 0
    else
        max_q = -Inf
        for child_id in tree.d_children[node_id]
            if tree.c_qvalues[child_id] > max_q
                max_q = tree.c_qvalues[child_id]
            end
        end
        new_v = max_q
    end
    tree.d_values[node_id] = new_v
    tree.d_visits[node_id] += 1
end

"""
Uses Max-Monte-Carlo backup to backup a chance node.
"""
function MaxUCT_backpropagate_c(tree::THTSTree{S, A}, mdp::MDP, node_id::Int, result::Float64) where {S, A}
    (state, action) = tree.state_actions[node_id]
    R_nc = reward(mdp, state, action)

    sum = 0
    for (child_id, prob) in tree.c_children[node_id]
        sum += tree.d_visits[child_id] * tree.d_values[child_id]
    end

    new_q = R_nc + sum / tree.c_visits[node_id]

    tree.c_qvalues[node_id] = new_q
    tree.c_visits[node_id] += 1
end



"""
Chooses greedy action for a state of the decision node\n
Returns action::A
"""
function greedy_action(tree::THTSTree{S, A}, node_id) where {S, A}
    children_c_nodes = tree.d_children[node_id]
    
    max_q = -Inf
    max_act = Nothing
    for chance_node_id in children_c_nodes
        if tree.c_qvalues[chance_node_id] > max_q
            max_q = tree.c_qvalues[chance_node_id]
            (s, a) = tree.state_actions[chance_node_id]
            max_act = a
        end
    end

    return max_act
end



function visit_d_node(tree::THTSTree{S, A}, node_id::Int) where {S, A} end
function visit_c_node(tree::THTSTree{S, A}, node_id::Int) where {S, A} end

function visit_d_node(tree::THTSTree{S, A}, mdp::MDP, node_id::Int, exploration_bias::Float64) where {S, A}
    state = tree.states[node_id]
    println("state: ", node_id)
    if isterminal(mdp, state)
        println("REACHED TERMINAL")
        acts = actions(mdp, state)
        res = reward(mdp, state, acts[1])
        println(res)
        return Float64(res)
    end

    if tree.d_visits[node_id] == 0
        # expand the children
        state = tree.states[node_id]
        acts = actions(mdp, state)

        for a in acts # create chance node, add it to d_nodes children
            add_chance_node(tree, state, a)
            chance_id = get_chance_id(tree, state, a)
            tree.c_qvalues[chance_id] = init_heuristic(mdp)
            push!(tree.d_children[node_id], chance_id)
        end
    end

    chance_node_id = select_chance_node(tree, node_id, exploration_bias)
    res = visit_c_node(tree, mdp, chance_node_id, exploration_bias)
    println("res_d", res)
    backpropagate_d(tree, node_id, res)
    return res
end

function visit_c_node(tree::THTSTree{S, A}, mdp::MDP, node_id::Int, exploration_bias::Float64) where {S, A}
    println("chance_node: ", tree.state_actions[node_id])
    if tree.c_visits[node_id] == 0
        # expand next d nodes, check for duplicates
        (state, action) = tree.state_actions[node_id]
        distr = transition(mdp, state, action)

        # THIS IS A MESS
        for (next_state, prob) in weighted_iterator(distr)
            if !haskey(tree.d_node_ids, next_state) # if decision node doesnt exist yet
                add_decision_node(tree, next_state)
                next_d_id = get_decision_id(tree, next_state)
                tree.d_values[next_d_id] = init_heuristic(mdp)
                tree.c_children[node_id][next_d_id] = prob
            else    # if decision node exists in the tree already
                next_d_id = get_decision_id(tree, next_state)
                # if !haskey(tree.c_children[node_id], next_d_id) # if not in the nodes children already
                tree.c_children[node_id][next_d_id] = prob
                # end
            end
        end
    end

    outcomes = tree.c_children[node_id]
    # select outcome based on probability
    next_state_id = select_outcome(outcomes)

    println("Here")
    println("ns: ", next_state_id)
    res = visit_d_node(tree, mdp, next_state_id, exploration_bias)
    println("res:", res)
    backpropagate_c(tree, node_id, res)
    return res
end

function base_thts(mdp::MDP, initial_state::S, max_iters::Int, exploration_bias::Float64) where {S, A}
    a = actions(mdp, initial_state)[1]
    tree = THTSTree{typeof(initial_state), typeof(a)}()
    add_decision_node(tree, initial_state)
    root_id = get_decision_id(tree, initial_state) # should be 1 tho

    for iter in 1:max_iters
        visit_d_node(tree, mdp, root_id, exploration_bias)
    end

    # return greedy_action(tree, root_id)
    return tree
end


