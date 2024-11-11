
using POMDPs
using POMDPLinter
using POMDPTools
using StatsBase
using FiniteHorizonPOMDPs
using Logging
using Test




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



##################### THTS #####################

function init_heuristic(mdp::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper)
    return 0
end

"""
Selects best action according to chosen criteria\n
Returns chance_id::Int of a chance node
"""
function select_chance_node(tree::THTSTree{S, A}, node_id::Int, exploration_bias::Float64) where {S, A}

    children_ids = tree.d_children[node_id]
    # println("chance nodes: ", children_ids)
    best_val = -Inf
    best_child = nothing

    Ck_nd = tree.d_visits[node_id]

    for child_id in children_ids
        Ck_nc = tree.c_visits[child_id]
        Qk_nc = tree.c_qvalues[child_id]
        # println("Qk_nc: ", Qk_nc)
        if Ck_nc == 0
            ucb1_val = Inf
        else
            ucb1_val = exploration_bias * sqrt(log(Ck_nd)/Ck_nc) + Qk_nc
        end

        # println("ucb1: ", ucb1_val)
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
function select_outcome(outcomes::Dict{Int, Float64})
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
function backpropagate_c(tree::THTSTree{S, A}, mdp::MDP, node_id::Int) where {S, A}
    tree.c_visits[node_id] += 1
end

"""
Backpropagates the result into the decision node
"""
function backpropagate_d(tree::THTSTree{S, A}, mdp::MDP, node_id::Int) where {S, A}
    tree.d_visits[node_id] += 1
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
end


function DPUCT_backpropagate_c(tree::THTSTree{S, A}, mdp::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, node_id::Int) where {S, A}
    tree.c_visits[node_id] += 1
    (state, action) = tree.state_actions[node_id]
    R_nc = reward(mdp, state, action)

    sum = 0
    Pk_nc = 0
    for (child_id, prob) in tree.c_children[node_id]
        sum += prob * tree.d_values[child_id]
        Pk_nc += prob
    end

    new_q =  R_nc + sum / Pk_nc
    
    tree.c_qvalues[node_id] = new_q
end





"""
Uses Max-Monte-Carlo backup to backup a chance node.
"""
function MaxUCT_backpropagate_c(tree::THTSTree{S, A}, mdp::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, node_id::Int) where {S, A}
    tree.c_visits[node_id] += 1
    (state, action) = tree.state_actions[node_id]
    R_nc = reward(mdp, state, action)

    sum = 0
    for (child_id, prob) in tree.c_children[node_id]
        sum += tree.d_visits[child_id] * tree.d_values[child_id]
    end

    new_q = R_nc + sum / tree.c_visits[node_id]

    tree.c_qvalues[node_id] = new_q
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

function visit_d_node(tree::THTSTree{S, A}, mdp::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, solver::THTSSolver, node_id::Int) where {S, A}
    state = tree.states[node_id]
    if isterminal(mdp, state)
        return
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

        if solver.enable_UCTStar
            backpropagate_d(tree, mdp, node_id)
            return
        end
    end

    chance_node_id = select_chance_node(tree, node_id, solver.exploration_constant)
    visit_c_node(tree, mdp, solver, chance_node_id)
    backpropagate_d(tree, mdp, node_id)

    return
end

function visit_c_node(tree::THTSTree{S, A}, mdp::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, solver::THTSSolver, node_id::Int) where {S, A}
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
                tree.c_children[node_id][next_d_id] = prob
            end
        end
        if solver.enable_UCTStar
            solver.backup_function(tree, mdp, node_id)
            return
        end
    end

    outcomes = tree.c_children[node_id]
    # select outcome based on probability
    next_state_id = select_outcome(outcomes)

    visit_d_node(tree, mdp, solver, next_state_id)
    solver.backup_function(tree, mdp, node_id)
    # DPUCT_backpropagate_c(tree, mdp, node_id)
    return
end

function base_thts(mdp::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, solver::THTSSolver, initial_state::S) where {S}
    a = actions(mdp, initial_state)[1]
    tree = THTSTree{typeof(initial_state), typeof(a)}()
    add_decision_node(tree, initial_state)
    root_id = get_decision_id(tree, initial_state) # should be 1 tho

    for iter in 1:solver.iterations
        visit_d_node(tree, mdp, solver, root_id)
        if solver.verbose && iter % (solver.iterations ÷ 5) == 0
            v_table = make_v_table(tree)
            q_table = make_q_table(tree)
            solver.verbose && @debug "V_TABLE for iter $iter: $v_table"
            solver.verbose && @debug "Q_TABLE for iter $iter: $q_table"
        end
    end

    return greedy_action(tree, root_id)
end


function make_v_table(tree::THTSTree)
    v_table = Dict()
    for (state, node_id) in tree.d_node_ids
        v_table[state] = tree.d_values[node_id]
    end
    return v_table
end

function make_q_table(tree::THTSTree)
    q_table = Dict()
    for (state, node_id) in tree.c_node_ids
        q_table[state] = tree.c_qvalues[node_id]
    end
    return q_table

end

mutable struct THTSSolver <: Solver
    exploration_constant::Float64
    iterations::Int
    verbose::Bool 
    backup_function::Function
    enable_UCTStar::Bool

    # Default constructor
    function THTSSolver(exploration_constant;
        iterations::Int = 10,
        verbose::Bool = false,
        backup_function::Function = backpropagate_c,
        enable_UCTStar::Bool = false
        )    

        return new(exploration_constant, iterations, verbose, backup_function, enable_UCTStar)
    end
end

function get_next_state(mdp::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, state::S, action::A) where {S, A}
    distr = transition(mdp, state, action)
    
    nest_states = Vector{S}()
    weights = Vector{Float64}()
    for (ns, prob) in weighted_iterator(distr)
        push!(nest_states, ns)
        push!(weights, Float64(prob))
    end

    w = Weights(weights)
    next_state = sample(nest_states, w)
    # println(next_state_id)
    return next_state

end



# function solve(solver::THTSSolver, mdp::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, kwargs...)
#     path = []

#     if solver.verbose
#         io = open("thts_log.txt", "w")
#         try
#             global_logger(SimpleLogger(io, Logging.Debug))
#             # Call your functions that involve logging here
#             path = get_path(solver, mdp, kwargs)
#         finally
#             close(io) 
#         end
#     else
#         path = get_path(solver, mdp, kwargs)
#     end
#     return path
# end

# function get_path(solver::THTSSolver, mdp::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, kwargs...)
#     path = []
#     state = get_initial_state(mdp)

    
#     while !isterminal(mdp, state)
#         solver.verbose && @info "STARTING THTS with initial state $state"
#         best_action = base_thts(mdp, solver, state)
#         solver.verbose && @info "BEST ACTION for state $state chosen as $best_action"
#         push!(path, (state, best_action))
#         state = get_next_state(mdp, state, best_action)
#     end

#     push!(path, (state))

#     return path

# end

function solve(solver::THTSSolver, mdp::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, kwargs...)
    path = []
    state = get_initial_state(mdp)

    
    while !isterminal(mdp, state)
        solver.verbose && @info "STARTING THTS with initial state $state"
        best_action = base_thts(mdp, solver, state)
        solver.verbose && @info "BEST ACTION for state $state chosen as $best_action"
        push!(path, (state, best_action))
        state = get_next_state(mdp, state, best_action)
    end

    push!(path, (state))

    return path
end



function get_initial_state(mdp::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper)
    all_s = stage_states(mdp, 1)
    all_states = collect(all_s)

    return all_states[50]
end
