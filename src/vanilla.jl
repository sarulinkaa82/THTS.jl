using Revise
using POMDPs, POMDPTools
using POMDPModels
using StatsBase

mutable struct THTSSolver <: Solver
    exploration_constant::Float64
    iterations::Int
    verbose::Bool 
    include_Q::Bool

    function THTSSolver(exploration_constant;
        iterations::Int = 10,
        verbose::Bool = false,
        include_Q::Bool = false)    

        return new(exploration_constant, iterations, verbose, include_Q)
    end
end

mutable struct StateNode
    state::Any
    action::Union{Nothing, Any}
    parent::Union{Nothing, StateNode}
    children::Vector{StateNode}
    visits::Int

    function StateNode(state, parent, action)
        return new(state, action, parent, StateNode[], 0)
    end
end

function print_node(node::StateNode)
    child_nodes = node.children
    children = []
    for child_node in child_nodes
        push!(children, child_node.state)
    end
    name = node.state
    action = node.action
    visits = node.visits
    parent = nothing
    if !isnothing(node.parent)
        parent = node.parent.state
    end
    
    println("($name, $action, $visits, $parent, $children)")
end


function solve(solver::THTSSolver, mdp::MDP; kwargs...)
    root = init_root(mdp)
    q_table = Dict{Any, Float64}()
    q_table[root.state] = 0
    # println(states(mdp))
    # println(initialstate(mdp))
    # print_node(root)

    for iter in 1:solver.iterations
        # println()
        # println("iteration: $iter")
        curr_node = root

        # SELECTION
        next_node = select_node(mdp, curr_node, solver.exploration_constant, q_table)
        println("Selected: ")
        print_node|(next_node)

        # EXPANSION
        # print_node(next_node)
        if !isterminal(mdp, next_node.state)
            next_node = expand(mdp, next_node)
        end
        # println("Expanded:")
        # print_node(next_node)

        # SIMULATION
        result = simulation(mdp, next_node, 50)
        # println("sim from $(next_node.state), result: ", result)

        # BACKPROPAGATION
        backpropagate(mdp, next_node, Float64(result), q_table)
        # println("Backpropagated")
        curr_node = next_node
    end

    # print_node(root)
    best = find_best(mdp, root, q_table)

    println("best action: ", best.action)
    # print_tree(root)
    # for c in root.children
    #     println(c.state)
    # end
    # println(best)
    return q_table
end

function find_best(mdp::MDP, node::StateNode, q_table::Dict{Any, Float64})
    max_child = nothing
    max_val = -Inf
    for c_node in node.children
        val = q_table[c_node.state] / c_node.visits
        # println("$(c_node.state): $val")
        if val > max_val
            max_val = val
            max_child = c_node
        end
    end
    return max_child
end


function init_root(mdp::MDP)
    # init_state_distr = initialstate(mdp)
    # println(init_state_distr)
    all_states = states(mdp)
    init_state = all_states[1]
    init_node = StateNode(init_state, nothing, nothing)
    return init_node
end



function select_node(mdp::MDP, node::StateNode, expl_const::Float64, q_table::Dict{Any, Float64})
    new_node = node
    i = 0
    while node.visits > 0 && !isterminal(mdp, new_node.state) && !isempty(new_node.children) && i < 10  # while visited and not terminal, select a node with uct_value
        # print_node(new_node)
        # println("diving")
        i += 1
        new_node = best_ucb_node(mdp, new_node, expl_const, q_table)
    end
    return new_node

end

function best_ucb_node(mdp::MDP, node::StateNode, expl_const::Float64, q::Dict)
    max_ucb = -Inf
    max_child = Nothing

    # println("Children of $(node.state)")
    # for c in node.children
    #     print_node(c)
    # end

    for child_node in node.children # compute best ucb

        if !haskey(q, child_node.state)
            q[child_node.state] = 0
        end
        if child_node.visits == 0
            ucb = Inf
        else
            ucb = (get(q, child_node.state, 0) / child_node.visits) + expl_const * sqrt(log(node.visits) / child_node.visits)
        end
        # println("$(child_node.state): $ucb")
        if ucb > max_ucb
            max_ucb = ucb
            max_child = child_node
        end
    end
    max_name = max_child.state
    # println("Chosen: $max_name")

    
    @assert !isnothing(max_child) "Uct should not return nothing!"
    return max_child
end

function expand(mdp::MDP, node::StateNode)
    acts = actions(mdp, node.state)

    for action in acts

        distr = transition(mdp, node.state, action)
        # println(distr)
        for (next_state, prob) in weighted_iterator(distr)
            # println("expanding: ", node.state)
            next_node = StateNode(next_state, node, action)
            # println("creating:")
            # print_node(next_node)
            push!(node.children, next_node)
        end
    end
    return node
end

function simulation(mdp::MDP, node::StateNode, max_depth::Int)
    depth = 0
    res = 0
    state = node.state
    while !isterminal(mdp, state) && depth < max_depth
        depth += 1
        acts = actions(mdp, state)
        rand_action = rand(acts)
        next_state_distr = transition(mdp, state, rand_action)
        next_states = Vector{Any}()
        weights = Vector{Float64}()
        for (new_state, prob) in weighted_iterator(next_state_distr)
            push!(next_states, new_state)
            push!(weights, Float64(prob))
        end
        # println(weights)
        # println(next_states)

        w = Weights(weights)
        next_state = sample(next_states, w)
        res += reward(mdp, state, rand_action, next_state)
        state = next_state
    end

    action = rand(actions(mdp, state))
    @assert !isnothing(action) "Action should not be null!"
    res += reward(mdp, state, action)
    return res
end

function backpropagate(mdp::MDP, node::StateNode, sim_result::Float64, q::Dict{Any, Float64})
    while !isnothing(node)
        node.visits += 1
        q[node.state] += sim_result
        node = node.parent
    end
end


function print_tree(node::StateNode, depth::Int = 0)
    println("  "^depth * "State: ", node.state, ", Visits: ", node.visits)
    
    for child in node.children
        print_tree(child, depth + 1)
    end
end





