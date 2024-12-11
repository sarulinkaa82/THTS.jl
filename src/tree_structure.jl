
struct THTSTree{S,A}
    d_node_ids::Dict{S, Int}
    c_node_ids::Dict{Tuple{S, A}, Int}

    # Decision nodes data
    states::Vector{S}
    d_visits::Vector{Int}
    d_values::Vector{Float64}
    d_children::Vector{Vector{Int}}

    # Chance nodes data
    state_actions::Vector{Tuple{S, A}}
    c_visits::Vector{Int}
    c_qvalues::Vector{Float64}
    c_children::Vector{Dict{Int, Float64}}

    # Constructor to initialize empty dictionaries and arrays 
    function THTSTree{S, A}() where  {S, A}
        new( # Pre-allocating might help here a bit
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


function add_decision_node!(tree::THTSTree{S, A}, state::S) where {S, A}
    if !haskey(tree.d_node_ids, state)
        state_id = length(tree.d_node_ids) + 1
        tree.d_node_ids[state] = state_id
        push!(tree.states, state)
        push!(tree.d_visits, 0)
        push!(tree.d_values, 0)
        push!(tree.d_children, Int[])
    end
end

function add_chance_node!(tree::THTSTree{S, A}, state::S, action::A) where {S, A}
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

function get_decision_id(tree::THTSTree{S, A}, state::S) where {S, A} #klidne bych byl explicitni a pojmenoval to get_decision_node_id
    if !haskey(tree.d_node_ids, state)
        throw("ERROR, attempting to acces id of a decision node that doesnt exist!")
    end

    return tree.d_node_ids[state]
end


function get_chance_id(tree::THTSTree{S, A}, state::S, action::A) where {S, A} #dtto
    state_action = (state, action)
    if !haskey(tree.c_node_ids, state_action)
        throw("ERROR, attempting to acces id of a chance node that doesnt exist!")
    end

    return tree.c_node_ids[state_action]
end

