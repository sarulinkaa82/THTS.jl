using D3Trees

function prepare_d3tree(tree::THTSTree, root_state; title="THTS Tree")
    nodes = Vector{Dict{String, Any}}()
    decision_queue = [(root_state, 1)]  # Start with the root state, (state, parent node index)
    decision_ids = Set{Int}()  # To keep track of processed decision nodes
    # Traverse the decision node
    i = 0
    while !isempty(decision_queue)
        state, parent_idx = popfirst!(decision_queue)
        i += 1
        if i > 10000
            break
        end
        println("processin $state")
        


        # Get the decision node ID
        if haskey(tree.d_node_ids, state)
            d_id = tree.d_node_ids[state]
        else
            error("State $state not found in decision node IDs.")
        end

        # Process the decision node
        current_idx = length(nodes) + 1
        push!(nodes, Dict(
            "type" => :state,
            "tag" => string(state),
            "tt_tag" => "Visits: $(tree.d_visits[d_id])\nValue: $(tree.d_values[d_id])",
            "n" => tree.d_visits[d_id],
            "total_n" => tree.d_visits[d_id],
            "parent_n" => parent_idx == 1 ? missing : nodes[parent_idx]["n"],
            "child_d3ids" => Int[]
        ))
        decision_ids = union(decision_ids, Set([d_id]))

        # If this is not the root node, add it as a child to its parent
        if parent_idx != 1
            push!(nodes[parent_idx]["child_d3ids"], current_idx)
        end

        # Add the children (chance nodes) to the queue
        for c_id in tree.d_children[d_id]
            if !haskey(tree.c_node_ids, tree.state_actions[c_id])
                error("Chance node ID $c_id not found in chance node IDs.")
            end

            # Process the chance node
            current_chance_idx = length(nodes) + 1
            chance_node = Dict(
                "type" => :action,
                "tag" => string(tree.state_actions[c_id]),
                "tt_tag" => "Visits: $(tree.c_visits[c_id])\nQ-value: $(tree.c_qvalues[c_id])",
                "n" => tree.c_visits[c_id],
                "q" => tree.c_qvalues[c_id],
                "parent_n" => tree.d_visits[d_id],
                "child_d3ids" => Int[]
            )
            push!(nodes, chance_node)
            push!(nodes[current_idx]["child_d3ids"], current_chance_idx)

            # Add children states to the queue for further processing
            for (child_id, _) in tree.c_children[c_id]
                child_state = tree.states[child_id]
                push!(decision_queue, (child_state, current_chance_idx))
            end
        end
    end

    # Fix parent-child relationships for visualization
    for node in nodes
        for child_id in node["child_d3ids"]
            if child_id <= length(nodes)
                nodes[child_id]["parent_n"] = node["n"]
            end
        end
    end

    return D3Tree(nodes; title=title)
end
