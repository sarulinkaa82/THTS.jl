using POMDPs
using POMDPModels, POMDPLinter
using POMDPTools

Base.@kwdef struct MausamKolobov <: MDP{String, String}
    size     = 6
    discount = 1
end


# states

function POMDPs.states(mdp::MausamKolobov)
    # s5 is the goal state
    states_ = ["s0", "s1", "s2", "s3", "s4", "s5"]
    return states_
end

function POMDPs.stateindex(mdp::MausamKolobov, s::String)
    all_states = POMDPs.states(mdp)
    for index in eachindex(all_states) # linear indexing
        if all_states[index] == s
            return index
        end
    end
    return 1
end

function POMDPs.initialstate(mdp::MausamKolobov)
    init_state_prob = zeros(Int64, mdp.size)
    init_state_prob[1] = 1
    return init_state_prob
end


# actions

function POMDPs.actions(mdp::MausamKolobov)
    return ["a00", "a01", "a1", "a20", "a21", "a3", "a40", "a41", "exit"]
end

function POMDPs.actions(mdp::MausamKolobov, s::String)
    state_index = POMDPs.stateindex(mdp, s)
    all_actions = POMDPs.actions(mdp)

    if state_index == 1
        return all_actions[1:2]
    elseif state_index == 2
        return all_actions[3:3]
    elseif state_index == 3
        return all_actions[4:5]
    elseif state_index == 4
        return all_actions[6:6]
    elseif state_index == 5
        return all_actions[7:8]
    else
        return [all_actions[9]]
    end   
end

function POMDPs.actionindex(mdp::MausamKolobov, a::String)
    all_actions = POMDPs.actions(mdp)
    for index in eachindex(all_actions) # linear indexing
        if all_actions[index] == a
            return index
        end
    end
    return 1
end


# transitions

function POMDPs.isterminal(mdp::MausamKolobov, s::String)
    if s == "s5"
        return true
    else
        return false
    end
end

function POMDPs.transition(mdp::MausamKolobov, s::String, a::String)
    
    # if isterminal(mdp, s)
    #     return Deterministic("tt")
    # end

    if s != "s4"
        # return deterministic for the next state according to state-action policy_matrix
        # this is kinda ugly im sorry
        if s == "s0"
            if a == "a00"
                return Deterministic("s2")
            else
                return Deterministic("s1")
            end

        elseif s == "s1"
            return Deterministic("s2")

        elseif s == "s2"
            if a == "a20"
                return Deterministic("s4")
            else
                return Deterministic("s1")
            end

        elseif s == "s3"
            return Deterministic("s4")

        elseif s == "s5"
            return Deterministic("s5")
        end
    else # s == "s4" - stochastic
        if a == "a40"
            return Deterministic("s5")
        else

            destinations = ["s3", "s5"]
            probabilities = [0.4, 0.6]
            return SparseCat(destinations, probabilities)
        end
    
    end
    
end


# rewards

function POMDPs.reward(mdp::MausamKolobov, s::String, a::String)
    if a == "a40"
        return -5
    elseif a == "a41"
        return -2
    elseif s == "s5"
        return 5
    else
        return -1 
    end
end


# discount

function POMDPs.discount(mdp::MausamKolobov)
    discountt = 0.95
    return discountt
end
