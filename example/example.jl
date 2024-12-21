using Revise
using THTS
using DataFrames
using Plots
using CSV
using FiniteHorizonPOMDPs
using Random


########## TESTING ##############

include("MausamKolobov.jl")
m = MausamKolobov()
fhm = fixhorizon(m, 25)


solver = THTSSolver(3, iterations = 5000, verbose = true, backup_function = MaxUCT_backpropagate_c, enable_UCTStar = false)
path = solve(solver, fhm)


## CONVERGENCE GRAPHS
is = get_initial_state(fhm)
res = base_thts(fhm, solver, is)

data = CSV.File("benchmarking/iteration_values.csv") |> DataFrame
deltas = diff(data.Value)
iterations = data.Iteration[2:end]

p = plot(layout=(3, 1), size=(800, 900))
plot!(p[1], data.Iteration, data.Value, xlabel="Iteration", ylabel="Value", title="Value over Iterations", legend=false)
plot!(p[2], iterations, deltas, xlabel="Iteration", ylabel="Delta Value", title="Delta of Value over Iterations", legend=false)
plot!(p[3], iterations[100:end], deltas[100:end], xlabel="Iteration", ylabel="Delta Value", title="Delta of Value (from Iteration 100)", legend=false)

savefig(p, "UCTSTar_convergence_h_02.png")


## TESTING ACCURACY

function test_accuracy()
    hit = 0
    for i in 1:1000
        act = base_thts(mdp, MaxUCTSolver, is)
        if act == "a41"
            hit += 1
        end
    end
    return hit / 1000
end

test_accuracy()
# horizon 15, iters 100
# maxuct acc - s0 = 0.98, s2 = 1.0, s4 = 0.85
# dpuct acc -  s0 = 1.00, s2 = 1.0, s4 = 1.00, 

# max uct horizon 5 acc s4 = 0.94
# mac uct horizon 2 acc s4 = 1.00




using MCTS
MCTSSdata = values_after_run(100, fhm)
p = scatter(MCTSSdata)

sol = THTSSolver(1.4, iterations = 5000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = true)
starData = values_after_run(100, fhm, sol)
p = scatter(starData)
savefig(p1, "example/UCTStar_values2.png")


scatter(
    MCTSSdata, label="MCTS", color=:blue, markershape=:circle,
    xlabel="Run", ylabel="Value", title="Values after 5000 iterations"
)
scatter!(MaxData, label="MaxUCT", color=:red, markershape=:circle)
scatter!(dpData, label="DP-UCT", color=:green, markershape=:circle)
scatter!(starData, label="UCT*", color=:purple, markershape=:circle)
savefig("algorithm_values.png")

# Exact State Values computation 
m = MausamKolobov()
function get_state_V(mdp::MDP, horizon::Int)
    all_states = ordered_states(mdp)

    V = fill(0.0, horizon + 1, length(all_states))

    for t in horizon:-1:1
        # v = zeros(length(all_states))

        for state in all_states
            max_val = -Inf

            for action in actions(mdp, state)
                distr = transition(mdp, state, action)
                val = 0
                for (ns, prob) in weighted_iterator(distr)
                    ns_i = stateindex(mdp, ns)
                    val += prob * (reward(mdp, state, action, ns) + V[t + 1, ns_i])
                end

                if max_val < val
                    max_val = val
                end
            end
            s_i = stateindex(mdp, state)
            V[t, s_i] = max_val
        end
        # println(V)

    end
    return V
end
v = get_state_V(m, 25)
