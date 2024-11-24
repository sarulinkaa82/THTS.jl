using Revise
using THTS
using FiniteHorizonPOMDPs
using MCTS
using POMDPModels
using BenchmarkTools

include("MausamKolobov.jl")
include("CustomDomains.jl")

include("utils.jl")

mdp = MausamKolobov()
fhm = fixhorizon(mdp, 25)


# Measure average and median reward of MaxUCT, DPUCT, UCT* and MCTS
avg_acc_reward(fhm, 100)

MaxUCTSolver = THTSSolver(7.0, iterations = 1000, backup_function = MaxUCT_backpropagate_c, enable_UCTStar = false)
DPUCTSolver = THTSSolver(3.4, iterations = 1000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = false)
UCTStarSolver = THTSSolver(3.4, iterations = 1000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = true)
MSolver = MCTSSolver(exploration_constant = 6.4, n_iterations = 1000, depth = 45)
is = get_initial_state(fhm)


max_b = @benchmark base_thts(fhm, MaxUCTSolver, is)
dp_b = @benchmark base_thts(fhm, DPUCTSolver, is)
star_b = @benchmark base_thts(fhm, UCTStarSolver, is)
mcts_b = @benchmark one_mcts(fhm, MSolver, is)