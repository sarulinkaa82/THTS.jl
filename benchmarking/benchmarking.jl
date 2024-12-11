using Revise
using THTS
using FiniteHorizonPOMDPs, POMDPModels
using MCTS
using BenchmarkTools
using CSV, DataFrames, PrettyTables
using Printf
include("benchmark_utils.jl")

include("MausamKolobov.jl")
include("CustomDomains.jl")


mdp = MausamKolobov()
mdp = fixhorizon(mdp, 25)

domain_size, grid_matrix = generate_test_domain("benchmarking/data/maze-7-A2.txt")
mdp = CustomDomain(size = domain_size, grid = grid_matrix)
mdp = fixhorizon(mdp, 40)

# Measure average and median reward of MaxUCT, DPUCT, UCT* and MCTS
avg_acc_reward(mdp, 500)

# Empty files storing benchmarling results 
# rewrite_benchmark_files()


measure_improvement("Outcomes")

partial_run_res = benchmark_partial_run(mdp, "Baseline - maze 15", 50)
complete_run_res = benchmark_complete_run(mdp, "Baseline - maze-25-A2",20)

a = partial_run_res[1]
a = partial_run_res[2]
a = partial_run_res[3]
a = partial_run_res[4]


MaxUCTSolver = THTSSolver(7.0, iterations = 1000, backup_function = MaxUCT_backpropagate_c, enable_UCTStar = false)
is = get_initial_state(mdp)
max_b = base_thts(mdp, MaxUCTSolver, is)

process_benchmark_data(3, "benchmarking/results/partial_run_memory_results.csv", "benchmarking/results/partial_memory_scale")
