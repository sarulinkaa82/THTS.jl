using Revise
using THTS
using FiniteHorizonPOMDPs, POMDPModels
using MCTS
using BenchmarkTools
using CSV, DataFrames, PrettyTables
using Printf
using Plots, StatsPlots
using Random
include("benchmark_utils.jl")

include("MausamKolobov.jl")
include("CustomDomains.jl")


mdp = MausamKolobov()
mdp = fixhorizon(mdp, 25)

domain_size, grid_matrix = generate_test_domain("benchmarking/data/maze-7-A2.txt")
mdp = CustomDomain(size = domain_size, grid = grid_matrix)
mdp = fixhorizon(mdp, 40)

# Measure average and median reward of MaxUCT, DPUCT, UCT* and MCTS
data = avg_acc_reward(mdp, 50)
# Create the boxplot
boxplot(data,
    labels=["MaxUCT", "DPUCT", "UCTStar", "MCTS"],  # Names for each group
    xlabel="Algorithms",
    ylabel="Accumulated Reward",
    title="Accumulated values after partial runs",
    xticks=(1:4, ["MaxUCT", "DPUCT", "UCTStar", "MCTS"]),  # Set custom x-axis labels
    legend=false
)

# Empty files storing benchmarling results 
# rewrite_benchmark_files()


measure_improvement("Retry base")
process_benchmark_data(3, "benchmarking/results/partial_run_memory_results.csv", "benchmarking/results/partial_memory_scale")

get_convergence(mdp)
savefig("convergence_graph.png")




bench_to_table("benchmarking/results/partial_run_time_results.csv", "Partial run results", "tab:partial_bench")


get_value_stat()
savefig("example/results/algorithm_values.png")