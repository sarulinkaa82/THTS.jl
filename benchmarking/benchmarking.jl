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
mdp = fixhorizon(mdp, 5)

is = get_initial_state(mdp)
MaxUCTSolver = THTSSolver(7.0, iterations = 100000, backup_function = MaxUCT_backpropagate_c, max_time = 1)
@elapsed base_thts(mdp, MaxUCTSolver, is)


domain_size, grid_matrix = generate_test_domain("benchmarking/data/maze-7-A2.txt")
mdp = CustomDomain(size = domain_size, grid = grid_matrix)
fhm = fixhorizon(mdp, 40)

# Measure average and median reward of MaxUCT, DPUCT, UCT* and MCTS
data = avg_acc_reward(fhm, 50)
# Create the boxplot
boxplot(data,
    labels=["MaxUCT", "DPUCT", "UCTStar", "MCTS"],  # Names for each group
    xlabel="Algorithms",
    ylabel="Accumulated Reward",
    title="Accumulated values after partial runs",
    xticks=(1:4, ["MaxUCT", "DPUCT", "UCTStar", "MCTS"]),  # Set custom x-axis labels
    legend=false
)

df = DataFrame(data, :auto)  # Convert to DataFrame
CSV.write("array3.csv", df)

data
savefig("example/results/accumulated_rewards.png")
# Empty files storing benchmarling results 
rewrite_benchmark_files()


measure_improvement("Heuristic")
process_benchmark_data(1, "benchmarking/results/partial_run_memory_results.csv", "benchmarking/results/partial_memory_scale")

get_convergence(fhm)
savefig("benchmarking/convergence_graph.pdf")




bench_to_table("benchmarking/results/partial_run_memory_results.csv", "", "tab:partial_memory_bench")


get_value_stat()
savefig("example/results/algorithm_values.png")


mdp = MausamKolobov()
mdp = fixhorizon(mdp, 25)

domain_size, grid_matrix = generate_test_domain("benchmarking/data/maze-7-A2.txt")
mdp = CustomDomain(size = domain_size, grid = grid_matrix)
fhm = fixhorizon(mdp, 20)


is = get_initial_state(fhm)
MaxUCTSolver = THTSSolver(7.0, iterations = 2000, backup_function = DPUCT_backpropagate_c, max_time = 1.0, enable_UCTStar = true, heuristic = euclidean_heuristic)
base_thts(fhm, MaxUCTSolver, is)
THTS.solve(MaxUCTSolver, fhm)


fhm.horizon

# Convert columns from KB to MB
file_path = "benchmarking/results/complete_run_memory_results.csv"
raw_data = CSV.File(file_path)
df = DataFrame(raw_data)
for col in names(df)[2:end]
    df[!, col] .= round.(df[!, col] ./ 1024, digits=2)
end
println(df)

CSV.write("benchmarking/results/complete_run_memory_results.csv", df)

header_t = (
    ["              Message          ", "MaxUCT", "DPUCT ", "UCTStar", "MCTS"],
    ["             [String]          ", " [MB] ", " [MB] ", "  [MB] ", "[MB]"])
open("benchmarking/results/complete_run_memory_results.txt", "w") do io
    pretty_table(io, df, header = header_t, header_alignment=:center)
end


mdp = MausamKolobov()
mdp = fixhorizon(mdp, 5)
is = get_initial_state(mdp)
MaxUCTSolver = THTSSolver(7.0, iterations = 10, backup_function = DPUCT_backpropagate_c, enable_UCTStar = true)
tree = base_thts(mdp, MaxUCTSolver, is)
using D3Trees
t = prepare_d3tree(tree, is)
inchrome(t)