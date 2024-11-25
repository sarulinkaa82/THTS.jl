using Revise
using THTS
using FiniteHorizonPOMDPs, POMDPModels
using MCTS
using BenchmarkTools
using CSV, DataFrames, PrettyTables
using Printf

include("MausamKolobov.jl")
include("CustomDomains.jl")

include("benchmark_utils.jl")

mdp = MausamKolobov()
mdp = fixhorizon(mdp, 25)

# Measure average and median reward of MaxUCT, DPUCT, UCT* and MCTS
avg_acc_reward(fhm, 100)

# Empty files storing benchmarling results 
rewrite_benchmark_files()

benchmark_partial_run(mdp, "Baseline - MausamKolobov domain")
benchmark_complete_run(mdp, "Baseline - MausamKolobov domain")


