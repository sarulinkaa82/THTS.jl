using Revise
using THTS
using FiniteHorizonPOMDPs, POMDPModels
using MCTS
using BenchmarkTools
using CSV, DataFrames, PrettyTables
using Printf

include("MausamKolobov.jl")
include("CustomDomains.jl")

include("utils.jl")

mdp = MausamKolobov()
mdp = fixhorizon(mdp, 25)


# Measure average and median reward of MaxUCT, DPUCT, UCT* and MCTS
avg_acc_reward(fhm, 100)


benchmark_one_run(mdp, "Baseline - MausamKolobov domain")
rewrite_benchmark_files()
