using Revise
using THTS
using Test
using FiniteHorizonPOMDPs

@testset "MaxUCT" begin
    include("MausamKolobov.jl")
    m = MausamKolobov()
    fhm = fixhorizon(m, 25)
    solver = THTSSolver(6.4, iterations = 5000, backup_function = MaxUCT_backpropagate_c) # There should an option to seed the algorithm
    
    all_s = stage_states(fhm, 1)
    all_states = collect(all_s)

    state_test1 = all_states[1]
    act1 = base_thts(fhm, solver, state_test1)
    @test act1 == "a00"

    state_test2 = all_states[3]
    act2 = base_thts(fhm, solver, state_test2)
    @test act2 == "a20"

    state_test3 = all_states[5]
    act3 = base_thts(fhm, solver, state_test3)
    @test act3 == "a41"
end

@testset "DPUCT" begin
    include("MausamKolobov.jl")
    m = MausamKolobov()
    fhm = fixhorizon(m, 25)
    solver = THTSSolver(6.4, iterations = 5000, backup_function = DPUCT_backpropagate_c) # There should an option to seed the algorithm
    
    all_s = stage_states(fhm, 1)
    all_states = collect(all_s)

    state_test1 = all_states[1]
    act1 = base_thts(fhm, solver, state_test1)
    @test act1 == "a00"

    state_test2 = all_states[3]
    act2 = base_thts(fhm, solver, state_test2)
    @test act2 == "a20"

    state_test3 = all_states[5]
    act3 = base_thts(fhm, solver, state_test3)
    @test act3 == "a41"
end

@testset "UCTStar" begin
    include("MausamKolobov.jl")
    m = MausamKolobov()
    fhm = fixhorizon(m, 25)
    solver = THTSSolver(6.4, iterations = 5000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = true) # There should an option to seed the algorithm
    
    all_s = stage_states(fhm, 1)
    all_states = collect(all_s)

    state_test1 = all_states[1]
    act1 = base_thts(fhm, solver, state_test1)
    @test act1 == "a00"

    state_test2 = all_states[3]
    act2 = base_thts(fhm, solver, state_test2)
    @test act2 == "a20"

    state_test3 = all_states[5]
    act3 = base_thts(fhm, solver, state_test3)
    @test act3 == "a41"
end
