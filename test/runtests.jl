using THTS
using Test

@testset "action choices" begin
    include("MausamKolobov.jl")
    m = MausamKolobov()
    fhm = fixhorizon(m, 15)
    solver = THTSSolver(1.4, iterations = 100)
    
    all_s = stage_states(fhm, 1)
    all_states = collect(all_s)

    state_test1 = all_states[1]
    act1 = base_thts(fhm, state_test1, 100, 1.4)
    @test act1 == "a00"

    state_test2 = all_states[3]
    act2 = base_thts(fhm, state_test2, 100, 1.4)
    @test act2 == "a20"

    state_test3 = all_states[5]
    act3 = base_thts(fhm, state_test3, 100, 1.4)
    @test act3 == "a41"
end
