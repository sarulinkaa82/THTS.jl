using Statistics


function avg_acc_reward(mdp::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, iters::Int)
    MaxUCTSolver = THTSSolver(7.0, iterations = 1000, backup_function = MaxUCT_backpropagate_c, enable_UCTStar = false)
    DPUCTSolver = THTSSolver(3.4, iterations = 1000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = false)
    UCTStarSolver = THTSSolver(3.4, iterations = 1000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = true)
    MCTSSolver = MCTS.MCTSSolver(exploration_constant=6.4, n_iterations = 1000, depth = 45)
    
    x_reward = []
    d_reward = []
    u_reward = []
    m_reward = []


    for i in 1:iters
        xr = THTS.solve(MaxUCTSolver, mdp)
        dr = THTS.solve(DPUCTSolver, mdp)
        ur = THTS.solve(UCTStarSolver, mdp)
        mr = run_mcts(MCTSSolver, mdp)

        push!(x_reward, xr)
        push!(d_reward, dr)
        push!(u_reward, ur)
        push!(m_reward, mr)

        println("Iteration $i finished")
    end

    x_mean = mean(x_reward)
    d_mean = mean(d_reward)
    u_mean = mean(u_reward)
    m_mean = mean(m_reward)

    x_median = median(x_reward)
    d_median = median(d_reward)
    u_median = median(u_reward)
    m_median = median(m_reward)
    

    println("MaxUCT mean r: $x_mean, median r: $x_median")
    println("DPUCT mean r: $d_mean, median r: $d_median")
    println("UCT* mean r: $u_mean, median r: $u_median")
    println("MCTS mean r $m_mean, median r: $m_median")
end


function one_mcts(fhm::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, solver::MCTSSolver, init_state)
    planner = MCTS.solve(solver, fhm)
    MCTS.plan!(planner, init_state)
    best_action = action(planner, init_state)
    return best_action
end

function rewrite_benchmark_files()
    open("benchmarking/results/partial_run_time_results.csv", "w") do io
        CSV.write(io, DataFrame(Message = String[], MaxUCT  = Float64[], DPUCT = Float64[], UCTStar = Float64[], MCTS = Float64[]))
    end

    open("benchmarking/results/partial_run_memory_results.csv", "w") do io
        CSV.write(io, DataFrame(Message = String[], MaxUCT = Float64[], DPUCT = Float64[], UCTStar = Float64[], MCTS = Float64[]))
    end

    open("benchmarking/results/complete_run_time_results.csv", "w") do io
        CSV.write(io, DataFrame(Message = String[], MaxUCT  = Float64[], DPUCT = Float64[], UCTStar = Float64[], MCTS = Float64[]))
    end

    open("benchmarking/results/complete_run_memory_results.csv", "w") do io
        CSV.write(io, DataFrame(Message = String[], MaxUCT  = Float64[], DPUCT = Float64[], UCTStar = Float64[], MCTS = Float64[]))
    end
end


function benchmark_partial_run(fhm::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, message::String, sample_n::Int)
    MaxUCTSolver = THTSSolver(7.0, iterations = 1000, backup_function = MaxUCT_backpropagate_c, enable_UCTStar = false)
    DPUCTSolver = THTSSolver(3.4, iterations = 1000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = false)
    UCTStarSolver = THTSSolver(3.4, iterations = 1000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = true)
    MSolver = MCTSSolver(exploration_constant = 6.4, n_iterations = 1000, depth = 45)
    is = get_initial_state(fhm)

    println("Running $sample_n samples")
    max_b = @benchmarkable base_thts($fhm, $MaxUCTSolver, $is)
    max_res = run(max_b, samples = sample_n, seconds = 120)
    println("Benchmarking MaxUCT finished")

    dp_b = @benchmarkable base_thts($fhm, $DPUCTSolver, $is)
    dp_res = run(dp_b, samples = sample_n, seconds = 120)
    println("Benchmarking DPUCT finished")
    
    star_b = @benchmarkable base_thts($fhm, $UCTStarSolver, $is)
    star_res = run(star_b, samples = sample_n, seconds = 120)
    println("Benchmarking UCT* finished")

    mcts_b = @benchmarkable one_mcts($fhm, $MSolver, $is)
    mcts_res = run(mcts_b, samples = sample_n, seconds = 120)
    println("Benchmarking MCTS finished")

    max_time = median(max_res).time
    dp_time = median(dp_res).time
    star_time = median(star_res).time
    mcts_time = median(mcts_res).time

    max_ft =  @sprintf("%.2f", max_time / 1e6)
    dp_ft =  @sprintf("%.2f", dp_time / 1e6)
    star_ft =  @sprintf("%.2f", star_time / 1e6)
    mcts_ft =  @sprintf("%.2f", mcts_time / 1e6)
    
    open("benchmarking/results/partial_run_time_results.csv", "a") do io
        CSV.write(io, DataFrame(Message = [message], MaxUCT = [max_ft], DPUCT = [dp_ft], 
        UCTStar = [star_ft], MCTS = [mcts_ft]), append=true)
    end

    max_mem = median(max_res).memory
    dp_mem = median(dp_res).memory
    star_mem = median(star_res).memory
    mcts_mem = median(mcts_res).memory
    
    max_fm = @sprintf("%.0f", max_mem / 1024)
    dp_fm = @sprintf("%.0f", dp_mem / 1024)
    star_fm = @sprintf("%.0f", star_mem / 1024)
    mcts_fm = @sprintf("%.0f", mcts_mem / 1024)

    open("benchmarking/results/partial_run_memory_results.csv", "a") do io
        CSV.write(io, DataFrame(Message = [message], MaxUCT = [max_fm], DPUCT = [dp_fm], 
                                UCTStar = [star_fm], MCTS = [mcts_fm]), append=true)
    end

    header_t = (
    ["              Message          ", "MaxUCT", "DPUCT ", "UCTStar", "MCTS"],
    ["             [String]          ", " [ms] ", " [ms] ", "  [ms] ", "[ms]"])

    t_data = CSV.File("benchmarking/results/partial_run_time_results.csv") |> DataFrame
    open("benchmarking/results/partial_run_time_results.txt", "w") do io
        pretty_table(io, t_data, header = header_t, header_alignment=:center)
    end

    header_m = (
        ["              Message          ", "  MaxUCT ", " DPUCT", "UCTStar", "MCTS "],
        ["              [String]         ", "  [KiB]  ",  " [KiB]"," [KiB] ", "[KiB]"])
    
    m_data = CSV.File("benchmarking/results/partial_run_memory_results.csv") |> DataFrame
    open("benchmarking/results/partial_run_memory_results.txt", "w") do io
        pretty_table(io,m_data, header = header_m, header_alignment=:center)
    end

    return (max_res, dp_res, star_res, mcts_res)

end



function benchmark_complete_run(fhm::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, message::String, sample_n::Int)
    MaxUCTSolver = THTSSolver(7.0, iterations = 1000, backup_function = MaxUCT_backpropagate_c, enable_UCTStar = false)
    DPUCTSolver = THTSSolver(3.4, iterations = 1000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = false)
    UCTStarSolver = THTSSolver(3.4, iterations = 1000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = true)
    MSolver = MCTSSolver(exploration_constant = 6.4, n_iterations = 1000, depth = 45)

    println("Running $sample_n samples")
    max_b = @benchmarkable THTS.solve($MaxUCTSolver, $fhm)
    max_res = run(max_b, samples = sample_n, seconds = 120)
    println("Benchmarking MaxUCT finished")

    dp_b = @benchmarkable THTS.solve($DPUCTSolver, $fhm)
    dp_res = run(dp_b, samples = sample_n, seconds = 120)
    println("Benchmarking DPUCT finished")
    
    star_b = @benchmarkable THTS.solve($UCTStarSolver, $fhm)
    star_res = run(star_b, samples = sample_n, seconds = 120)
    println("Benchmarking UCT* finished")
    
    mcts_b = @benchmarkable run_mcts($MSolver, $fhm)
    mcts_res = run(mcts_b, samples = sample_n, seconds = 120)
    println("Benchmarking MCTS finished")

    max_time = median(max_res).time
    dp_time = median(dp_res).time
    star_time = median(star_res).time
    mcts_time = median(mcts_res).time

    max_ft =  @sprintf("%.2f", max_time / 1e6)
    dp_ft =  @sprintf("%.2f", dp_time / 1e6)
    star_ft =  @sprintf("%.2f", star_time / 1e6)
    mcts_ft =  @sprintf("%.2f", mcts_time / 1e6)
    
    open("benchmarking/results/complete_run_time_results.csv", "a") do io
        CSV.write(io, DataFrame(Message = [message], MaxUCT = [max_ft], DPUCT = [dp_ft], 
        UCTStar = [star_ft], MCTS = [mcts_ft]), append=true)
    end

    max_mem = median(max_res).memory
    dp_mem = median(dp_res).memory
    star_mem = median(star_res).memory
    mcts_mem = median(mcts_res).memory
    
    max_fm = @sprintf("%.0f", max_mem / 1024)
    dp_fm = @sprintf("%.0f", dp_mem / 1024)
    star_fm = @sprintf("%.0f", star_mem / 1024)
    mcts_fm = @sprintf("%.0f", mcts_mem / 1024)

    open("benchmarking/results/complete_run_memory_results.csv", "a") do io
        CSV.write(io, DataFrame(Message = [message], MaxUCT = [max_fm], DPUCT = [dp_fm], 
                                UCTStar = [star_fm], MCTS = [mcts_fm]), append=true)
    end

    header_t = (
    ["              Message          ", "MaxUCT", "DPUCT ", "UCTStar", "MCTS"],
    ["             [String]          ", " [ms] ", " [ms] ", "  [ms] ", "[ms]"])

    t_data = CSV.File("benchmarking/results/complete_run_time_results.csv") |> DataFrame
    open("benchmarking/results/complete_run_time_results.txt", "w") do io
        pretty_table(io, t_data, header = header_t, header_alignment=:center)
    end

    header_m = (
        ["              Message          ", "  MaxUCT ", " DPUCT", "UCTStar", "MCTS "],
        ["              [String]         ", "  [KiB]  ",  " [KiB]"," [KiB] ", "[KiB]"])
    
    m_data = CSV.File("benchmarking/results/complete_run_memory_results.csv") |> DataFrame
    open("benchmarking/results/complete_run_memory_results.txt", "w") do io
        pretty_table(io,m_data, header = header_m, header_alignment=:center)
    end
    
    return (max_res, dp_res, star_res, mcts_res)

end



function run_mcts(msolver::MCTSSolver, mdp::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper)
    path = []
    state = get_initial_state(mdp)
    acc_reward = 0

    while !isterminal(mdp, state)
        planner = MCTS.solve(msolver, mdp)
        MCTS.plan!(planner, state)
        best_action = action(planner, state)

        push!(path, (state, best_action))
        (state, r) = THTS.get_next_state(mdp, state, best_action)
        acc_reward += r
    end

    push!(path, (state))
    
    # println(path)
    return acc_reward
end
