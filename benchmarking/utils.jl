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


function run_mcts(msolver::MCTSSolver, mdp::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper)
    path = []
    # msolver = MCTSSolver(n_iterations=2000, exploration_constant=2.4, depth = 45)
    state = get_initial_state(mdp)
    acc_reward = 0

    while !isterminal(mdp, state)
        planner = MCTS.solve(msolver, fhm)
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

function one_mcts(fhm::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, solver::MCTSSolver, init_state)
    planner = MCTS.solve(solver, fhm)
    MCTS.plan!(planner, init_state)
    best_action = action(planner, init_state)
    return best_action
end

function rewrite_benchmark_files()
    open("benchmarking/benchmark_time_results.csv", "w") do io
        CSV.write(io, DataFrame(Message = String[], MaxUCT  = Float64[], DPUCT = Float64[], UCTStar = Float64[], MCTS = Float64[]))
    end

    open("benchmarking/benchmark_memory_results.csv", "w") do io
        CSV.write(io, DataFrame(Message = String[], MaxUCT = Float64[], DPUCT = Float64[], UCTStar = Float64[], MCTS = Float64[]))
    end
end


function benchmark_one_run(fhm::FiniteHorizonPOMDPs.FixedHorizonMDPWrapper, message::String)
    MaxUCTSolver = THTSSolver(7.0, iterations = 1000, backup_function = MaxUCT_backpropagate_c, enable_UCTStar = false)
    DPUCTSolver = THTSSolver(3.4, iterations = 1000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = false)
    UCTStarSolver = THTSSolver(3.4, iterations = 1000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = true)
    MSolver = MCTSSolver(exploration_constant = 6.4, n_iterations = 1000, depth = 45)
    is = get_initial_state(fhm)

    max_b = @benchmark base_thts($fhm, $MaxUCTSolver, $is)
    println("Benchmarking MaxUCT finished")
    dp_b = @benchmark base_thts($fhm, $DPUCTSolver, $is)
    println("Benchmarking DPUCT finished")
    star_b = @benchmark base_thts($fhm, $UCTStarSolver, $is)
    println("Benchmarking UCT* finished")
    mcts_b = @benchmark one_mcts($fhm, $MSolver, $is)
    println("Benchmarking MCTS finished")

    max_time = mean(max_b).time
    dp_time = mean(dp_b).time
    star_time = mean(star_b).time
    mcts_time = mean(mcts_b).time

    max_ft =  @sprintf("%.2f", max_time / 1e6)
    dp_ft =  @sprintf("%.2f", dp_time / 1e6)
    star_ft =  @sprintf("%.2f", star_time / 1e6)
    mcts_ft =  @sprintf("%.2f", mcts_time / 1e6)
    
    open("benchmarking/benchmark_time_results.csv", "a") do io
        CSV.write(io, DataFrame(Message = [message], MaxUCT = [max_ft], DPUCT = [dp_ft], 
        UCTStar = [star_ft], MCTS = [mcts_ft]), append=true)
    end

    max_mem = mean(max_b).memory
    dp_mem = mean(dp_b).memory
    star_mem = mean(star_b).memory
    mcts_mem = mean(mcts_b).memory
    
    max_fm = @sprintf("%.0f", max_mem / 1024)
    dp_fm = @sprintf("%.0f", dp_mem / 1024)
    star_fm = @sprintf("%.0f", star_mem / 1024)
    mcts_fm = @sprintf("%.0f", mcts_mem / 1024)

    open("benchmarking/benchmark_memory_results.csv", "a") do io
        CSV.write(io, DataFrame(Message = [message], MaxUCT = [max_fm], DPUCT = [dp_fm], 
                                UCTStar = [star_fm], MCTS = [mcts_fm]), append=true)
    end

    header_t = (
    ["              Message          ", "MaxUCT", "DPUCT ", "UCTStar", "MCTS"],
    ["             [String]          ", " [ms] ", " [ms] ", "  [ms] ", "[ms]"])

    t_data = CSV.File("benchmarking/benchmark_time_results.csv") |> DataFrame
    open("benchmarking/benchmark_time_results.txt", "w") do io
        pretty_table(io, t_data, header = header_t, header_alignment=:center)
    end

    header_m = (
        ["              Message          ", "  MaxUCT ", " DPUCT", "UCTStar", "MCTS "],
        ["              [String]         ", "  [KiB]  ",  " [KiB]"," [KiB] ", "[KiB]"])
    
    m_data = CSV.File("benchmarking/benchmark_memory_results.csv") |> DataFrame
    open("benchmarking/benchmark_memory_results.txt", "w") do io
        pretty_table(io,m_data, header = header_m, header_alignment=:center)
    end

end


