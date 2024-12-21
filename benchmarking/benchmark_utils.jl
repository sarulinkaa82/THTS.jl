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

    x_std = std(x_reward)
    d_std = std(d_reward)
    u_std = std(u_reward)
    m_std = std(m_reward)

    data = [x_reward, d_reward, u_reward, m_reward]

    
    println("MaxUCT mean r: $x_mean, median r: $x_median, std: $x_std")
    println("DPUCT mean r: $d_mean, median r: $d_median, std: $d_std")
    println("UCT* mean r: $u_mean, median r: $u_median, std: $u_std")
    println("MCTS mean r $m_mean, median r: $m_median, std: $m_std")
    return data
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

function measure_improvement(str::String)
    println("BENCHMARKING MausamKolobov...")
    name = str * " - MausamKolobov"
    mdp = MausamKolobov()
    mdp = fixhorizon(mdp, 25)
    benchmark_partial_run(mdp, name, 50)
    benchmark_complete_run(mdp, name,50)

    println("BENCHMARKING maze-7")
    name = str * " - maze-7-A2"
    domain_size, grid_matrix = generate_test_domain("benchmarking/data/maze-7-A2.txt")
    mdp = CustomDomain(size = domain_size, grid = grid_matrix)
    mdp = fixhorizon(mdp, 80)
    benchmark_partial_run(mdp, name, 50)
    benchmark_complete_run(mdp, name, 50)

    println("BENCHMARKING maze-15")
    name = str * " - maze-15-A2"
    domain_size, grid_matrix = generate_test_domain("benchmarking/data/maze-15-A2.txt")
    mdp = CustomDomain(size = domain_size, grid = grid_matrix)
    mdp = fixhorizon(mdp, 80)
    benchmark_partial_run(mdp, name, 50)
    benchmark_complete_run(mdp, name,50)
end

function process_benchmark_data(baseline_id::Int, csv_path::String, name::String)

    csv_name = name * ".csv"
    txt_name = name * ".txt"
    open(csv_name, "w") do io
        CSV.write(io, DataFrame(Message = String[], MaxUCT  = Float64[], DPUCT = Float64[], UCTStar = Float64[], MCTS = Float64[]))
    end
        
    df = CSV.read(csv_path, DataFrame)
    baseMax = df[baseline_id,"MaxUCT"]
    baseDPUCT = df[baseline_id,"DPUCT"]
    baseUCTStar = df[baseline_id,"UCTStar"]
    baseMCTS = df[baseline_id,"MCTS"]

    for row in eachrow(df)
        newMax = row["MaxUCT"] ./ baseMax
        newDPUCT = row["DPUCT"] ./baseDPUCT
        newUCTStar = row["UCTStar"] ./baseUCTStar
        newMCTS = row["MCTS"] ./baseMCTS

        message = row["Message"]

        max_ft =  @sprintf("%.2f", newMax)
        dp_ft =  @sprintf("%.2f", newDPUCT)
        star_ft =  @sprintf("%.2f", newUCTStar)
        mcts_ft =  @sprintf("%.2f", newMCTS)
        
        open(csv_name, "a") do io
            CSV.write(io, DataFrame(Message = [message], MaxUCT = [max_ft], DPUCT = [dp_ft], 
            UCTStar = [star_ft], MCTS = [mcts_ft]), append=true)
        end

        data = CSV.File(csv_name) |> DataFrame
        open(txt_name, "w") do io
            pretty_table(io, data)
        end

    end
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

function values_after_run(iters::Int, mdp, solver)
    values = []
    is = get_initial_state(mdp)
    println(is)
    for i in 1:iters
        tree = THTS.base_thts(mdp, solver, is)
        push!(values, tree.d_values[1])
    end

    return values
end

function values_after_run(iters::Int, mdp)
    msolver = MCTSSolver(n_iterations=5000, exploration_constant=2.4, depth = 25, rng=MersenneTwister(8))
    
    values = []
    is = get_initial_state(mdp)
    for i in 1:iters
        planner = MCTS.solve(msolver, mdp)
        is = get_initial_state(mdp)
        MCTS.plan!(planner, is)
        val = MCTS.value(planner, is)
        push!(values, val)
    end

    return values
end

function get_convergence(mdp)

    mctsdata = CSV.File("benchmarking/iteration_values copy.csv") |> DataFrame
    MaxUCTSolver = THTSSolver(7.0, iterations = 1000, backup_function = MaxUCT_backpropagate_c, enable_UCTStar = false, verbose = true)
    DPUCTSolver = THTSSolver(1.4, iterations = 1000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = false, verbose = true)
    UCTStarSolver = THTSSolver(1.4, iterations = 1000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = true, verbose = true)
    is = get_initial_state(mdp)
    base_thts(mdp, MaxUCTSolver, is)
    maxdata = CSV.File("benchmarking/iteration_values.csv") |> DataFrame
    base_thts(mdp, DPUCTSolver, is)
    dpdata = CSV.File("benchmarking/iteration_values.csv") |> DataFrame
    base_thts(mdp, UCTStarSolver, is)
    stardata = CSV.File("benchmarking/iteration_values.csv") |> DataFrame
    p = plot(label="MaxUCT", maxdata.Iteration, maxdata.Value, xlabel="Iteration", ylabel="Value", title="Value over Iterations")
    p = plot!(label="DP-UCT", maxdata.Iteration, dpdata.Value, xlabel="Iteration", ylabel="Value", title="Value over Iterations")
    p = plot!(label="UCT*", maxdata.Iteration, stardata.Value, xlabel="Iteration", ylabel="Value", title="Value over Iterations")
    p = plot!(label="MCTS", mctsdata.Iteration, mctsdata.Value, xlabel="Iteration", ylabel="Value", title="Value over Iterations")
end


function generate_custom_latex_table(data, headers, caption, label)
    # Construct the header row with bold formatting
    header_row = join(["\\bfseries " * h for h in headers], " & ") * " \\\\ \\Midrule"
    
    # Construct the data rows
    data_rows = [join(row, " & ") * " \\\\" for row in data]
    
    # Combine everything into the table structure
    latex_table = """
    \\begin{table}[h]
    \\begin{ctucolortab}
    \\caption{$caption}
    \\begin{tabular}{cccccc}
    $header_row
    $(join(data_rows, " \n"))
    \\end{tabular}
    \\end{ctucolortab}
    \\label{$label}
    \\end{table}
    """
    
    return latex_table
end



function get_value_stat()
    mdp = MausamKolobov()
    mdp = fixhorizon(mdp, 25)

    MaxUCTSolver = THTSSolver(7.0, iterations = 1000, backup_function = MaxUCT_backpropagate_c, enable_UCTStar = false)
    DPUCTSolver = THTSSolver(3.4, iterations = 1000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = false)
    UCTStarSolver = THTSSolver(3.4, iterations = 1000, backup_function = DPUCT_backpropagate_c, enable_UCTStar = true)

    maxuct = values_after_run(50, mdp, MaxUCTSolver)
    plot(maxuct)
    dpuct = values_after_run(50, mdp, DPUCTSolver)
    uctstar = values_after_run(50, mdp, UCTStarSolver)
    mcts = values_after_run(50, mdp)


    # Combine the data into a vector of vectors
    data = [maxuct, dpuct, uctstar, mcts]

    # Create the boxplot
    boxplot(data,
        labels=["MaxUCT", "DPUCT", "UCTStar", "MCTS"],  # Names for each group
        xlabel="Algorithms",
        ylabel="Value",
        title="Values after partial runs",
        xticks=(1:4, ["MaxUCT", "DPUCT", "UCTStar", "MCTS"]),  # Set custom x-axis labels
        legend=false
    )
    hline!([-5.992], label="Correct Value", linestyle=:dash, color=:green)
end

function bench_to_table(path::String, caption::String, label::String)
    df = CSV.read(path, DataFrame)
    headers = names(df)
    data = [collect(row) for row in eachrow(df)]

    latex_code = generate_custom_latex_table(data, headers, caption, label)
    print(latex_code)
end