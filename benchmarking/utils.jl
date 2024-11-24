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