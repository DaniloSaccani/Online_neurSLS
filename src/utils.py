import torch


def calculate_collisions(x, sys, min_dist):
    deltax = x[:, 0::4].repeat(sys.n_agents, 1, 1) - x[:, 0::4].repeat(sys.n_agents, 1, 1).transpose(0, 2)
    deltay = x[:, 1::4].repeat(sys.n_agents, 1, 1) - x[:, 1::4].repeat(sys.n_agents, 1, 1).transpose(0, 2)
    distance_sq = (deltax ** 2 + deltay ** 2)
    n_coll = ((0.0001 < distance_sq) * (distance_sq < min_dist**2)).sum()
    return n_coll

def set_params():
    # # # # # # # # Parameters # # # # # # # #
    min_dist = 1.  # min distance for collision avoidance
    t_end = 100
    n_agents = 2
    x0, xbar = get_ini_cond(n_agents)
    linear = True
    # # # # # # # # Hyperparameters # # # # # # # #
    learning_rate = 1e-3
    epochs = 1500
    Q = 10*torch.kron(torch.eye(n_agents), torch.diag(torch.tensor([1, 1, 1, 1])))
    alpha_u = 0.001  # Regularization parameter for penalizing the input
    alpha_ca = 100
    alpha_obst = 5e3
    n_xi = 5 # \xi dimension -- number of states of REN
    l = 5  # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN
    n_traj = 5  # number of trajectories collected at each step of the learning
    std_ini = 0.2  # standard deviation of initial conditions
    gamma_bar =torch.tensor(15)
    wmax = 0.1
    decayw = 12
    maxtimew = 70
    return min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, alpha_u, alpha_ca, alpha_obst, n_xi, \
           l, n_traj, std_ini, gamma_bar, wmax, decayw,maxtimew

def set_params_tracking():
    # # # # # # # # Parameters # # # # # # # #
    min_dist = 1.  # min distance for collision avoidance
    t_end = 310
    n_agents = 1
    # # # # # # # # Hyperparameters # # # # # # # #
    learning_rate = 1e-3
    epochs = 1200
    Q = torch.kron(torch.eye(n_agents), torch.diag(torch.tensor([10, 10, .1, .1])))
    alpha_u = 0.05  # Regularization parameter for penalizing the input
    alpha_ca = 100
    alpha_obst = 1e3
    n_xi = 8 # \xi dimension -- number of states of REN
    l = 8  # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN
    n_traj = 5  # number of trajectories collected at each step of the learning
    std_ini = 0.2  # standard deviation of initial conditions
    gamma_bar = torch.tensor(15)
    wmax = 0.1
    decayw = 12
    maxtimew = 70
    return min_dist, t_end, n_agents, learning_rate, epochs, Q, alpha_u, alpha_ca, alpha_obst, n_xi, \
           l, n_traj, std_ini, gamma_bar, wmax, decayw,maxtimew


def set_params_online():
    # # # # # # # # Parameters # # # # # # # #
    params = set_params()
    min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, \
        alpha_u, alpha_ca, alpha_obst, n_xi, l, n_traj, std_ini, gamma_bar, wmax, decayw, maxtimew = params
    learning_rate = 1e-2
    epochs = 400 # 35
    n_traj_ol = 3 # number of trajectories
    Horizon = 25 #7
    timeInstantOpt = 2
    sim_time = t_end  # this is the simulation time#
    # GAIN https://ch.mathworks.com/help/control/ref/dynamicsystem.norm.html
    gainF = 1
    #gainF = 10000
    return min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, alpha_u, alpha_ca, alpha_obst, n_xi, \
        l, n_traj, std_ini, gamma_bar, wmax, decayw,maxtimew, Horizon, timeInstantOpt, sim_time, gainF, n_traj_ol


def set_params_continueTrain():
    # # # # # # # # Parameters # # # # # # # #
    params = set_params()
    min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, \
        alpha_u, alpha_ca, alpha_obst, n_xi, l, n_traj, std_ini, gamma_bar, wmax, decayw, maxtimew = params
    learning_rate = 1e-3
    epochs = 600
    return min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, alpha_u, alpha_ca, alpha_obst, n_xi, \
        l, n_traj, std_ini, gamma_bar, wmax, decayw,maxtimew

def get_ini_cond(n_agents):
    # Corridor problem
    x0 = torch.tensor([2, -2, 0, 0,
                       -2, -2, 0, 0,
                       ])
    xbar = torch.tensor([-2, 2, 0, 0,
                         2., 2, 0, 0,
                         ])
    return x0, xbar
