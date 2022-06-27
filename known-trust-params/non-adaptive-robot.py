"""Here, the robot tries to estimate the human's weights but only uses them for performance computation for the trust update.
The weights of the objective function of the robot are fixed. 
Further, the true trust parameters of the human are known a priori. So, they are not updated via feedback"""

import numpy as np
import _context
from classes.POMDPSolver import Solver
from classes.HumanModels import BoundedRational
from classes.IRLModel import Posterior
from classes.ThreatSetter import ThreatSetter
from classes.RewardFunctions import Affine
import os
import argparse
import pickle
import time
from tqdm import tqdm

def run_one_simulation(args: argparse.Namespace, seed: int):

    # Output data: Trust feedback, trust estimation, posterior distribution, weights, healths, times, recommendations, actions
    data = {}

    ############################################# PARAMETERS THAT CAN BE MODIFIED ##################################################
    wh_rob = args.health_weight_robot           # Fixed health weight of the robot
    wt_rob = args.trust_weight                  # Trust increase reward weight
    kappa = args.kappa                          # Assumed rationality coefficient in the bounded rationality model
    stepsize = args.posterior_stepsize                    # stepsize in the posterior
    wh_hum = args.health_weight_human           # True health weight of the human. time weight = 1 - health weight
    trust_params = args.trust_params            # Human's true trust parameters in the beta distribution model [alpha_0, beta_0, ws, wf]. These are known by the robot
    directory = "./figures/Bounded Rationality/non-adaptive-learner/kappa" + str(kappa) + "/" + str(wh_hum) # Storage directory for the plots
    N = args.num_sites                          # Number of sites in a mission (Horizon for planning)
    num_missions = args.num_missions            # Number of "missions" of N sites each
    region_size = args.region_size              # Region is a group of houses with a specific value of prior threat probability

    # Reward function
    max_health = 110 # The constant in the affine reward function
    reward_fun = Affine(max_health=max_health)

    # Flags for controlling the resets in between missions
    RESET_SOLVER = args.reset_solver
    RESET_HUMAN = args.reset_human
    RESET_HEALTH = args.reset_health
    RESET_TIME = args.reset_time

    STORE_FIGS = args.store_figs     # Flag to decide whether to save the trust and posterior plots
    PRINT_FLAG = args.print_flag     # Flag to decide whether to print the data to the console output
    #################################################################################################################################

    wc_rob = 1 - wh_rob
    est_human_weights = {'health': None, 'time': None}
    rob_weights = {'health': wh_rob, 'time': wc_rob, 'trust': wt_rob}
    solver = Solver(N, rob_weights, trust_params.copy(), None, None, None, est_human_weights, hum_mod='bounded_rational', reward_fun=reward_fun)

    # Intialize posterior
    posterior = Posterior(kappa=kappa, stepsize=stepsize, reward_fun=reward_fun)

    # Initialize human model
    wc_hum = 1-wh_hum
    human_weights = {"health":wh_hum, "time": wc_hum}
    human = BoundedRational(trust_params, human_weights, reward_fun=reward_fun, kappa=1.0)
    
    if STORE_FIGS and not os.path.exists(directory):
        os.makedirs(directory)
    
    # THINGS TO LOOK FOR AND STORE AND PLOT/PRINT
    # Trust, posterior after every interaction, health, time, recommendation, action
    # # Initialize storage
    # N stuff
    recs = np.zeros((num_missions, N), dtype=int)
    acts = np.zeros((num_missions, N), dtype=int)
    weights = posterior.weights.copy()
    prior_levels_storage = np.zeros((num_missions, N), dtype=float)
    after_scan_storage = np.zeros((num_missions, N), dtype=float)
    threats_storage = np.zeros((num_missions, N), dtype=int)
    perf_actual = np.zeros((num_missions, N), dtype=int)
    perf_est = np.zeros((num_missions, N), dtype=int)
    
    # N+1 stuff
    trust_feedback = np.zeros((num_missions, N+1), dtype=float)
    trust_estimate = np.zeros((num_missions, N+1), dtype=float)
    times = np.zeros((num_missions, N+1), dtype=int)
    healths = np.zeros((num_missions, N+1), dtype=int)
    parameter_estimates = np.zeros((num_missions, N+1, 4), dtype=float)
    wh_means = np.zeros((num_missions, N+1), dtype=float)
    wh_map = np.zeros((num_missions, N+1), dtype=float)
    wh_map_prob = np.zeros((num_missions, N+1), dtype=float)
    posterior_dists = np.zeros((num_missions, N+1, len(posterior.dist)), dtype=float)

    # Initialize health and time
    health = 100
    current_time = 0

    for j in range(num_missions):

        if PRINT_FLAG:
            # For printing purposes
            table_data = [['prior', 'after_scan', 'rec', 'action', 'health', 'time', 'trust-fb', 'trust-est', 'perf-hum', 'perf-rob', 'wh-mean', 'wh-map']]

        # Initialize threats
        rng = np.random.default_rng(seed=seed+j)
        priors = rng.random(N // region_size)
        threat_setter = ThreatSetter(N, region_size, priors=priors, seed=seed+j)
        threat_setter.setThreats()
        priors = threat_setter.priors
        after_scan = threat_setter.after_scan
        prior_levels = np.zeros_like(after_scan)
        for i in range(threat_setter.num_regions):
            prior_levels[i*threat_setter.region_size:(i+1)*threat_setter.region_size] = priors[i]
        threats = threat_setter.threats
        solver.update_danger(threats, prior_levels, after_scan, reset=False)

        # Store threat infos
        prior_levels_storage[j, :] = prior_levels
        after_scan_storage[j, :] = after_scan
        threats_storage[j, :] = threats

        # Reset the solver to remove old performance history. But, we would need new parameters
        if RESET_SOLVER:
            solver.reset(human.get_mean())
        if RESET_HUMAN:
            human.reset()
        if RESET_HEALTH:
            health = 100
        if RESET_TIME:
            current_time = 0

        # For each site, get recommendation, choose action, update health, time, trust, posterior
        for i in range(N):

            # Get the recommendation
            rec = solver.get_recommendation(i, health, current_time, posterior)

            # Choose action
            action = human.choose_action(rec, after_scan[i], health, current_time)

            # Update health, time
            time_old = current_time
            health_old = health

            if action:
                current_time += 10.
            else:
                if threats[i]:
                    health -= 10.

            # Storage
            recs[j, i] = rec
            acts[j, i] = action
            times[j, i] = time_old
            healths[j, i] = health_old

            # Update posterior (UPDATE THIS BEFORE UPDATING TRUST)
            wh_means[j, i] = posterior.get_mean()
            prob, weight = posterior.get_map()
            wh_map[j, i] = weight
            wh_map_prob[j, i] = prob
            posterior_dists[j, i, :] = posterior.dist
            posterior.update(rec, action, human.get_mean(), health_old, time_old, after_scan[i])

            # Use the old values of health and time to compute the performance
            trust_est_old = solver.get_trust_estimate()
            trust_estimate[j, i] = trust_est_old
            solver.forward(i, rec, health_old, time_old, posterior)

            # Update trust (based on old values of health and time)
            trust_feedback[j, i] = human.get_feedback()
            human.update_trust(rec, threats[i], health_old, time_old)

            ############ TODO: Update trust parameters ###########################
            parameter_estimates[j, i, :] = np.array(solver.get_trust_params())
            ######################################################################

            # Storage
            perf_est[j, i] = solver.get_last_performance()
            perf_actual[j, i] = human.get_last_performance()

            if PRINT_FLAG:
                # Store stuff
                row = []
                row.append("{:.2f}".format(threat_setter.prior_levels[i]))
                row.append("{:.2f}".format(threat_setter.after_scan[i]))
                row.append(str(rec))
                row.append(str(action))
                row.append(str(health_old))
                row.append(str(time_old))
                row.append("{:.2f}".format(trust_feedback[j, i]))
                row.append("{:.2f}".format(trust_est_old))
                row.append(str(human.get_last_performance()))
                row.append(str(solver.get_last_performance()))
                row.append("{:.2f}".format(posterior.get_mean()))
                row.append("{:.2f}".format(posterior.get_map()[1]))
                table_data.append(row)

        if PRINT_FLAG:
            # Get the values after the last site
            row = ['', '', '', '', str(health), str(current_time), "{:.2f}".format(human.get_mean()), "{:.2f}".format(solver.get_trust_estimate()), str(human.get_last_performance()), str(solver.get_last_performance()), "{:.2f}".format(posterior.get_mean()), "{:.2f}".format(posterior.get_map()[1])]
            table_data.append(row)
            # Print
            col_print(table_data)
        
        # Store the final values after the last house
        trust_feedback[j, -1] = human.get_feedback()
        trust_estimate[j, -1] = solver.get_trust_estimate()
        healths[j, -1] = health
        times[j, -1] = current_time
        parameter_estimates[j, -1, :] = np.array(solver.get_trust_params())
        wh_means[j, -1] = posterior.get_mean()
        prob, weight = posterior.get_map()
        wh_map[j, -1] = weight
        wh_map_prob[j, -1] = prob
        posterior_dists[j, -1, :] = posterior.dist
    
    data['trust feedback'] = trust_feedback
    data['trust estimate'] = trust_estimate
    data['health'] = healths
    data['time'] = times
    data['recommendation'] = recs
    data['actions'] = acts
    data['weights'] = weights
    data['posterior'] = posterior_dists
    data['prior threat level'] = prior_levels_storage
    data['after scan level'] = after_scan_storage
    data['threat'] = threats_storage
    data['trust parameter estimates'] = parameter_estimates
    data['mean health weight'] = wh_means
    data['map health weight'] = wh_map
    data['map health weight probability'] = wh_map_prob
    data['performance estimates'] = perf_est
    data['performance actual'] = perf_actual

    return data

def main(args: argparse.Namespace):

    ############################################# PARAMETERS THAT CAN BE MODIFIED ##################################################
    num_simulations = args.num_simulations      # Number of simulations to run
    STORE_FIGS = args.store_figs                # Flag to decide whether to save the trust and posterior plots
    N = args.num_sites                          # Number of sites in a mission (Horizon for planning)
    num_missions = args.num_missions            # Number of "missions" of N sites each
    stepsize = args.posterior_stepsize          # Stepsize in the posterior distrbution over the weights
    num_weights = int(1/stepsize) + 1           # Number of weight samples in the posterior distribution
    wh_hum = args.health_weight_human           # True health weight of the human. time weight = 1 - health weight
    kappa = args.kappa                          # Assumed rationality coefficient in the bounded rationality model
    data_direc = "./data/Bounded Rationality/non-adaptive-learner/kappa" + str(kappa) + "/wh" + str(wh_hum) # Storage directory for the plots
    #################################################################################################################################

    data_all = {}
    data_all['trust feedback'] = np.zeros((num_simulations, num_missions, N+1), dtype=float)
    data_all['trust estimate'] = np.zeros((num_simulations, num_missions, N+1), dtype=float)
    data_all['health'] = np.zeros((num_simulations, num_missions, N+1), dtype=int)
    data_all['time'] = np.zeros((num_simulations, num_missions, N+1), dtype=int)
    data_all['recommendation'] = np.zeros((num_simulations, num_missions, N), dtype=int)
    data_all['actions'] = np.zeros((num_simulations, num_missions, N), dtype=int)
    data_all['weights'] = np.zeros((num_simulations, num_missions, N, num_weights), dtype=float)
    data_all['posterior'] = np.zeros((num_simulations, num_missions, N+1, num_weights), dtype=float)
    data_all['prior threat level'] = np.zeros((num_simulations, num_missions, N), dtype=float)
    data_all['after scan level'] = np.zeros((num_simulations, num_missions, N), dtype=float)
    data_all['threat'] = np.zeros((num_simulations, num_missions, N), dtype=int)
    data_all['trust parameter estimates'] = np.zeros((num_simulations, num_missions, N+1, 4), dtype=float)
    data_all['mean health weight'] = np.zeros((num_simulations, num_missions, N+1), dtype=float)
    data_all['map health weight'] = np.zeros((num_simulations, num_missions, N+1), dtype=float)
    data_all['map health weight probability'] = np.zeros((num_simulations, num_missions, N+1), dtype=float)
    data_all['performance estimates'] = np.zeros((num_simulations, num_missions, N), dtype=int)
    data_all['performance actual'] = np.zeros((num_simulations, num_missions, N), dtype=int)

    for i in tqdm(range(num_simulations)):
        data_one_simulation = run_one_simulation(args,  i * num_missions)
        for k, v in data_one_simulation.items():
            # print(k)
            data_all[k][i] = v

    ############################### STORING THE DATA #############################
    if not os.path.exists(data_direc):
        os.makedirs(data_direc)

    data_file = data_direc + '/' + time.strftime("%Y%m%d-%H%M%S") + '.pkl'
    with open(data_file, 'wb') as f:
        pickle.dump(data_all, f)

def col_print(table_data):
    
    string = ""
    for _ in range(len(table_data[0])):
        string += "{: ^12}"

    for row in table_data:
        print(string.format(*row))

if __name__ == "__main__":

    parser= argparse.ArgumentParser(description='Adaptive solver that adapts its weights according to learnt human weights')
    parser.add_argument('--trust-weight', type=float, help='trust weight for the robot (default: 10.0)', default=10.0)
    parser.add_argument('--kappa', type=float, help='rationality coeffiecient (default: 0.05)', default=0.05)
    parser.add_argument('--health-weight-robot', type=float, help='Fixed health weight of the robot (default: 0.7)', default=0.7)
    parser.add_argument('--health-weight-human', type=float, help='True health weight of the human (default: 0.9)', default=0.9)
    parser.add_argument('--trust-params', nargs=4, help='Trust parameters for the human, default=[90., 30., 20. 30.]', default=[90., 30., 20., 30.])
    parser.add_argument('--num-sites', type=int, help='Number of sites in a mission (default: 10)', default=10)
    parser.add_argument('--num-missions', type=int, help='Number of missions (default: 10)', default=10)
    parser.add_argument('--region-size', type=int, help='Size of a region with a certain probability of threat (default: 1)', default=1)
    parser.add_argument('--reset-solver', type=bool, help="Flag to reset the solver after each mission (default: False)", default=False)
    parser.add_argument('--reset-human', type=bool, help="Flag to reset the human after each mission (default: False)", default=False)
    parser.add_argument('--reset-health', type=bool, help="Flag to reset the health after each mission (default: False)", default=False)
    parser.add_argument('--reset-time', type=bool, help="Flag to reset the time after each mission (default: False)", default=False)
    parser.add_argument('--store-figs', type=bool, help="Flag to store the plots (default: False)", default=False)
    parser.add_argument('--print-flag', type=bool, help="Flag to print the data to output (default: False)", default=False)
    parser.add_argument('--num-simulations', type=int, help='Number of simulations to run (default: 10000)', default=10000)
    parser.add_argument('--posterior-stepsize', type=float, help='Stepsize in the posterior distribution (default(0.05)', default=0.05)

    main(parser.parse_args())
