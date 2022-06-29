"""Here, the robot tries to estimate the human's weights and uses them as its own weights and for performance computation for the trust update.
Further, the true trust parameters of the human are not known. They are not updated usign gradient descent after receiving trust feedback"""

import numpy as np
import _context
from classes.Utils import *
from classes.POMDPSolver import Solver
from classes.HumanModels import BoundedRational
from classes.IRLModel import Posterior
from classes.ThreatSetter import ThreatSetter
from classes.RewardFunctions import Affine
from classes.ParamsUpdater import Estimator
import os
import argparse
from tqdm import tqdm
import time
import pickle

def run_one_simulation(args: argparse.Namespace, seed: int):

    data = {}
    
    ############################################# PARAMETERS THAT CAN BE MODIFIED ##################################################
    wt_rob = args.trust_weight                  # Trust increase reward weight
    kappa = args.kappa                          # Assumed rationality coefficient in the bounded rationality model
    wh_hum = args.health_weight_human           # Humans true health weight. time weight = 1 - health weight
    trust_params = args.trust_params            # Human's true trust parameters in the beta distribution model [alpha_0, beta_0, ws, wf]. These are known by the robot
    directory = "./figures/Bounded-Rationality/adaptive-learner/kappa" + str(kappa) + "/" + str(wh_hum) # Storage directory for the plots
    N = args.num_sites                          # Number of sites in a mission (Horizon for planning)
    num_missions = args.num_missions            # Number of "missions" of N sites each
    region_size = args.region_size              # Region is a group of houses with a specific value of prior threat probability

    # Reward function
    max_health = 110 # The constant in the affine reward function
    reward_fun = Affine(max_health=max_health)

    # Trust parameter estimator
    num_iterations = args.num_gradient_steps
    gradient_stepsize = args.gradient_stepsize
    err_tol = args.tolerance
    estimator = Estimator(num_iterations, gradient_stepsize, err_tol)

    # Posterior
    stepsize = args.posterior_stepsize

    # Flags for controlling the resets in between missions
    RESET_SOLVER = args.reset_solver
    RESET_HUMAN = args.reset_human
    RESET_HEALTH = args.reset_health
    RESET_TIME = args.reset_time
    RESET_ESTIMATOR = args.reset_estimator

    STORE_FIGS = args.store_figs     # Flag to decide whether to save the trust and posterior plots
    PRINT_FLAG = args.print_flag     # Flag to decide whether to print the data to the console output
    #################################################################################################################################

    # Intialize posterior
    posterior = Posterior(kappa=kappa, stepsize=stepsize, reward_fun=reward_fun)

    wh_rob = posterior.get_mean()
    wc_rob = 1 - wh_rob
    est_human_weights = {'health': None, 'time': None}
    rob_weights = {'health': wh_rob, 'time': wc_rob, 'trust': wt_rob}
    solver = Solver(N, rob_weights, trust_params.copy(), None, None, None, est_human_weights, hum_mod='bounded_rational', reward_fun=reward_fun)

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

        # Store threats stuff
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
        if RESET_ESTIMATOR:
            estimator.reset()

        # Initialize the trust feedbacks and estimates. This serves as a general starting state of trust for a participant (more like propensity)
        # This is before any interaction. Serves as a starting point for trust parameters
        trust_feedback[j, 0] = human.get_feedback()

        # Get an initial guess on the parameters based on this feedback
        initial_guess = estimator.getInitialGuess(trust_feedback[j, 0])

        # Set the solver's trust params to this initial guess
        solver.update_params(initial_guess)
        trust_estimate[j, 0] = solver.get_trust_estimate()

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
                current_time += 10
            else:
                if threats[i]:
                    health -= 10

            times[i] = current_time
            healths[i] = health

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

            # Update the weights of the solver
            wh_rob = posterior.get_mean()
            wc_rob = 1 - wh_rob
            rob_weights = {'health': wh_rob, 'time': wc_rob, 'trust': wt_rob}
            solver.set_reward_weights(rob_weights)

            # Use the old values of health and time to compute the performance
            solver.forward(i, rec, health_old, time_old, posterior)
            trust_est_after = solver.get_trust_estimate()
            trust_estimate[j, i+1] = trust_est_after
            
            # Update trust (based on old values of health and time)
            human.update_trust(rec, threats[i], health_old, time_old)
            trust_fb_after = human.get_feedback()
            trust_feedback[j, i+1] = trust_fb_after

            # Update trust parameters
            opt_params = estimator.getParams(solver.trust_params, solver.get_last_performance(), trust_fb_after)
            solver.update_params(opt_params)
            parameter_estimates[j, i+1, :] = np.array(opt_params)

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
                row.append("{:.2f}".format(trust_fb_after))
                row.append("{:.2f}".format(trust_est_after))
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
        healths[j, -1] = health
        times[j, -1] = current_time
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
    num_simulations = args.num_simulations
    kappa = args.kappa                          # Assumed rationality coefficient in the bounded rationality model
    wh_hum = args.health_weight_human           # Humans true health weight. time weight = 1 - health weight
    N = args.num_sites                          # Number of sites in a mission (Horizon for planning)
    num_missions = args.num_missions            # Number of "missions" of N sites each
    stepsize = args.posterior_stepsize          # Stepsize in the posterior distrbution over the weights
    num_weights = int(1/stepsize) + 1           # Number of weight samples in the posterior distribution
    data_direc = "./data/Bounded-Rationality/adaptive-learner/kappa" + str(kappa) + "/wh" + str(wh_hum) # Storage directory for the plots
    #################################################################################################################################
    
    data_all = initialize_storage_dict(num_simulations, num_missions, N, num_weights)

    for i in tqdm(range(num_simulations)):
        data_one_simulation = run_one_simulation(args, i * num_missions)
        for k, v in data_one_simulation.items():
            # print(k)
            data_all[k][i] = v

    ############################### STORING THE DATA #############################
    data_all['args'] = args

    if not os.path.exists(data_direc):
        os.makedirs(data_direc)

    data_file = data_direc + '/' + time.strftime("%Y%m%d-%H%M%S") + '.pkl'
    with open(data_file, 'wb') as f:
        pickle.dump(data_all, f)

if __name__ == "__main__":

    parser= argparse.ArgumentParser(description='Adaptive solver that adapts its weights according to learnt human weights')

    # Add the common args to the parser
    parser = add_common_args(parser)

    # Add args specific to this script
    parser.add_argument('--health-weight-human', type=float, help='health weight for the human (default: 0.9)', default=0.9)

    args = parser.parse_args()

    main(args)

