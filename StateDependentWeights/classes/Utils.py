import argparse
import numpy as np

def add_common_args(parser: argparse.ArgumentParser):

    parser.add_argument('--trust-weight', type=float, help='trust weight for the robot (default: 10.0)', default=10.0)
    parser.add_argument('--kappa', type=float, help='rationality coeffiecient (default: 0.05)', default=0.05)
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
    parser.add_argument('--num-simulations', type=int, help='Number of simulations to run (default: 1000)', default=1000)
    parser.add_argument('--posterior-stepsize', type=float, help='Stepsize in the posterior distribution (default(0.05)', default=0.05)
    parser.add_argument('--num-gradient-steps', type=int, help='Number of iterations of gradient descent for trust parameter estimation (default: 200)', default=200)
    parser.add_argument('--gradient-stepsize', type=float, help='Stepsize for gradient descent (default: 0.0001)', default=0.001)
    parser.add_argument('--tolerance', type=float, help='Error tolerance for the gradient descent step (default: 0.01)', default=0.01)
    parser.add_argument('--reset-estimator', type=bool, help="Flag to reset the trust parameter estimator (default: False)", default=False)

    return parser

def initialize_storage_dict(num_simulations, num_missions, N, num_weights):

    data = {}

    data['trust feedback'] = np.zeros((num_simulations, num_missions, N+1), dtype=float)
    data['trust estimate'] = np.zeros((num_simulations, num_missions, N+1), dtype=float)
    data['health'] = np.zeros((num_simulations, num_missions, N+1), dtype=int)
    data['time'] = np.zeros((num_simulations, num_missions, N+1), dtype=int)
    data['recommendation'] = np.zeros((num_simulations, num_missions, N), dtype=int)
    data['actions'] = np.zeros((num_simulations, num_missions, N), dtype=int)
    data['weights'] = np.zeros((num_simulations, num_missions, N, num_weights), dtype=float)
    data['posterior'] = np.zeros((num_simulations, num_missions, N+1, num_weights), dtype=float)
    data['prior threat level'] = np.zeros((num_simulations, num_missions, N), dtype=float)
    data['after scan level'] = np.zeros((num_simulations, num_missions, N), dtype=float)
    data['threat'] = np.zeros((num_simulations, num_missions, N), dtype=int)
    data['trust parameter estimates'] = np.zeros((num_simulations, num_missions, N+1, 4), dtype=float)
    data['mean health weight'] = np.zeros((num_simulations, num_missions, N+1), dtype=float)
    data['map health weight'] = np.zeros((num_simulations, num_missions, N+1), dtype=float)
    data['map health weight probability'] = np.zeros((num_simulations, num_missions, N+1), dtype=float)
    data['performance estimates'] = np.zeros((num_simulations, num_missions, N), dtype=int)
    data['performance actual'] = np.zeros((num_simulations, num_missions, N), dtype=int)        

    return data

def col_print(table_data):
    
    string = ""
    for _ in range(len(table_data[0])):
        string += "{: ^12}"

    for row in table_data:
        print(string.format(*row))
