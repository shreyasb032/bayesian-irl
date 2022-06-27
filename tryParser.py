import argparse


parser= argparse.ArgumentParser(description='Adaptive solver that adapts its weights according to learnt human weights')
parser.add_argument('--trust-weight', type=float, help='trust weight for the robot (default: 10.0)', default=10.0)
parser.add_argument('--kappa', type=float, help='rationality coeffiecient (default: 0.05)', default=0.05)
parser.add_argument('--human-health-weight', type=float, help='health weight for the human (default: 0.9)', default=0.9)
parser.add_argument('--trust-params', nargs=4, help='Trust parameters for the human, default=[90., 30., 20. 30.]', default=[90., 30., 20., 30.])
parser.add_argument('--num-sites', type=int, help='Number of sites in a mission (default: 10)', default=10)
parser.add_argument('--num-missions', type=int, help='Number of missions (default: 10)', default=10)
parser.add_argument('--region-size', type=int, help='Size of a region with a certain probability of threat (default: 1)', default=1)
parser.add_argument('--reset-solver', type=bool, help="Flag to reset the solver after each mission (default: False)", default=False)
parser.add_argument('--reset-human', type=bool, help="Flag to reset the human after each mission (default: False)", default=False)
parser.add_argument('--reset-health', type=bool, help="Flag to reset the health after each mission (default: False)", default=False)
parser.add_argument('--reset-time', type=bool, help="Flag to reset the time after each mission (default: False)", default=False)
parser.add_argument('--store-figs', type=bool, help="Flag to store the plots (default: False)", default=False)

args = parser.parse_args()

print(args)
