from classes.DataReader import PickleReader
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot(fig, ax: plt.Axes, mean, std=None, color=None, title=None, x_label=None, y_label=None, plot_label=None, xlim=None, ylim=None):

    ax.plot(mean, label=plot_label, c=color, linewidth=2)

    if std is not None:
        ax.fill_between(np.arange(len(mean)), mean+std, mean-std, color=color, alpha=0.7)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    return fig, ax

def plot_posterior(fig, ax: plt.Axes, weights, mean, std,  fill=False):
    colors = list(mcolors.TABLEAU_COLORS)

    num_missions = mean.shape[0]

    for i in range(num_missions):
        ax.plot(weights, mean[i, :], linewidth=2, label='Mission #{:d}'.format(i+1), c=colors[i])
        if fill:
            ax.fill_between(weights, mean[i, :]+std[i, :], mean[i, :]-std[i, :], alpha=0.5, color=colors[i])
    
    ax.legend()
    ax.set_title("Posterior Distribution after each mission", fontsize=16)
    ax.set_xlabel(r'Health Weight $w_h$', fontsize=14)
    ax.set_ylabel(r'$P(w_h)$', fontsize=14)
    
    return fig, ax

def analyze(data: dict):

    # Most shapes are (num_simulations, num_missions, num_sites or num_sites+1)

    num_simulations, num_missions, num_sites = data["actions"].shape

    ############################## TRUST At end of mission ##################################################
    fig, ax = plt.subplots()

    trust_feedback = data['trust feedback']
    mean = np.mean(trust_feedback, axis=0)
    std = np.std(trust_feedback, axis=0)

    fig, ax = plot(fig, ax, mean[:, -1], std[:, -1], 'tab:blue', plot_label='feedback', title="Trust at end of mission", x_label='Mission index', y_label='Trust')
    
    trust_estimate = data['trust estimate']
    mean = np.mean(trust_estimate, axis=0)
    std = np.std(trust_estimate, axis=0)

    fig, ax = plot(fig, ax, mean[:, -1], std[:, -1], 'tab:orange', plot_label='estimate', title="Trust at end of mission", x_label='Mission index', y_label='Trust')

    ax.legend()
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True)

    ############################ Trust after each site #######################################################

    fig, ax = plt.subplots()

    trust_feedback = data['trust feedback']
    mean = np.mean(trust_feedback, axis=0)
    std = np.std(trust_feedback, axis=0)

    fig, ax = plot(fig, ax, mean.flatten(), std.flatten(), 'tab:blue', plot_label='feedback', title="Trust after each site", x_label='Site index', y_label='Trust')
    
    trust_estimate = data['trust estimate']
    mean = np.mean(trust_estimate, axis=0)
    std = np.std(trust_estimate, axis=0)

    fig, ax = plot(fig, ax, mean.flatten(), std.flatten(), 'tab:orange', plot_label='estimate', title="Trust after each site", x_label='Site index', y_label='Trust')

    ax.legend()
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True)

    ########################################### POSTERIOR ################################################
    fig, ax = plt.subplots()

    posterior_dists = data['posterior']
    weights = data['weights'][0, 0, 0, :]

    mean = np.mean(posterior_dists, axis=0)[:, -1, :]
    std = np.std(posterior_dists, axis=0)[:, -1, :]

    # mean and std have shape (num_missions, N+1)
    fig, ax = plot_posterior(fig, ax, weights, mean, std)
    ax.grid(True)

    ######################################### Mean weight after each site #################################################
    fig, ax = plt.subplots()

    wh_means = data['mean health weight']   # Shape (num_simulations, num_missions, N+1)
    mean = np.mean(wh_means, axis=0)        # Shape (num_missions, N+1)
    std = np.std(wh_means, axis=0)          # Shape (num_missions, N+1)

    fig, ax = plot(fig, ax, mean.flatten(), std.flatten(), 'tab:blue', plot_label='mean weight', title="Estimated health weight after each site", x_label='Site index', y_label='Mean health weight')

    ax.set_ylim([-0.05, 1.05])
    ax.grid(True)

    ########################################### Mean weight after each mission ############################################

    fig, ax = plt.subplots()

    wh_means = data['mean health weight']   # Shape (num_simulations, num_missions, N+1)
    mean = np.mean(wh_means, axis=0)        # Shape (num_missions, N+1)
    std = np.std(wh_means, axis=0)          # Shape (num_missions, N+1)

    fig, ax = plot(fig, ax, mean[:, -1], std[:, -1], 'tab:blue', plot_label='mean weight', title="Estimated health weight after each mission", x_label='Mission index', y_label='Mean health weight')
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True)

    plt.show()


    # data['trust feedback']
    # data['trust estimate']
    # data['health']
    # data['time']
    # data['recommendation']
    # data['actions']
    # data['weights']
    # data['posterior']
    # data['prior threat level']
    # data['after scan level']
    # data['threat'] 
    # data['trust parameter estimates']
    # data['mean health weight']
    # data['map health weight']
    # data['map health weight probability']
    # data['performance estimates']
    # data['performance actual']

def main(args: argparse.Namespace):

    filepath = args.path
    PRINT_TRUST_PARAMS = args.print_trust_params

    if filepath is None:
        raise "filepath cannot be none"

    reader = PickleReader(filepath)
    reader.read_data()

    analyze(reader.data)

    if PRINT_TRUST_PARAMS:
        i = 0
        parameter_estimates = reader.data["trust parameter estimates"]
        num_simulations, num_missions, _, _ = parameter_estimates.shape

        while i < num_simulations:
            for j in range(num_missions):
                print(parameter_estimates[i, j, -1, :])
            
            i += 1
            input("Press Enter...")

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Exploring the stored simulated data')

    parser.add_argument('--path', required=True, help='Filepath for the data file. No default')
    parser.add_argument('--print-trust-params', help='Flag to control whether to print the trust parameters (default: False)', default=False)

    args = parser.parse_args()

    main(args)

