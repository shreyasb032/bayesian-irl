from classes.DataReader import PickleReader
import numpy as np
import argparse
import matplotlib.pyplot as plt

def plot(fig, ax: plt.Axes, mean, std=None, color=None, title=None, x_label=None, y_label=None, plot_label=None, xlim=None, ylim=None):

    ax.plot(mean, label=plot_label, c=color, linewidth=2)

    if std is not None:
        ax.fill_between(np.arange(len(mean)), mean+std, mean-std, color=color, alpha=0.7)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    return fig, ax


def analyze(data: dict):

    # Most shapes are (num_simulations, num_missions, num_sites or num_sites+1)

    num_simulations, num_missions, num_sites = data["actions"].shape

    ########################### TRUST ##################################################
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


    ########################################### POSTERIOR ###############################
    fig, ax = plt.subplots()

    posterior_dists = data['posterior']

    mean = np.mean(posterior_dists, axis=0)
    std = np.std(posterior_dists, axis=0)


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

def main(filepath: str):

    if filepath is None:
        raise "filepath cannot be none"

    reader = PickleReader(filepath)
    reader.read_data()

    analyze(reader.data)



if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Exploring the stored simulated data')

    parser.add_argument('--path', required=True, help='Filepath for the data file. No default')

    args = parser.parse_args()

    main(args.path)

