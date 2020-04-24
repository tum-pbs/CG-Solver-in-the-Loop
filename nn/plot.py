# coding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import pickle



def plot_iterations(plot_dir="iter_plot"):
    print('Creating Accuracy/Iterations Plot...')

    # Plot Maximum of Residuum
    plt.ylabel('Iterations')

    fig, ax = plt.subplots()
    width = 0.25

    with open(plot_dir + 'acc.data', 'rb') as file:
        accuracies = pickle.load(file)
        x = np.arange(len(accuracies))

    with open(plot_dir + 'solverbased_iter.data', 'rb') as file:
        mean_it_solverbased = pickle.load(file)

    with open(plot_dir + 'supervised_iter.data', 'rb') as file:
        mean_it_supervised = pickle.load(file)

    with open(plot_dir + 'zeroguess_iter.data', 'rb') as file:
        mean_it_zero= pickle.load(file)

    rects_solverbased = ax.bar(x - width, mean_it_solverbased, width, label='Solver-based', color='blue')
    rects_supervised = ax.bar(x, mean_it_supervised, width, label='Supervised', color='green')
    rects_zero = ax.bar(x + width, mean_it_zero, width, label='Zero Guess', color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(accuracies)
    ax.set_ylabel('Iterations')
    ax.set_xlabel('Target Accuracy')
    ax.set_title('Iterations for target Accuracy')

    ax.legend()
    fig.tight_layout()

    path = plot_dir + 'iterations_plot'
    plt.savefig(path)
    plt.close()
    print('Saved Accuracy/Iterations Plot to %s' % path)

def plot_residuum(plot_dir="residuum_plot"):

    # --- Load Data ---

    with open(plot_dir + '/solverbased_resmax.data', 'rb') as file:
        solverbased_resmax = pickle.load(file)

    with open(plot_dir + '/solverbased_resmean.data', 'rb') as file:
        solverbased_resmean = pickle.load(file)

    with open(plot_dir + '/solverbased_points.data', 'rb') as file:
        solverbased_pointdata = pickle.load(file)



    with open(plot_dir + '/supervised_resmax.data', 'rb') as file:
        supervised_resmax = pickle.load(file)

    with open(plot_dir + '/supervised_resmean.data', 'rb') as file:
        supervised_resmean = pickle.load(file)

    with open(plot_dir + '/supervised_points.data', 'rb') as file:
        supervised_pointdata = pickle.load(file)



    with open(plot_dir + '/tompson_resmax.data', 'rb') as file:
        tompson_resmax = pickle.load(file)

    with open(plot_dir + '/tompson_resmean.data', 'rb') as file:
        tompson_resmean = pickle.load(file)

    with open(plot_dir + '/tompson_points.data', 'rb') as file:
        tompson_pointdata = pickle.load(file)



    with open(plot_dir + '/zeroguess_resmax.data', 'rb') as file:
        zeroguess_resmax = pickle.load(file)

    with open(plot_dir + '/zeroguess_resmean.data', 'rb') as file:
        zeroguess_resmean = pickle.load(file)

    with open(plot_dir + '/zeroguess_points.data', 'rb') as file:
        zeroguess_pointdata = pickle.load(file)


    # --- Plot ---

    fig, ax = plt.subplots()

    # Residuum Max
    ax.set_ylabel('Residuum Max')
    ax.set_yscale('log')
    ax.set_xlabel('Iterations')

    ax.plot(supervised_resmax[0], supervised_resmax[1], color="green", label="Supervised")
    ax.plot(tompson_resmax[0], tompson_resmax[1], color="orange", label="Tompson")
    ax.plot(solverbased_resmax[0], solverbased_resmax[1], color="blue", label="Solver-based")
    ax.plot(zeroguess_resmax[0], zeroguess_resmax[1], color="red", label="Zero Guess")

    #plt.scatter(x=it, y=all_maxima, s=0.01, c=(0, 0, 1.0), alpha=0.25)
    #plt.scatter(x=it, y=all_maxima_noguess, s=0.01, c=(1.0, 0, 0), alpha=0.25)

    ax.legend()
    fig.tight_layout()

    path = plot_dir + "/residuum_max_plot"
    plt.savefig(path)
    plt.close()
    print('Saved Residuum Max Plot to %s' % path)


plot_residuum()