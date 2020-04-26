# coding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import pickle



def plot_iterations(plot_dir="iter_plot/"):
    print('Creating Accuracy/Iterations Plot...')

    # Plot Maximum of Residuum
    plt.ylabel('Iterations')

    fig, ax = plt.subplots()
    width = 0.15

    with open(plot_dir + 'acc.data', 'rb') as file:
        accuracies = pickle.load(file)
        x = np.arange(len(accuracies))

    with open(plot_dir + 'solverbased_iter.data', 'rb') as file:
        mean_it_solverbased = pickle.load(file)

    with open(plot_dir + 'supervised_iter.data', 'rb') as file:
        mean_it_supervised = pickle.load(file)

    with open(plot_dir + 'tompson_iter.data', 'rb') as file:
        mean_it_tompson = pickle.load(file)

    with open(plot_dir + 'zeroguess_iter.data', 'rb') as file:
        mean_it_zero= pickle.load(file)

    rects_solverbased = ax.bar(x - 1.5*width, mean_it_solverbased, width, label='Solver-based', color='blue')
    rects_supervised = ax.bar(x - 0.5*width, mean_it_supervised, width, label='Supervised', color='green')
    rects_tompson = ax.bar(x + 0.5*width, mean_it_tompson, width, label='Tompson', color='orange')
    rects_zero = ax.bar(x + 1.5*width, mean_it_zero, width, label='Zero Guess', color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(accuracies)
    ax.set_ylabel('Iterations')
    ax.set_xlabel('Target Accuracy')
    ax.set_title('Iterations for target Accuracy')
    ax.tick_params(axis='both', labelsize=8)

    ax.legend()
    fig.tight_layout()

    path = plot_dir + 'iterations_plot'
    plt.savefig(path, dpi=200)
    plt.close()
    print('Saved Accuracy/Iterations Plot to %s' % path)

def plot_residuum(plot_dir="residuum_plot/"):

    # --- Load Data ---

    with open(plot_dir + 'solverbased_resmax.data', 'rb') as file:
        solverbased_resmax = pickle.load(file)

    with open(plot_dir + 'solverbased_resmean.data', 'rb') as file:
        solverbased_resmean = pickle.load(file)

    with open(plot_dir + 'solverbased_points.data', 'rb') as file:
        solverbased_pointdata = pickle.load(file)



    with open(plot_dir + 'supervised_resmax.data', 'rb') as file:
        supervised_resmax = pickle.load(file)

    with open(plot_dir + 'supervised_resmean.data', 'rb') as file:
        supervised_resmean = pickle.load(file)

    with open(plot_dir + 'supervised_points.data', 'rb') as file:
        supervised_pointdata = pickle.load(file)



    with open(plot_dir + 'tompson_resmax.data', 'rb') as file:
        tompson_resmax = pickle.load(file)

    with open(plot_dir + 'tompson_resmean.data', 'rb') as file:
        tompson_resmean = pickle.load(file)

    with open(plot_dir + 'tompson_points.data', 'rb') as file:
        tompson_pointdata = pickle.load(file)



    with open(plot_dir + 'zeroguess_resmax.data', 'rb') as file:
        zeroguess_resmax = pickle.load(file)

    with open(plot_dir + 'zeroguess_resmean.data', 'rb') as file:
        zeroguess_resmean = pickle.load(file)

    with open(plot_dir + 'zeroguess_points.data', 'rb') as file:
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

    ax.scatter(x=supervised_pointdata[0], y=supervised_pointdata[2], s=0.01, c="green", alpha=0.25)
    ax.scatter(x=tompson_pointdata[0], y=tompson_pointdata[2], s=0.01, c="orange", alpha=0.25)
    ax.scatter(x=solverbased_pointdata[0], y=solverbased_pointdata[2], s=0.01, c="blue", alpha=0.25)
    ax.scatter(x=zeroguess_pointdata[0], y=zeroguess_pointdata[2], s=0.01, c="red", alpha=0.25)

    ax.legend()
    fig.tight_layout()

    path = plot_dir + "residuum_max_plot"
    plt.savefig(path, dpi=200)
    plt.close()
    print('Saved Residuum Max Plot to %s' % path)

def plot_images(plot_dir="pressure_images/", individual_images=True):
    true_images = np.load(plot_dir + "true_images.npy")
    untrained_images = np.load(plot_dir + "untrained_images.npy")
    supervised_images = np.load(plot_dir + "supervised_images.npy")
    solverbased_images = np.load(plot_dir + "solverbased_images.npy")
    tompson_images = np.load(plot_dir + "tompson_images.npy")

    if individual_images:
        # Plot true images
        for i, image in zip(range(len(true_images)), true_images):
            plt.imshow(image, cmap='bwr', origin='lower')
            plt.savefig(plot_dir + "true/true" + "_%s" % i, dpi=200)
            plt.close()

        # Plot untrained images
        for i, image in zip(range(len(untrained_images)), untrained_images):
            plt.imshow(image, cmap='bwr', origin='lower')
            plt.savefig(plot_dir + "untrained/untrained" + "_%s" % i, dpi=200)
            plt.close()

        # Plot supervised images
        for i, image in zip(range(len(supervised_images)), supervised_images):
            plt.imshow(image, cmap='bwr', origin='lower')
            plt.savefig(plot_dir + "supervised/supervised" + "_%s" % i, dpi=200)
            plt.close()

        # Plot solver-based images
        for i, image in zip(range(len(solverbased_images)), solverbased_images):
            plt.imshow(image, cmap='bwr', origin='lower')
            plt.savefig(plot_dir + "solverbased/solverbased" + "_%s" % i, dpi=200)
            plt.close()

        # Plot tompson images
        for i, image in zip(range(len(tompson_images)), tompson_images):
            plt.imshow(image, cmap='bwr', origin='lower')
            plt.savefig(plot_dir + "tompson/tompson" + "_%s" % i, dpi=200)
            plt.close()


    # Comparison Plot
    example_indices = (11, 2, 9, 15)
    f, axarr = plt.subplots(len(example_indices), 4)

    #turn off subplot tick labels
    for row in axarr:
        for ax in row:
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # Labels
    axarr[-1][0].set_xlabel('True')
    axarr[-1][1].set_xlabel('Tompson')
    axarr[-1][2].set_xlabel('Solver-Based')
    axarr[-1][3].set_xlabel('Supervised')

    # Images
    for img, i in zip(example_indices, range(len(example_indices))):
        axarr[i][0].imshow(true_images[img], cmap='bwr', origin='lower')
        axarr[i][1].imshow(tompson_images[img], cmap='bwr', origin='lower')
        axarr[i][2].imshow(solverbased_images[img], cmap='bwr', origin='lower')
        axarr[i][3].imshow(supervised_images[img], cmap='bwr', origin='lower')



    path = plot_dir + "img_comparison"
    plt.savefig(path, dpi=300)
    plt.close()

plot_images(individual_images=False)