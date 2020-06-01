# coding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle


def plot_iterations(to_plot, labels, plot_dir="iter_plot/"):
    print('Creating Accuracy/Iterations Plot...')

    # Plot Maximum of Residuum
    plt.ylabel('Iterations')

    fig, ax = plt.subplots()
    width = 0.15

    with open(plot_dir + 'acc.data', 'rb') as file:
        accuracies = pickle.load(file)
        x = np.arange(len(accuracies))

    count = min(len(to_plot), len(colors), len(labels))
    pos = -(float(count) / 2.0 - 0.5)

    for name, color, l in zip(to_plot, colors, labels):
        with open(plot_dir + name + '_iter.data', 'rb') as file:
            mean_it = pickle.load(file)

        ax.bar(x + pos * width, mean_it, width, label=l, color=color)
        pos += 1.0

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

def plot_residuum(to_plot, labels, cropped=True, plot_dir="residuum_plot/"):

    # --- Load Data ---
    resmax = {}
    resmean = {}
    pointdata = {}

    for name in to_plot:
        prefix = plot_dir + name

        if cropped is False:
            prefix += "_full"

        with open(prefix + '_resmax.data', 'rb') as file:
            resmax[name] = pickle.load(file)

        with open(prefix + '_resmean.data', 'rb') as file:
            resmean[name] = pickle.load(file)

        with open(prefix + '_points.data', 'rb') as file:
            pointdata[name] = pickle.load(file)


    # --- Plot ---

    fig, ax = plt.subplots()

    # Residuum Max
    ax.set_ylabel('Residuum Max')
    ax.set_yscale('log')
    ax.set_xlabel('Iterations')

    for name, color, l in zip(to_plot, colors, labels):
        ax.plot(resmax[name][0], resmax[name][1], color=color, label=l)
        ax.scatter(x=pointdata[name][0], y=pointdata[name][2], s=0.01, c=color, alpha=0.25)

    ax.legend()
    fig.tight_layout()

    path = plot_dir + "residuum_max_plot"
    plt.savefig(path, dpi=200)
    plt.close()
    print('Saved Residuum Max Plot to %s' % path)

def plot_residuum_zoomed(to_plot, labels, cropped=True, plot_dir="residuum_plot/"):

    # --- Load Data ---
    resmax = {}
    resmean = {}

    for name in to_plot:
        prefix = plot_dir + name

        if cropped is False:
            prefix += "_full"

        with open(prefix + '_resmax.data', 'rb') as file:
            resmax[name] = pickle.load(file)

        with open(prefix + '_resmean.data', 'rb') as file:
            resmean[name] = pickle.load(file)


    # --- Plot ---

    fig, ax = plt.subplots()

    # Residuum Max
    ax.set_ylabel('Residuum Max')
    ax.set_yscale('log')
    ax.set_xlabel('Iterations')

    for name, color, l in zip(to_plot, colors, labels):

        resmax_data = resmax[name][1]
        ax.plot(range(70), resmax_data[:70], color=color, label=l)

    ax.legend()
    fig.tight_layout()

    path = plot_dir + "residual_zoomed"
    plt.savefig(path, dpi=200)
    plt.close()
    print('Saved Residuum Max Plot to %s' % path)

def plot_images(to_plot, labels, plot_dir="pressure_images/", individual_images=True):

    img_data = {}

    # Load Image data
    for name in to_plot:
        img_data[name] = data = np.load(plot_dir + name + "_images.npy")

        # plot individual images if option active
        if individual_images:
            for i, image in zip(range(len(data)), data):
                plt.imshow(image, cmap='terrain', origin='lower')

                path = plot_dir + name + "/"

                if not os.path.exists(path):
                    os.makedirs(path)

                plt.savefig(path + name + "_%s" % i, dpi=200)
                plt.close()


    # Comparison Plot
    example_indices = (11, 2, 9, 15)
    f, axarr = plt.subplots(len(example_indices), len(to_plot))

    #turn off subplot tick labels
    for row in axarr:
        for ax in row:
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # Labels
    for l, i in zip(labels, range(len(to_plot))):
        axarr[-1][i].set_xlabel(l)

    # Images
    for name, j in zip(to_plot, range(len(to_plot))):
        for img, i in zip(example_indices, range(len(example_indices))):
            axarr[i][j].imshow(img_data[name][img], cmap='terrain', origin='lower')

    path = plot_dir + "img_comparison"
    plt.savefig(path, dpi=300)
    plt.close()


def plot_images_residual(to_plot, labels, plot_dir="residual_images/"):

    img_data = {}
    #TODO load input divergence as first column

    # Load Residual Image data
    for name in to_plot:
        img_data[name] = data = np.load(plot_dir + name + "_res_i1_images.npy")


    # Comparison Plot
    example_indices = (6, 7, 17)
    f, axarr = plt.subplots(len(example_indices), len(to_plot))
    plt.subplots_adjust(hspace=0.1, wspace=0.0)

    #turn off subplot tick labels
    for row in axarr:
        for ax in row:
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # Labels
    for l, i in zip(labels, range(len(to_plot))):
        axarr[-1][i].set_xlabel(l)

    for j in range(len(example_indices)):
        axarr[j][0].set_ylabel("Sample %s" % j)

    # Images
    for sample, i in zip(example_indices, range(len(example_indices))):
        row_imgs = np.array([img_data[k][sample] for k in img_data])
        max = np.max(row_imgs)
        min = np.min(row_imgs)

        colorbar = None

        for name, j in zip(to_plot, range(len(to_plot))):
            img = axarr[i][j].imshow(img_data[name][sample], vmax=max, vmin=min, cmap='terrain', origin='lower')

            if colorbar is None:
                colorbar = f.colorbar(img, ax=axarr[i][-1])

    path = plot_dir + "cg_residual_img.pdf"
    plt.savefig(path, dpi=300, format='pdf')
    plt.close()

# define what to plot
colors = ("blue", "green", "orange", "red", "purple", "black")
#list = ("solverbased_i1", "solverbased_i2", "solverbased_i5", "solverbased", "solverbased_i15")
#label_list = ("SOL-1", "SOL-2", "SOL-5", "SOL-10", "SOL-15")

list = ("sol5", "sup", "tom")
label_list = ("SOL-5", "NON", "SOL-DIV")

plot_images_residual(list, label_list)