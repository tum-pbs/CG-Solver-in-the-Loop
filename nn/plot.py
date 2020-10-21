# coding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

import os
import numpy as np
import pickle


def plot_iterations(to_plot, labels, plot_dir="iter_plot/"):
    print('Creating Accuracy/Iterations Plot...')

    # Plot Maximum of Residuum
    plt.ylabel('Iterations')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
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
    ax.set_xticklabels(("$10^{-1}$", "", "$10^{-2}$", "", "$10^{-3}$", "", "$10^{-4}$", "", "$10^{-5}$", "", "$10^{-6}$"))
    ax.set_ylabel('Iterations')
    ax.set_xlabel('Target Accuracy')
    #ax.set_title('Iterations for Target Accuracy')
    ax.tick_params(axis='both', labelsize=18)
    plt.grid(True, axis='y')
    plt.grid(False, axis='x')

    ax.legend(fontsize=18)
    fig.tight_layout()

    path = plot_dir + 'iterations_plot.pdf'
    plt.savefig(path, dpi=200, format='pdf')
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
    fig.set_size_inches(8, 5)

    # Residuum Max
    ax.set_ylabel('Residuum Max')
    ax.set_yscale('log')
    ax.set_xlabel('Iterations')

    for name, color, l in zip(to_plot, colors, labels):
        ax.plot(resmax[name][0], resmax[name][1], color=color, label=l)
        #ax.scatter(x=pointdata[name][0], y=pointdata[name][2], s=0.01, c=color, alpha=0.25)

    ax.legend(fontsize=18)
    fig.tight_layout()

    #ax.set_title('Residual Error over Iterations')
    ax.tick_params(axis='both', labelsize=18)
    plt.grid(True, axis='y')
    plt.grid(False, axis='x')

    path = plot_dir + "residuum_max_plot.pdf"
    plt.savefig(path, dpi=100, format='pdf')
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
    fig.set_size_inches(8, 5)

    # Residuum Max
    ax.set_ylabel('Residuum Max')
    ax.set_yscale('log')
    ax.set_xlabel('Iterations')

    for name, color, l in zip(to_plot, colors, labels):

        resmax_data = resmax[name][1]
        ax.plot(range(70), resmax_data[:70], color=color, label=l)

    ax.legend(fontsize=18)
    fig.tight_layout()

    #ax.set_title('Residual Error over Iterations')
    ax.tick_params(axis='both', labelsize=18)
    plt.grid(True, axis='y')
    plt.grid(False, axis='x')

    path = plot_dir + "residual_zoomed.pdf"
    plt.savefig(path, dpi=200, format='pdf')
    plt.close()
    print('Saved Residuum Max Plot to %s' % path)


def plot_images_pressure(to_plot, labels, plot_dir="pressure_images/"):

    img_data = {}
    true_pressure = np.load(plot_dir + "true_images.npy")

    # Load Residual Image data
    for name in to_plot:
        img_data[name] = np.load(plot_dir + name + "_images.npy")


    # Comparison Plot
    #example_indices = (11, 2, 9, 15)
    example_indices = (11, 2, 9)

    f, axarr = plt.subplots(len(example_indices), len(to_plot))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    f.set_size_inches(len(to_plot) * 1.0 + 0.3, len(example_indices) * 1.0)

    #turn off subplot tick labels
    for row in axarr:
        for ax in row:
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # Labels
    for l, i in zip(labels, range(len(to_plot))):
        axarr[-1][i].set_xlabel(l)

    for j in range(len(example_indices)):
        axarr[j][0].set_ylabel("Sample %s" % j, fontsize=12)

    # Images
    for sample, i in zip(example_indices, range(len(example_indices))):
        row_imgs = np.array([img_data[k][sample] for k in img_data])
        mean = np.reshape(np.mean(row_imgs, axis=(1, 2)), [-1, 1, 1])  # subtract mean per image

        row_imgs = row_imgs - mean
        imgs = true_pressure[sample] - row_imgs
        #imgs = row_imgs

        max = np.max(imgs)
        min = np.min(imgs)

        colorbar = None

        for j in range(len(to_plot)):
            img = axarr[i][j].imshow(imgs[j], vmax=max, vmin=min, cmap='terrain', origin='lower')

            if colorbar is None:
                colorbar = f.colorbar(img, ax=axarr[i], shrink=0.9)
                #colorbar.ax.tick_params(labelsize='x-small')

    path = plot_dir + "cg_pressure_img.pdf"
    plt.savefig(path, dpi=300, format='pdf')
    plt.close()

def plot_images_residual(to_plot, labels, plot_dir="residual_images/"):

    img_data = {}

    # Load Residual Image data
    for name in to_plot:
        img_data[name] = data = np.load(plot_dir + name + "_res_i1_images.npy")


    # Comparison Plot
    example_indices = (6, 7)
    #example_indices = (17, 12)

    rows = len(example_indices)
    columns = len(img_data.keys())

    f, axarr = plt.subplots(rows, columns)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    f.set_size_inches(columns * 1.0 + 0.3, rows * 1.0)


    #turn off subplot tick labels
    for row in axarr:
        for ax in row:
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # Labels
    for l, i in zip(labels, range(columns)):
        axarr[-1][i].set_xlabel(l)

    for j in range(rows):
        axarr[j][0].set_ylabel("Sample %s" % j)

    # Images
    for sample, i in zip(example_indices, range(rows)):
        row_imgs = np.array([img_data[k][sample] for k in img_data])
        max = np.max(row_imgs)
        min = np.min(row_imgs)

        colorbar = None

        for name, j in zip(to_plot, range(columns)):
            img = axarr[i][j].imshow(img_data[name][sample], vmax=max, vmin=min, cmap='terrain', origin='lower')

            if colorbar is None:
                colorbar = f.colorbar(img, ax=axarr[i])

    #plt.tight_layout(pad=0.5)
    path = plot_dir + "cg_residual_img.pdf"
    plt.savefig(path, dpi=300, format='pdf')
    plt.close()

def plot_iterations_for_target(to_plot, labels, plot_dir="iter_plot_single_target/"):

    # --- Load Data ---
    it = {}

    for name in to_plot:
        prefix = plot_dir + name

        with open(prefix + '_iterations.data', 'rb') as file:
            it[name] = pickle.load(file)

    plt.rcParams.update(
        {
            'grid.alpha': 0.3,
            'grid.color': 'gray',
            'grid.linestyle': 'solid',
            'grid.linewidth': 0.1,
            'font.family': 'serif',
            'font.serif': ['Times', 'Times New Roman'],
            'font.sans-serif': ['Helvetica', 'Arial'],
            #'font.size': 20,
            'font.size': 10,
            'legend.fontsize': 10,
            'text.usetex': True
        }
    )

    fig, ax = plt.subplots()
    fig.set_size_inches(len(to_plot)*1.0, 4.8)

    # Actual plotting
    width = 0.7

    for x, name, c in zip(range(len(to_plot)), to_plot, colors):

        # Compute mean and std
        mean = np.mean(it[name])
        std = np.std(it[name])

        print("Mean: %s" % mean)
        print("Standard Deviation: %s" % std)

        ax.bar(x, mean, width, color=c, yerr=std, error_kw=dict(lw=0.5, capsize=3, capthick=0.5))

    ax.set_ylabel("Iterations for Accuracy $10^{-3}$", fontsize=14)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.set_xticks(range(len(to_plot)))
    ax.set_xticklabels(labels, fontsize=14)
    plt.grid(True, axis='y')
    plt.grid(False, axis='x')
    plt.tight_layout(pad=0.5)

    path = plot_dir + 'cg_target_iterations.pdf'
    plt.savefig(path, dpi=300, format='pdf')
    plt.close()
    print('Saved "iterations for target" figure to %s' % path)

def plot_sim_residual(to_plot, labels, plot_dir="sim_residual/"):

    # --- Load Data ---
    residual = {}

    for name in to_plot:
        with open(plot_dir + 'residual_' + name + '.data', 'rb') as file:
            residual[name] = pickle.load(file)


    # --- Plot ---

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)

    # Residuum Max
    ax.set_ylabel('Residual Divergence (Max)')
    ax.set_yscale('log')
    ax.set_xlabel('Simulation Step')

    for name, color, l in zip(to_plot, colors, labels):
        ax.plot(residual[name][0], residual[name][1], color=color, label=l)

    ax.legend(fontsize=18)
    fig.tight_layout()

    #ax.set_title('Residual Error over Iterations')
    ax.tick_params(axis='both', labelsize=18)
    plt.grid(True, axis='y')
    plt.grid(False, axis='x')

    path = plot_dir + "sim_residual_plot.pdf"
    plt.savefig(path, dpi=100, format='pdf')
    plt.close()
    print('Saved Simulation Residual to %s' % path)

# define what to plot

plt.rcParams.update(
    {
        'grid.alpha': 0.3,
        'grid.color': 'gray',
        'grid.linestyle': 'solid',
        'grid.linewidth': 0.1,
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman'],
        'font.sans-serif': ['Helvetica', 'Arial'],
        'font.size': 14,
        'legend.fontsize': 10,
        'text.usetex': True
    }
)

# #TODO Figure 4
# colors = (cm.tab20c( 7,20), cm.tab20c(13,20), cm.Greens(0.5))
# list = ("zeroguess", "tompson_scalar", "solverbased_i5")
# label_list = ("SRC", "SOL$_{DIV}$", "SOL$_5$")
#
# plot_iterations_for_target(list, label_list)


# #TODO Full Iterations Plot
# list = ("zeroguess", "tompson_scalar", "supervised", "solverbased_i5")
# label_list = ("Zero", "PHY", "SUP", "SOL$_5$")
# colors = (cm.tab20c( 7,20), cm.tab20c(13,20), cm.tab20c(17,20), cm.Greens(0.5))
#
# plot_iterations(list, label_list)
#
#
#TODO Residual Plot
list = ("zeroguess", "solverbased_i5", "tompson_scalar", "supervised")
label_list = ("Zero", "SOL$_5$",  "PHY", "SUP")
colors = (cm.tab20c( 7,20), cm.Greens(0.5), cm.tab20c(13,20), cm.tab20c(17,20))

plot_residuum(list, label_list, cropped=False)
#
#
# #TODO Ablation Plots
# list = ("solverbased_i1", "solverbased_i2", "solverbased_i5", "solverbased", "solverbased_i15")
# label_list = ("SOL$_1$", "SOL$_2$",  "SOL$_5$", "SOL$_{10}$", "SOL$_{15}$")
# colors = (cm.Greens(0.3), cm.Greens(0.5), cm.Greens(0.7), cm.Greens(0.9), cm.Greens(1.1))
#
# plot_residuum_zoomed(list, label_list, cropped=True)
# plot_iterations(list, label_list)
#
#
# #TODO Residual Image Comparison
# list = ["sol5", "tom", "sup"]
# label_list = ["SOL$_5$",  "SOL$_{DIV}$", "NON"]
#
# plot_images_residual(list, label_list)


# #TODO Pressure Image Comparison
# list = ["solverbased_i5", "tompson_scalar", "supervised"]
# label_list = ["SOL$_5$",  "SOL$_{DIV}$", "NON"]
#
# plot_images_pressure(list, label_list)

# #TODO Simulation Residual Divergence Plot
# list = ("cg", "solverbased_i5")
# label_list = ("CG", "SOL")
# colors = (cm.tab20c( 7,20), cm.Greens(0.5), cm.tab20c(13,20), cm.tab20c(17,20), cm.tab20c(19,20))
#
# plot_sim_residual(list, label_list)