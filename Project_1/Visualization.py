# %%
import matplotlib.pyplot as plt
import torch

# %%
def plot_pairs(nb, data, classes, target):
    fig = plt.figure(constrained_layout=True, figsize=(8, 4*nb))
    fig.suptitle('Visualization of pair selection.' + '\n', fontsize=16)
    target_str = [r"larger ($>$)", r"smaller or equal ($\leq$)"]
    subfigs = fig.subfigures(nrows=nb, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f'Pair {row} (target: {target_str[int(target[row][0])]})', fontsize=15)

        axs = subfig.subplots(nrows=1, ncols=2)
        for col, ax in enumerate(axs):
            ax.imshow(data[row, col, :, :], cmap="gray")
            ax.set_title(f'Class: {classes[row, col]}', fontsize = 14)
            ax.axis("off")

def plot_single(data):
    fig = plt.figure(constrained_layout=True, figsize=(8, 4*data.shape[0]))
    fig.suptitle(f'Visualization of tensor' + '\n', fontsize=16)
    axs = fig.subplots(nrows=data.shape[0], ncols=1)
    for row, ax in enumerate(axs):
        ax.imshow(data[row], cmap="gray")
        ax.set_title('\n', fontsize = 14)
        ax.axis("off")

def plot_analysis(mean, std, epochs, nb_trials, name):

    fig, ax = plt.subplots(1,1, figsize=(8, 6), dpi = 80)
    plt.rcParams['font.size'] = '16'

    x = range(1, epochs+1)

    mean0, = ax.plot(x, mean, color="darkblue")
    std1, = ax.plot(x, mean+std, color="blue", alpha=0.97)
    std2, = ax.plot(x, mean-std, color="blue", alpha=0.97)
    fill = ax.fill_between(x, mean+std, mean-std, color="blue", alpha=0.5)

    ax.set_title("Mean and standard deviation of test_accuracy after" + '\n' + "{} runs with {} epochs".format(nb_trials, epochs))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.grid()
    ax.grid(which="minor", linestyle=':', lw=.5)
    ax.tick_params(axis="both", direction="in", top = True, right=True, which="both")
    ax.legend([mean0, (std1, fill ,std2)], ["mean", "standard deviation"], prop={"size":14})
    fig.tight_layout()
    fig.savefig("./Plots/" + "Analysis_" + name.replace(' ', '_') + ".png")

def plot_comparison(results1, results2, epochs, nb_trials):
    
    name1, mean1, std1 = results1
    name2, mean2, std2 = results2
    
    fig, ax = plt.subplots(1,1, figsize=(8, 6), dpi = 80)
    plt.rcParams['font.size'] = '16'
    x = range(1, epochs+1)

    mean10, = ax.plot(x, mean1, color="darkblue")
    std11, = ax.plot(x, mean1+std1, color="blue", alpha=0.97)
    std12, = ax.plot(x, mean1-std1, color="blue", alpha=0.97)
    fill1 = ax.fill_between(x, mean1+std1, mean1-std1, color="blue", alpha=0.5)

    mean20, = ax.plot(x, mean2, color="darkorange")
    std21, = ax.plot(x, mean2+std2, color="orange", alpha=0.97)
    std22, = ax.plot(x, mean2-std2, color="orange", alpha=0.97)
    fill2 = ax.fill_between(x, mean2+std2, mean2-std2, color="orange", alpha=0.5)

    ax.set_title("Mean and standard deviation of test_accuracy after" + '\n' + "{} runs with {} epochs".format(nb_trials, epochs))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.grid()
    ax.grid(which="minor", linestyle=':', lw=.5)
    ax.tick_params(axis="both", direction="in", top = True, right=True, which="both")
    ax.legend([(mean10, std11, fill1 ,std12), (mean20, std21, fill2 ,std22)], [name1, name2], prop={"size":14})
    fig.tight_layout()
    fig.savefig("./Plots/" + "Comparison_" + name1.replace(' ', '_') + '_' + name2.replace(' ', '_') + ".png")
# %%
