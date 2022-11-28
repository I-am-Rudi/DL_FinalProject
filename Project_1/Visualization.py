# %%
import matplotlib.pyplot as plt

# %%
def plot_pairs(nb, data, classes, target):
    fig = plt.figure(constrained_layout=True, figsize=(8, 4*nb))
    fig.suptitle('Visualization of pair selection.' + '\n', fontsize=16)
    target_str = [r"larger ($>$)", r"smaller or equal ($\leq$)"]
    subfigs = fig.subfigures(nrows=nb, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f'Pair {row} (target: {target_str[int(target[row])]})', fontsize=15)

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

def plot_analysis(mean, std, epochs, nb_trials):

    fig, ax = plt.subplots(1,1, figsize=(8, 6), dpi = 80)
    plt.rcParams['font.size'] = '16'

    x = range(1, epochs+1)

    mean0, = ax.plot(x, mean, color="darkblue")
    std1, = ax.plot(x, mean+std, color="blue", alpha=0.97)
    std2, = ax.plot(x, mean-std, color="blue", alpha=0.97)
    fill = ax.fill_between(x, mean+std, mean-std, color="blue", alpha=0.5)
    
    print(mean0, std1, std2, fill)

    ax.set_title("Mean and standard deviation of test_accuracy after" + '\n' + "{} runs with {} epochs".format(nb_trials, epochs))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.grid()
    ax.grid(which="minor", linestyle=':', lw=.5)
    ax.tick_params(axis="both", direction="in", top = True, right=True, which="both")
    ax.legend([mean0, (std1, fill ,std2)], ["mean", "standard deviation"], prop={"size":14})
    fig.tight_layout()
    fig.show()