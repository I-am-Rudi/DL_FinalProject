# %%
import numpy as np
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
