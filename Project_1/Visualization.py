# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def plot_pairs(nb, data, classes, target):
    fig = plt.figure(constrained_layout=True, figsize=(8, 4*nb))
    fig.suptitle('Visualization of pair selection.')
    data = np.array(data, dtype='float')
    target_str = [r"larger ($>$)", r"smaller or equal ($\leq$)"]
    subfigs = fig.subfigures(nrows=nb, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f'Pair {row} (target: {target_str[int(target[row])]})')

        axs = subfig.subplots(nrows=1, ncols=2)
        for col, ax in enumerate(axs):
            ax.imshow(data[row, col, :, :])
            ax.set_title(f'Class: {classes[row, col]}')


