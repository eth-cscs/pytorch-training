import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_matrix(matrix, tokens, show_values=False, fmt=".2f", value_color="black"):
    plt.matshow(matrix, cmap='Blues')
    plt.xticks(range(matrix.shape[-1]), tokens, rotation=90)
    plt.yticks(range(matrix.shape[-1]), tokens)
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", top=False, left=False)

    if show_values:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = format(matrix[i, j], fmt)
                plt.text(j, i, value, ha='center', va='center', color=value_color)

    plt.show()
