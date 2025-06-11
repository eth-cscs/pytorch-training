import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_matrix(matrix, tokens):
    plt.matshow(matrix, cmap='Blues')
    plt.xticks(range(matrix.shape[-1]), tokens, rotation=90)
    plt.yticks(range(matrix.shape[-1]), tokens)
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_xticks(np.arange(matrix.shape[1]+1)-.5, minor=True)
    plt.gca().set_yticks(np.arange(matrix.shape[1]+1)-.5, minor=True)
    plt.grid(which="minor", color="w", linestyle='-', linewidth=2)
    plt.gca().tick_params(which="minor", top=False, left=False)
    plt.show()
