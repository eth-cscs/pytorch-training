import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm


def get_line_data(nsamples=1000, seed=42):
    '''
    generates a random vector $x \in [-0.5, 1.5]$ 
    evaluate it in a linear function $y = 2x$
    added uniformly distributed noise to $y$ and 
    that give us $y \in [-1.5, 1.5]$
    '''
    ref_slope = 2.0
    ref_offset = 0.0
    ground_truth = [ref_slope, ref_offset]

    np.random.seed(seed)
    noise = np.random.random((nsamples, 1)) - 0.5    # -0.5 to center the noise
    x_train = np.random.random((nsamples, 1)) - 0.5  # -0.5 to center x around 0
    y_train = ref_slope * x_train + ref_offset + noise
    
    return x_train, y_train, ground_truth


def get_curve_data(nsamples=1000):
    ''' Random data for polynomial regression '''
    f = lambda x: 2 - 2*x - x**2 + x**3

    noise = np.random.normal(size=(nsamples, 1))
    x = np.array(sorted(np.random.random((nsamples, 1))*4.5 - 2))
    y = f(x) + noise

    ground_truth = np.arange(-2.3, 2.8, 0.005)
    ground_truth = (ground_truth, f(ground_truth))
    
    return x, y, ground_truth


def plot_loss(history, ax=None, figsize=(6,4)):
    ''' Plots the line regression loss history '''
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)

    loss_hist = history[:, 2] if history.ndim > 1 else history

    ax.plot(loss_hist, 'r.-')
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.grid()

    return ax


def plot_line(history, x_train, y_train, ax=None, figsize=(6,4)):
    ''' Plots the line regression, before and after '''
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)

    slope_hist = history[:, 0]
    offset_hist = history[:, 1]

    ax.plot(x_train, y_train, '.', label='data', alpha=0.2)
    ax.plot(x_train, slope_hist[0] * x_train + offset_hist[0], 'r-', label='model (step 0)')
    ax.plot(x_train, slope_hist[-1] * x_train + offset_hist[-1], 'g-', label='model (trained)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    ax.legend()

    return ax


def plot_track(history, x_train, y_train, ground_truth, ax=None, figsize=(6,4)):
    ''' Plots the line regression track '''
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)

    slope_hist = history[:, 0]
    offset_hist = history[:, 1]
    ref_slope, ref_offset = ground_truth

    def loss_function_field(m, n, xref, yref):
        ''' Utility function for ploting the MSE loss '''
        return np.mean(np.square(yref - m * xref - n ))

    _m = np.arange(-0, 4.01, 0.1)
    _n = np.arange(-0.5, 0.51, 0.1)
    M, N = np.meshgrid(_m, _n)

    Z = np.zeros(M.shape)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Z[i, j] = loss_function_field(M[i, j], N[i, j], x_train, y_train)

    cp = ax.contour(M, N, Z, 15, vmin=Z.min(), vmax=Z.max(), alpha=0.99, colors='k', linestyles='--')
    cpf = ax.contourf(M, N, Z, vmin=Z.min(), vmax=Z.max(), alpha=0.8, cmap=plt.cm.RdYlBu_r)
    ax.clabel(cp, cp.levels[:6])
    if not hasattr(ax, 'colorbar'):
        ax.colorbar = ax.figure.colorbar(cpf, ax=ax)
    m = slope_hist[-1]
    n = offset_hist[-1]
    ax.plot(slope_hist, offset_hist, '.-', lw=2, c='k')
    ax.plot([ref_slope], [ref_offset], 'rx', ms=10)
    ax.set_xlim([_m.min(), _m.max()])
    ax.set_ylim([_n.min(), _n.max()])
    ax.set_xlabel('Slope')
    ax.set_ylabel('Offset')

    return ax


def plot_regression(history, x_train, y_train, ground_truth, axs=None, figsize=(18,4)):
    ''' Plots the line regression history '''
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=figsize)
    history = np.array(history)

    for ax in axs:
        ax.clear()
    plot_loss(history, axs[0])
    plot_line(history, x_train, y_train, axs[1])
    plot_track(history, x_train, y_train, ground_truth, axs[2])

    return axs


class DynamicRegressionPlot:
    ''' Manages a dynamic line regression plot '''

    def __init__(self, ground_truth, display_every=10, figsize=(16,4)):
        self.ground_truth = ground_truth
        self.display_every = display_every
        self.figsize = figsize
        self.graph_fig = None

    def update_graph(self, history, x_train, y_train):
        if len(history) % self.display_every == 0:
            if self.graph_fig is None:
                self.graph_fig, self.graph_axs = plt.subplots(1, 3, figsize=self.figsize)
                self.graph_out = display(self.graph_fig, display_id=True)
            plot_regression(history, x_train, y_train, self.ground_truth, self.graph_axs)
            self.graph_out.update(self.graph_fig)
            plt.close()
