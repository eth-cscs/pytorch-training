import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms


def show_sample(image_ex, label_ex):
    matplotlib.rcParams['figure.figsize'] = (9, 9)
    plt.imshow(image_ex, cmap='Blues', alpha=0.5)
    plt.title(label_ex)
    ax = plt.gca()
    ax.set_xticks(range(0, image_ex.shape[0] + 1, 1))
    ax.set_yticks(range(0, image_ex.shape[0] + 1, 1))
    plt.xlim((0, image_ex.shape[0]))
    plt.ylim((0, image_ex.shape[0]))
    ax.grid(color='k', linestyle=':', linewidth=1, alpha=0.2)

    for tick in [*ax.xaxis.get_major_ticks(),
                 *ax.yaxis.get_major_ticks()]:
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    for (j, i), label in np.ndenumerate(image_ex):
        ax.text(i, j, f'{label:0.1f}', ha='center', va='center',
                fontsize=7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.show()
    matplotlib.rcParams['figure.figsize'] = (6, 4)


# Define a transform for visualization
visualize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
])


def show_predictions(images, predicted_labels, num_col=8):
    num_samples = predicted_labels.shape[0]
    num_rows = num_samples // num_col
    fig, axs = plt.subplots(num_rows, num_col,
                            figsize=(num_col * 1.3, num_rows * 1.3))

    for col in range(num_rows):
        for row in range(num_col):
            image = visualize_transform(images[num_col * col + row].cpu())
            axs[col, row].imshow(image, cmap='Blues')
            axs[col, row].axis('off')
            axs[col, row].text(24, 3,
                               f'{predicted_labels[num_col * col + row]}')

    plt.show()


def show_act_functions(x, functions, num_col=8):

    num_samples = len(functions)
    num_rows = math.ceil(num_samples / num_col)
    fig, axs = plt.subplots(num_rows, num_col,
                            figsize=(num_col * 4, num_rows * 3))

    for col in range(num_rows):
        for row in range(num_col):
            if num_col * col + row > num_samples - 1:
                break

            if len(axs.shape) > 1:
                p = axs[col, row]
            else:
                p = axs[col + row]

            fun = functions[num_col * col + row]
            p.plot(x, fun()(x), label=fun.__name__, c='k')
            # axs[col, row].axis('off')
            p.legend()
            p.grid(ls=':', alpha=0.5)

    plt.show()
