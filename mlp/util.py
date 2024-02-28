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


def show_predictions(images, predicted_labels, num_cols=8):
    num_samples = predicted_labels.shape[0]
    num_rows = (num_samples + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols,
                            figsize=(num_cols * 1.3, num_rows * 1.3))

    axs = axs.reshape([num_rows, num_cols])
    
    for row in range(num_rows):
        for col in range(num_cols):
            index = num_cols * row + col
            
            axs[row, col].axis('off')
            if index >= num_samples:
                continue
            image = visualize_transform(images[index].cpu())
            axs[row, col].imshow(image, cmap='Blues')
            axs[row, col].text(24, 3, f'{predicted_labels[index]}')

    plt.show()


def show_act_functions(x, functions, num_cols=8):

    num_samples = len(functions)
    num_rows = (num_samples + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols,
                            figsize=(num_cols * 4, num_rows * 3))

    axs = axs.reshape([num_rows, num_cols])

    
    for row in range(num_rows):
        for col in range(num_cols):            
            index = num_cols * row + col
            if index >= num_samples:
                break

            p = axs[row, col]
                
            fun = functions[index]
            p.plot(x, fun()(x), label=fun.__name__, c='k')
            p.legend()
            p.grid(ls=':', alpha=0.5)

    plt.show()
