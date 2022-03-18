import os

import numpy as np
import torch
import matplotlib.pyplot as plt


def save_images(model, dataset, dirpath, idxs, iteration, device):
    fig, axs = plt.subplots(1, 3)
    for idx in idxs:
        plot_name = f"it-{iteration}_idx{idx}.png"
        x, y, c = dataset[idx]
        pred = model(x[None, ...].to(device))
        axs[0].imshow(x.numpy()[0])
        axs[1].imshow(y.numpy()[0])
        axs[2].imshow(pred.detach().numpy()[0, 0])
        fig.savefig(os.path.join(dirpath, plot_name))
        plt.close()
