import os

import numpy as np
import torch
import matplotlib.pyplot as plt


def save_images(model, dataset, dirpath, idxs, iteration, device):
    fig, axs = plt.subplots(len(idxs), 3)
    for i, idx in enumerate(idxs):
        plot_name = f"it-{iteration}_idx{idx}.png"
        x, y, c = dataset[idx]
        pred = model(x[None, ...].to(device))
        axs[i][0].imshow(x.numpy()[0])
        axs[i][1].imshow(y.numpy()[0])
        axs[i][2].imshow(pred.detach().cpu().numpy()[0, 0])
    fig.savefig(os.path.join(dirpath, plot_name))
    plt.close()
