import os

import numpy as np
import torch
import matplotlib.pyplot as plt


def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    # if len(ref) == 0:
    #     return misc.constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [x if isinstance(x, torch.Tensor) else torch.array(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))

def translate2d(tx, ty, **kwargs):
    return matrix(
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
        **kwargs)

def save_images(model, dataset, dirpath, idxs, iteration, device):
    n_images = len(idxs)
    fig, axs = plt.subplots(n_images, 3)
    for i, idx in enumerate(idxs):
        plot_name = f"it-{iteration}_idx{idx}.png"
        # x, y, c = dataset[idx]
        x, y = dataset[idx]
        pred = model(x[None, ...].to(device))
        if n_images > 1:
          axs[i][0].imshow(x.numpy()[0])
          axs[i][1].imshow(y.numpy()[0])
          axs[i][2].imshow(pred.detach().cpu().numpy()[0, 0])
        else:
          axs[0].imshow(x.numpy()[0])
          axs[1].imshow(y.numpy()[0])
          axs[2].imshow(pred.detach().cpu().numpy()[0, 0])
    fig.savefig(os.path.join(dirpath, plot_name))
    plt.close()
