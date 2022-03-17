import os
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt

from trainer import trainer


def main():
    # initialization
    params = {
        "dirpath": "D:\data\pixelation\*.npy",
        "random_seed": 24,
        "batch_size": 64,
        "num_workers": 3,
        "G_lr": 2e-3,
        "D_lr": 2e-3,
        "n_epochs": 3,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "log_freq": 10,
        "runs_dirpath": r"C:\Users\nplak\PycharmProjects\PixelStylizer\runs"
    }

    all_runs_paths = glob.glob(os.path.join(params['runs_dirpath'], "*"))
    if all_runs_paths:
        current_run = max([int(x.split('\\')[-1]) for x in all_runs_paths]) + 1
    else:
        current_run = 0
    params['save_dirpath'] = os.path.join(params['runs_dirpath'], str(current_run))
    os.makedirs(params['save_dirpath'])

    trainer(params)

if __name__ == "__main__":
    main()

