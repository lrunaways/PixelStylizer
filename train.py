import os
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

from trainer import trainer

def create_parser():
    parser = argparse.ArgumentParser(description="Train")

    # Arguments for stata and models saving
    parser.add_argument("--exp_info", type=str, default="ganwe-lt5", help="")
    parser.add_argument(
        "--dirpath", type=str, default="D://data//pixelation//*.npy", help=""
    )
    parser.add_argument(
        "--runs_dirpath", type=str, default=r"C://Users//nplak//PycharmProjects//PixelStylizer//runs", help=""
    )
    parser.add_argument(
        "--random_seed", type=int, default=24, help="seed"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="seed"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="seed"
    )
    parser.add_argument(
        "--G_lr", type=float, default=2e-3, help="seed"
    )
    parser.add_argument(
        "--D_lr", type=float, default=2e-3, help="seed"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=30, help="seed"
    )
    parser.add_argument(
        "--log_freq", type=int, default=16, help="seed"
    )
    return parser

def main(args):
    # initialization
    params = {
        "exp_info": args.exp_info,
        "dirpath": args.dirpath,
        "random_seed": args.random_seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "G_lr": args.G_lr,
        "D_lr": args.D_lr,
        "n_epochs": args.n_epochs,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "log_freq": args.log_freq,
        "runs_dirpath": args.runs_dirpath,
    }

    all_runs_paths = glob.glob(os.path.join(params['runs_dirpath'], "*"))
    if all_runs_paths:
        current_run = max([int(x.split('\\')[-1].split('-')[0]) for x in all_runs_paths]) + 1
    else:
        current_run = 0
    current_run_name = str(current_run) + '-' + params['exp_info']
    params['save_dirpath'] = os.path.join(params['runs_dirpath'], current_run_name)
    os.makedirs(params['save_dirpath'])

    trainer(params)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)

