import os
import copy


import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_dataloaders
from networks import BasicGAN
from losses import colour_loss_fn, GANLoss #, generator_loss
from utils import save_images

def val_loop(model, dataloader):
   pass


def gan_train_loop(train_dataloader, val_dataloader, gan_model, device, log_freq, epoch, save_dirpath):
    gan_loss = GANLoss(gan_model.G, gan_model.D)

    gan_model.to(device)
    gan_model.train()

    for i, (x, y_real, colours) in enumerate(tqdm(train_dataloader)):
        gan_model.opt_G
        gan_loss.accumulate_gradients(phase, x, y_real. gain=1.0)
        if i % log_freq == 0:
            iteration = i + epoch * len(train_dataloader)
            gan_model.G.eval()
            save_images(gan_model.G, val_dataloader.dataset, save_dirpath, [0], iteration, device)
            gan_model.G.train()



def train_loop(train_dataloader, val_dataloader,
               model, opt, device, log_freq, epoch, save_dirpath):
    model.train()
    loss_fn = torch.nn.L1Loss()
    # loss_fn = gan_loss
    c_w = 100
    for i, (x, y_real, colours) in enumerate(tqdm(train_dataloader)):
        x, y_real, colours = x.to(device), y_real.to(device), colours.to(device)
        y_generated = model(x)

        model.zero_grad()

        colour_loss = colour_loss_fn(y_generated, colours)
        loss = loss_fn(y_generated, y_real)
        total_loss = loss + c_w*colour_loss
        print(f"mse: {loss.item()} colour: {colour_loss.item()}")

        if i % log_freq == 0:
            iteration = i + epoch * len(train_dataloader)
            model.eval()
            save_images(model.G, val_dataloader.dataset, save_dirpath, [0], iteration, device)
            model.train()

        total_loss.backward()
        opt.step()



def trainer(params):
    np.random.seed(params['random_seed'])
    torch.manual_seed(params['random_seed'])

    torch.backends.cudnn.benchmark = True    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.

    train_dataloader, val_dataloader = get_dataloaders(
        dirpath=params['dirpath'],
        batch_size=params['batch_size'],
        num_workers=params['num_workers']
    )

    # Construct networks and optimizer
    GAN = BasicGAN(G_lr=params['G_lr'], D_lr=params['D_lr'], device=params['device'])


    # Resume from existing pickle
    pass

    # Setup augmentation
    pass

    for epoch in tqdm(range(params['n_epochs'])):

        # Train
        gan_train_loop(train_dataloader, GAN, device=params['device'])
        # train_loop(train_dataloader, val_dataloader,
        #            GAN.G, GAN.opt_G,
        #            device=params['device'],
        #            log_freq=params['log_freq'],
        #            epoch=epoch,
        #            save_dirpath=params['save_dirpath'])

        # Save image snapshot
        # save_images()
        #
        # # Save network snapshot.
        # GAN.save_snapshot()
        #
        # # Evaluate metrics.
        # val_loop(GAN['G'], val_dataloader, device=params['device'])
