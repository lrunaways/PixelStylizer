import os
import copy


import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_dataloaders
from networks import BasicGAN
from losses import colour_loss_fn, GANLoss #, generator_loss
from utils import save_images, save_images_real
from augment import AugmentPipe

def val_loop(model, dataloader):
   pass


def gan_train_loop(train_dataloader, val_dataloader,
                   G_augmentator, D_augmentator, gan_model,
                   device, log_freq, epoch, save_dirpath, G_phase, D_phase, ada_interval=16, ada_kimg=20):
    # augment_p               = 0,        # Initial value of augmentation probability.
    # ada_interval            = 4,        # How often to perform ADA adjustment?
    # ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    gan_loss = GANLoss(gan_model.G, gan_model.D, augment_pipe=D_augmentator)

    gan_model.to(device)
    gan_model.train()

    gen_loss = {}
    disk_loss = {}
    for batch_idx, (x, y_real) in enumerate(tqdm(train_dataloader)):
    # for i, (x, y_real, colours) in enumerate(tqdm(train_dataloader)):
        if G_augmentator is not None:
            x, y_real = G_augmentator(x, y_real)
        gan_model.zero_grad()
        # x, y_real, colours = x.to(device), y_real.to(device), colours.to(device)
        x, y_real = x.to(device), y_real.to(device)
        gen_loss_ = gan_loss.accumulate_gradients(G_phase, x, y_real, gain=1.0)
        # disk_loss_ = gan_loss.accumulate_gradients(D_phase, x, y_real, gain=1.0)
        gan_model.opt_step()

        for key in gen_loss_:
            gen_loss[key] = gen_loss.get(key, gen_loss_[key])*0.8 + gen_loss_[key]*0.2
        print(f'G basic loss: {gen_loss["basic_loss"]}')
        # for key in disk_loss_:
        #     disk_loss[key] = disk_loss.get(key, disk_loss_[key])*0.8 + disk_loss_[key]*0.2

        if batch_idx % log_freq == 0:
            # DGLossRatio = gen_loss.get('loss_Gadv', 0) / disk_loss['loss_Dgen']
            # print(f"Dgen_acc: {disk_loss_['Dgen_acc']}, Dgen_real: {disk_loss_['Dreal_acc']}, GDLossRatio: {DGLossRatio}, loss_Gadv: {gen_loss.get('loss_Gadv', 0)}")
            # print(f"loss_Gadv: {gen_loss['loss_Gadv']}")
            # print(f"loss_Dgen: {disk_loss['loss_Dgen']}, ")
            iteration = batch_idx + epoch * len(train_dataloader)
            gan_model.G.eval()
            save_images(gan_model.G, val_dataloader.dataset, save_dirpath, [0, 100, 200], iteration, device)
            save_images_real(gan_model.G, save_dirpath, iteration, device)
            gan_model.G.train()
        # Execute ADA heuristic.
        # if (gan_loss.augment_pipe is not None) and (batch_idx % ada_interval == 0):
        #     print()
        #     print(f"loss_Dreal: {float(disk_loss_['loss_Dreal'].detach())}")
        #     adjust = np.sign(disk_loss_['real_logits_sign'].cpu() - 0.6) * (train_dataloader.batch_size * ada_interval) / (ada_kimg * 1000)
        #     gan_loss.augment_pipe.p = torch.max(gan_loss.augment_pipe.p + adjust, torch.tensor(0.0, device=device))
        #     print(f"New augment p: {gan_loss.augment_pipe.p}")



# def train_loop(train_dataloader, val_dataloader,
#                model, opt, device, log_freq, epoch, save_dirpath):
#     model.train()
#     loss_fn = torch.nn.L1Loss()
#     # loss_fn = gan_loss
#     c_w = 100
#     for i, (x, y_real, colours) in enumerate(tqdm(train_dataloader)):
#         x, y_real, colours = x.to(device), y_real.to(device), colours.to(device)
#         y_generated = model(x)
#
#         model.zero_grad()
#
#         colour_loss = colour_loss_fn(y_generated, colours)
#         loss = loss_fn(y_generated, y_real)
#         total_loss = loss + c_w*colour_loss
#         print(f"mse: {loss.item()} colour: {colour_loss.item()}")
#
#         if i % log_freq == 0:
#             iteration = i + epoch * len(train_dataloader)
#             model.eval()
#             save_images(model, val_dataloader.dataset, save_dirpath, [0, 100, 200], iteration, device)
#             model.train()
#
#         total_loss.backward()
#         opt.step()



def trainer(params):
    np.random.seed(params['random_seed'])
    torch.manual_seed(params['random_seed'])

    # torch.backends.cudnn.benchmark = True    # Improves training speed.
    # torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    # torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.

    train_dataloader, val_dataloader = get_dataloaders(
        dirpath=params['dirpath'],
        batch_size=params['batch_size'],
        num_workers=params['num_workers']
    )

    # Construct networks and optimizer
    GAN = BasicGAN(G_lr=params['G_lr'], D_lr=params['D_lr'], device=params['device'])

    # dummy forward to initialize Lazy modules
    # x, y, _ = next(iter(train_dataloader))
    x, y = next(iter(train_dataloader))
    x, y = x.to(params['device']), y.to(params['device'])
    GAN.G(x)
    GAN.D(torch.concat([x, y], axis=1))
    # del x, y, _
    del x, y

    # Resume from existing pickle
    pass

    # Setup augmentation


    # G_augmentator = AugmentPipe(xflip=0.5, rotate90=0.5)
    # D_augmentator = AugmentPipe(xflip=1, rotate90=1, xint=1, scale=1, rotate=0, aniso=1, xfrac=1,
    #                             brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
    # D_augmentator.p = torch.tensor(0.0, device=params['device'])

    G_augmentator = None
    D_augmentator = None
    #TODO: endless iterations instead of epochs
    #TODO: bhwc format

    for epoch in tqdm(range(params['n_epochs'])):

        # Train
        losses = gan_train_loop(train_dataloader, val_dataloader,
                       G_augmentator=G_augmentator,
                       D_augmentator=D_augmentator,
                       gan_model=GAN, device=params['device'],
                       log_freq=params['log_freq'],
                       epoch=epoch,
                       save_dirpath=params['save_dirpath'],
                       G_phase=params['G_phase'],
                       D_phase='Dboth',
                       )

        print(1)

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
