import os

import numpy as np
import torch


class BasicG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = []
        n_blocks = 7
        for i in range(n_blocks):
            self.blocks.extend([
                torch.nn.Sequential(
                        torch.nn.LazyConv2d(1 if i == n_blocks-1 else 64, 3, padding='same', padding_mode="reflect"),
                        torch.nn.LazyBatchNorm2d(),
                        torch.nn.LeakyReLU(),
                )
            ])
        self.blocks = torch.nn.ModuleList(self.blocks)

    def forward(self, x):
        input_x = x
        #TODO: attention checkerboard
        #TODO: check add reversed checkerboard to equalize checkerboard addition
        checkerboard = torch.zeros_like(x)
        checkerboard[:, :, 0::2, 1::2] = 0.1
        checkerboard[:, :, 1::2, 0::2] = 0.1
        x = torch.concat([x, checkerboard, checkerboard*(-1) + 0.2], axis=1)
        for i in range(len(self.blocks)):
            if i > 1 and i < len(self.blocks) - 1:
                x = self.blocks[i](x) + x
            else:
                x = self.blocks[i](x)
        return x


class BasicD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = []
        n_blocks = 4
        for i in range(n_blocks):
            self.blocks.extend([
                torch.nn.Sequential(
                        torch.nn.LazyConv2d(1 if i == n_blocks-1 else 64, 2, stride=2),
                        torch.nn.LeakyReLU(),
                )
            ])
        self.blocks = torch.nn.ModuleList(self.blocks)

    def forward(self, x):
        for i in range(len(self.blocks)):
            if i > 0 and i < len(self.blocks) - 1:
                x = self.blocks[i](x)
            else:
                x = self.blocks[i](x)
        x = torch.flatten(x, start_dim=1)
        x = torch.mean(x, axis=-1)
        return x


class BasicGAN:
    def __init__(self, G_lr, D_lr=None, device='cpu'):
        self.G = BasicG().to(device)
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=G_lr)
        self.D = BasicD().to(device)
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=D_lr)

    def train(self):
        self.G.train()
        self.D.train()

    def eval(self):
        self.G.eval()
        self.D.eval()

    def to(self, device):
        self.G.to(device)
        self.D.to(device)

    def zero_grad(self):
        self.opt_G.zero_grad(set_to_none=True)
        self.opt_D.zero_grad(set_to_none=True)

    def opt_step(self):
        self.opt_G.step()
        self.opt_D.step()

    def save_snapshot(self):
        pass

    def load_snapshot(self):
        pass
