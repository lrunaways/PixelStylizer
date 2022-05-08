import os

import numpy as np
import torch
from torch.nn.utils.parametrizations import spectral_norm

# class Mapper(torch.nn.Module):
#     def __init__(self,
#                  z_dim,  # Input latent (Z) dimensionality.
#                  w_dim,  # Intermediate latent (W) dimensionality.
#                  num_ws,  # Number of intermediate latents to output.
#                  num_layers=2,  # Number of mapping layers.
#                  lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
#                 ):
#         super().__init__()
#         self.z_dim = z_dim
#         self.w_dim = w_dim
#         self.num_ws = num_ws
#         self.num_layers = num_layers
#         input_ = torch.nn.Linear(z_dim, w_dim, bias=True)
#         intermediate = []
#         for i in range(num_layers):
#             intermediate.append(torch.nn.Linear(z_dim, w_dim, bias=True))
#             intermediate.append(torch.nn.Linear(z_dim, w_dim, bias=True))
#         out = torch.nn.Linear(w_dim, num_ws, bias=True)

class Noise(torch.nn.Module):
    def __init__(self):
        super(Noise, self).__init__()
        self.noise = torch.nn.Parameter(torch.Tensor([1e-2]))

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise
        return x + x * noise


class BasicG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = []
        n_blocks = 7
        for i in range(n_blocks):
            lrelu_slope = 0.2 if i != n_blocks - 1 else 1.0
            self.blocks.extend([
                torch.nn.Sequential(
                        torch.nn.LazyConv2d(1 if i == n_blocks-1 else 128, 3, padding='same', padding_mode="reflect"),
                        Noise(),
                        torch.nn.LazyBatchNorm2d(),
                        torch.nn.LeakyReLU(negative_slope=lrelu_slope),
                )
            ])

        self.blocks = torch.nn.ModuleList(self.blocks)

    def forward(self, x):
        # input_x = x
        checkerboard = torch.zeros_like(x)
        checkerboard[:, :, 0::2, 1::2] = 0.01
        # checkerboard[:, :, 1::2, 0::2] = 0.01
        x = torch.concat([x, checkerboard], axis=1)
        for i in range(len(self.blocks)):
            if i > 1 and i < len(self.blocks) - 1:
                x = self.blocks[i](x) + x
            else:
                x = self.blocks[i](x)
        return x


class BasicD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.input_norm = torch.nn.LazyBatchNorm2d()

        self.blocks = []
        n_blocks = 4
        for i in range(n_blocks):
            lrelu_slope = 0.2 if i != n_blocks - 1 else 1.0
            self.blocks.extend([
                torch.nn.Sequential(
                        spectral_norm(
                            torch.nn.Conv2d(
                                3 if i == 0 else 128,
                                1 if i == n_blocks-1 else 128,  4, stride=2)
                        ),
                        torch.nn.LeakyReLU(negative_slope=lrelu_slope),
                )
            ])
        self.blocks = torch.nn.ModuleList(self.blocks)

    def calc_grad(self, x):
        grad_x = (x[:, 0:1, :-1, :] - x[:, 0:1, 1:, :])
        grad_y = (x[:, 0:1, :, :-1] - x[:, 0:1, :, 1:])
        grad = torch.nn.functional.pad(grad_x, (0, 0, 1, 0)) + torch.constant_pad_nd(grad_y, (0, 1, 0, 0))
        grad = (grad - grad.min())/(grad.max() - grad.min()) * 2 - 1
        return grad

    def forward(self, x):
        # x = self.input_norm(x)
        grad = self.calc_grad(x)
        x = torch.concat([x, grad], axis=1)
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
