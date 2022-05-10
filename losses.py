import os

import numpy as np
import torch
import matplotlib.pyplot as plt


def colour_loss_fn(y_pred, colours):
    colour_loss = (torch.min(torch.abs((y_pred - colours)), axis=1)[0])
    colour_loss *= colour_loss > 1 / 255
    colour_loss = colour_loss.mean()
    return colour_loss


# def discriminator_loss(disc_real_output, disc_generated_output):
#
#     real_loss = -tf.math.reduce_mean(tf.math.minimum(disc_real_output - 1, tf.zeros_like(disc_real_output)))
#     generated_loss = -tf.math.reduce_mean(tf.math.minimum(-disc_generated_output - 1, tf.zeros_like(disc_real_output)))
#
#     metric_real = acc(tf.ones_like(disc_real_output), tf.sigmoid(disc_real_output))
#     metric_gen = acc(tf.zeros_like(disc_generated_output), tf.sigmoid(disc_generated_output))
#
#     total_disc_loss = (real_loss + generated_loss)
#     total_disc_metric = (metric_real + metric_gen ) /2
#     return total_disc_loss, total_disc_metric
#
# def generator_loss_backward(gen_logits, gain, perceptual_weight):
#   loss_Gmain = torch.nn.functional.softplus(-gen_logits)
#   loss_Gmain = loss_Gmain.mean().mul(gain)
#   perceptual_loss = 0 #perceptual_loss_object(gen_output, targets) #  mae_object(gen_output_ent, targets_ent)
#
#   total_gen_loss = loss_Gmain*1.0 + perceptual_weight*perceptual_loss
#   total_gen_loss.backward()

class GANLoss:
    def __init__(self, G, D, augment_pipe=None, r1_gamma=10):
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.l1_loss = torch.nn.L1Loss()

    def run_G(self, x):
        # z = torch.randn((x.shape[0], 256), device=x.device)
        y_gen = self.G(x)
        return y_gen

    def run_D(self, x, y):
        if self.augment_pipe is not None:
            x, y = self.augment_pipe(x, y)
        disc_input = torch.concat([x, y], axis=1)
        logits = self.D(disc_input)
        return logits

    def accumulate_gradients(self, phase, x, y, gain=1.0):
        assert phase in ['Gboth', 'Gadv', 'Gbasic', 'Dmain', 'Dreg', 'Dboth']
        if phase[0] == 'G':
            self.G.requires_grad_(True)
            self.D.requires_grad_(False)
        elif phase[0] == 'D':
            self.G.requires_grad_(False)
            self.D.requires_grad_(True)
        loss = {}
        # Gadv: Maximize logits for generated images.
        if phase in ['Gboth', 'Gadv', 'Gbase']:
            y_gen = self.run_G(x)
            loss = 0
            if phase in ['Gboth', 'Gbase']:
                basic_loss = self.l1_loss(y_gen, y)
                loss['basic_loss'] = basic_loss
                loss += basic_loss
            if phase in ['Gboth', 'Gadv']:
                gen_logits = self.run_D(x, y_gen)
                loss_Gadv = torch.nn.functional.softplus(-gen_logits)
                # loss_Gadv = (1 - gen_logits)**2
                loss_Gadv = loss_Gadv.mean()
                loss['loss_Gadv'] = loss_Gadv
                loss += loss_Gadv
            (loss).mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        if phase in ['Dmain', 'Dboth']:
            y_gen = self.run_G(x)
            gen_logits = self.run_D(x, y_gen)
            loss_Dgen = torch.nn.functional.softplus(gen_logits)
            # loss_Dgen = (gen_logits)**2
            loss_Dgen = loss_Dgen.mean()
            loss['loss_Dgen'] = loss_Dgen
            loss['Dgen_acc'] = (torch.sigmoid(gen_logits) < 0.5).to(dtype=float).mean()
            loss_Dgen.mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization./
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            x_tmp = x.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
            y_tmp = y.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
            real_logits = self.run_D(x_tmp, y_tmp)

            loss_Dreal = 0
            if phase in ['Dmain', 'Dboth']:
                loss_Dreal = torch.nn.functional.softplus(-real_logits)
                # loss_Dreal = (1-real_logits)**2
                loss_Dreal = loss_Dreal.mean()
                loss['real_logits_sign'] = real_logits.mean().sign().detach()
                loss['Dreal_acc'] = (torch.sigmoid(real_logits) > 0.5).to(dtype=float).mean()
                loss['loss_Dreal'] = loss_Dreal

            loss_Dr1 = 0
            if phase in ['Dreg', 'Dboth']:
                r1_grads = \
                torch.autograd.grad(
                    outputs=[real_logits.sum()], inputs=[x_tmp, y_tmp],
                    create_graph=True, only_inputs=True)[0]
                r1_penalty = r1_grads.square().sum([1, 2, 3])
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                loss_Dr1 = loss_Dr1.mean()
                loss['loss_Dr1'] = loss_Dr1
            (loss_Dreal + loss_Dr1).mul(gain).backward()
        return loss

def additional_entropy(y_true, y_pred):
    pass