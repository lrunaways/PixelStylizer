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
        y_gen = self.G(x)
        return y_gen

    def run_D(self, x, y):
        disc_input = torch.concat([x, y], axis=1)
        if self.augment_pipe is not None:
            disc_input = self.augment_pipe(disc_input)
        logits = self.D(disc_input)
        return logits

    def accumulate_gradients(self, phase, x, y, gain=1.0):
        assert phase in ['Gboth', 'Gadv', 'Gbasic', 'Dmain', 'Dreg', 'Dboth']

        if phase in ['Gboth', 'Gbasic']:
            basic_loss = self.l1_loss(x, y)
            basic_loss.mean().mul(gain).backward()

        # Gadv: Maximize logits for generated images.
        if phase in ['Gboth', 'Gadv']:
            y_gen = self.run_G(x)
            gen_logits = self.run_D(x, y_gen)
            loss_Gmain = torch.nn.functional.softplus(-gen_logits)
            loss_Gmain.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            y_gen = self.run_G(x)
            gen_logits = self.run_D(x, y_gen)
            loss_Dgen = torch.nn.functional.softplus(gen_logits)
            loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            x_tmp = x.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
            y_tmp = y.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
            real_logits = self.run_D(x_tmp, y_tmp)

            loss_Dreal = 0
            if phase in ['Dmain', 'Dboth']:
                loss_Dreal = torch.nn.functional.softplus(-real_logits)

            loss_Dr1 = 0
            if phase in ['Dreg', 'Dboth']:
                r1_grads = \
                torch.autograd.grad(
                    outputs=[real_logits.sum()], inputs=[x_tmp, y_tmp],
                    create_graph=True, only_inputs=True)[0]
                r1_penalty = r1_grads.square().sum([1, 2, 3])
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
            (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

def additional_entropy(y_true, y_pred):
    pass