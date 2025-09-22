from __future__ import print_function

import numpy as np

import sys
import statistics as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchbraid
import torchbraid.utils

from mpi4py import MPI
import math

from hf_src import *
from torchbraid.utils import LPDropout as Dropout

__all__ = [ 'OpenLayer', 'CloseLayer', 'StepLayer', 'parse_args', 'ParallelNet' ]

# Ported Huggingface layers

####################################################################################
####################################################################################
# Network architecture is Open + ViT Encoders + Classifier layers with layer norm


class OpenLayer(nn.Module):
    """
    Just the embedding
    """
    def __init__(self, config):
        super(OpenLayer, self).__init__()
        self.embedding = ViTEmbeddings(config)

        # Place two layer in open/close layer 
        self.layer1 = ViTLayer(config)
        self.layer2 = ViTLayer(config)
  
    def forward(self, pixel_values):
        embedding_output = self.embedding(
            pixel_values
        )
        embedding_output = self.layer1(embedding_output, dt=1)
        embedding_output = self.layer2(embedding_output, dt=1)
        return embedding_output

class CloseLayer(nn.Module):
    """
    Close; final layer norm and classifier
    """
    def __init__(self, config):
        super(CloseLayer, self).__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.layernorm = nn.Identity()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        
        # Place ONE layer in open/close layer 
        self.layer1 = ViTLayer(config)
        self.layer2 = ViTLayer(config)

    def forward(self, encoder_outputs):
        encoder_outputs = self.layer1(encoder_outputs, dt=1)
        encoder_outputs = self.layer2(encoder_outputs, dt=1)
        sequence_output = self.layernorm(encoder_outputs)
        logits = self.classifier(sequence_output[:, 0, :])

        return logits

class StepLayer(nn.Module):
    """
    Just the encoder layer
    """
    def __init__(self, config):
        super(StepLayer, self).__init__()

        self.layer = ViTLayer(config)
    def forward(self, x, dt: float=1.0):
        # layer_outputs = layer_module(hidden_states)
        # hidden_states = layer_outputs
        x = self.layer(x, dt=dt)

        return x

####################################################################################
####################################################################################

# Parallel network class
# local_steps: number of ResNet layers per processor
# all other parameter definitions are in argument parser comments below
class ParallelNet(nn.Module):
    def __init__(self, config = None,
        local_steps=8, Tf=1.0,max_levels=1, bwd_max_iters=1,
               fwd_max_iters=2, print_level=0, braid_print_level=0, cfactor=4,
               fine_fcf=False, skip_downcycle=True, fmg=False, relax_only_cg=0,
               user_mpi_buf=False, comm_lp=MPI.COMM_WORLD):
        super(ParallelNet, self).__init__()

        # Use the same config (except for hidden layers which is contorlled elsewhere)
        # as bert-large-uncased
        if config is None: 
            config = ViTConfig(
                num_labels=1000,  
                hidden_size=768,  # Default hidden size for ViT
                num_hidden_layers=12,  # Number of transformer layers
                num_attention_heads=12,  # Number of attention heads
                image_size=224,  # Input image size
                patch_size=16,  # Patch size
                intermediate_size=3072,  # Feed-forward layer size
                hidden_dropout_prob=0.0,  # Dropout probability
                attention_probs_dropout_prob=0.0,  # Dropout for attention
            )

        step_layer = lambda: StepLayer(config)
        self.comm_lp = comm_lp
        numprocs = self.comm_lp.Get_size()

        # Seperate max_levels to forward and backawrd
        if not isinstance(max_levels, int):
            max_fwd_levels = max_levels[0]
            max_bwd_levels = max_levels[1]
        else:
            max_fwd_levels = max_levels
            max_bwd_levels = max_levels


        self.parallel_nn = torchbraid.LayerParallel(comm_lp, step_layer, local_steps*numprocs, Tf,
                                                    max_fwd_levels=max_fwd_levels, max_bwd_levels=max_bwd_levels,
                                                    max_iters=2, user_mpi_buf=user_mpi_buf)
        
        self.parallel_nn.setBwdResidualCompute(True)
        self.parallel_nn.setBwdMaxIters(bwd_max_iters)
        self.parallel_nn.setFwdMaxIters(fwd_max_iters)
        self.parallel_nn.setPrintLevel(print_level, True)
        self.parallel_nn.setPrintLevel(braid_print_level, False)
        self.parallel_nn.setCFactor(cfactor)
        self.parallel_nn.setSkipDowncycle(skip_downcycle)
        self.parallel_nn.setBwdRelaxOnlyCG(relax_only_cg)
        self.parallel_nn.setFwdRelaxOnlyCG(relax_only_cg)
        if fmg:
          self.parallel_nn.setFMG()

        self.parallel_nn.setNumRelax(1)  # FCF relaxation default on coarse levels
        if not fine_fcf:
            self.parallel_nn.setNumRelax(0, level=0)  # Set F-Relaxation only on the fine grid
        else:
            self.parallel_nn.setNumRelax(1, level=0)  # Set FCF-Relaxation on the fine grid

        # this object ensures that only the LayerParallel code runs on ranks!=0
        compose = self.compose = self.parallel_nn.comp_op()

        # by passing this through 'compose' (mean composition: e.g. OpenLayer o channels)
        # on processors not equal to 0, these will be None (there are no parameters to train there)
        self.open_nn = compose(OpenLayer, config)
        self.close_nn = compose(CloseLayer, config)

    def saveSerialNet(self, name):
        # Model can be reloaded in serial format with: model = torch.load(filename)
        serial_nn = self.parallel_nn.buildSequentialOnRoot()
        if self.comm_lp.Get_rank() == 0:
            s_net = SerialNet(serial_nn=serial_nn, open_nn=self.open_nn,
                              close_nn=self.close_nn)
            s_net.eval()
            torch.save(s_net, name)

    def forward(self, x):
        # We need the mask to be passed through, so using global allows us 
        # to pass through without interfering with the parallel_nn
        
        # by passing this through 'o' (mean composition: e.g. self.open_nn o x)
        # this makes sure this is run on only processor 0
        x = self.compose(self.open_nn, x)

        x = self.parallel_nn(x)

        # Go through the classifier layers
        out = self.compose(self.close_nn, x)

        return out
    
    # Let's not do dropout for this yet
    # 
    def new_mask(self, x=None): 
      """
      Generates a new mask given a sample input
      """
      raise Exception
      for layer in self.parallel_nn.local_layers.modules():
        # print(layer)
        if isinstance(layer, Dropout):
          # print('Generating')
          layer.generate_new_mask(x)

# Serial Network Class (used by the saveSerialNet functionality in ParallelNet)
class SerialNet(nn.Module):
    def __init__(self, serial_nn=None, open_nn=None, close_nn=None):
        super(SerialNet, self).__init__()

        self.open_nn = open_nn
        self.serial_nn = serial_nn
        self.close_nn = close_nn

    def forward(self, x):
        x = self.open_nn(x)
        x = self.serial_nn(x)
        out = self.close_nn(x)

        return out
