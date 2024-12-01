from __future__ import print_function

import numpy as np

import sys
import statistics as stats
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchbraid
import torchbraid.utils

from torchbraid.mgopt import root_print, compute_levels

from timeit import default_timer as timer

from mpi4py import MPI

from _utils.block import Block

from _utils.feed_forward_network import FeedForward
from _utils.multi_head_attention import MultiHeadAttention

__all__ = [ 'OpenLayer', 'CloseLayer', 'StepLayer', 'parse_args', 'ParallelNet' ]

####################################################################################
####################################################################################
# Network architecture is Open + ResNet + Close
# The StepLayer defines the ResNet (ODENet)


class OpenLayer(nn.Module):
  def __init__(self, model_dimension, vocabulary_size, context_window, **kwargs):
    super().__init__()
    torch.manual_seed(0)
    self.emb    = nn.Embedding(vocabulary_size, model_dimension)
    self.posenc = nn.Embedding( context_window, model_dimension)

  def forward(self, x):
    B, T = x.shape
    positions = torch.arange(T, device=x.device)
    x = self.emb(x) + self.posenc(positions)  # (B,T,C) + (T,C) = (B,T,C)

    return x
# end layer

class CloseLayer(nn.Module):
  def __init__(self, model_dimension, vocabulary_size, **kwargs):
    super().__init__()
    torch.manual_seed(0)
    self.ln = nn.LayerNorm(model_dimension) # final layer norm
    self.classifier = nn.Linear(model_dimension, vocabulary_size)
    # self.apply(init_weights)

  def forward(self, x, **kwargs):
    #print(f'Norm from transformers 1: {torch.norm(x.float()):.3e}')
    x = self.ln(x) # (B,T,C)
    #print(f'Norm from transformers 2: {torch.norm(x.float()):.3e}')
    x = self.classifier(x) # (B,T,vocabulary_size)

    return x
# end layer

#class BertFeedForward(torch.nn.Module): 
#    "Implements FFN equation." 
#    def __init__(self, d_model, middle_dim=2048, dropout=0.0): 
#        super(FeedForward, self).__init__() 
#         
#        self.fc1 = torch.nn.Linear(d_model, middle_dim) 
#        self.fc2 = torch.nn.Linear(middle_dim, d_model) 
#        self.activation = torch.nn.GELU() 
# 
#    def forward(self, x): 
#        out = self.activation(self.fc1(x)) 
#        out = self.fc2(out)
#        return out 
#

# f = open('llog3.txt', 'w')
# f.close()
class StepLayer(nn.Module):
  def __init__(self, model_dimension, num_heads, context_window, dropout, **kwargs):
  #def __init__(self, **kwargs):
    super().__init__()
    # torch.manual_seed(0)
    #print(f'{kwargs=}')
    #self.block = Block(model_dimension, num_heads, context_window, dropout, **kwargs)
    #self.block = Block(**kwargs)

    # Self definition
    head_size = model_dimension // num_heads
    self.sa = MultiHeadAttention(
     num_heads, head_size, model_dimension, context_window, dropout
    )
    self.ffwd = FeedForward(model_dimension, dropout)
    # self.ffwd = FeedForward(model_dimension, model_dimension * 1, dropout)
    self.ln1 = nn.LayerNorm(model_dimension)
    self.ln2 = nn.LayerNorm(model_dimension)


  def forward(self, x,dt:float=1.0): 
    #print(f'{dt=}')
    x = x + dt * self.sa(self.ln1(x))
    #x1 = self.block(x, dt)
    return x

####################################################################################
####################################################################################

# Parallel network class
# local_steps: number of ResNet layers per processor
# all other parameter definitions are in argument parser comments below
class ParallelNet(nn.Module):
  def __init__(self, model_dimension, num_heads, vocabulary_size, context_window, 
               local_steps=8, Tf=1.0, max_levels=1, bwd_max_iters=1,
               fwd_max_iters=2, print_level=0, braid_print_level=0, cfactor=4,
               fine_fcf=False, skip_downcycle=True, fmg=False, relax_only_cg=0,
               user_mpi_buf=False, comm_lp=MPI.COMM_WORLD):
    super(ParallelNet, self).__init__()

    step_layer = lambda: StepLayer(model_dimension=model_dimension, num_heads=num_heads, context_window=context_window, dropout=0.1)
    self.comm_lp = comm_lp
    numprocs = self.comm_lp.Get_size()

    if not isinstance(max_levels, int):
        max_fwd_levels = max_levels[0]
        max_bwd_levels = max_levels[1]
    else:
        max_fwd_levels = max_levels
        max_bwd_levels = max_levels


    self.parallel_nn = torchbraid.LayerParallel(comm_lp, step_layer, local_steps*numprocs, Tf,
                                                max_fwd_levels=max_fwd_levels, 
                                                max_bwd_levels=max_bwd_levels,
                                                max_iters=2, user_mpi_buf=user_mpi_buf)
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
    self.open_nn = compose(OpenLayer, model_dimension, vocabulary_size, context_window)
    self.close_nn = compose(CloseLayer, model_dimension, vocabulary_size)

  def saveSerialNet(self, name):
    # Model can be reloaded in serial format with: model = torch.load(filename)
    serial_nn = self.parallel_nn.buildSequentialOnRoot()

    if self.comm_lp.Get_rank() == 0:
      s_net = SerialNet(-1, -1, -1,-1, serial_nn=serial_nn, open_nn=self.open_nn, close_nn=self.close_nn)
      s_net.eval()
      torch.save(s_net, name)

  def forward(self, x):
    # by passing this through 'o' (mean composition: e.g. self.open_nn o x)
    # this makes sure this is run on only processor 0
    x = self.compose(self.open_nn, x)
    # t0_CB = time.time()
    x = self.parallel_nn(x)
    # t1_CB = time.time()
    x = self.compose(self.close_nn, x)
    # print(f'CB: {t1_CB - t0_CB :<4f}')

    return x

# Serial Network Class (used by the saveSerialNet functionality in ParallelNet)
class SerialNet(nn.Module):
  def __init__(self, model_dimension, num_heads, vocabulary_size, context_window, local_steps=8, Tf=1.0, serial_nn=None, open_nn=None, close_nn=None):
    super(SerialNet, self).__init__()

    self.serial_nn = serial_nn
    self.open_nn = open_nn
    self.close_nn = close_nn

  def forward(self, x):
    x = self.open_nn(x)
    x = self.serial_nn(x)
    x = self.close_nn(x)
    return x


####################################################################################
####################################################################################

def int_or_tuple(value):
  try:
    # Check if the value is an integer
    return int(value)
  except ValueError:
    # Check if the value is a tuple-like object
    try:
      elements = tuple(map(int, value.strip('()').split(',')))
      return elements
    except ValueError:
      raise argparse.ArgumentTypeError("Invalid value. Must be an integer or a tuple-like object.")


# Parse command line 
def parse_args():
  """
  Return back an args dictionary based on a standard parsing of the command line inputs
  """

  # Command line settings
  parser = argparse.ArgumentParser(description='Parser for GPT2 network')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')

  # artichtectural settings
  parser.add_argument('--steps', type=int, default=32, metavar='N',
                      help='Number of times steps in the resnet layer (default: 32)')
  parser.add_argument('--Tf',type=float,default=1.0,
                      help='Final time for ResNet layer-parallel part')
  parser.add_argument('--serial-file', type=str, default=None,
                      help='Save network to file in serial (not parallel) format')

  # algorithmic settings (batching)
  parser.add_argument('--percent-data', type=float, default=0.05, metavar='N',
                      help='how much of the data to read in and use for training/testing')
  parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                      help='input batch size for training (default: 50)')
  parser.add_argument('--epochs', type=int, default=3, metavar='N',
                      help='number of epochs to train (default: 3)')
  parser.add_argument('--endepochs', type=int, default=-1, metavar='N',
          help='Early epoch stopping; defaulting of negative 1 means we run until epoch is reached (default: 3)')
  parser.add_argument('--lr', type=float, default=6e-4, metavar='LR',
                      help='learning rate (default:6e-4)')

  # algorithmic settings (layer-parallel)
  parser.add_argument('--lp-max-levels', type=int_or_tuple, default=(1,2), metavar='N',
                      help='Layer parallel max number of levels (default: (1,2)) which is one forward and two backwards')
  parser.add_argument('--lp-bwd-max-iters', type=int, default=1, metavar='N',
                      help='Layer parallel max backward iterations (default: 1)')
  parser.add_argument('--lp-fwd-max-iters', type=int, default=2, metavar='N',
                      help='Layer parallel max forward iterations (default: 2)')
  parser.add_argument('--lp-print-level', type=int, default=0, metavar='N',
                      help='Layer parallel internal print level (default: 0)')
  parser.add_argument('--lp-braid-print-level', type=int, default=0, metavar='N',
                      help='Layer parallel braid print level (default: 0)')
  parser.add_argument('--lp-cfactor', type=int, default=4, metavar='N',
                      help='Layer parallel coarsening factor (default: 4)')
  parser.add_argument('--lp-fine-fcf',action='store_true', default=False,
                      help='Layer parallel fine FCF for forward solve, on or off (default: False)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--warm-up', action='store_true', default=False,
                      help='Warm up for GPU timings (default: False)')
  parser.add_argument('--lp-user-mpi-buf',action='store_true', default=False,
                      help='Layer parallel use user-defined mpi buffers (default: False)')
  parser.add_argument('--lp-use-downcycle', action='store_true', default=False,
                      help='Layer parallel use downcycle on or off (default: False)')

  # data parallelism
  parser.add_argument('--dp-size', type=int, default=1, metavar='N',
                      help='Data parallelism (used if value != 1)')

  ## save model
  parser.add_argument('--output_fn',type=str, default=None,#required=True,
                      help='Output filename (for model saving)')
  parser.add_argument('--models_dir',type=str, default=None,#required=True,
                      help='Models directory (for model saving)')

  ## additional arguments
  parser.add_argument('--context_window', type=int, default=256)
  parser.add_argument('--input_text', type=str, default='wikipedia')
  parser.add_argument('--tokenization', type=str, default='gpt2')
  parser.add_argument('--model_dimension', type=int, default=384)
  parser.add_argument('--num_heads', type=int, default=6)

  parser.add_argument('--enforce_serial', action='store_true')

  ##
  # Do some parameter checking
  rank  = MPI.COMM_WORLD.Get_rank()
  procs = MPI.COMM_WORLD.Get_size()
  args = parser.parse_args()

  if procs % args.dp_size != 0:
    root_print(rank, 1, 1, 'Data parallel size must be an even multiple of the number of processors: %d %d'
               % (procs, args.dp_size) )
    sys.exit(0)
  else:
    procs_lp = int(procs / args.dp_size)

  ##
  # Compute number of parallel-in-time multigrid levels 
  def compute_levels(num_steps, min_coarse_size, cfactor):
    from math import log, floor
    # Find L such that ( max_L min_coarse_size*cfactor**L <= num_steps)
    levels = floor(log(float(num_steps) / min_coarse_size, cfactor)) + 1

    if levels < 1:
      levels = 1
    return levels

  #if args.lp_max_levels < 1:
  #  min_coarse_size = 3
  #  args.lp_max_levels = compute_levels(args.steps, min_coarse_size, args.lp_cfactor)

  if args.steps % procs_lp != 0:
    root_print(rank, 1, 1, 'Steps must be an even multiple of the number of layer parallel processors: %d %d'
               % (args.steps, procs_lp) )
    sys.exit(0)

  return args


####################################################################################
####################################################################################
