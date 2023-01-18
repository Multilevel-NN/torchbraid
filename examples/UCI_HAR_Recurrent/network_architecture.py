from __future__ import print_function

import numpy as np

import sys
import statistics as stats
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchbraid
import torchbraid.utils

from torchbraid.mgopt import root_print, compute_levels

from timeit import default_timer as timer

from mpi4py import MPI


__all__ = [ 'ImplicitGRUBlock', 'CloseLayer', 'parse_args', 'ParallelNet' ]

####################################################################################
####################################################################################
def imp_gru_cell_fast(dt : float, x_red_r : torch.Tensor, x_red_z : torch.Tensor, x_red_n : torch.Tensor, 
                      h : torch.Tensor, lin_rh_W : torch.Tensor, lin_zh_W : torch.Tensor,
                      lin_nr_W : torch.Tensor, lin_nr_b : torch.Tensor) -> torch.Tensor:

  r   =    torch.sigmoid(x_red_r +     F.linear(h,lin_rh_W))
  n   =    torch.   tanh(x_red_n + r * F.linear(h,lin_nr_W, lin_nr_b))
  dtz = dt*torch.sigmoid(x_red_z +     F.linear(h,lin_zh_W))

  return torch.div(torch.addcmul(h,dtz,n),1.0+dtz)

def imp_gru_cell(dt : float, x : torch.Tensor, h : torch.Tensor,
                 lin_rx_W : torch.Tensor, lin_rx_b : torch.Tensor, lin_rh_W : torch.Tensor,
                 lin_zx_W : torch.Tensor, lin_zx_b : torch.Tensor, lin_zh_W : torch.Tensor,
                 lin_nx_W : torch.Tensor, lin_nx_b : torch.Tensor, lin_nr_W : torch.Tensor, 
                 lin_nr_b : torch.Tensor) -> torch.Tensor:

  r   =      torch.sigmoid(F.linear(x, lin_rx_W, lin_rx_b) +     F.linear(h, lin_rh_W))
  n   =      torch.   tanh(F.linear(x, lin_nx_W, lin_nx_b) + r * F.linear(h, lin_nr_W,lin_nr_b))
  dtz = dt * torch.sigmoid(F.linear(x, lin_zx_W, lin_zx_b) +     F.linear(h, lin_zh_W))

  return torch.div(torch.addcmul(h, dtz, n), 1.0 + dtz)


class ImplicitGRUBlock(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(ImplicitGRUBlock, self).__init__()

    #

    self.lin_rx = [None,None]
    self.lin_rh = [None,None]
    self.lin_rx[0] = nn.Linear(input_size, hidden_size, True)
    self.lin_rh[0] = nn.Linear(hidden_size, hidden_size, False)

    self.lin_zx = [None,None]
    self.lin_zh = [None,None]
    self.lin_zx[0] = nn.Linear(input_size, hidden_size, True)
    self.lin_zh[0] = nn.Linear(hidden_size, hidden_size, False)

    self.lin_nx = [None,None]
    self.lin_nr = [None,None]
    self.lin_nx[0] = nn.Linear(input_size, hidden_size, True)
    self.lin_nr[0] = nn.Linear(hidden_size, hidden_size, True)

    #

    self.lin_rx[1] = nn.Linear(hidden_size, hidden_size, True)
    self.lin_rh[1] = nn.Linear(hidden_size, hidden_size, False)

    self.lin_zx[1] = nn.Linear(hidden_size, hidden_size, True)
    self.lin_zh[1] = nn.Linear(hidden_size, hidden_size, False)

    self.lin_nx[1] = nn.Linear(hidden_size, hidden_size, True)
    self.lin_nr[1] = nn.Linear(hidden_size, hidden_size, True)

    # record the layers so that they are handled by backprop correctly
    layers =  self.lin_rx + self.lin_rh + \
              self.lin_zx + self.lin_zh + \
              self.lin_nx + self.lin_nr
    self.lin_layers = nn.ModuleList(layers)

  def reduceX(self, x):
    x_red_r = self.lin_rx[0](x)
    x_red_z = self.lin_zx[0](x)
    x_red_n = self.lin_nx[0](x)

    return (x_red_r, x_red_z, x_red_n)

  def fastForward(self, level, tstart, tstop, x_red, h_prev):
    dt = tstop-tstart

    h_prev = h_prev[0]
    h0 = imp_gru_cell_fast(dt, *x_red, h_prev[0],
                           self.lin_rh[0].weight,
                           self.lin_zh[0].weight,
                           self.lin_nr[0].weight, self.lin_nr[0].bias)
    h1 = imp_gru_cell(dt, h0, h_prev[1],
                      self.lin_rx[1].weight, self.lin_rx[1].bias, self.lin_rh[1].weight,
                      self.lin_zx[1].weight, self.lin_zx[1].bias, self.lin_zh[1].weight,
                      self.lin_nx[1].weight, self.lin_nx[1].bias, self.lin_nr[1].weight, self.lin_nr[1].bias)

    # Note: we return a tuple with a single element
    return (torch.stack((h0, h1)), )


  def forward(self, level, tstart, tstop, x, h_prev):
    dt = tstop-tstart

    h_prev = h_prev[0]
    h0 = imp_gru_cell(dt, x, h_prev[0],
                      self.lin_rx[0].weight, self.lin_rx[0].bias, self.lin_rh[0].weight,
                      self.lin_zx[0].weight, self.lin_zx[0].bias, self.lin_zh[0].weight,
                      self.lin_nx[0].weight, self.lin_nx[0].bias, self.lin_nr[0].weight, self.lin_nr[0].bias)
    h1 = imp_gru_cell(dt, h0, h_prev[1],
                      self.lin_rx[1].weight, self.lin_rx[1].bias, self.lin_rh[1].weight,
                      self.lin_zx[1].weight, self.lin_zx[1].bias, self.lin_zh[1].weight,
                      self.lin_nx[1].weight, self.lin_nx[1].bias, self.lin_nr[1].weight, self.lin_nr[1].bias)

    # Note: we return a tuple with a single element
    return (torch.stack((h0, h1)), )

class CloseLayer(nn.Module):
  def __init__(self, hidden_size, num_classes):
    super(CloseLayer, self).__init__()
    self.fc = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    x = self.fc(x)
    return x

####################################################################################
####################################################################################
# Parallel network class, primarily builds a RNN_Parallel network object
# num_layers: number of layers per processor
# for definitions of layer-parallel (and other parameters) see more advanced scripts and notebooks
class ParallelNet(nn.Module):
  def __init__(self,
               input_size=9,
               hidden_size=100,
               Tf=None,
               num_layers=2,
               num_classes=6,
               num_steps=32,
               max_levels=3,
               max_iters=1,
               fwd_max_iters=2,
               print_level=0,
               braid_print_level=0,
               cfactor=4,
               skip_downcycle=True):
    super(ParallelNet, self).__init__()

    self.RNN_model = ImplicitGRUBlock(input_size, hidden_size)

    if Tf==None:
      Tf = float(num_steps) * MPI.COMM_WORLD.Get_size() # when using an implicit method with GRU
    self.Tf = Tf
    self.dt = Tf / float(num_steps * MPI.COMM_WORLD.Get_size())

    self.parallel_rnn = torchbraid.RNN_Parallel(MPI.COMM_WORLD,
                                                self.RNN_model,
                                                num_steps,hidden_size,num_layers,
                                                Tf,
                                                max_fwd_levels=max_levels,
                                                max_bwd_levels=max_levels,
                                                max_iters=max_iters)

    if fwd_max_iters > 0:
      self.parallel_rnn.setFwdMaxIters(fwd_max_iters)

    self.parallel_rnn.setPrintLevel(print_level)

    cfactor_dict = dict()
    cfactor_dict[-1] = cfactor
    self.parallel_rnn.setCFactor(cfactor_dict)
    self.parallel_rnn.setSkipDowncycle(skip_downcycle)
    self.parallel_rnn.setNumRelax(1)            # FCF on all levels, by default
    self.parallel_rnn.setFwdNumRelax(1, level=0) # F-Relaxation on the fine grid (by default)
    self.parallel_rnn.setBwdNumRelax(0, level=0) # F-Relaxation on the fine grid (by default)

    # this object ensures that only the RNN_Parallel code runs on ranks!=0
    compose = self.compose = self.parallel_rnn.comp_op()

    self.close_rnn = compose(CloseLayer, hidden_size, num_classes)

    self.hidden_size = hidden_size
    self.num_layers = num_layers

  def forward(self, x):
    h = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    hn = self.parallel_rnn(x, h)

    x = self.compose(self.close_rnn,hn[-1,:,:])

    return x

####################################################################################
####################################################################################

# Parse command line
def parse_args():
  """
  Return back an args dictionary based on a standard parsing of the command line inputs
  """

  # Training settings
  parser = argparse.ArgumentParser(description='UCI-HAR example argument parser')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--percent-data', type=float, default=1.0, metavar='N',
                      help='how much of the data to read in and use for training/testing')

  # artichtectural settings
  parser.add_argument('--sequence-length', type=int, default=128, metavar='N',
                      help='number of time steps in each sequence (default: 128)')
  parser.add_argument('--tf',type=float,default=128.0,
                      help='Final time (default=128.0, to match sequence length)')
  parser.add_argument('--input-size', type=int, default=9, metavar='N',
                      help='size of input at each time step (default: 9)')
  parser.add_argument('--hidden-size', type=int, default=100, metavar='N',
                      help='number of units in each hidden layer (default: 100)')
  parser.add_argument('--num-layers', type=int, default=2, metavar='N',
                      help='number of hidden layers (default: 2)')
  parser.add_argument('--num-classes', type=int, default=6, metavar='N',
                      help='number of classes (default: 6)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='Disable CUDA training (default: False)')

  # algorithmic settings (gradient descent and batching)
  parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                      help='input batch size for training (default: 100)')
  parser.add_argument('--epochs', type=int, default=5, metavar='N',
                      help='number of epochs to train (default: 7)')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.001)')
  parser.add_argument('--warm-up',action='store_true', default=False,
                      help='Do a warm-up epoch, gives more consistent timings on GPUs (default: False)')

  # algorithmic settings (parallel or serial)
  parser.add_argument('--force-lp', action='store_true', default=False,
                      help='Use layer parallel even if there is only 1 MPI rank (default off)')
  parser.add_argument('--lp-max-levels', type=int, default=3, metavar='N',
                      help='Layer parallel levels (default: 3)')
  parser.add_argument('--lp-max-iters', type=int, default=1, metavar='N',
                      help='Layer parallel iterations (default: 1)')
  parser.add_argument('--lp-fwd-max-iters', type=int, default=2, metavar='N',
                      help='Layer parallel (forward) iterations (default: 4, value of -1 implies uses --lp-iters)')
  parser.add_argument('--lp-print-level', type=int, default=0, metavar='N',
                      help='Layer parallel internal print level (default: 0)')
  parser.add_argument('--lp-braid-print-level', type=int, default=0, metavar='N',
                      help='Layer parallel braid print level (default: 0)')
  parser.add_argument('--lp-cfactor', type=int, default=4, metavar='N',
                      help='Layer parallel coarsening factor (default: 4)')
  parser.add_argument('--lp-use-downcycle',action='store_true', default=False,
                      help='Layer parallel use downcycle on or off (default: False)')
  parser.add_argument('--lp-fwd-finerelax', type=int, default=1, metavar='N',
                      help='Forward fine relaxation (default: 1, F-relax)')
  parser.add_argument('--lp-fwd-relax', type=int, default=3, metavar='N',
                      help='Forward relaxation (default: 3, FCF-relax)')

  args = parser.parse_args()

  ##
  # Do some parameter checking
  rank  = MPI.COMM_WORLD.Get_rank()
  procs = MPI.COMM_WORLD.Get_size()
  args = parser.parse_args()

  # ATTENTION: Added a new variable num_steps (sequence_len should be defined before here)
  # In order to evenly distribute sub-sequences across the given processors,
  # the number of processors (procs) must be one of the factors of sequence length (e.g., when sequence length = 28 (MNIST), the feasible number of processors are 1, 2, 4, 7, 14 and 28)
  if args.sequence_length % procs != 0:
    root_print(rank,'The number of processors must be a factor of sequence length')
    sys.exit(0)


  ##
  # Compute number of parallel-in-time multigrid levels
  def compute_levels(num_steps, min_coarse_size, cfactor):
    from math import log, floor
    # Find L such that ( max_L min_coarse_size*cfactor**L <= num_steps)
    levels = floor(log(float(num_steps) / min_coarse_size, cfactor)) + 1

    if levels < 1:
      levels = 1
    return levels

  if args.lp_max_levels < 1:
    min_coarse_size = 3
    args.lp_max_levels = compute_levels(args.steps, min_coarse_size, args.lp_cfactor)

  return args


####################################################################################
####################################################################################
