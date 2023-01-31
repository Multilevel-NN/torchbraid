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


__all__ = [ 'OpenFlatLayer', 'CloseLayer', 'StepLayer', 'parse_args', 'ParallelNet' ]

####################################################################################
####################################################################################
# Network architecture is Open + ResNet + Close
# The StepLayer defines the ResNet (ODENet)

class OpenFlatLayer(nn.Module):
  ''' 
  Opening layer has no parameters, replicates image number of channels times
  '''
  def __init__(self, channels):
    super(OpenFlatLayer, self).__init__()
    self.channels = channels

  def forward(self, x):
    s = len(x.shape) * [1]
    s[1] = self.channels
    x = x.repeat(s)
    return x

class CloseLayer(nn.Module):
  '''
  Dense closing classification layer
  '''
  def __init__(self, channels):
    super(CloseLayer, self).__init__()
    self.fc1 = nn.Linear(channels * 28 * 28, 32)
    self.fc2 = nn.Linear(32, 10)

  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


class StepLayer(nn.Module):
  '''
  ResNet composed of convolutional layers
  '''
  def __init__(self, channels):
    super(StepLayer, self).__init__()
    ker_width = 3
    self.conv1 = nn.Conv2d(channels, channels, ker_width, padding=1)
    self.conv2 = nn.Conv2d(channels, channels, ker_width, padding=1)

  def forward(self, x):
    return F.relu(self.conv2(F.relu(self.conv1(x))))


####################################################################################
####################################################################################

# Parallel network class
# local_steps: number of ResNet layers per processor
# all other parameter definitions are in argument parser comments below
class ParallelNet(nn.Module):
  def __init__(self, channels=12, local_steps=8, Tf=1.0, max_levels=1, bwd_max_iters=1,
               fwd_max_iters=2, print_level=0, braid_print_level=0, cfactor=4,
               fine_fcf=False, skip_downcycle=True, fmg=False, relax_only_cg=0,
               user_mpi_buf=False, comm_lp=MPI.COMM_WORLD):
    super(ParallelNet, self).__init__()

    step_layer = lambda: StepLayer(channels)
    self.comm_lp = comm_lp
    numprocs = self.comm_lp.Get_size()

    self.parallel_nn = torchbraid.LayerParallel(comm_lp, step_layer, local_steps*numprocs, Tf,
                                                max_fwd_levels=max_levels, max_bwd_levels=max_levels,
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

    # by passing this through 'compose' (mean composition: e.g. OpenFlatLayer o channels)
    # on processors not equal to 0, these will be None (there are no parameters to train there)
    self.open_nn = compose(OpenFlatLayer, channels)
    self.close_nn = compose(CloseLayer, channels)

  def saveSerialNet(self, name):
    # Model can be reloaded in serial format with: model = torch.load(filename)
    serial_nn = self.parallel_nn.buildSequentialOnRoot()
    if self.comm_lp.Get_rank() == 0:
      s_net = SerialNet(-1, -1, -1, serial_nn=serial_nn, open_nn=self.open_nn, close_nn=self.close_nn)
      s_net.eval()
      torch.save(s_net, name)

  def forward(self, x):
    # by passing this through 'o' (mean composition: e.g. self.open_nn o x)
    # this makes sure this is run on only processor 0

    x = self.compose(self.open_nn, x)
    x = self.parallel_nn(x)
    x = self.compose(self.close_nn, x)

    return x

# Serial Network Class (used by the saveSerialNet functionality in ParallelNet)
class SerialNet(nn.Module):
  def __init__(self, channels=12, local_steps=8, Tf=1.0, serial_nn=None, open_nn=None, close_nn=None):
    super(SerialNet, self).__init__()

    if open_nn is None:
      self.open_nn = OpenFlatLayer(channels)
    else:
      self.open_nn = open_nn

    if serial_nn is None:
      step_layer = lambda: StepLayer(channels)
      numprocs = 1
      parallel_nn = torchbraid.LayerParallel(MPI.COMM_SELF, step_layer, numprocs * local_steps, Tf,
                                             max_fwd_levels=1, max_bwd_levels=1, max_iters=1)
      parallel_nn.setPrintLevel(0, True)
      self.serial_nn = parallel_nn.buildSequentialOnRoot()
    else:
      self.serial_nn = serial_nn

    if close_nn is None:
      self.close_nn = CloseLayer(channels)
    else:
      self.close_nn = close_nn

  def forward(self, x):
    x = self.open_nn(x)
    x = self.serial_nn(x)
    x = self.close_nn(x)
    return x


####################################################################################
####################################################################################

# Parse command line 
def parse_args():
  """
  Return back an args dictionary based on a standard parsing of the command line inputs
  """

  # Command line settings
  parser = argparse.ArgumentParser(description='MNIST example argument parser')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--dataset', default='digits',
                      help='if digits use digits MNIST, if fashion use fashion MNIST')

  # artichtectural settings
  parser.add_argument('--steps', type=int, default=32, metavar='N',
                      help='Number of times steps in the resnet layer (default: 32)')
  parser.add_argument('--channels', type=int, default=3, metavar='N',
                      help='Number of channels in resnet layer (default: 4)')
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
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')

  # algorithmic settings (layer-parallel)
  parser.add_argument('--lp-max-levels', type=int, default=3, metavar='N',
                      help='Layer parallel max number of levels (default: 3)')
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

  if args.lp_max_levels < 1:
    min_coarse_size = 3
    args.lp_max_levels = compute_levels(args.steps, min_coarse_size, args.lp_cfactor)

  if args.steps % procs_lp != 0:
    root_print(rank, 1, 1, 'Steps must be an even multiple of the number of layer parallel processors: %d %d'
               % (args.steps, procs_lp) )
    sys.exit(0)

  return args


####################################################################################
####################################################################################
