"""
This file contains:
  - Generic multilevel solver.
    Based on PyAMG multilevel solver, https://github.com/pyamg/pyamg
    PyAMG is released under the MIT license.
  - MG/Opt implementations of the multilevel solver
"""


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


__all__ = [ 'parse_args', 'ParallelNet' ]

####################################################################################
####################################################################################
# Classes and functions that define the basic network types for MG/Opt and TorchBraid.

class OpenLayer(nn.Module):
  ''' Opening layer (not ODE-net, not parallelized in time) '''
  def __init__(self,channels):
    super(OpenLayer, self).__init__()
    self.channels = channels
    self.pre = nn.Sequential(
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )


  def forward(self, x):
    # this bit of python magic simply replicates each image in the batch
    s = len(x.shape)*[1]
    s[1] = self.channels
    #x = self.pre(x)    # Comment out to turn off open-layer pooling
    x = x.repeat(s)
    return x
# end layer


class CloseLayer(nn.Module):
  ''' Closing layer (not ODE-net, not parallelized in time) '''
  def __init__(self,channels):
    super(CloseLayer, self).__init__()
    self.avg = nn.AvgPool2d(2)
    self.fc = nn.Linear(channels*28*28, 10)  # coarsen in total by 2, when pooling is used

  def forward(self, x):
    #x = self.avg(x)    # Comment out to turn off close-layer pooling
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x
# end layer


class StepLayer(nn.Module):
  ''' Single ODE-net layer will be parallelized in time ''' 
  def __init__(self,channels):
    super(StepLayer, self).__init__()
    ker_width = 3
    self.conv = nn.Conv2d(channels,channels,ker_width,padding=1)

  def forward(self, x):
    y = self.conv(x)
    y.tanh_()
    return y 
# end layer


class ParallelNet(nn.Module):
  ''' Full parallel ODE-net based on StepLayer,  will be parallelized in time ''' 
  def __init__(self, channels=8, global_steps=8, Tf=1.0, max_fwd_levels=1, max_bwd_levels=1, max_iters=1, max_fwd_iters=0, 
                     print_level=0, braid_print_level=0, fwd_cfactor=4, bwd_cfactor=4, fine_fwd_fcf=False, 
                     fine_bwd_fcf=False, fwd_nrelax=1, bwd_nrelax=1, skip_downcycle=True, fmg=False, fwd_relax_only_cg=0, 
                     bwd_relax_only_cg=0, CWt=1.0, fwd_finalrelax=False, user_mpi_buf=False):
    super(ParallelNet, self).__init__()

    step_layer = lambda: StepLayer(channels)

    self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD,step_layer,global_steps,Tf,max_fwd_levels=max_fwd_levels,max_bwd_levels=max_bwd_levels,max_iters=max_iters,user_mpi_buf=user_mpi_buf)

    if max_fwd_iters>0:
      self.parallel_nn.setFwdMaxIters(max_fwd_iters)
    self.parallel_nn.setPrintLevel(print_level,True)
    #self.parallel_nn.setPrintLevel(2,False)
    self.parallel_nn.setPrintLevel(braid_print_level,False)
    self.parallel_nn.setFwdCFactor(fwd_cfactor)
    self.parallel_nn.setBwdCFactor(bwd_cfactor)
    self.parallel_nn.setSkipDowncycle(skip_downcycle)
    self.parallel_nn.setBwdRelaxOnlyCG(bwd_relax_only_cg)
    self.parallel_nn.setFwdRelaxOnlyCG(fwd_relax_only_cg)
    self.parallel_nn.setCRelaxWt(CWt)
    self.parallel_nn.setMinCoarse(2)

    if fwd_finalrelax:
        self.parallel_nn.setFwdFinalFCRelax()

    if fmg:
      self.parallel_nn.setFMG()

    self.parallel_nn.setFwdNumRelax(fwd_nrelax)  # Set relaxation iters for forward solve 
    self.parallel_nn.setBwdNumRelax(bwd_nrelax)  # Set relaxation iters for backward solve
    if not fine_fwd_fcf:
      self.parallel_nn.setFwdNumRelax(0,level=0) # F-Relaxation on the fine grid for forward solve
    else:
      self.parallel_nn.setFwdNumRelax(1,level=0) # FCF-Relaxation on the fine grid for forward solve
    if not fine_bwd_fcf:
      self.parallel_nn.setBwdNumRelax(0,level=0) # F-Relaxation on the fine grid for backward solve
    else:
      self.parallel_nn.setBwdNumRelax(1,level=0) # FCF-Relaxation on the fine grid for backward solve

    # this object ensures that only the LayerParallel code runs on ranks!=0
    compose = self.compose = self.parallel_nn.comp_op()
    
    # by passing this through 'compose' (mean composition: e.g. OpenLayer o channels) 
    # on processors not equal to 0, these will be None (there are no parameters to train there)
    self.open_nn = compose(OpenLayer,channels)
    self.close_nn = compose(CloseLayer,channels)

  def forward(self, x):
    # by passing this through 'o' (mean composition: e.g. self.open_nn o x) 
    # this makes sure this is run on only processor 0

    x = self.compose(self.open_nn,x)
    x = self.parallel_nn(x)
    x = self.compose(self.close_nn,x)

    return x
# end ParallelNet 
####################################################################################
####################################################################################



####################################################################################
####################################################################################
# Parsing functions

def parse_args():
  """
  Return back an args dictionary based on a standard parsing of the command line inputs
  """
  
  # Command line settings
  parser = argparse.ArgumentParser(description='MG/Opt Solver Parameters')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--use-fashion',action='store_true', default=False, 
                      help='Enables the use of Fashion MNIST if on, and Digit MNIST otherwise')
  
  # artichtectural settings
  parser.add_argument('--steps', type=int, default=4, metavar='N',
                      help='Number of times steps in the resnet layer (default: 4)')
  parser.add_argument('--channels', type=int, default=8, metavar='N',
                      help='Number of channels in resnet layer (default: 8)')
  parser.add_argument('--tf',type=float,default=1.0,
                      help='Final time')

  # algorithmic settings (gradient descent and batching)
  parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                      help='input batch size for training (default: 50)')
  parser.add_argument('--NIepochs', type=int, default=2, metavar='N',
                      help='number of epochs per Nested Iteration (default: 2)')
  parser.add_argument('--epochs', type=int, default=2, metavar='N',
                      help='number of epochs to train (default: 2)')
  parser.add_argument('--samp-ratio', type=float, default=1.0, metavar='N',
                      help='number of samples as a ratio of the total number of samples')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')
  parser.add_argument('--zero-init-guess', type=int, default=0, metavar='N',
                      help='Zero initial guess for all parameters when starting NI (parallel reproducibility), 0:False, 1:True (default: 0)')
  parser.add_argument('--preserve-optim', type=int, default=1, metavar='N',
                      help='Preserve optimizer state between MGOpt V-cycles, 0:False, 1:True (default: 1)')
  
  # algorithmic settings (parallel or serial)
  parser.add_argument('--lp-fwd-levels', type=int, default=3, metavar='N',
                      help='Layer parallel levels for forward solve (default: 3)')
  parser.add_argument('--lp-bwd-levels', type=int, default=3, metavar='N',
                      help='Layer parallel levels for backward solve (default: 3)')
  parser.add_argument('--lp-iters', type=int, default=2, metavar='N',
                      help='Layer parallel iterations (default: 2)')
  parser.add_argument('--lp-fwd-iters', type=int, default=-1, metavar='N',
                      help='Layer parallel (forward) iterations (default: -1, default --lp-iters)')
  parser.add_argument('--lp-print', type=int, default=0, metavar='N',
                      help='Layer parallel internal print level (default: 0)')
  parser.add_argument('--lp-braid-print', type=int, default=0, metavar='N',
                      help='Layer parallel braid print level (default: 0)')
  parser.add_argument('--lp-fwd-cfactor', type=int, default=4, metavar='N',
                      help='Layer parallel coarsening factor for forward solve (default: 4)')
  parser.add_argument('--lp-bwd-cfactor', type=int, default=4, metavar='N',
                      help='Layer parallel coarsening factor for backward solve (default: 4)')
  parser.add_argument('--lp-fwd-nrelax-coarse', type=int, default=1, metavar='N',
                      help='Layer parallel relaxation steps on coarse grids for forward solve (default: 1)')
  parser.add_argument('--lp-bwd-nrelax-coarse', type=int, default=1, metavar='N',
                      help='Layer parallel relaxation steps on coarse grids for backward solve (default: 1)')
  parser.add_argument('--lp-fwd-finefcf',action='store_true', default=False, 
                      help='Layer parallel fine FCF for forward solve, on or off (default: False)')
  parser.add_argument('--lp-bwd-finefcf',action='store_true', default=False, 
                      help='Layer parallel fine FCF for backward solve, on or off (default: False)')
  parser.add_argument('--lp-fwd-finalrelax',action='store_true', default=False, 
                      help='Layer parallel do final FC relax after forward cycle ends (always on for backward). (default: False)')
  parser.add_argument('--lp-use-downcycle',action='store_true', default=False, 
                      help='Layer parallel use downcycle on or off (default: False)')
  parser.add_argument('--lp-use-fmg',action='store_true', default=False, 
                      help='Layer parallel use FMG for one cycle (default: False)')
  parser.add_argument('--lp-bwd-relaxonlycg',action='store_true', default=0, 
                      help='Layer parallel use relaxation only on coarse grid for backward cycle (default: False)')
  parser.add_argument('--lp-fwd-relaxonlycg',action='store_true', default=0, 
                      help='Layer parallel use relaxation only on coarse grid for forward cycle (default: False)')
  parser.add_argument('--lp-use-crelax-wt', type=float, default=1.0, metavar='CWt',
                      help='Layer parallel use weighted C-relaxation on backwards solve (default: 1.0).  Not used for coarsest braid level.')
  parser.add_argument('--lp-user-mpi-buf',action='store_true', default=False, 
                      help='Layer parallel use user-defined mpi buffers (default: False)')

  # algorithmic settings (nested iteration)
  parser.add_argument('--ni-levels', type=int, default=3, metavar='N',
                      help='Number of nested iteration levels (default: 3)')
  parser.add_argument('--ni-rfactor', type=int, default=2, metavar='N',
                      help='Refinment factor for nested iteration (default: 2)')
  
  # algorithmic settings (MG/Opt)
  parser.add_argument('--mgopt-printlevel', type=int, default=1, metavar='N',
                      help='Print level for MG/Opt, 0 least, 1 some, 2 a lot') 
  parser.add_argument('--mgopt-iter', type=int, default=1, metavar='N',
                      help='Number of MG/Opt iterations to optimize over a batch')
  parser.add_argument('--mgopt-levels', type=int, default=2, metavar='N',
                      help='Number of MG/Opt levels to use')
  parser.add_argument('--mgopt-nrelax-pre', type=int, default=1, metavar='N',
                      help='Number of MG/Opt pre-relaxations on each level')
  parser.add_argument('--mgopt-nrelax-post', type=int, default=1, metavar='N',
                      help='Number of MG/Opt post-relaxations on each level')
  parser.add_argument('--mgopt-nrelax-coarse', type=int, default=3, metavar='N',
                      help='Number of MG/Opt relaxations to solve the coarsest grid')


  ##
  # Do some parameter checking
  rank  = MPI.COMM_WORLD.Get_rank()
  procs = MPI.COMM_WORLD.Get_size()
  args = parser.parse_args()
  
  if args.lp_bwd_levels==-1:
    min_coarse_size = 3
    args.lp_bwd_levels = compute_levels(args.steps, min_coarse_size, args.lp_bwd_cfactor)

  if args.lp_fwd_levels==-1:
    min_coarse_size = 3
    args.lp_fwd_levels = compute_levels(args.steps,min_coarse_size,args.lp_fwd_cfactor)

  if args.steps % procs!=0:
    root_print(rank, 1, 1, 'Steps must be an even multiple of the number of processors: %d %d' % (args.steps,procs) )
    sys.exit(0)
  
  ni_levels = args.ni_levels
  ni_rfactor = args.ni_rfactor
  if args.steps % ni_rfactor**(ni_levels-1) != 0:
    root_print(rank, 1, 1, 'Steps combined with the coarsest nested iteration level must be an even multiple: %d %d %d' % (args.steps,ni_rfactor,ni_levels-1))
    sys.exit(0)
  
  if args.steps / ni_rfactor**(ni_levels-1) % procs != 0:
    root_print(rank, 1, 1, 'Coarsest nested iteration must fit on the number of processors')
    sys.exit(0)

  return args


####################################################################################
####################################################################################
