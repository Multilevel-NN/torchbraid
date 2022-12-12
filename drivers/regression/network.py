"""
This file contains:
   - the regression network
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

from timeit import default_timer as timer

from mpi4py import MPI

####################################################################################
####################################################################################
# Classes and functions that define the basic network types for MG/Opt and TorchBraid.

class OpenLayer(nn.Module):
  ''' Opening layer (not ODE-net, not parallelized in time) '''
  def __init__(self,input_size,width):
    super(OpenLayer, self).__init__()

    self.fc = nn.Linear(input_size,width,bias=True)

  def forward(self, x):
    # this bit of python magic simply replicates each image in the batch
    return torch.tanh(self.fc(x))
# end layer


class CloseLayer(nn.Module):
  ''' Closing layer (not ODE-net, not parallelized in time) '''
  def __init__(self,width,output_size):
    super(CloseLayer, self).__init__()

    self.fc = nn.Linear(width,output_size,bias=False)

  def forward(self, x):
    return self.fc(x)
# end layer


class StepLayer(nn.Module):
  ''' Single ODE-net layer will be parallelized in time ''' 
  def __init__(self,width):
    super(StepLayer, self).__init__()
    ker_width = 3
    self.lin = nn.Linear(width,width,bias=True)

  def forward(self, x):
    return torch.tanh(self.lin(x))
# end layer


class ParallelNet(nn.Module):
  ''' Full parallel ODE-net based on StepLayer,  will be parallelized in time ''' 
  def __init__(self, 
               # network architecture
               input_size,width,output_size, 
               local_steps=8, Tf=1.0,  
               # braid parameters
               max_fwd_levels=1, max_bwd_levels=1, max_iters=1, max_fwd_iters=0, 
               print_level=0, braid_print_level=0, 
               fwd_cfactor=4, bwd_cfactor=4, fine_fwd_fcf=False, fine_bwd_fcf=False, 
               fwd_nrelax=1, bwd_nrelax=1, skip_downcycle=True, fmg=False, fwd_relax_only_cg=0, 
               bwd_relax_only_cg=0, CWt=1.0, fwd_finalrelax=False):
    super(ParallelNet, self).__init__()

    step_layer = lambda: StepLayer(width)
    self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD,
                                                step_layer,
                                                local_steps,
                                                Tf,
                                                max_fwd_levels=max_fwd_levels,
                                                max_bwd_levels=max_bwd_levels,
                                                max_iters=max_iters)
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
    self.open_nn = compose(OpenLayer,input_size,width)
    self.close_nn = compose(CloseLayer,width,output_size)

  def forward(self, x):
    # by passing this through 'o' (mean composition: e.g. self.open_nn o x) 
    # this makes sure this is run on only processor 0

    x = self.compose(self.open_nn,x)
    x = self.parallel_nn(x)
    x = self.compose(self.close_nn,x)

    return x
# end ParallelNet 
