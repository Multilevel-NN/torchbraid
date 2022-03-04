#@HEADER
# ************************************************************************
# 
#                        Torchbraid v. 0.1
# 
# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC 
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. 
# Government retains certain rights in this software.
# 
# Torchbraid is licensed under 3-clause BSD terms of use:
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name National Technology & Engineering Solutions of Sandia, 
# LLC nor the names of the contributors may be used to endorse or promote 
# products derived from this software without specific prior written permission.
# 
# Questions? Contact Eric C. Cyr (eccyr@sandia.gov)
# 
# ************************************************************************
#@HEADER

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import sys
import numpy as np
import statistics as stats

import torchbraid

import faulthandler
faulthandler.enable()

from mpi4py import MPI

def root_print(rank,s):
  if rank==0:
    print(s)

class ConvBlock(nn.Module):
  def __init__(self,dim,num_ch):
    super(ConvBlock, self).__init__()
    self.lin = nn.Conv1d(num_ch,num_ch,kernel_size=3,padding=1,bias=False)

  def forward(self, x):
    return F.relu(self.lin(x))
# end layer

class TestLayerParallel_MultiNODE(unittest.TestCase):

  def test_NoMGRIT(self):
    tolerance = 1e-15
    Tf = 4.0
    max_levels = 1
    max_iters = 1

    self.forwardBackwardProp(tolerance,Tf,max_levels,max_iters)

  def test_MGRIT(self):
    tolerance = 1e-7
    Tf = 8.0
    max_levels = 2
    max_iters = 5

    self.forwardBackwardProp(tolerance,Tf,max_levels,max_iters,print_level=0)

  def forwardBackwardProp(self,tolerance,Tf,max_levels,max_iters,print_level=0):
    rank = MPI.COMM_WORLD.Get_rank()

    root_print(rank,'\n')
    root_print(rank,self.id())

    dim = 10
    num_ch = 3
    num_samp = 8

    conv_block = lambda: ConvBlock(dim,num_ch)
    #bn_block = lambda: nn.MaxPool1d(3)
    bn_block = lambda: nn.BatchNorm1d(num_ch)

    basic_block = [conv_block,bn_block,conv_block]
    num_steps   = [        4,         1,         3]

    # this is the torchbraid class being tested 
    #######################################
    parallel_net = torchbraid.LayerParallel(MPI.COMM_WORLD,basic_block,num_steps,Tf,max_levels=max_levels,max_iters=max_iters)
    parallel_net.setPrintLevel(print_level)
    parallel_net.setSkipDowncycle(False)

    # this is the reference torch "solution"
    #######################################
    serial_net = parallel_net.buildSequentialOnRoot()

    # run forward/backward propgation
    xs = torch.rand(num_samp,num_ch,dim) # forward initial cond
    xs.requireds_grad = False

    if rank==0:
      ys = serial_net(xs)

    # propagation with torchbraid
    #######################################
    xp = xs.clone()
    xp.requires_grad = False

    yp = parallel_net(xp)

    yp_root = parallel_net.getFinalOnRoot(yp)

    if rank==0:
      # check error
      forward_error = torch.norm(ys-yp_root)/torch.norm(ys)
      root_print(rank,f'Forward Error: {forward_error}')
      self.assertLessEqual(forward_error,tolerance,
                           "Relative error in the forward proppgation, serial to parallel comparison.")

  # forwardPropSerial

if __name__ == '__main__':
  unittest.main()
