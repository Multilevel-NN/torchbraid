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

import inspect

import torch
import torch.nn as nn

from mpi4py import MPI

import copy

from torchbraid.rnn_braid_function import BraidFunction
from torchbraid.utils import ContextTimerManager

import torchbraid.rnn_apps as apps
from torchbraid.lp_module import LPModule

import numpy as np

class RNN_Serial(nn.Module):
  """
  Helper class to build a serial RNN from the parallel version.
  This makes comparison to the serial version easier
  """
  def __init__(self,RNN_model,num_layers,hidden_size,dt=1.0):
    super(RNN_Serial,self).__init__()
    self.num_layers  = num_layers
    self.hidden_size = hidden_size
    self.dt          = dt

    self.RNN_model = RNN_model
  # end __init__

  def forward(self,x,h_c=None):
    if h_c is None:
      h = torch.zeros(self.num_layers, x.size(0), self.hidden_size, x.device)
      c = torch.zeros(self.num_layers, x.size(0), self.hidden_size, x.device)
      h_c = (h,c)
    elif isinstance(h_c,torch.Tensor):
      h_c = (h_c,)

    num_steps = x.shape[1]
    for i in range(num_steps):
      h_c = self.RNN_model(0,0.0,self.dt,x[:,i,:],h_c)

    if len(h_c)==1:
      return h_c[0]
    return h_c
# end RNN_Serial

class RNN_Parallel(LPModule):

  ##################################################

  def __init__(self,comm,basic_block,num_steps,hidden_size,num_layers,Tf,max_fwd_levels=1,max_bwd_levels=1,max_iters=10):
    super().__init__(comm)

    self.num_layers  = num_layers
    self.hidden_size = hidden_size
    self.RNN_models  = basic_block

    # RNN_torchbraid_apps.py -> ForwardBraidApp
    self.fwd_app = apps.ForwardBraidApp(comm,self.RNN_models,num_steps,Tf,max_fwd_levels,max_iters,self.timer_manager)
    self.bwd_app = apps.BackwardBraidApp(self.fwd_app,self.timer_manager)
  # end __init__

  def zero_grad(self):
    self.RNN_models.zero_grad()

  def getFastForwardInfo(self):
    return self.fwd_app.getFastForwardInfo()

  def forward(self,x,h_c=None):
    # we are doing this to take adavtage of
    # pytorch's autograd which functions "naturally"
    # with the torch.autograd.function

    if h_c is None:
      h = torch.zeros(self.num_layers, x.size(0), self.hidden_size,device=x.device)
      c = torch.zeros(self.num_layers, x.size(0), self.hidden_size,device=x.device)
      h_c = (h,c)

    params = list(self.parameters())
    if isinstance(h_c, torch.Tensor):
      return BraidFunction.apply(self.fwd_app,self.bwd_app,1,x,h_c,*params)
    else:
      return BraidFunction.apply(self.fwd_app,self.bwd_app,len(h_c),x,*h_c,*params)
  # end forward

  def buildInit(self,t):
    # prefix_rank  = self.comm.Get_rank()
    # print("Rank %d RNN_Parallel -> buildInit() - start" % prefix_rank)

    g = self.g0.clone()
    if t>0:
      t_h,t_c = g.tensors()
      t_h[:] = 0.0
      t_c[:] = 0.0

    # print("Rank %d RNN_Parallel -> buildInit() - end" % prefix_rank)
    return g

# end RNN_Parallel
