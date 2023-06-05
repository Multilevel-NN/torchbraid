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

from torchbraid.gru_braid_function import BraidFunction
from torchbraid.utils import ContextTimerManager

import torchbraid.gru_apps as apps
from torchbraid.lp_module import LPModule

import numpy as np

class GRU_Serial(nn.Module):
  """
  Helper class to build a serial GRU from the parallel version.
  This makes comparison to the serial version easier
  """
  def __init__(self,GRU_model,num_layers,hidden_size,dt=1.0):
    super(GRU_Serial,self).__init__()
    self.num_layers  = num_layers
    self.hidden_size = hidden_size
    self.dt          = dt

    self.GRU_model = GRU_model
  # end __init__

  def forward(self,x,h=None):
    if h is None:
      h = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

    if isinstance(h, torch.Tensor):
      h = (h,)
        
    num_steps = x.shape[1]
    for i in range(num_steps):
      h = self.GRU_model(0,0.0,self.dt,x[:,i,:],h)

    return h[0]
# end Serial

class GRU_Parallel(LPModule):

  ##################################################

  def __init__(self,comm,basic_block,num_steps,hidden_size,num_layers,Tf,max_fwd_levels=1,max_bwd_levels=1,max_iters=10):
    super().__init__(comm)

    self.num_layers  = num_layers
    self.hidden_size = hidden_size
    self.GRU_models  = basic_block

    # gru_apps.py -> ForwardBraidApp
    self.fwd_app = apps.ForwardBraidApp(comm,self.GRU_models,num_steps,Tf,max_fwd_levels,max_iters,self.timer_manager)
    self.bwd_app = apps.BackwardBraidApp(self.fwd_app,self.timer_manager)
  # end __init__

  def zero_grad(self):
    self.GRU_models.zero_grad()

  def getFastForwardInfo(self):
    return self.fwd_app.getFastForwardInfo()

  def forward(self,x,h=None):
    # we are doing this to take adavtage of
    # pytorch's autograd which functions "naturally"
    # with the torch.autograd.function

    if h is None:
      h = torch.zeros(self.num_layers, x.size(0), self.hidden_size,device=x.device)

    params = list(self.parameters())
    return BraidFunction.apply(self.fwd_app,self.bwd_app,x,h,*params)
  # end forward

# end GRU_Parallel
