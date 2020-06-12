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

# cython: profile=True
# cython: linetrace=True

import torch
import torch.nn as nn

from mpi4py import MPI

import copy

from torchbraid.torchbraid_function import BraidFunction
from torchbraid.utils import ContextTimerManager

import torchbraid.torchbraid_apps as apps

##
# Define your Python Braid Vector

#  a python level module
##########################################################

class ODEBlock(nn.Module):
  def __init__(self,layer,dt):
    super(ODEBlock, self).__init__()

    self.dt = dt
    self.layer = layer

  def forward(self, x):
    return x + self.dt*self.layer(x)
# end ODEBlock

class LayerParallel(nn.Module):

  def __init__(self,comm,layer_block,num_steps,Tf,max_levels=1,max_iters=10):
    super(LayerParallel,self).__init__()

    self.comm = comm

    # optional parameters
    global_steps = num_steps*comm.Get_size()

    self.dt = Tf/global_steps
  
    self.layer_block = layer_block
    self.layer_models = [layer_block() for i in range(num_steps)]
    self.local_layers = nn.Sequential(*self.layer_models)

    self.timer_manager = ContextTimerManager()

    self.fwd_app = apps.ForwardBraidApp(comm,self.layer_models,num_steps,Tf,max_levels,max_iters,self.timer_manager)
    self.bwd_app = apps.BackwardBraidApp(self.fwd_app,self.timer_manager)

    self.param_size = 0
  # end __init__

  def zero_grad(self):
    for l in self.fwd_app.layer_models:
      l.zero_grad()
    self.local_layers.zero_grad()

  def getTimerManager(self):
    """
    Get a TimerContextManager that describes how much time is taken by what.
    """
    return self.timer_manager

  def setPrintLevel(self,print_level):
    self.fwd_app.setPrintLevel(print_level)
    self.bwd_app.setPrintLevel(print_level)

  def setNumRelax(self,relax,level=-1):
    self.fwd_app.setNumRelax(relax,level=level)
    self.bwd_app.setNumRelax(relax,level=level)

  def setCFactor(self,cfactor):
    self.fwd_app.setCFactor(cfactor)
    self.bwd_app.setCFactor(cfactor)

  def setSkipDowncycle(self,skip):
    self.fwd_app.setSkipDowncycle(skip)
    self.bwd_app.setSkipDowncycle(skip)

  def getMPIData(self):
    return self.fwd_app.getMPIData()

  def forward(self,x):
    # we are doing this to take adavtage of
    # pytorch's autograd which functions "naturally"
    # with the torch.autograd.function
    params = list(self.parameters())
    return BraidFunction.apply(self.fwd_app,self.bwd_app,x,*params) 
  # end forward

  def buildInit(self,t):
    x = self.x0.clone()
    if t>0:
      t_x = x.tensor()
      t_x[:] = 0.0
    return x

  # This method copies the layerr parameters and can be used for verification
  def buildSequentialOnRoot(self):
    ode_layers    = [ODEBlock(copy.deepcopy(l),self.dt) for l in self.layer_models]
    remote_layers = ode_layers
    build_seq_tag = 12         # this 
    comm          = self.getMPIData().getComm()
    my_rank       = self.getMPIData().getRank()
    num_ranks     = self.getMPIData().getSize()

    # short circuit for serial case
    if num_ranks==1:
      return nn.Sequential(*remote_layers)

    if my_rank==0:
      for i in range(1,self.getMPIData().getSize()):
        remote_layers += comm.recv(source=i,tag=build_seq_tag)
      return nn.Sequential(*remote_layers)
    else:
      comm.send(ode_layers,dest=0,tag=build_seq_tag)
      return None
  # end buildSequentialOnRoot

  def getFinal(self):
    return  self.fwd_app.getFinal()

  def getFinalOnRoot(self):
    build_seq_tag = 99        # this 
    comm          = self.getMPIData().getComm()
    my_rank       = self.getMPIData().getRank()
    num_ranks     = self.getMPIData().getSize()

    # short circuit for serial case
    if num_ranks==1:
      return self.getFinal()

    # send the output of the last layer to the root
    if my_rank==0:
      remote_final = comm.recv(source=num_ranks-1,tag=build_seq_tag)
      return remote_final
    elif my_rank==num_ranks-1:
      final = self.getFinal()
      comm.send(final,dest=0,tag=build_seq_tag)

    return None

  def copyVectorFromRoot(self,vec):
    build_seq_tag = 99        # this 
    comm          = self.getMPIData().getComm()
    my_rank       = self.getMPIData().getRank()
    num_ranks     = self.getMPIData().getSize()

    # short circuit for serial case
    if num_ranks==1:
      return vec

    # send the output of the last layer to the root
    if my_rank==0:
      for dest in range(1,num_ranks):
        comm.send(vec,dest,tag=build_seq_tag)
      return vec
    else:
      result = comm.recv(source=0,tag=build_seq_tag)
      return result

  def getTimersString(self):
    """
    Print the timers recored by the model.
    """
    comm     = self.comm
    my_rank  = self.comm.Get_rank() 
    num_proc = self.comm.Get_size() 

    local_result = self.timer_manager.getResultString()
    result = comm.gather(local_result,root=0)

    if my_rank==0:
      format_str = "\n   *** Proc = {rank:<8d} ***\n"

      result_str = ''
      for remote,s in zip(range(0,num_proc),result):
        result_str += format_str.format(rank=remote)
        result_str += s
      # for remote

      result = result_str
    # end if my_rank

    return result
  # end getTimersString

# end LayerParallel
