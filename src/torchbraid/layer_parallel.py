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

from torchbraid.braid_function import BraidFunction
from torchbraid.utils import ContextTimerManager

import torchbraid.odenet_apps as apps
from torchbraid.lp_module import LPModule

import numpy as np

##
# Define your Python Braid Vector

#  a python level module
##########################################################
class FixDTBlock(nn.Module):
  """Build a module that removes the dt from the forward evaluation
     this eases consistency with the PyTorch modus operandi
  """
  def __init__(self,layer,dt):
    super(FixDTBlock, self).__init__()

    self.dt = dt
    self.layer = layer

  def forward(self, x):
    return self.layer(self.dt,x)
# end FixDTBlock

class LayerParallel(LPModule):

  def __init__(self,comm,layer_blocks,global_steps,Tf,max_fwd_levels=1,max_bwd_levels=1,max_iters=10,spatial_ref_pair=None,user_mpi_buf=False, nsplines=0, splinedegree=1):
    """
    This takes a number of arguments to construct a layer parallel list.
    The big piece here is layer_block and global_steps. If layer_block is a functor then those
    layers will be constructed as needed. In this case global_steps is an integer that says how
    many time steps to take in an assume NODE.

    If layer_block and global_steps are lists (they must be the same length). Each functor
    in layer_block is paired with an integer number of global steps. The functor will be used
    that number of times to create the neural network layers. If the number of global_steps for
    a block is equal to 1, then that is treated as normal layer (not a NODE layer). For all
    blocks with global_steps greater than 1, the layer is treated as a NODE. Note that the time
    step used doesn't really care about this, and if you sum the global steps (say equal to N_total),
    then dt=Tf/N_total.
    """
    super().__init__(comm)

    # this allows layers to be defined in a varied way, with different numbers of repititions 
    global_steps = self.makeList(global_steps) # this conditional converts a single element to a list
    layer_blocks = self.makeList(layer_blocks)

    assert(len(global_steps)==len(layer_blocks)) # sanity check
    layers = zip(global_steps,layer_blocks)

    self.fwd_app = apps.ForwardODENetApp(comm,layers,Tf,max_fwd_levels,max_iters,self.timer_manager,
                                         spatial_ref_pair=spatial_ref_pair,user_mpi_buf=user_mpi_buf,
                                         nsplines=nsplines, splinedegree=splinedegree)
    self.bwd_app = apps.BackwardODENetApp(self.fwd_app,self.timer_manager,max_levels=max_bwd_levels)

    self.layer_models = [l for l in self.fwd_app.layer_models]
    self.local_layers = nn.Sequential(*self.layer_models)

    self.dt = self.fwd_app.dt

  # end __init__

  def makeList(self,data):
    """
    Conditionally convert a single element to a list, or return a list

    For instance: makeList(7) == [7], while makeList([7]) == [7]
    """
    if isinstance(data,list):
      return data
    return [data]

  def zero_grad(self):
    for l in self.fwd_app.layer_models:
      if l==None: continue
      l.zero_grad()
    self.local_layers.zero_grad()

  def setFwdStorage(self, storage):
    self.fwd_app.setStorage(storage)

  def setBwdStorage(self, storage):
    self.bwd_app.setStorage(storage)

  def setMinCoarse(self, mc):
    self.fwd_app.setMinCoarse(mc)
    self.bwd_app.setMinCoarse(mc)

  def setFMG(self):
    self.fwd_app.setFMG()
    self.bwd_app.setFMG()

  def setFwdFinalFCRelax(self):
    self.fwd_app.finalRelax()

  def setBwdFinalFCRelax(self):
    self.bwd_app.finalRelax()

  def setBwdRelaxOnlyCG(self, flag):
    self.bwd_app.setRelaxOnlyCG(flag)

  def setFwdRelaxOnlyCG(self, flag):
    self.fwd_app.setRelaxOnlyCG(flag)

  def setCRelaxWt(self, CWt):
    self.bwd_app.setCRelaxWt(CWt)
    #
    # Probably leave commented out for forward solve, more interested
    # in "mixing" adjoint data.
    #self.fwd_app.setCFactor(CWt)

  def forward(self,x):
    # we are doing this to take adavtage of
    # pytorch's autograd which functions "naturally"
    # with the torch.autograd.function
    params = list(self.parameters())

    if self.training:
      self.fwd_app.trainNetwork()
      self.bwd_app.trainNetwork() # for consistency, though the bwd_app should *only* be used in training
    else:
      self.fwd_app.evalNetwork()
      self.bwd_app.evalNetwork()

    return BraidFunction.apply(self.fwd_app,self.bwd_app,x,*params) 
  # end forward

  def getFineTimePoints(self):
    return self.fwd_app.getTimePoints()


  # This method copies the layer parameters and can be used for verification
  def buildSequentialOnRoot(self):
    ode_layers    = [FixDTBlock(copy.deepcopy(l),self.dt) for l in self.layer_models]
    remote_layers = ode_layers
    build_seq_tag = 12         # this 
    comm          = self.getMPIComm()
    my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()

    # short circuit for serial case
    if num_ranks==1:
      return nn.Sequential(*remote_layers)

    if my_rank==0:
      for i in range(1,self.getMPIComm().Get_size()):
        remote_layers += comm.recv(source=i,tag=build_seq_tag)
      remote_layers = nn.Sequential(*remote_layers)

      if hasattr(self,'device'):
        return remote_layers.to(self.device)
      else:
        return remote_layers
    else:
      ode_layers_cpu = [o.cpu() for o in ode_layers]
      comm.send(ode_layers_cpu,dest=0,tag=build_seq_tag)
      return None
  # end buildSequentialOnRoot


# end LayerParallel
