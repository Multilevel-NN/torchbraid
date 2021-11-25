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

import numpy as np

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
    y = self.dt*self.layer(x)
    y.add_(x)
    return y
# end ODEBlock

class LayerParallel(nn.Module):
  
  class ExecLP:
    """Helper class for btorchuilding composite neural network modules

    This class is used by customers of the LayerParallel module to
    allow construction of composite neural networks with the proper
    parallel execution and gradients.

    One naming convection is to use 'o' for a class of this type
    signifying object compoistion.
    """

    def __init__(self,rank):
      """Constructor setting the LP rank of this processor"""
      self.my_rank = rank

    def __call__(self,op,*args,**kwargs):
      """Call an operator conditionally based on being on rank 0
         
         If op is a class, than this returns None on processors other
         than rank 0.
      """
      
      if self.my_rank==0:
        return op(*args,**kwargs)

      # this helps with makign constructos consistent
      if inspect.isclass(op):
        return None

      # blindly assume that all the arguments are torch
      # tensors, and propagate this through
      value = torch.zeros(1)
      for a in args:
        if a.requires_grad:
          value += torch.norm(a)

       # so this is all a hack to get this thing to work
      return torch.zeros(1)*value

  def __init__(self,comm,layer_block,num_steps,Tf,max_levels=1,max_iters=10,spatial_ref_pair=None, sc_levels=None):
    super(LayerParallel,self).__init__()

    self.comm = comm

    self.exec_helper = self.ExecLP(comm.Get_rank())

    # optional parameters
    global_steps = num_steps*comm.Get_size()

    self.dt = Tf/global_steps
  
    self.layer_block = layer_block
    self.layer_models = [layer_block() for i in range(num_steps)]
    self.local_layers = nn.Sequential(*self.layer_models)

    self.timer_manager = ContextTimerManager()

    self.fwd_app = apps.ForwardODENetApp(comm,self.layer_models,num_steps,Tf,max_levels,max_iters,self.timer_manager,
                                         spatial_ref_pair=spatial_ref_pair, layer_block=layer_block, sc_levels=sc_levels)
    self.bwd_app = apps.BackwardODENetApp(self.fwd_app,self.timer_manager)

    self.enable_diagnostics = False
  # end __init__

  def comp_op(self):
    """Short for compose operator, returns a functor that allows contstruction of composite neural 
       networks using this LayerParallel module.
    """
    return self.exec_helper

  def zero_grad(self):
    for l in self.fwd_app.layer_models:
      if l==None: continue
      l.zero_grad()
    self.local_layers.zero_grad()

  def getTimerManager(self):
    """
    Get a TimerContextManager that describes how much time is taken by what.
    """
    return self.timer_manager

  def setPrintLevel(self,print_level,tb_print=False):
    """
    Set the print level for this module. If tb_print (torchbraid print) is
    set to true this method sets the internal printing diagnostics. If it is
    false, the print level is passed along to xbraid. 
    """

    self.fwd_app.setPrintLevel(print_level,tb_print)
    self.bwd_app.setPrintLevel(print_level,tb_print)

  def setNumRelax(self,relax,level=-1):
    self.fwd_app.setNumRelax(relax,level=level)
    self.bwd_app.setNumRelax(relax,level=level)

  def setMaxIters(self,max_iters):
    self.fwd_app.setNumRelax(max_iters)
    self.bwd_app.setNumRelax(max_iters)

  def setFwdMaxIters(self,max_iters):
    self.fwd_app.setMaxIters(max_iters)

  def setBwdMaxIters(self,max_iters):
    self.bwd_app.setMaxIters(max_iters)

  def setFMG(self):
    self.fwd_app.setFMG()
    self.bwd_app.setFMG()

  def setCFactor(self,cfactor):
    self.fwd_app.setCFactor(cfactor)
    self.bwd_app.setCFactor(cfactor)

  def setSkipDowncycle(self,skip):
    self.fwd_app.setSkipDowncycle(skip)
    self.bwd_app.setSkipDowncycle(skip)

  def getMPIComm(self):
    return self.fwd_app.getMPIComm()

  def forward(self,x):
    # we are doing this to take advantage of
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

  def diagnostics(self,enable):
    """
    This method tells torchbraid to keep track of the feature vectors
    and parameters for eventual output. This is to help debug stability
    questions and other potential issues
    """

    self.enable_diagnostics = enable

    self.fwd_app.diagnostics(enable)
    self.bwd_app.diagnostics(enable)

  def getDiagnostics(self):
    """
    Get a dictionary with the diagnostics return by the forward application.
    This method does some parallal communication (gather to root), for printing
    purposes.

    Note the method diagnostics(enable) must be called prior to this with
    enable=True
    """

    assert(self.enable_diagnostics)

    local_diag = self.fwd_app.getSolnDiagnostics()

    # communicate everyone's diagnostics to root
    diag_vector = self.comm.gather(local_diag) 
    serial_net = self.buildSequentialOnRoot()

    result = dict()
    result['rank'] = self.comm.Get_rank()
    result['timestep_index'] = []
    result['step_in'] = []
    result['step_out'] = []
    result['params'] = []

    # build up diagnostics on root
    ########################################
    if self.comm.Get_rank()==0:
      for d in diag_vector:
        result['timestep_index'] += d['timestep_index']
        result['step_in'] += d['step_in']
        result['step_out'] += d['step_out']

      for c in serial_net.children():
        params = []
        for i,p in enumerate(c.parameters()):
          params += [torch.norm(p).item()]
        params = np.array(params)

        result['params'] += [params]
      # end for d,c
    # end rank==0
          
    return result

  # This method copies the layer parameters and can be used for verification
  def buildSequentialOnRoot(self):
    ode_layers    = [ODEBlock(copy.deepcopy(l),self.dt) for l in self.layer_models]
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
      return nn.Sequential(*remote_layers)
    else:
      comm.send(ode_layers,dest=0,tag=build_seq_tag)
      return None
  # end buildSequentialOnRoot

  def getFinalOnRoot(self,vec):
    build_seq_tag = 99        # this 
    comm          = self.getMPIComm()
    my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()

    # short circuit for serial case
    if num_ranks==1:
      return vec

    # send the output of the last layer to the root
    if my_rank==0:
      remote_final = comm.recv(source=num_ranks-1,tag=build_seq_tag)
      return remote_final
    elif my_rank==num_ranks-1:
      final = vec
      comm.send(final,dest=0,tag=build_seq_tag)

    return None

  def copyVectorFromRoot(self,vec):
    build_seq_tag = 99        # this 
    comm          = self.getMPIComm()
    my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()

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
