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

import inspect
import torch
import torch.nn as nn

from mpi4py import MPI

import copy

from torchbraid.utils import ContextTimerManager
from torchbraid.rnn_braid_function import BraidFunction

import torchbraid.rnn_apps as apps

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
      h = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
      c = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
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

class RNN_Parallel(nn.Module):
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
  # end ExecLP

  ##################################################

  def __init__(self,comm,basic_block,num_steps,hidden_size,num_layers,Tf,model_compute_steps=False,max_levels=1,max_iters=10,abs_tol=1e-9):
    super(RNN_Parallel,self).__init__()

    self.comm        = comm
    self.num_layers  = num_layers
    self.hidden_size = hidden_size
    self.RNN_models  = basic_block

    self.exec_helper = self.ExecLP(comm.Get_rank())
    self.timer_manager = ContextTimerManager()

    # RNN_torchbraid_apps.py -> ForwardBraidApp
    self.fwd_app = apps.ForwardBraidApp(comm,self.RNN_models,num_steps,Tf,max_levels,max_iters,self.timer_manager,abs_tol)
    self.bwd_app = apps.BackwardBraidApp(self.fwd_app,self.timer_manager,abs_tol)
  # end __init__

  def comp_op(self):
    """Short for compose operator, returns a functor that allows contstruction of composite neural 
       networks using this LayerParallel module.
    """
    return self.exec_helper

  def zero_grad(self):
    self.RNN_models.zero_grad()

  def getFastForwardInfo(self):
    return self.fwd_app.getFastForwardInfo()

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

  def setFwdNumRelax(self,relax,level=-1):
    self.fwd_app.setNumRelax(relax,level=level)

  def setBwdNumRelax(self,relax,level=-1):
    self.bwd_app.setNumRelax(relax,level=level)

  def setMaxIters(self,max_iters):
    self.fwd_app.setNumRelax(max_iters)
    self.bwd_app.setNumRelax(max_iters)

  def getFwdMaxIters(self):
    return self.fwd_app.getMaxIters()

  def getBwdMaxIters(self):
    return self.bwd_app.getMaxIters()

  def setFwdMaxIters(self,max_iters):
    self.fwd_app.setMaxIters(max_iters)

  def setBwdMaxIters(self,max_iters):
    self.bwd_app.setMaxIters(max_iters)

  def setCFactor(self,cfactor):
    self.fwd_app.setCFactor(cfactor)
    self.bwd_app.setCFactor(cfactor)

  def setSkipDowncycle(self,skip):
    self.fwd_app.setSkipDowncycle(skip)
    self.bwd_app.setSkipDowncycle(skip)

  def getMPIComm(self):
    return self.fwd_app.getMPIComm()

  def forward(self,x,h_c=None):
    # we are doing this to take adavtage of
    # pytorch's autograd which functions "naturally"
    # with the torch.autograd.function

    if h_c is None:
      h = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
      c = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
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

    # prefix_rank  = self.comm.Get_rank()
    # print("Rank %d RNN_Parallel -> copyVectorFromRoot() - called" % prefix_rank)

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

  def getFwdStats(self):
    itr, res = self.fwd_app.getBraidStats()
    return itr,res

  def getBwdStats(self):
    itr, res = self.bwd_app.getBraidStats()
    return itr,res

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
