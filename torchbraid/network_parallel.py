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

import torchbraid.resnet_apps as apps

import numpy as np

def distributeNetworkFromRoot(comm,network):
  """
  Function takes the network on the root and distributes children across multiple processors

  comm: Communicator to use for distribution
  network: defined on the root (ignored elsewhere), children will be distributed

  returns: Network's on each processor, on the root it will contain references to the
           layers selected for this processors. The children will be distributed
           accordinng to processor rank in a linear way. The work will not be load balanced.
  """

  num_ranks = comm.Get_size()
  my_rank = comm.Get_rank()

  if my_rank==0:
    children = [c for c in network.children()]

    assert(len(children)>=num_ranks)

    even = len(children) // num_ranks
    remain = len(children) % num_ranks

    # figure out how many layers to distribute to each processor
    # (adds an additional for each remaining processors)
    count = (num_ranks+1)*[0]
    offset = 0
    for i in range(num_ranks):
      count[i+1] = offset+even 
      offset += even
      if remain>0:
        count[i+1] += 1
        remain -= 1
        offset += 1
    assert(offset==len(children))

    # now that the counts are setup properly, it should be trivial
    all_children = num_ranks*[None]
    for i in range(num_ranks):
      all_children[i] = children[count[i]:count[i+1]]
  else:
    all_children = None

  return comm.scatter(all_children,root=0)

# end distributedNetworkFromRoot

class NetworkParallel(nn.Module):
  
  class ExecLP:
    """Helper class for building composite neural network modules

    This class is used by customers of the NetworkParallel module to
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

  def __init__(self,comm,layers,max_levels=1,max_iters=10):
    super(NetworkParallel,self).__init__()

    self.comm = comm

    self.exec_helper = self.ExecLP(comm.Get_rank())

    self.layers = layers[:] # copy the list (so we don't modify the passed in value)
                            # note that this _does_not_ copy the layers, just the list
    self.local_layers = nn.Sequential(*self.layers)

    self.timer_manager = ContextTimerManager()

    self.fwd_app = apps.ForwardResNetApp(comm,self.layers,max_levels,max_iters,self.timer_manager)
    self.bwd_app = apps.BackwardResNetApp(self.fwd_app,self.timer_manager)
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

  def setRelaxOnlyCG(self, flag):
    self.bwd_app.setRelaxOnlyCG(flag)
    
    # Probably leave commented out for forward solve, still need to accurately
    # process incoming data.  It's the gradient solve that we believe can be
    # made more inexact.
    #self.fwd_app.setRelaxOnlyCG(flag)

  def setCFactor(self,cfactor):
    self.fwd_app.setCFactor(cfactor)
    self.bwd_app.setCFactor(cfactor)

  def setSkipDowncycle(self,skip):
    self.fwd_app.setSkipDowncycle(skip)
    self.bwd_app.setSkipDowncycle(skip)

  def getMPIComm(self):
    return self.fwd_app.getMPIComm()

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
