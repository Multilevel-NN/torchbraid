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

from torchbraid.utils import ContextTimerManager
from torchbraid.rnn_braid_function import BraidFunction

import torchbraid.rnn_apps as apps

##
# Define your Python Braid Vector

#  a python level module
##########################################################

class RNN_Parallel(nn.Module):

  def __init__(self,comm,basic_block,num_steps,hidden_size,num_layers,Tf,max_levels=1,max_iters=10):
    super(RNN_Parallel,self).__init__()

    self.comm = comm

    self.basic_block = basic_block
    self.RNN_models = basic_block()

    self.timer_manager = ContextTimerManager()

    # RNN_torchbraid_apps.py -> ForwardBraidApp
    self.fwd_app = apps.ForwardBraidApp(comm,self.RNN_models,num_steps,hidden_size,num_layers,Tf,max_levels,max_iters,self.timer_manager)
    # self.fwd_app = apps.ForwardBraidApp(comm,self.RNN_models,num_steps,Tf,max_levels,max_iters,self.timer_manager)
    # self.bwd_app = apps.BackwardBraidApp(self.fwd_app,self.timer_manager)

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
    # self.bwd_app.setPrintLevel(print_level)

  def setNumRelax(self,relax,level=-1):
    self.fwd_app.setNumRelax(relax,level=level)
    # self.bwd_app.setNumRelax(relax,level=level)

  def setCFactor(self,cfactor):
    self.fwd_app.setCFactor(cfactor)
    # self.bwd_app.setCFactor(cfactor)

  def setSkipDowncycle(self,skip):
    self.fwd_app.setSkipDowncycle(skip)
    # self.bwd_app.setSkipDowncycle(skip)

  def getMPIComm(self):
    return self.fwd_app.getMPIComm()

  def forward(self,x):
    # prefix_rank  = self.comm.Get_rank()
    # print("Rank %d RNN_Parallel -> forward() - called" % prefix_rank)

    # we are doing this to take adavtage of
    # pytorch's autograd which functions "naturally"
    # with the torch.autograd.function
    params = list(self.parameters())  # TODO: Need to modify 07/14
    # return BraidFunction.apply(self.fwd_app,self.bwd_app,x,*params)
    #############################
    ##### CRITICAL CHANGE!! #####
    #############################
    return BraidFunction.apply(self.fwd_app,x,*params)
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

  def getFinal(self):           # TODO: Need to modify 07/14

    # prefix_rank  = self.comm.Get_rank()
    # print("Rank %d RNN_Parallel -> getFinal() - called" % prefix_rank)

    # print("type of self.fwd_app.getFinal(): ",type(self.fwd_app.getFinal()))

    return  self.fwd_app.getFinal()

  def getFinalOnRoot(self):

    # prefix_rank  = self.comm.Get_rank()
    # print("Rank %d RNN_Parallel -> getFinalOnRoot() - called" % prefix_rank)
    
    build_seq_tag = 99        # this 
    comm          = self.getMPIComm()
    my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()
    
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

  def getTimersString(self):
    
    # prefix_rank  = self.comm.Get_rank()
    # print("Rank %d RNN_Parallel -> getTimersString() - called" % prefix_rank)

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
