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

from torchbraid.utils import ContextTimerManager

import numpy as np

class LPModule(nn.Module):
  """
  Class abstraction for layer parallel modules

  This is code which is shared between the LayerParallel and GRU_Parallel classes
  """
  class ExecLP:
    """Helper class for btorchuilding composite neural network modules

    This class is used by customers of the LayerParallel module to
    allow construction of composite neural networks with the proper
    parallel execution and gradients.

    One naming convection is to use 'o' for a class of this type
    signifying object composition.
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

      # determine the parallel device
      device = None
      for a in args:
        if hasattr(a,'device'):
          device = a.device
          break
      # take the first device


      # blindly assume that all the arguments are torch
      # tensors, and propagate this through
      value = torch.zeros(1,device=device)
      for a in args:
        if a.requires_grad:
          value += torch.norm(a)

      # so this is all a hack to get this thing to work
      if 'mgopt_term' in kwargs:
        return torch.zeros(1,device=device)*value - kwargs['mgopt_term']
      else:
        return torch.zeros(1,device=device)*value

  def __init__(self, comm):
    super().__init__()
    self.comm = comm
    self.exec_helper = self.ExecLP(comm.Get_rank())
    self.timer_manager = ContextTimerManager()

    self.enable_diagnostics = False

  def comp_op(self):
    """Short for compose operator, returns a functor that allows contstruction of composite neural 
       networks using this LayerParallel module.
    """
    return self.exec_helper

  def extra_repr(self):
    return f'parallel rank = {self.getMPIComm().Get_rank()} of {self.getMPIComm().Get_size()}'

  def repr_helper(self,parent):
    # call representation function in parallel
    repr = nn.Module.__repr__(parent)

    if self.getMPIComm().Get_rank()==0:
      return repr # only return representation on processor zero
    else:
      return '--empty (module print)--' # this will print garbage on a line (how to fix this?)

  def __repr__(self):
    main_str = nn.Module.__repr__(self)

    comm    = self.getMPIComm()
    all_str = comm.gather(main_str,root=0)
    if comm.Get_rank()==0:
      return '\n<...> '.join(all_str)
    else:
      return '--ignore parallel print out--'

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

  def setFwdNumRelax(self,relax,level=-1):
    self.fwd_app.setNumRelax(relax,level=level)

  def setBwdNumRelax(self,relax,level=-1):
    self.bwd_app.setNumRelax(relax,level=level)

  def setNumRelax(self,max_iters,level=-1):
    self.fwd_app.setNumRelax(max_iters,level)
    self.bwd_app.setNumRelax(max_iters,level)

  def setMaxIters(self,max_iters):
    self.fwd_app.setMaxIters(max_iters)
    self.bwd_app.setMaxIters(max_iters)

  def setFwdMaxLevels(self,max_levels):
    self.fwd_app.setMaxLevels(max_levels)

  def setBwdMaxLevels(self,max_levels):
    self.bwd_app.setMaxLevels(max_levels)

  def getFwdMaxLevels(self):
    return self.fwd_app.getMaxLevels()

  def getBwdMaxLevels(self):
    return self.bwd_app.getMaxLevels()

  def setFwdMaxIters(self,max_iters):
    self.fwd_app.setMaxIters(max_iters)

  def setBwdMaxIters(self,max_iters):
    self.bwd_app.setMaxIters(max_iters)

  def getFwdMaxIters(self):
    return self.fwd_app.getMaxIters()

  def getBwdMaxIters(self):
    return self.bwd_app.getMaxIters()

  def setCFactor(self,cfactor):
    self.fwd_app.setCFactor(cfactor)
    self.bwd_app.setCFactor(cfactor)

  def setFwdCFactor(self,cfactor):
    self.fwd_app.setCFactor(cfactor)

  def setBwdCFactor(self,cfactor):
    self.bwd_app.setCFactor(cfactor)

  def setSkipDowncycle(self,skip):
    self.fwd_app.setSkipDowncycle(skip)
    self.bwd_app.setSkipDowncycle(skip)

  def setSkipFwdDowncycle(self, skip):
    self.fwd_app.setSkipDowncycle(skip)

  def setSkipBwdDowncycle(self, skip):
    self.bwd_app.setSkipDowncycle(skip)

  def getMPIComm(self):
    return self.fwd_app.getMPIComm()

  def getFwdStats(self):
    itr, res = self.fwd_app.getBraidStats()
    return itr,res

  def getBwdStats(self):
    itr, res = self.bwd_app.getBraidStats()
    return itr,res

  def getFinalOnRoot(self,vec):
    build_seq_tag = 99        # this
    comm          = self.getMPIComm()
    my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()

    # short circuit for serial case
    if num_ranks==1:
      return vec

    if vec.device.type=='cuda':
      torch.cuda.synchronize()

    # send the output of the last layer to the root
    if my_rank==0:
      remote_final = comm.recv(source=num_ranks-1,tag=build_seq_tag)
      remote_final = remote_final.to(vec.device)
      return remote_final
    elif my_rank==num_ranks-1:
      comm.send(vec.cpu(),dest=0,tag=build_seq_tag)

    if vec.device.type=='cuda':
      torch.cuda.synchronize()

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
        comm.send(vec.cpu(),dest,tag=build_seq_tag)
      return vec
    else:
      result = comm.recv(source=0,tag=build_seq_tag)
      if hasattr(vec,'device'):
        result = result.to(vec.device)
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

  def diagnostics(self,enable):
    """
    This method tells torchbraid, to keep track of the feature vectors
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
