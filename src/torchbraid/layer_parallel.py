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
class FixDTBlock(nn.Module):
  """Build a module that removes teh dt from the forward evaluation
     this eases consistency with the PyTorch modus operandi
  """
  def __init__(self,layer,dt):
    super(FixDTBlock, self).__init__()

    self.dt = dt
    self.layer = layer

  def forward(self, x):
    return self.layer(self.dt,x)
# end FixDTBlock

class LayerParallel(nn.Module):
  
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

  def __init__(self,comm,layer_blocks,global_steps,Tf,max_fwd_levels=1,max_bwd_levels=1,max_iters=10,spatial_ref_pair=None,user_mpi_buf=False, nsplines=0, splinedegree=1, gpu_direct_commu=False):
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
    super(LayerParallel,self).__init__()

    self.comm = comm
    self.exec_helper = self.ExecLP(comm.Get_rank())
    self.timer_manager = ContextTimerManager()

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

    self.enable_diagnostics = False
  # end __init__

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

    comm      = self.getMPIComm() 
    all_str = comm.gather(main_str,root=0)
    if comm.Get_rank()==0:
      return '\n<...> '.join(all_str)
    else:
      return '--ignore parallel print out--'

  def makeList(self,data):
    """
    Conditionally convert a single element to a list, or return a list

    For instance: makeList(7) == [7], while makeList([7]) == [7]
    """
    if isinstance(data,list):
      return data
    return [data]

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

  def setFwdStorage(self, storage):
    self.fwd_app.setStorage(storage)

  def setBwdStorage(self, storage):
    self.bwd_app.setStorage(storage)

  def setMinCoarse(self, mc):
    self.fwd_app.setMinCoarse(mc)
    self.bwd_app.setMinCoarse(mc)

  def setFwdNumRelax(self,relax,level=-1):
    self.fwd_app.setNumRelax(relax,level=level)
  
  def setBwdNumRelax(self,relax,level=-1):
    self.bwd_app.setNumRelax(relax,level=level)

  def setNumRelax(self,max_iters,level=-1):
    self.fwd_app.setNumRelax(max_iters,level)
    self.bwd_app.setNumRelax(max_iters,level)

  def setFwdAbsTol(self,abs_tol):
    self.fwd_app.setAbsTol(abs_tol)

  def setBwdAbsTol(self,abs_tol):
    self.bwd_app.setAbsTol(abs_tol)

  def setMaxIters(self,max_iters):
    self.fwd_app.setMaxIters(max_iters)
    self.bwd_app.setMaxIters(max_iters)

  def setFwdMaxIters(self,max_iters):
    self.fwd_app.setMaxIters(max_iters)

  def setBwdMaxIters(self,max_iters):
    self.bwd_app.setMaxIters(max_iters)

  def getFwdMaxIters(self):
    return self.fwd_app.getMaxIters()

  def getBwdMaxIters(self):
    return self.bwd_app.getMaxIters()

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

  def getFwdStats(self):
    itr, res = self.fwd_app.getBraidStats()
    return itr,res

  def getBwdStats(self):
    itr, res = self.bwd_app.getBraidStats()
    return itr,res

  def getFineTimePoints(self):
    return self.fwd_app.getTimePoints()

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
      remote_final = remote_final.to(vec.device)
      return remote_final
    elif my_rank==num_ranks-1:
      final = vec
      comm.send(final.cpu(),dest=0,tag=build_seq_tag)

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


# end LayerParallel
