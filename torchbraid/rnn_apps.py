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
import traceback
import numpy as np

from braid_vector import BraidVector

import torchbraid_app as parent

import sys

from mpi4py import MPI

class ForwardBraidApp(parent.BraidApp):

  def __init__(self,comm,RNN_models,local_num_steps,hidden_size,num_layers,Tf,max_levels,max_iters,timer_manager,abs_tol):
    parent.BraidApp.__init__(self,'RNN',comm,local_num_steps,Tf,max_levels,max_iters,spatial_ref_pair=None,require_storage=True,abs_tol=abs_tol)

    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.RNN_models = RNN_models

    comm          = self.getMPIComm()
    my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()

    # build up the core
    self.py_core = self.initCore()

    # force evaluation of gradients at end of up-cycle
    self.finalRelax()

    self.timer_manager = timer_manager
    self.use_deriv = False

    self.user_dt_ratio = self._dt_ratio_

    self.seq_shapes = None
    self.backpropped = dict()
  # end __init__

  def _dt_ratio_(self,level,tstart,tstop,fine_dt): 
    return np.sqrt(np.sqrt((tstop-tstart)/fine_dt))

  def setDtRatio(self,user_dt_ratio):
    self.user_dt_ratio = user_dt_ratio

  def dt_ratio(self,level,tstart,tstop):
    return self.user_dt_ratio(level,tstart,tstop,self.dt)
  # end dt_ratio

  def getTensorShapes(self):
    return list(self.shape0)+self.seq_shapes

  def getSequenceVector(self,t,tf,level):
    index = self.getLocalTimeStepIndex(t,tf,level)
    if index<0: 
      pre_str = "\n{}: WARNING: getSequenceVector index negative at {}: {}\n".format(self.my_rank,t,index)
      stack_str = utils.stack_string('{}: |- '.format(self.my_rank))
      print(pre_str+stack_str)
 
    if index<self.x.shape[1]:
      value = self.x[:,index,:]
    else:
      # this is a sentinnel
      value = self.x[:,0,:].detach().clone()
      
    return value

  def clearTempLayerWeights(self):
    layer = self.temp_layer

    for dest_p in list(layer.parameters()):
      dest_p.data = torch.empty(())
  # end setLayerWeights

  def setLayerWeights(self,t,tf,level,weights):
    layer = self.getLayer(t,tf,level)

    with torch.no_grad():
      for dest_p,src_w in zip(list(layer.parameters()),weights):
        dest_p.data = src_w
  # end setLayerWeights

  def initializeVector(self,t,x):
    seq_x = self.getSequenceVector(t,None,level=0)
    x.addWeightTensors((seq_x,))

  def run(self,x,h_c):
    num_ranks     = self.mpi_comm.Get_size()
    my_rank       = self.mpi_comm.Get_rank()
    comm = self.mpi_comm

    assert(x.shape[1]==self.local_num_steps)

    self.x = x
    self.seq_shapes = [x[:,0,:].shape]

    with self.timer("run:precomm"):
      recv_request = None
      if my_rank<num_ranks-1:
        neighbor_x = torch.zeros(x[:,0,:].shape)
        recv_request = comm.Irecv(neighbor_x.numpy(),source=my_rank+1,tag=22)

      # send deta vector to the left
      send_request = None
      if my_rank>0:
        send_request = comm.Isend(np.ascontiguousarray(self.x[:,0,:].numpy()),dest=my_rank-1,tag=22)

      if recv_request:
        recv_request.Wait()
        self.x = torch.cat((self.x,neighbor_x.unsqueeze(1)), 1)

      if send_request:
        send_request.Wait()
    # end wit htimer

    # run the braid solver
    y = self.runBraid(h_c)

    with self.timer("run:postcomm"):
      y = comm.bcast(y,root=num_ranks-1)

    # y is a tuple with the final h,c components
    return y
  # end forward

  def timer(self,name):
    return self.timer_manager.timer("ForWD::"+name)

  def eval(self,g0,tstart,tstop,level,done):
    """
    Method called by "my_step" in braid. This is
    required to propagate from tstart to tstop, with the initial
    condition x. The level is defined by braid
    """

    # there are two paths by which eval is called:
    #  1. x is a BraidVector: my step has called this method
    #  2. x is a torch tensor: called internally (probably at the behest
    #                          of the adjoint)

    index = self.getLocalTimeStepIndex(tstart,tstop,level)
    seq_x = g0.weightTensors()[0]

    t_h,t_c = g0.tensors()
    if not done:
      with torch.no_grad():
        t_yh,t_yc = self.RNN_models(seq_x,t_h,t_c)

        if level!=0:
          dt_ratio = self.dt_ratio(level,tstart,tstop)

          t_yh = (1.0-dt_ratio)*t_h + dt_ratio*t_yh
          t_yc = (1.0-dt_ratio)*t_c + dt_ratio*t_yc
    else:
      with torch.enable_grad():
        h = t_h.detach()
        c = t_c.detach()
        h.requires_grad = True
        c.requires_grad = True
        t_yh,t_yc = self.RNN_models(seq_x,h,c)

        if level!=0:
          dt_ratio = self.dt_ratio(level,tstart,tstop)

          t_yh = (1.0-dt_ratio)*h + dt_ratio*t_yh
          t_yc = (1.0-dt_ratio)*c + dt_ratio*t_yc
      self.backpropped[tstart,tstop] = ((h,c),(t_yh,t_yc))

    seq_x = self.getSequenceVector(tstop,None,level)

    g0.addWeightTensors((seq_x,))
    g0.replaceTensor(t_yh,0)
    g0.replaceTensor(t_yc,1)
  # end eval

  def getPrimalWithGrad(self,tstart,tstop,level):
    """ 
    Get the forward solution associated with this
    time step and also get its derivative. This is
    used by the BackwardApp in computation of the
    adjoint (backprop) state and parameter derivatives.
    Its intent is to abstract the forward solution
    so it can be stored internally instead of
    being recomputed.
    """
    
    if level==0 and (tstart,tstop) in self.backpropped:
      x,y = self.backpropped[(tstart,tstop)]
      return y,x

    b_x = self.getUVector(0,tstart)
    t_x = b_x.tensors()

    x = tuple([v.detach() for v in t_x])

    xh,xc = x 
    xh.requires_grad = True
    xc.requires_grad = True

    seq_x = b_x.weightTensors()[0]

    with torch.enable_grad():
      yh,yc = self.RNN_models(seq_x,xh,xc)

      if level!=0:
        dt_ratio = self.dt_ratio(level,tstart,tstop)

        yh = (1.0-dt_ratio)*xh + dt_ratio*yh
        yc = (1.0-dt_ratio)*xc + dt_ratio*yc
   
    return (yh,yc), x
  # end getPrimalWithGrad

# end ForwardBraidApp

##############################################################

class BackwardBraidApp(parent.BraidApp):

  def __init__(self,fwd_app,timer_manager,abs_tol):
    # call parent constructor
    parent.BraidApp.__init__(self,'RNN',fwd_app.getMPIComm(),
                          fwd_app.local_num_steps,
                          fwd_app.Tf,
                          fwd_app.max_levels,
                          fwd_app.max_iters,spatial_ref_pair=None,require_storage=True,abs_tol=abs_tol)

    self.fwd_app = fwd_app

    # build up the core
    self.py_core = self.initCore()

    # reverse ordering for adjoint/backprop
    self.setRevertedRanks(1)

    # force evaluation of gradients at end of up-cycle
    self.finalRelax()

    self.timer_manager = timer_manager
  # end __init__

  def __del__(self):
    self.fwd_app = None

  def getTensorShapes(self):
    return self.shape0

  def timer(self,name):
    return self.timer_manager.timer("BckWD::"+name)

  def run(self,x):

    try:
      self.RNN_models = self.fwd_app.RNN_models

      f = self.runBraid(x)

      self.grads = [p.grad.detach().clone() for p in self.RNN_models.parameters()]

      # required otherwise we will re-add teh gradients
      self.RNN_models.zero_grad() 

      self.RNN_models = None
    except:
      print('\n**** Torchbraid Internal Exception ****\n')
      traceback.print_exc()

    return f
  # end forward

  def eval(self,w,tstart,tstop,level,done):
    """
    Evaluate the adjoint problem for a single time step. Here 'w' is the
    adjoint solution. The variables 'x' and 'y' refer to the forward
    problem solutions at the beginning (x) and end (y) of the type step.
    """
    try:
        # we need to adjust the time step values to reverse with the adjoint
        # this is so that the renumbering used by the backward problem is properly adjusted
        t_y,t_x = self.fwd_app.getPrimalWithGrad(self.Tf-tstop,self.Tf-tstart,level)

        # play with the parameter gradients to make sure they are on apprpriately,
        # store the initial state so we can revert them later
        required_grad_state = []
        if done!=1:
          for p in self.RNN_models.parameters(): 
            required_grad_state += [p.requires_grad]
            p.requires_grad = False

        # perform adjoint computation
        t_w = w.tensors()
        for v,w_d in zip(t_y,t_w):
          v.backward(w_d,retain_graph=True)

        # this little bit of pytorch magic ensures the gradient isn't
        # stored too long in this calculation (in particulcar setting
        # the grad to None after saving it and returning it to braid)
        for wv,xv in zip(t_w,t_x):
          wv.copy_(xv.grad.detach()) 
          xv.grad = None

        # revert the gradient state to where they started
        if done!=1:
          for p,s in zip(self.RNN_models.parameters(),required_grad_state):
            p.requires_grad = s
    except:
      print('\n**** Torchbraid Internal Exception ****\n')
      traceback.print_exc()

  # end eval

# end BackwardODENetApp
