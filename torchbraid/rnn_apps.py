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
  # end __init__

  def run(self,x,h_c):
    num_ranks     = self.mpi_comm.Get_size()
    my_rank       = self.mpi_comm.Get_rank()
    comm = self.mpi_comm

    assert(x.shape[1]==self.local_num_steps)

    self.x = x

    # send deta vector to the left
    if my_rank>0:
      comm.send(self.x[:,0,:],dest=my_rank-1,tag=22)
    if my_rank<num_ranks-1:
      neighbor_x = comm.recv(source=my_rank+1,tag=22)
      self.x = torch.cat((self.x,neighbor_x.unsqueeze(1)), 1)

    # run the braid solver
    y = self.runBraid(h_c)

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
    t_h,t_c = g0.tensors()
    with torch.no_grad():
      t_yh,t_yc = self.RNN_models(self.x[:,index,:],t_h,t_c)

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
    
    b_x = self.getUVector(0,tstart)
    t_x = b_x.tensors()

    x = tuple([v.detach() for v in t_x])

    xh,xc = x 
    xh.requires_grad = True
    xc.requires_grad = True

    index = self.getLocalTimeStepIndex(tstart,tstop,level)
    with torch.enable_grad():
      yh,yc = self.RNN_models(self.x[:,index,:],xh,xc)
   
    return ((yh,yc), x), self.RNN_models
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
      f = self.runBraid(x)

      # this code is due to how braid decomposes the backwards problem
      # The ownership of the time steps is shifted to the left (and no longer balanced)
      first = 1
      if self.getMPIComm().Get_rank()==0:
        first = 0

      self.grads = [p.grad.detach().clone() for p in self.fwd_app.RNN_models.parameters()]

      # required otherwise we will re-add teh gradients
      self.fwd_app.RNN_models.zero_grad() 
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
        (t_y,t_x),layer = self.fwd_app.getPrimalWithGrad(self.Tf-tstop,self.Tf-tstart,level)

        # t_x should have no gradient (for memory reasons)
        for v in t_x:
          assert(v.grad is None)

        # play with the parameter gradients to make sure they are on apprpriately,
        # store the initial state so we can revert them later
        required_grad_state = []
        for p in layer.parameters(): 
          required_grad_state += [p.requires_grad]
          if done!=1:
            # if you are not on the fine level, compute no parameter gradients
            p.requires_grad = False

        # perform adjoint computation
        t_w = w.tensors()
        for v in t_w:
          v.requires_grad = False
        for v,w_d in zip(t_y,t_w):
          v.backward(w_d,retain_graph=True)

        # this little bit of pytorch magic ensures the gradient isn't
        # stored too long in this calculation (in particulcar setting
        # the grad to None after saving it and returning it to braid)
        for wv,xv in zip(t_w,t_x):
          wv.copy_(xv.grad.detach()) 

        # revert the gradient state to where they started
        for p,s in zip(layer.parameters(),required_grad_state):
          p.requires_grad = s
    except:
      print('\n**** Torchbraid Internal Exception ****\n')
      traceback.print_exc()
  # end eval

# end BackwardODENetApp
