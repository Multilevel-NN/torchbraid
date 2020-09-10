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
from torchbraid_app import BraidApp
from torchbraid_app import BraidVector

import sys
import traceback

from mpi4py import MPI

class ForwardODENetApp(BraidApp):

  def __init__(self,comm,layer_models,local_num_steps,Tf,max_levels,max_iters,timer_manager,spatial_ref_pair=None):
    BraidApp.__init__(self,'FWDApp',comm,local_num_steps,Tf,max_levels,max_iters,spatial_ref_pair=spatial_ref_pair)

    # note that a simple equals would result in a shallow copy...bad!
    self.layer_models = [l for l in layer_models]

    comm          = self.getMPIData().getComm()
    my_rank       = self.getMPIData().getRank()
    num_ranks     = self.getMPIData().getSize()

    # send everything to the left (this helps with the adjoint method)
    if my_rank>0:
      comm.send(self.layer_models[0],dest=my_rank-1,tag=22)
    if my_rank<num_ranks-1:
      neighbor_model = comm.recv(source=my_rank+1,tag=22)
      self.layer_models.append(neighbor_model)

    # build up the core
    self.py_core = self.initCore()

    self.timer_manager = timer_manager
    self.use_deriv = False
  # end __init__

  def __del__(self):
    pass

  def updateParallelWeights(self):
    # send everything to the left (this helps with the adjoint method)
    comm          = self.getMPIData().getComm()
    my_rank       = self.getMPIData().getRank()
    num_ranks     = self.getMPIData().getSize()

    if my_rank>0:
      comm.send(self.layer_models[0],dest=my_rank-1,tag=22)
    if my_rank<num_ranks-1:
      neighbor_model = comm.recv(source=my_rank+1,tag=22)
      self.layer_models[-1] = neighbor_model

  def run(self,x):
    self.soln_store = dict()

    # turn on derivative path (as requried)
    self.use_deriv = self.training

    # run the braid solver
    with self.timer("runBraid"):

      # do boundary exchange for parallel weights
      if self.use_deriv:
        self.updateParallelWeights()

      y = self.runBraid(x)

      # reset derivative papth
      self.use_deriv = False

    return y
  # end forward

  def getSolnDiagnostics(self):
    """
    Compute and return a vector of all the local solutions.
    This does no parallel computation. The result is a dictionary
    with hopefully self explanatory names.
    """

    # make sure you could store this
    assert(self.enable_diagnostics)
    assert(self.soln_store is not None)

    result = dict()
    result['timestep_index'] = []
    result['step_in'] = []
    result['step_out'] = []
    for ts in sorted(self.soln_store):
      x,y = self.soln_store[ts]

      result['timestep_index'] += [ts]
      result['step_in']        += [torch.norm(x).item()]
      result['step_out']       += [torch.norm(y).item()]

    return result 

  def timer(self,name):
    return self.timer_manager.timer("ForWD::"+name)

  def getLayer(self,t,tf,level):
    index = self.getLocalTimeStepIndex(t,tf,level)
    return self.layer_models[index]

  def parameters(self):
    return [list(l.parameters()) for l in self.layer_models]

  def eval(self,y,tstart,tstop,level,force_deriv=False,x=None):
    """
    Method called by "my_step" in braid. This is
    required to propagate from tstart to tstop, with the initial
    condition x. The level is defined by braid
    """

    # this function is used twice below to define an in place evaluation
    def in_place_eval(t_y,t_x,tstart,tstop,level):
      # get some information about what to do
      dt = tstop-tstart
      layer = self.getLayer(tstart,tstop,level) # resnet "basic block"

      t_y.zero_()
      with torch.enable_grad():
        if level==0:
          t_x.requires_grad = True 

        t_y.copy_(t_x)
        t_y.add_(dt*layer(t_x))
    # end in_place_eval

    # there are two paths by which eval is called:
    #  1. x is a BraidVector: my step has called this method
    #  2. x is a torch tensor: called internally (probably for the adjoint) 

    if isinstance(y,BraidVector):
      # FIXME: this is a source of memory growth, I'd like to remove it
      t_x = y.tensor().detach()
      t_y = y.tensor().detach().clone()

      in_place_eval(t_y,t_x,tstart,tstop,level)

      # store off the solution for later adjoints
      if level==0:
        ts_index = self.getGlobalTimeStepIndex(tstart,tstop,0)
        self.soln_store[ts_index] = (t_y,t_x)

      # change the pointer under the hood of teh braid vector
      y.tensor_ = t_y.detach().clone()
    else: 
      in_place_eval(y,x,tstart,tstop,level)
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
    
    ts_index = self.getGlobalTimeStepIndex(tstart,tstop,level)
    layer = self.getLayer(tstart,tstop,level)

    # the idea here is store it internally, failing
    # that the values need to be recomputed locally. This may be
    # because you are at a processor boundary, or decided not
    # to start the value 
    if ts_index in self.soln_store:
      t_y = self.soln_store[ts_index][0]
      t_x = self.soln_store[ts_index][1]

      return [t_y,t_x],layer

    # value wasn't found, recompute it and return.
    t_x = self.soln_store[ts_index-1][0]
    x_o = t_x.detach()
    x_o.requires_grad = t_x.requires_grad

    y = x_o.detach().clone()
    self.eval(y,tstart,tstop,0,force_deriv=True,x=x_o)
    return (y, x_o), layer

  # end getPrimalWithGrad

# end ForwardODENetApp

##############################################################

class BackwardODENetApp(BraidApp):

  def __init__(self,fwd_app,timer_manager):
    # call parent constructor
    BraidApp.__init__(self,'BWDApp',
                           fwd_app.getMPIData().getComm(),
                           fwd_app.local_num_steps,
                           fwd_app.Tf,
                           fwd_app.max_levels,
                           fwd_app.max_iters,
                           spatial_ref_pair=fwd_app.spatial_ref_pair)

    self.fwd_app = fwd_app

    # build up the core
    self.py_core = self.initCore()

    # reverse ordering for adjoint/backprop
    self.setRevertedRanks(1)

    self.timer_manager = timer_manager
  # end __init__

  def __del__(self):
    self.fwd_app = None

  def timer(self,name):
    return self.timer_manager.timer("BckWD::"+name)

  def run(self,x):

    try:
      f = self.runBraid(x)


      # this code is due to how braid decomposes the backwards problem
      # The ownership of the time steps is shifted to the left (and no longer balanced)
      first = 1
      if self.getMPIData().getRank()==0:
        first = 0

      self.grads = []

      # preserve the layerwise structure, to ease communication
      # - note the prection of the 'None' case, this is so that individual layers
      # - can have gradient's turned off
      my_params = self.fwd_app.parameters()
      for sublist in my_params[first:]:
        sub_gradlist = [] 
        for item in sublist:
          if item.grad is not None:
            sub_gradlist += [ item.grad.clone() ] 
          else:
            sub_gradlist += [ None ]

        self.grads += [ sub_gradlist ]
      # end for sublist

      for m in self.fwd_app.layer_models:
         m.zero_grad()
    except:
      print('\n**** Torchbraid Internal Exception ****\n')
      traceback.print_exc()

    return f
  # end forward

  def eval(self,w,tstart,tstop,level):
    """
    Evaluate the adjoint problem for a single time step. Here 'w' is the
    adjoint solution. The variables 'x_old' and 'x_new' refer to the forward
    problem solutions at the beginning (x_old) and end (x_new) of the type step.
    """
    try:
        # we need to adjust the time step values to reverse with the adjoint
        # this is so that the renumbering used by the backward problem is properly adjusted
        (t_x_new,t_x_old),layer = self.fwd_app.getPrimalWithGrad(self.Tf-tstop,self.Tf-tstart,level)

        # t_x_old should have no gradient (for memory reasons)
        assert(t_x_old.grad is None)

        # we are going to change the required gradient, make sure they return
        # to where they started!
        required_grad_state = []

        # play with the layers gradient to make sure they are on apprpriately
        for p in layer.parameters(): 
          required_grad_state += [p.requires_grad]
          if level==0:
            if not p.grad is None:
              p.grad.data.zero_()
          else:
            # if you are not on the fine level, compute no gradients
            p.requires_grad = False

        # perform adjoint computation
        t_w = w.tensor()
        t_w.requires_grad = False
        t_x_new.backward(t_w,retain_graph=True)

        # this little bit of pytorch magic ensures the gradient isn't
        # stored too long in this calculation (in particulcar setting
        # the grad to None after saving it and returning it to braid)
        t_grad = t_x_old.grad.detach() 
        t_x_old.grad = None

        for p,s in zip(layer.parameters(),required_grad_state):
          p.requires_grad = s
    except:
      print('\n**** Torchbraid Internal Exception ****\n')
      traceback.print_exc()

    w.tensor_ = t_grad
    #return BraidVector(t_grad,level) 
  # end eval

# end BackwardODENetApp
