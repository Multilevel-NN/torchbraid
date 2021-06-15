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

import torch

from braid_vector import BraidVector
from torchbraid_app import BraidApp
import utils 

import sys
import traceback
import resource
import copy

from mpi4py import MPI

class ForwardODENetApp(BraidApp):

  def __init__(self,comm,local_num_steps,Tf,max_levels,max_iters,timer_manager,spatial_ref_pair=None, layer_block=None):
    """
    """
    BraidApp.__init__(self,'FWDApp',comm,local_num_steps,Tf,max_levels,max_iters,spatial_ref_pair=spatial_ref_pair,require_storage=True)

    sys.stdout.flush()

    comm          = self.getMPIComm()
    my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()
    self.my_rank = my_rank
    self.layer_block = layer_block

    owned_layers = self.end_layer-self.start_layer+1
    if my_rank==num_ranks-1:
      # the last time step should not create a layer, there is no step being
      # taken on that final step
      owned_layers -= 1

    self.layer_models = [self.layer_block() for _ in range(owned_layers)]

    self.timer_manager = timer_manager
    self.use_deriv = False

    self.parameter_shapes = []
    for p in self.layer_models[0].parameters(): 
      self.parameter_shapes += [p.data.size()]

    self.temp_layer = layer_block()
    self.clearTempLayerWeights()
  # end __init__

  def __del__(self):
    pass

  def getTensorShapes(self):
    return list(self.shape0)+self.parameter_shapes

  def setVectorWeights(self,layer_index,level,x):
    if layer_index<len(self.layer_models) and layer_index>=0:
      layer = self.layer_models[layer_index]
    else:
      layer = None

    if layer!=None:
      weights = [p.data for p in layer.parameters()]
    else:
      weights = []
    x.addWeightTensors(weights)

  def clearTempLayerWeights(self):
    layer = self.temp_layer

    for dest_p in list(layer.parameters()):
      dest_p.data = torch.empty(())
  # end clearTempLayerWeights

  def setLayerWeights(self,t,tf,level,weights):
    layer = self.temp_layer

    with torch.no_grad():
      for dest_p,src_w in zip(list(layer.parameters()),weights):
        dest_p.data = src_w
  # end setLayerWeights

  def initializeVector(self,t,x):
    index = self.getGlobalTimeIndex(t)-self.start_layer
    self.setVectorWeights(index,0,x)

  def run(self,x):
    # turn on derivative path (as requried)
    self.use_deriv = self.training

    # run the braid solver
    with self.timer("runBraid"):

      y = self.runBraid(x)

      # reset derivative papth
      self.use_deriv = False

    if y is not None:
      return y[0]
    else:
      return None
  # end forward

  def timer(self,name):
    return self.timer_manager.timer("ForWD::"+name)

  def getLayer(self,t,tf,level):
    index = self.getLocalTimeStepIndex(t,tf,level)
    if index < 0:
      return self.temp_layer

    return self.layer_models[index]

  def parameters(self):
    params = []
    for l in self.layer_models:
      if l!=None:
        params += [list(l.parameters())]

    return params

  def eval(self,y,tstart,tstop,level,done,x=None):
    """
    Method called by "my_step" in braid. This is
    required to propagate from tstart to tstop, with the initial
    condition x. The level is defined by braid
    """

    self.setLayerWeights(tstart,tstop,level,y.weightTensors())
    layer = self.temp_layer

    t_y = y.tensor().detach()

    # no gradients are necessary here, so don't compute them
    dt = tstop-tstart
    with torch.no_grad():
      k = torch.norm(t_y).item()
      q = dt*layer(t_y)
      kq = torch.norm(q).item()
      t_y.add_(q)
      del q

    tstop_index = self.getGlobalTimeIndex(tstop)
    self.setVectorWeights(tstop_index-self.start_layer,level,y)
  # end eval

  def getPrimalWithGrad(self,tstart,tstop):
    """ 
    Get the forward solution associated with this
    time step and also get its derivative. This is
    used by the BackwardApp in computation of the
    adjoint (backprop) state and parameter derivatives.
    """

    b_x = self.getUVector(0,tstart)

    ts_index = self.getGlobalTimeIndex(tstart)-self.start_layer

    assert(ts_index<len(self.layer_models))
    layer = self.layer_models[ts_index]

    t_x = b_x.tensor()
    x = t_x.detach()
    y = t_x.detach().clone()

    x.requires_grad = True 
    dt = tstop-tstart
    with torch.enable_grad():
      y = x + dt * layer(x)
    return (y, x), layer
  # end getPrimalWithGrad

# end ForwardODENetApp

##############################################################

class BackwardODENetApp(BraidApp):

  def __init__(self,fwd_app,timer_manager):
    # call parent constructor
    BraidApp.__init__(self,'BWDApp',
                           fwd_app.getMPIComm(),
                           fwd_app.local_num_steps,
                           fwd_app.Tf,
                           fwd_app.max_levels,
                           fwd_app.max_iters,
                           spatial_ref_pair=fwd_app.spatial_ref_pair)

    self.fwd_app = fwd_app

    # build up the core
    self.initCore()

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
      if f is not None:
        f = f[0]

      self.grads = []

      # preserve the layerwise structure, to ease communication
      # - note the prection of the 'None' case, this is so that individual layers
      # - can have gradient's turned off
      my_params = self.fwd_app.parameters()
      for sublist in my_params:
        sub_gradlist = [] 
        for item in sublist:
          if item.grad is not None:
            sub_gradlist += [ item.grad.clone() ] 
          else:
            sub_gradlist += [ None ]

        self.grads += [ sub_gradlist ]
      # end for sublist

      for l in self.fwd_app.layer_models:
         if l==None: continue
         l.zero_grad()
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
        (t_y,t_x),layer = self.fwd_app.getPrimalWithGrad(self.Tf-tstop,
                                                         self.Tf-tstart)
                                                         

        # t_x should have no gradient (for memory reasons)
        assert(t_x.grad is None)

        # we are going to change the required gradient, make sure they return
        # to where they started!
        required_grad_state = []

        # play with the layers gradient to make sure they are on apprpriately
        for p in layer.parameters(): 
          required_grad_state += [p.requires_grad]
          if done==1:
            if not p.grad is None:
              p.grad.data.zero_()
          else:
            # if you are not on the fine level, compute no parameter gradients
            p.requires_grad = False

        # perform adjoint computation
        t_w = w.tensor()
        t_w.requires_grad = False
        t_y.backward(t_w)

        # this little bit of pytorch magic ensures the gradient isn't
        # stored too long in this calculation (in particulcar setting
        # the grad to None after saving it and returning it to braid)
        t_w.copy_(t_x.grad.detach()) 

        for p,s in zip(layer.parameters(),required_grad_state):
          p.requires_grad = s
    except:
      print('\n**** Torchbraid Internal Exception: ' 
           +'backward eval: rank={}, level={}, time interval=({:.2f},{:.2f}) ****\n'.format(self.fwd_app.my_rank,level,tstart,tstop))
      traceback.print_exc()
  # end eval

# end BackwardODENetApp
