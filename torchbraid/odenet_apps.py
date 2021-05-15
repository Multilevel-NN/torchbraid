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
import numpy as np

from bsplines import BsplineBasis

from mpi4py import MPI

class ForwardODENetApp(BraidApp):

  def __init__(self,comm,layer_models,local_num_steps,Tf,max_levels,max_iters,timer_manager,spatial_ref_pair=None, layer_block=None, nsplines=0, splinedegree=1):
    """
    """
    BraidApp.__init__(self,'FWDApp',comm,local_num_steps,Tf,max_levels,max_iters,spatial_ref_pair=spatial_ref_pair,require_storage=True)

    # note that a simple equals would result in a shallow copy...bad! (SG: why would that be bad?)
    self.layer_models = [l for l in layer_models]

    comm          = self.getMPIComm()
    my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()
    self.my_rank = my_rank
    self.layer_block = layer_block

    # SpliNet 
    self.splinet=False
    if nsplines>0:
      self.splinet=True
      self.splinebasis = BsplineBasis(nsplines, splinedegree, Tf)

      # compute index offset for local layer storage in the layer_models vector
      nKnots = nsplines - splinedegree + 1
      spline_dknots = Tf / (nKnots - 1)
      if my_rank == 0: # First processor's time-interval includes t0_local=0.0. Others exclude t0_local, owning only (t0_local, tf_local]!
        self.splineoffset =  int( (self.t0_local ) / spline_dknots ) 
      else:
        self.splineoffset =  int( (self.t0_local + self.dt) / spline_dknots ) # index offset for local storage

      # print(my_rank, ": Spline offset: ", self.splineoffset)

    # send everything to the left (this helps with the adjoint method)
    # SG: This is probably not needed anymore. At least for the splinet version, it is not. 
    if not self.splinet:
      if my_rank>0:
        comm.send(list(self.layer_models[0].parameters()), dest=my_rank-1,tag=22)
      if my_rank<num_ranks-1:
        neighbor_model = comm.recv(source=my_rank+1,tag=22)
        new_model = layer_block()
        with torch.no_grad():
          for dest_p, src_w in zip(list(new_model.parameters()), neighbor_model):
            dest_p.data = src_w
        self.layer_models.append(new_model)
      else:
        # this is a sentinel at the end of the processors and layers
        self.layer_models.append(None)

    # build up the core
    self.py_core = self.initCore()

    self.timer_manager = timer_manager
    self.use_deriv = False

    self.parameter_shapes = []
    for p in layer_models[0].parameters(): 
      self.parameter_shapes += [p.data.size()]

    print(comm.Get_rank(), ": Creating the temp_layer.")
    self.temp_layer = layer_block()
    self.clearTempLayerWeights()
    print(comm.Get_rank(), ": Done.")
  # end __init__

  def __del__(self):
    pass

  def getTensorShapes(self):
    return list(self.shape0)+self.parameter_shapes

  def setVectorWeights(self,t,tf,level,x):
    if self.splinet: 
      with torch.no_grad():
        # Evaluate the splines at time t and get interval k such that t \in [\tau_k, \tau_k+1] for splineknots \tau
        splines, k = self.splinebasis.eval(t)
        # Add up sum over p+1 non-zero splines(t) times weights coeffients, l=0,\dots,p
        l = 0 # first one here, because I didn't know how to set the shape of 'weights' correctly...
        layermodel_localID = k + l - self.splineoffset
        assert layermodel_localID >= 0 and layermodel_localID < len(self.layer_models)
        layer = self.layer_models[layermodel_localID]
        weights = [splines[l] * p.data for p in layer.parameters()] # l=0
        # others: l=1,dots, p
        for l in range(1,len(splines)):
          layermodel_localID = k + l - self.splineoffset
          if t== self.Tf and l==len(splines)-1: # There is one more spline at Tf, which is zero at Tf and therefore it is not stored. Skip. 
            continue
          assert layermodel_localID >= 0 and layermodel_localID < len(self.layer_models)
          layer = self.layer_models[layermodel_localID]
          for dest_w, src_p in zip(weights, list(layer.parameters())):  
              dest_w.add_(src_p.data, alpha=splines[l])

    else:
      layer = self.getLayer(t,tf,level)
      if layer!=None:
        weights = [p.data for p in layer.parameters()]
      else:
        weights = []

    x.addWeightTensors(weights)
  # end setVectorWeights

  def clearTempLayerWeights(self):
    layer = self.temp_layer

    for dest_p in list(layer.parameters()):
      dest_p.data = torch.empty(())
  # end clearLayerWeights

  def setLayerWeights(self,t,tf,level,weights):
    layer = self.getLayer(t,tf,level)

    with torch.no_grad():
      for dest_p,src_w in zip(list(layer.parameters()),weights):
        dest_p.data = src_w
  # end setLayerWeights

  def initializeVector(self,t,x):
    # print(self.my_rank, ": InitializeVector(t=", t, ")")
    self.setVectorWeights(t,0.0,0,x)

  def updateParallelWeights(self):
    # send everything to the left (this helps with the adjoint method)
    comm          = self.getMPIComm()
    my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()

    if my_rank>0:
      comm.send(list(self.layer_models[0].parameters()), dest=my_rank-1,tag=22)
    if my_rank<num_ranks-1:
      neighbor_model = comm.recv(source=my_rank+1,tag=22)
      new_model = self.layer_block()
      with torch.no_grad():
        for dest_p, src_w in zip(list(new_model.parameters()), neighbor_model):
          dest_p.data = src_w
      self.layer_models[-1] = new_model
    else:
      # this is a sentinel at the end of the processors and layers
      self.layer_models.append(None)


  def run(self,x):
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

    if y is not None:
      return y[0]
    else:
      return None
  # end forward

  def timer(self,name):
    return self.timer_manager.timer("ForWD::"+name)

  def getLayer(self,t,tf,level):

    # if it is a splinet, use weights from temp_layer. It always contains the sum over spline-coeffs times spline weights, as set in setVectorWeights(tstop)
    if self.splinet:
      return self.temp_layer

    # if not a splinet, get the layer either from storage layer_models[i] or, if it is not stored on this proc, from the temp_layer 
    index = self.getLocalTimeStepIndex(t,tf,level)
    if index < 0:
      #pre_str = "\n{}: WARNING: getLayer index negative at {}: {}\n".format(self.my_rank,t,index)
      #stack_str = utils.stack_string('{}: |- '.format(self.my_rank))
      #print(pre_str+stack_str)
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

    # this function is used twice below to define an in place evaluation
    def in_place_eval(t_y,tstart,tstop,level,t_x=None):
      # get some information about what to do
      dt = tstop-tstart
      layer = self.getLayer(tstart,tstop,level) # if splinet, this returns temp_layer

      # print(self.my_rank, ": FWDeval level ", level, " ", tstart, "->", tstop)

      if t_x==None:
        t_x = t_y
      else:
        t_y.copy_(t_x)

      q = dt*layer(t_x)
      t_y.add_(q)
      del q
    # end in_place_eval

    # there are two paths by which eval is called:
    #  1. x is a BraidVector: my step has called this method
    #  2. x is a torch tensor: called internally (probably for the adjoint) 

    if isinstance(y,BraidVector):
      # SG: If splinet, this will pass y.weights to temp_layer
      self.setLayerWeights(tstart,tstop,level,y.weightTensors())

      t_y = y.tensor().detach()

      # no gradients are necessary here, so don't compute them
      with torch.no_grad():
        in_place_eval(t_y,tstart,tstop,level)

      if y.getSendFlag():
        self.clearTempLayerWeights()

        y.releaseWeightTensors()
        y.setSendFlag(False)
      # wipe out any sent information

      self.setVectorWeights(tstop,0.0,level,y) # if splinet, this will set y.weights to the sum over spline-coeffs(tstop) times spline-weights 

    else: 
      x.requires_grad = True 
      with torch.enable_grad():
        in_place_eval(y,tstart,tstop,level,t_x=x)
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
    
    layer = self.getLayer(tstart,tstop,level)

    # SG: does that contain layer weights?
    b_x = self.getUVector(0,tstart)
    t_x = b_x.tensor()

    self.setLayerWeights(tstart,tstop,level,b_x.weightTensors())

    x = t_x.detach()
    y = t_x.detach().clone()

    x.requires_grad = t_x.requires_grad

    self.eval(y,tstart,tstop,0,done=0,x=x)
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
      if f is not None:
        f = f[0]

      # Maybe communicate the splines here? 
      if self.fwd_app.splinet:
        print(self.getMPIComm().Get_rank(), "I will comunicate!")
        for i,layer in enumerate(self.fwd_app.layer_models):
            if layer is not None:
                for p in layer.parameters():
                  print(self.getMPIComm().Get_rank(), "Sp", i, "Grad", p.grad)
        # req = []
        # for splinecomm,i in self.fwd_app.spline_comm_vec:
        #   if splinecomm is not MPI.COMM_NULL: #?? 
        #     # pack the buffer
        #     grad = self.fwd_app.layer_models[i].grad() #?? 
        #     buffer = utils.pack_buffer(grad) #?? 

        #     # Non-blocking allreduce on this splines communicator
        #     req=splinecomm.Iallreduce(MPI.IN_PLACE, buffer, MPI.SUM)

        # # Finish up communication
        # # for splinecomm,i in self.fwd_app.spline_comm_vec:
        #   # if splinecomm is not MPI.COMM_NULL:
        #     MPI.Request.Wait(req)
        #     utils.unpack_buffer(self.fwd_app.layer_models[i].grad(), buffer)

      # this code is due to how braid decomposes the backwards problem
      # The ownership of the time steps is shifted to the left (and no longer balanced)
      first = 1
      if self.getMPIComm().Get_rank()==0:
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
        # print(self.fwd_app.my_rank, ": BWDeval level ", level, " ", tstart, "->", tstop, "done", done)
          
        # we need to adjust the time step values to reverse with the adjoint
        # this is so that the renumbering used by the backward problem is properly adjusted
        (t_y,t_x),layer = self.fwd_app.getPrimalWithGrad(self.Tf-tstop,self.Tf-tstart,level)

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
            # if you are not done, compute no parameter gradients
            p.requires_grad = False

        # perform adjoint computations
        t_w = w.tensor()
        t_w.requires_grad = False
        t_y.backward(t_w)

        # The above set's the gradient of the layer.parameters(), which, in case of a SpliNet is the templayer. Need to spread those sensitivities to each spline layer in the layer_models.
        if self.fwd_app.splinet:
          if done==1:
            with torch.no_grad(): # No idea if this is needed here... 
              splines, k = self.fwd_app.splinebasis.eval(self.Tf-tstop)

              # Spread derivavites to d+1 non-zero splines(t) times weights:
              # \bar L_{k+l} += splines[l] * layer.parameters().gradient
              for l in range(len(splines)): 
                # Get the layer whose gradient is to be updated
                layermodel_localID = k + l - self.fwd_app.splineoffset
                if (self.Tf-tstop == 0.0) and l==len(splines)-1: # There is one more spline at Tf, which is zero at Tf and therefore it is not stored. Skip. 
                  continue
                assert layermodel_localID >= 0 and layermodel_localID < len(self.fwd_app.layer_models)
                layer_out = self.fwd_app.layer_models[layermodel_localID]

                # Update the gradient of this layer
                for dest, src in zip(list(layer_out.parameters()), list(layer.parameters())):  
                  if dest.grad == None:
                    dest.grad = src.grad * splines[l]
                  else:
                    dest.grad.add_(src.grad, alpha=splines[l])

        # this little bit of pytorch magic ensures the gradient isn't
        # stored too long in this calculation (in particulcar setting
        # the grad to None after saving it and returning it to braid)
        t_w.copy_(t_x.grad.detach()) 

        for p,s in zip(layer.parameters(),required_grad_state):
          p.requires_grad = s
    except:
      print('\n**** Torchbraid Internal Exception ****\n')
      traceback.print_exc()
  # end eval

# end BackwardODENetApp
