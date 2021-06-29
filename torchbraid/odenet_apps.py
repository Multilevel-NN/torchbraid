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
from bsplines import BsplineBasis

from mpi4py import MPI

class ForwardODENetApp(BraidApp):

  def __init__(self,comm,local_num_steps,Tf,max_levels,max_iters,timer_manager,spatial_ref_pair=None, layer_block=None, nsplines=0, splinedegree=1):
    """
    """
    BraidApp.__init__(self,'FWDApp',comm,local_num_steps,Tf,max_levels,max_iters,spatial_ref_pair=spatial_ref_pair,require_storage=True)

    sys.stdout.flush()

    comm          = self.getMPIComm()
    my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()
    self.my_rank = my_rank
    self.layer_block = layer_block

    # If this is a SpliNet, create spline basis and overwrite local self.start_layer/end_layer 
    self.splinet = False
    if nsplines>0:
      self.splinet = True
      if comm.Get_rank() == 0:
        print("Torchbraid will create a SpliNet with ", nsplines, " spline basis functions of degree", splinedegree)

      self.splinebasis = BsplineBasis(nsplines, splinedegree, Tf)
      spline_dknots = Tf / (nsplines - splinedegree) # spacing of spline knots
      if comm.Get_rank() == 0: # First processor's time-interval includes t0_local=0.0. Others exclude t0_local, owning only (t0_local, tf_local]!
        self.start_layer = int( (self.t0_local ) / spline_dknots )
      else:
        self.start_layer = int( (self.t0_local + self.dt) / spline_dknots )
      self.end_layer = int( (self.tf_local ) / spline_dknots ) + splinedegree

    # Number of locally stored layers
    owned_layers = self.end_layer-self.start_layer+1
    if my_rank==num_ranks-1:
      # the last time step should not create a layer, there is no step being
      # taken on that final step
      owned_layers -= 1

    # Now creating the trainable layers
    self.layer_models = [self.layer_block() for _ in range(owned_layers)]

    self.timer_manager = timer_manager
    self.use_deriv = False

    self.parameter_shapes = []
    for p in self.layer_models[0].parameters(): 
      self.parameter_shapes += [p.data.size()]

    self.temp_layer = layer_block()
    self.clearTempLayerWeights()


    # If this is a SpliNet, create communicators for shared weights
    if self.splinet:
      # For each spline basis function, create one communicator that contains all processors that store this spline.
      self.spline_comm_vec = []
      for i in range(nsplines):
        group = comm.Get_group()  # all processors, then exclude those who don't store i
        exclude = []
        for k in range(comm.Get_size()):
          # recompute start_layer and end_layer for all other processors.
          dt = Tf/(local_num_steps*comm.Get_size())
          t0loc = k*local_num_steps*dt  
          tfloc = (k+1)*local_num_steps*dt
          if k == 0:
            startlayer = int( t0loc / spline_dknots )
          else :
            startlayer = int( (t0loc+self.dt) / spline_dknots )
          endlayer = int( tfloc / spline_dknots ) + splinedegree
          if k == comm.Get_size()-1:
            endlayer = endlayer-1
          if i < startlayer or i > endlayer:
            exclude.append(k)
        newgroup = group.Excl(exclude)
        # Finally create the communicator and store it. 
        thiscomm = comm.Create(newgroup) # This will be MPI.COMM_NULL on all processors that are excluded
        self.spline_comm_vec.append(thiscomm)  
      
      # for i,commsp in enumerate(self.spline_comm_vec):
      #   if commsp != MPI.COMM_NULL:
      #     # print(comm.Get_rank(), ": In communicator ", i, ": I'm rank ", self.spline_comm_vec[i].Get_rank(), "out of", self.spline_comm_vec[i].Get_size())
      #     if commsp.Get_rank() == 0:
      #       print("comm ", i," has size ", commsp.Get_size())
  # end __init__

  def __del__(self):
    pass

  def getTensorShapes(self):
    return list(self.shape0)+self.parameter_shapes

  def setVectorWeights(self,t,x):

    if self.splinet: 
      # Evaluate the splines at time t and get interval k such that t \in [\tau_k, \tau_k+1] for splineknots \tau
      with torch.no_grad():
        splines, k = self.splinebasis.eval(t)
        # Add up sum over p+1 non-zero splines(t) times weights coeffients, l=0,\dots,p
        l = 0 # first one here, because I didn't know how to set the shape of 'weights' correctly...
        layermodel_localID = k + l - self.start_layer
        assert layermodel_localID >= 0 and layermodel_localID < len(self.layer_models)
        layer = self.layer_models[layermodel_localID]
        weights = [splines[l] * p.data for p in layer.parameters()] # l=0
        # others: l=1,dots, p
        for l in range(1,len(splines)):
          layermodel_localID = k + l - self.start_layer
          if t== self.Tf and l==len(splines)-1: # There is one more spline at Tf, which is zero at Tf and therefore it is not stored. Skip. 
            continue
          assert layermodel_localID >= 0 and layermodel_localID < len(self.layer_models)
          layer = self.layer_models[layermodel_localID]
          for dest_w, src_p in zip(weights, list(layer.parameters())):  
              dest_w.add_(src_p.data, alpha=splines[l])

    else: 
      layer_index = self.getGlobalTimeIndex(t) - self.start_layer
      if layer_index<len(self.layer_models) and layer_index>=0:
        layer = self.layer_models[layer_index]
      else:
        layer = None

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
  # end clearTempLayerWeights

  def setLayerWeights(self,t,tf,level,weights):
    layer = self.temp_layer

    with torch.no_grad():
      for dest_p,src_w in zip(list(layer.parameters()),weights):
        dest_p.data = src_w
  # end setLayerWeights

  def initializeVector(self,t,x):
    self.setVectorWeights(t,x)

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

    # print(self.my_rank, ": FWDeval level ", level, " ", tstart, "->", tstop)
    self.setLayerWeights(tstart,tstop,level,y.weightTensors())
    layer = self.temp_layer
    # print(self.my_rank, ": FWDeval level ", level, " ", tstart, "->", tstop, [p.data for p in layer.parameters()])

    t_y = y.tensor().detach()

    # no gradients are necessary here, so don't compute them
    dt = tstop-tstart
    with torch.no_grad():
      k = torch.norm(t_y).item()
      q = dt*layer(t_y)
      kq = torch.norm(q).item()
      t_y.add_(q)
      del q

    # This connects weights at tstop with the vector y. For a SpliNet, the weights at tstop are evaluated using the spline basis function. 
    self.setVectorWeights(tstop,y)
  # end eval

  def getPrimalWithGrad(self,tstart,tstop):
    """ 
    Get the forward solution associated with this
    time step and also get its derivative. This is
    used by the BackwardApp in computation of the
    adjoint (backprop) state and parameter derivatives.
    """

    b_x = self.getUVector(0,tstart)

    # Set the layer at tstart. For a SpliNet, get the layer weights from x at tstart, otherwise, get layer and weights from storage.
    if self.splinet:
      self.clearTempLayerWeights()
      self.setLayerWeights(tstart,tstop,0,b_x.weightTensors())
      layer = self.temp_layer
    else:
      ts_index = self.getGlobalTimeIndex(tstart)-self.start_layer
      assert(ts_index<len(self.layer_models))
      assert(ts_index >= 0)
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

      # Communicate the spline gradients here. Alternatively, this could be done in braid_function.py: "backward(ctx, grad_output)" ?
      if self.fwd_app.splinet:
        # req = []
        for i,splinecomm in enumerate(self.fwd_app.spline_comm_vec):
          if splinecomm != MPI.COMM_NULL: 
            # print(splinecomm.Get_rank(), ": I will pack spline ", i)
            # pack the spline into a buffer and initiate non-blocking allredude
            splinelayer = self.fwd_app.layer_models[i - self.fwd_app.start_layer]
            buf = utils.pack_buffer([p.grad for p in splinelayer.parameters()])
            req=splinecomm.Iallreduce(MPI.IN_PLACE, buf, MPI.SUM)

        # Finish up communication. TODO: Queue all requests.
        # for i, splinecomm in enumerate(self.fwd_app.spline_comm_vec):
          # if splinecomm != MPI.COMM_NULL:
            MPI.Request.Wait(req)
            utils.unpack_buffer([p.grad for p in splinelayer.parameters()], buf)

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
      # print(self.getMPIComm().Get_rank(), " self.grads=", self.grads)

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
        (t_y,t_x),layer = self.fwd_app.getPrimalWithGrad(self.Tf-tstop,
                                                         self.Tf-tstart)
                                                         
        # print(self.fwd_app.my_rank, "--> FWD with layer ", [p.data for p in layer.parameters()])

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

        # The above set's the gradient of the layer.parameters(), which, in case of SpliNet, is the templayer -> need to spread those sensitivities to the layer_models
        if self.fwd_app.splinet and done==1:
          with torch.no_grad(): # No idea if this is actually needed here... 
            splines, k = self.fwd_app.splinebasis.eval(self.Tf-tstop)
            # Spread derivavites to d+1 non-zero splines(t) times weights:
            # \bar L_{k+l} += splines[l] * layer.parameters().gradient
            for l in range(len(splines)): 
              # Get the layer whose gradient is to be updated
              layermodel_localID = k + l - self.fwd_app.start_layer
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
      print('\n**** Torchbraid Internal Exception: ' 
           +'backward eval: rank={}, level={}, time interval=({:.2f},{:.2f}) ****\n'.format(self.fwd_app.my_rank,level,tstart,tstop))
      traceback.print_exc()
  # end eval

# end BackwardODENetApp
