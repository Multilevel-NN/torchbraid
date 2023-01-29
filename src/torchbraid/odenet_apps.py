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

from collections import OrderedDict # for 

import torch
import torch.nn as nn

from .braid_vector import BraidVector
from torchbraid.torchbraid_app import BraidApp
from torchbraid.bsplines import BsplineBasis
import torchbraid.utils
import itertools

import sys
import traceback
import resource
import copy

from bisect import bisect_right
from mpi4py import MPI

class ForwardODENetApp(BraidApp):
  class ODEBlock(nn.Module):
    """This is a helper class to wrap layers that should be ODE time steps."""
    def __init__(self,layer):
      super(ForwardODENetApp.ODEBlock, self).__init__()

      self.layer = layer

    def forward(self,dt, x):
      y = dt*self.layer(x)
      y.add_(x)
      return y
  # end ODEBlock

  class PlainBlock(nn.Module):
    """This is a helper class to wrap layers that are not ODE time steps."""
    def __init__(self,layer):
      super(ForwardODENetApp.PlainBlock, self).__init__()

      self.layer = layer

    def forward(self,dt, x):
      return self.layer(x)
  # end ODEBlock

  def __init__(self,comm,layers,Tf,max_levels,max_iters,timer_manager,spatial_ref_pair=None,user_mpi_buf=False,nsplines=0, splinedegree=1):
    """
    """
    self.layer_blocks,num_steps = self.buildLayerBlocks(layers)

    BraidApp.__init__(self,'FWDApp',
                      comm,
                      num_steps,
                      Tf,
                      max_levels,
                      max_iters,
                      spatial_ref_pair=spatial_ref_pair,
                      user_mpi_buf=user_mpi_buf,
                      require_storage=True)

    self.finalRelax()

    # set timing data output file
    self.setTimerFile("braid_forward_timings")

    comm          = self.getMPIComm()
    my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()
    self.my_rank = my_rank

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
      if comm.Get_rank() == num_ranks-1: # Last processor's time-interval excludes tf_local because no step is being done from there.
        self.end_layer = int( (self.tf_local - self.dt ) / spline_dknots ) + splinedegree
      else:
        self.end_layer = int( (self.tf_local ) / spline_dknots ) + splinedegree

    # Number of locally stored layers
    owned_layers = self.end_layer-self.start_layer+1
    if my_rank==num_ranks-1 and not self.splinet:
      # the last time step should not create a layer, there is no step being
      # taken on that final step
      owned_layers -= 1

    # Now creating the trainable layers
    self.layer_models = [self.buildLayerBlock(self.start_layer+i) for i in range(owned_layers)]

    if self.use_cuda:
      torch.cuda.synchronize()

    self.timer_manager = timer_manager
    self.use_deriv = False

    self.parameter_shapes = []
    for layer_constr in self.layer_blocks[1]:
      # build the layer on the proper device
      layer = layer_constr()
      #self.parameter_shapes += [[p.data.size() for p in layer.parameters()]]

      # sanity check
      sd = layer.state_dict()
      for k in sd:
        assert isinstance(sd[k],torch.Tensor)
      self.parameter_shapes += [[sd[k].size() for k in layer.state_dict()]]

    self.temp_layers = dict()

    # If this is a SpliNet, create communicators for shared weights
    if self.splinet:
      # For each spline basis function, create one communicator that contains all processors that store this spline.
      self.spline_comm_vec = []
      for i in range(nsplines):
        group = comm.Get_group()  # all processors, then exclude those who don't store i
        exclude = []
        for k in range(comm.Get_size()):
          # recompute start_layer and end_layer for all other processors.
          dt = Tf/(self.local_num_steps*comm.Get_size())
          t0loc = k*self.local_num_steps*dt  
          tfloc = (k+1)*self.local_num_steps*dt
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
      
  # end __init__

  def __del__(self):
    pass

  def buildShapes(self,x):
    """Do a dry run to determine all the shapes that need to be built."""
    shapes = [x.shape]
    for layer_constr in self.layer_blocks[1]:
      # build the layer on the proper device
      layer = layer_constr().to(self.device) 
       
      x = layer(x)
      shapes += [x.shape]

    return shapes

  def buildLayerBlocks(self,layers):
    # this block of code prepares the data for easy sorting
    [counts,layer_blocks] = list(zip(*layers))
    layer_indices = list(itertools.accumulate(counts))

    num_steps = layer_indices[-1]
    return (layer_indices,layer_blocks,counts),num_steps

  def buildLayerBlock(self,i):
    """
    This function returns a block (e.g. a lambda that constructs the layer).
    """
    ind = bisect_right(self.layer_blocks[0],i)
    layer = self.layer_blocks[1][ind]()
    if self.layer_blocks[2][ind]==1:
      # if its just one time step, assume the user wants only a scalar
      layer = self.PlainBlock(layer)
    else:
      layer = self.ODEBlock(layer)

    return layer.to(self.device)

  def getTempLayer(self,t):
    """
    This function returns a pytorch layer module. A dictionary is used
    to cache the construction and search. These will be used to make sure
    that any parallel communication is correctly handled even if the layer
    is of a different type.
    """
    i = self.getGlobalTimeIndex(t)
    ind = bisect_right(self.layer_blocks[0],i)

    # using a dictionary to cache previously built temp layers
    if ind in self.temp_layers:
      result = self.temp_layers[ind]
    else:
      result = self.buildLayerBlock(i)
      self.temp_layers[ind] = result
    
    return result

  def getFeatureShapes(self,tidx,level):
    i = self.getFineTimeIndex(tidx,level)
    ind = bisect_right(self.layer_blocks[0],i)
    return [self.shape0[ind],]

  def getParameterShapes(self,tidx,level):
    if len(self.parameter_shapes)<=0:
      return []
    i = self.getFineTimeIndex(tidx,level)
    ind = bisect_right(self.layer_blocks[0],i)

    return self.parameter_shapes[ind]

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
        # weights = [p.data for p in layer.parameters()]
        sd = layer.state_dict()
        weights = [sd[k] for k in sd]
      else:
        weights = []

    x.addWeightTensors(weights)
  # end setVectorWeights

  def setLayerWeights(self,t,tf,level,weights):
    layer = self.getTempLayer(t)

    with torch.no_grad():
      #for dest_p,src_w in zip(list(layer.parameters()),weights):
      #  dest_p.data = src_w

      sd = layer.state_dict()
      keys = [k for k in sd]
      assert len(keys)==len(weights)

      pairs = zip(keys,weights)
      layer.load_state_dict(OrderedDict(pairs))
  # end setLayerWeights

  def initializeVector(self,t,x):
    self.setVectorWeights(t,x)

  def run(self,x):
    # turn on derivative path (as requried)
    self.use_deriv = self.training
    
    # instead of doing runBraid, can execute tests
    #self.testBraid(x)

    # run the braid solver
    self.getMPIComm().Barrier()
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

    self.setLayerWeights(tstart,tstop,level,y.weightTensors())
    layer = self.getTempLayer(tstart)

    t_y = y.tensor().detach()

    # no gradients are necessary here, so don't compute them
    dt = tstop-tstart
    with torch.no_grad():
      ny = layer(dt,t_y)
      y.replaceTensor(ny) 

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
      self.setLayerWeights(tstart,tstop,0,b_x.weightTensors())
      layer = self.getTempLayer(tstart)
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
      y = layer(dt,x)
    return (y, x), layer
  # end getPrimalWithGrad

# end ForwardODENetApp

##############################################################

class BackwardODENetApp(BraidApp):

  def __init__(self,fwd_app,timer_manager,max_levels=-1):
    # call parent constructor
    if max_levels == -1:
        max_levels = fwd_app.max_levels
    BraidApp.__init__(self,'BWDApp',
                           fwd_app.getMPIComm(),
                           fwd_app.getMPIComm().Get_size()*fwd_app.local_num_steps,
                           fwd_app.Tf,
                           max_levels,
                           fwd_app.max_iters,
                           spatial_ref_pair=fwd_app.spatial_ref_pair,
                           user_mpi_buf=fwd_app.user_mpi_buf)


    self.fwd_app = fwd_app

    # build up the core
    self.initCore()

    # reverse ordering for adjoint/backprop
    self.setRevertedRanks(1)

    # force evaluation of gradients at end of up-cycle
    self.finalRelax()

    # set timing data output file
    self.setTimerFile("braid_backward_timings")

    self.timer_manager = timer_manager
  # end __init__

  def __del__(self):
    self.fwd_app = None

  def timer(self,name):
    return self.timer_manager.timer("BckWD::"+name)

  def run(self,x):
    
    # instead of doing runBraid, can execute tests
    #self.testBraid(x)

    try:

      with self.timer("runBraid"):
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
            buf = torchbraid.utils.pack_buffer([p.grad for p in splinelayer.parameters()])
            req=splinecomm.Iallreduce(MPI.IN_PLACE, buf, MPI.SUM)

            # Finish up communication. TODO: Queue all requests.
            MPI.Request.Wait(req)
            torchbraid.utils.unpack_buffer([p.grad for p in splinelayer.parameters()], buf)
      # end splinet

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

  def getFeatureShapes(self,tidx,level):
    fine_idx = self.getFineTimeIndex(tidx,level)
    # need to map back to the global fine index on the forward grid
    return self.fwd_app.getFeatureShapes(self.num_steps-fine_idx,0)

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
        w.replaceTensor(t_x.grad.detach().clone()) 

        for p,s in zip(layer.parameters(),required_grad_state):
          p.requires_grad = s
    except:
      print('\n**** Torchbraid Internal Exception: ' 
           +'backward eval: rank={}, level={}, time interval=({:.2f},{:.2f}) ****\n'.format(self.fwd_app.my_rank,level,tstart,tstop))
      traceback.print_exc()
  # end eval

# end BackwardODENetApp
