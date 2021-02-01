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

def getMaxMemory(comm,message):
  usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

  total_usage = comm.reduce(usage,op=MPI.SUM)
  min_usage = comm.reduce(usage,op=MPI.MIN)
  max_usage = comm.reduce(usage,op=MPI.MAX)
  if comm.Get_rank()==0:
    result = '%.2f MiB, (min,avg,max)=(%.2f,%.2f,%.2f)' % (total_usage/2**20, min_usage/2**20,total_usage/comm.Get_size()/2**20,max_usage/2**20)
    print(message.format(result))

def getLocalMemory(comm,message):
  usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

  result = '%.2f MiB' % (usage/2**20)
  print(('{}) ' + message).format(comm.Get_rank(),result))

class ForwardODENetApp(BraidApp):

  def __init__(self,comm,layer_models,local_num_steps,Tf,max_levels,max_iters,timer_manager,spatial_ref_pair=None, layer_block=None):
    """
    """
    BraidApp.__init__(self,'FWDApp',comm,local_num_steps,Tf,max_levels,max_iters,spatial_ref_pair=spatial_ref_pair,require_storage=True)

    # note that a simple equals would result in a shallow copy...bad!
    self.layer_models = [l for l in layer_models]

    comm          = self.getMPIComm()
    my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()
    self.my_rank = my_rank

    # send everything to the left (this helps with the adjoint method)
    if my_rank>0:
      # comm.send(self.layer_models[0],dest=my_rank-1,tag=22)
      comm.send(list(self.layer_models[0].parameters()), dest=my_rank-1,tag=22)

    self.layer_block = layer_block
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

    # self.temp_layer = copy.deepcopy(self.layer_models[0])
    self.temp_layer = layer_block()
    self.clearTempLayerWeights()
  # end __init__

  def __del__(self):
    pass

  def getTensorShapes(self):
    return list(self.shape0)+self.parameter_shapes

  def setVectorWeights(self,t,tf,level,x):
    layer = self.getLayer(t,tf,level)
    if layer!=None:
      weights = [p.data for p in layer.parameters()]
    else:
      weights = []
    x.addWeightTensors(weights)

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
    self.setVectorWeights(t,0.0,0,x)

  def updateParallelWeights(self):
    # send everything to the left (this helps with the adjoint method)
    comm          = self.getMPIComm()
    my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()

    # if my_rank>0:
    #   comm.send(self.layer_models[0],dest=my_rank-1,tag=22)
    # if my_rank<num_ranks-1:
    #   neighbor_model = comm.recv(source=my_rank+1,tag=22)
    #   self.layer_models[-1] = neighbor_model

    if my_rank>0:
      # comm.send(self.layer_models[0],dest=my_rank-1,tag=22)
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
    # with self.timer("runBraid"):

    # do boundary exchange for parallel weights
    if self.use_deriv:
      self.updateParallelWeights()

    y = self.runBraid(x)

    # reset derivative papth
    self.use_deriv = False

    return y
  # end forward

  def timer(self,name):
    return self.timer_manager.timer("ForWD::"+name)

  def getLayer(self,t,tf,level):
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

  def eval(self,y,tstart,tstop,level,x=None):
    """
    Method called by "my_step" in braid. This is
    required to propagate from tstart to tstop, with the initial
    condition x. The level is defined by braid
    """

    # this function is used twice below to define an in place evaluation
    def in_place_eval(t_y,tstart,tstop,level,t_x=None):
      # get some information about what to do
      dt = tstop-tstart
      layer = self.getLayer(tstart,tstop,level) # resnet "basic block"

      #print(self.my_rank, ": FWDeval level ", level, " ", tstart, "->", tstop, " using layer ", layer.getID(), ": ", layer.linearlayer.weight[0].data)

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

      self.setVectorWeights(tstop,0.0,level,y)

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

    b_x = self.getUVector(0,tstart)
    t_x = b_x.tensor()

    self.setLayerWeights(tstart,tstop,level,b_x.weightTensors())

    x = t_x.detach()
    y = t_x.detach().clone()

    x.requires_grad = t_x.requires_grad

    self.eval(y,tstart,tstop,0,x=x)
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

  def eval(self,w,tstart,tstop,level):
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
        assert(t_x.grad is None)

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
      print('\n**** Torchbraid Internal Exception ****\n')
      traceback.print_exc()
  # end eval

# end BackwardODENetApp
