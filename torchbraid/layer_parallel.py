# cython: profile=True
# cython: linetrace=True

import torch
import torch.nn as nn

from mpi4py import MPI

import copy

from torchbraid_function import BraidFunction

import torchbraid_apps as apps

##
# Define your Python Braid Vector

#  a python level module
##########################################################

class ODEBlock(nn.Module):
  def __init__(self,layer,dt):
    super(ODEBlock, self).__init__()

    self.dt = dt
    self.layer = layer

  def forward(self, x):
    return x + self.dt*self.layer(x)
# end ODEBlock

class LayerParallel(nn.Module):

  def __init__(self,comm,layer_block,num_steps,Tf,max_levels=1,max_iters=10):
    super(Model,self).__init__()

    # optional parameters
    global_steps = num_steps*comm.Get_size()

    self.dt = Tf/global_steps
  
    self.layer_block = layer_block
    self.layer_models = [layer_block() for i in range(num_steps)]
    self.local_layers = nn.Sequential(*self.layer_models)

    self.fwd_app = apps.ForewardBraidApp(comm,self.layer_models,num_steps,Tf,max_levels,max_iters)
    self.bwd_app = apps.BackwardBraidApp(self.fwd_app)

    self.param_size = 0
  # end __init__

  def zero_grad(self):
    for l in self.fwd_app.layer_models:
      l.zero_grad()
    self.local_layers.zero_grad()

  def setPrintLevel(self,print_level):
    self.fwd_app.setPrintLevel(print_level)
    self.bwd_app.setPrintLevel(print_level)

  def setNumRelax(self,relax):
    self.fwd_app.setNumRelax(relax)
    self.bwd_app.setNumRelax(relax)

  def setCFactor(self,cfactor):
    self.fwd_app.setCFactor(cfactor)
    self.bwd_app.setCFactor(cfactor)

  def setSkipDowncycle(self,skip):
    self.fwd_app.setSkipDowncycle(skip)
    self.bwd_app.setSkipDowncycle(skip)

  def getMPIData(self):
    return self.fwd_app.getMPIData()

  def forward(self,x):
    # we are doing this to take adavtage of
    # pytorch's autograd which functions "naturally"
    # with the torch.autograd.function
    params = list(self.parameters())
    return BraidFunction.apply(self.fwd_app,self.bwd_app,x,*params) 
  # end forward

  def buildInit(self,t):
    x = self.x0.clone()
    if t>0:
      t_x = x.tensor()
      t_x[:] = 0.0
    return x

  # This method copies the layerr parameters and can be used for verification
  def buildSequentialOnRoot(self):
    ode_layers    = [ODEBlock(copy.deepcopy(l),self.dt) for l in self.layer_models]
    remote_layers = ode_layers
    build_seq_tag = 12         # this 
    comm          = self.getMPIData().getComm()
    my_rank       = self.getMPIData().getRank()
    num_ranks     = self.getMPIData().getSize()

    # short circuit for serial case
    if num_ranks==1:
      return nn.Sequential(*remote_layers)

    if my_rank==0:
      for i in range(1,self.getMPIData().getSize()):
        remote_layers += comm.recv(source=i,tag=build_seq_tag)
      return nn.Sequential(*remote_layers)
    else:
      comm.send(ode_layers,dest=0,tag=build_seq_tag)
      return None
  # end buildSequentialOnRoot

  def getFinal(self):
    return  self.fwd_app.getFinal()

  def getFinalOnRoot(self):
    build_seq_tag = 99        # this 
    comm          = self.getMPIData().getComm()
    my_rank       = self.getMPIData().getRank()
    num_ranks     = self.getMPIData().getSize()

    # short circuit for serial case
    if num_ranks==1:
      return self.getFinal()

    # send the output of the last layer to the root
    if my_rank==0:
      remote_final = comm.recv(source=num_ranks-1,tag=build_seq_tag)
      return remote_final
    elif my_rank==num_ranks-1:
      final = self.getFinal()
      comm.send(final,dest=0,tag=build_seq_tag)

    return None

  def copyVectorFromRoot(self,vec):
    build_seq_tag = 99        # this 
    comm          = self.getMPIData().getComm()
    my_rank       = self.getMPIData().getRank()
    num_ranks     = self.getMPIData().getSize()

    # short circuit for serial case
    if num_ranks==1:
      return vec

    # send the output of the last layer to the root
    if my_rank==0:
      for dest in range(1,num_ranks):
        comm.send(vec,dest,tag=build_seq_tag)
      return vec
    else:
      result = comm.recv(source=0,tag=build_seq_tag)
      return result

# end Model
