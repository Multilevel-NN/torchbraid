# cython: profile=True
# cython: linetrace=True

import torch
from torchbraid_app import BraidApp
from torchbraid_app import BraidVector

from mpi4py import MPI

class ForewardBraidApp(BraidApp):

  def __init__(self,comm,layer_models,local_num_steps,Tf,max_levels,max_iters):
    BraidApp.__init__(self,comm,local_num_steps,Tf,max_levels,max_iters)

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

    self.x_final = None

    # build up the core
    self.py_core = self.initCore()
  # end __init__

  def run(self,x):
    y = self.runBraid(x)
    return y
  # end forward

  def getLayer(self,t,tf,level):
    index = self.getLocalTimeStepIndex(t,tf,level)
    return self.layer_models[index]

  def parameters(self):
    return [list(l.parameters()) for l in self.layer_models]

  def eval(self,x,tstart,tstop,level):
    dt = tstop-tstart

    with torch.no_grad(): 
      t_x = x.tensor()
      layer = self.getLayer(tstart,tstop,x.level())
      t_y = t_x+dt*layer(t_x)
      return BraidVector(t_y,x.level()) 
  # end eval

  def evalFwdWithGrad(self,tstart,tstop,level):
    dt = tstop-tstart
    finegrid = 0
 
    ts_index = self.getGlobalTimeStepIndex(tstart,tstop,level)

    # get the primal vector from the forward app
    px = self.getUVector(finegrid,ts_index)

    t_px = px.tensor().clone()
    t_px.requires_grad = True

    layer = self.getLayer(tstart,tstop,level)

    # enables gradient calculation 
    with torch.enable_grad():
      # turn off parameter gradients below the fine grid
      #  - for posterity record their old value
      grad_list = []
      if level!=0:
        for p in layer.parameters():
          grad_list += [p.requires_grad]
          p.requires_grad = False
      else:
        # clean up parameter gradients on fine level, 
        # they are only computed once
        layer.zero_grad()

      t_py = t_px+dt*layer(t_px)

      # turn gradients back on
      if level!=0:
        for p,g in zip(layer.parameters(),grad_list):
          p.requires_grad = g
    # end with torch.enable_grad
    
    return t_py,t_px
# end ForwardBraidApp

##############################################################

class BackwardBraidApp(BraidApp):

  def __init__(self,fwd_app):
    # call parent constructor
    BraidApp.__init__(self,fwd_app.getMPIData().getComm(),
                           fwd_app.local_num_steps,
                           fwd_app.Tf,
                           fwd_app.max_levels,
                           fwd_app.max_iters)

    self.fwd_app = fwd_app

    # build up the core
    self.py_core = self.initCore()

    # setup adjoint specific stuff
    self.fwd_app.setStorage(0)

    # reverse ordering for adjoint/backprop
    self.setRevertedRanks(1)
  # end __init__

  def run(self,x):
    f = self.runBraid(x)

    my_params = self.fwd_app.parameters()

    # this code is due to how braid decomposes the backwards problem
    # The ownership of the time steps is shifted to the left (and no longer balanced)
    first = 1
    if self.getMPIData().getRank()==0:
      first = 0

    # preserve the layerwise structure, to ease communication
    self.grads = [ [item.grad.clone() for item in sublist] for sublist in my_params[first:]]
    for m in self.fwd_app.layer_models:
       m.zero_grad()

    return f
  # end forward

  def eval(self,x,tstart,tstop,level):
    # we need to adjust the time step values to reverse with the adjoint
    # this is so that the renumbering used by the backward problem is properly adjusted
    t_py,t_px = self.fwd_app.evalFwdWithGrad(self.Tf-tstop,self.Tf-tstart,level)

    t_x = x.tensor()
    t_py.backward(t_x)

    return BraidVector(t_px.grad,level) 
  # end eval
# end BackwardBraidApp
