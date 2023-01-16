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

from torchbraid.torchbraid_app import BraidApp
import torchbraid.utils

import sys

from timeit import default_timer as timer
from mpi4py import MPI

class ForwardBraidApp(BraidApp):

  def __init__(self,comm,RNN_models,local_num_steps,Tf,max_levels,max_iters,timer_manager,user_mpi_buf=False):
    BraidApp.__init__(self,'FWRNN',
                      comm,
                      comm.Get_size()*local_num_steps,
                      Tf,
                      max_levels,
                      max_iters,
                      spatial_ref_pair=None,
                      user_mpi_buf=user_mpi_buf,
                      require_storage=True)

    self.RNN_models = RNN_models

    self.finalRelax()

    # set timing data output file
    self.setTimerFile("braid_forward_timings")

    self.my_rank = self.getMPIComm().Get_rank()

    self.timer_manager = timer_manager
    self.use_deriv = False

    self.seq_shapes = None
    self.backpropped = dict()
    self.seq_x_reduced = dict()

    self.has_fastforward = hasattr(self.RNN_models,'fastForward')
    self.fastforward_time  = 0.0
    self.fastforward_calls = 0
    if self.has_fastforward:
      assert(hasattr(self.RNN_models,'reduceX'))
  # end __init__

  def getFastForwardInfo(self):
    return self.fastforward_time, self.fastforward_calls

  def computeStep(self,level,tstart,tstop,seq_x,u,allow_ff):
    """
    This method handles a fast forward evaluation. If the user has not
    defined a RNN cell that can handle fast forward, a standard computation
    is performed. If the cell can handle the fast forward, then the reduced
    version of the sequence is computed or pulled from a dictionary. If its
    computed, the computed reduced value of the sequence is stored for
    reuse. Regardless of it being computed or pulled from storage, the fastForward
    version of the RNNcell is called (which used the reduced version of the
    sequence variable.
    """

    if allow_ff:
      if tstart not in self.seq_x_reduced:
        # don't differentiate this

        start_timer = timer()
        with torch.no_grad():
          seq_x_reduce = self.RNN_models.reduceX(seq_x)
        stop_timer = timer()

        self.fastforward_time  +=  stop_timer-start_timer
        self.fastforward_calls += 1

        # store for later reuse
        self.seq_x_reduced[tstart] = seq_x_reduce
      else:
        seq_x_reduce = self.seq_x_reduced[tstart]

      return self.RNN_models.fastForward(level,tstart,tstop,seq_x_reduce,u)
    else:
      return self.RNN_models(level,tstart,tstop,seq_x,u)

  def timer(self,name):
    return self.timer_manager.timer("ForWD::"+name)

  def getFeatureShapes(self,tidx,level):
    return self.shape0

  def getParameterShapes(self,tidx,level):
    return self.seq_shapes

  def getDataVectorIndex(self,t):
    shift = 1 if self.my_rank>0 else 0

    return self.getGlobalTimeIndex(t)-self.start_layer+shift

  def getSequenceVector(self,t):
    index = self.getDataVectorIndex(t)
    if index<0:
      my_rank = self.my_rank
      pre_str  = "\n{}: WARNING: getSequenceVector index negative at {}: {}\n".format(my_rank,t,index)
      pre_str += "\n{}:          step index = {}, start_layer = {}, stop_layer = {}\n".format(my_rank,self.getTimeStepIndex(),self.start_layer,self.end_layer)
      stack_str = utils.stack_string('{}: |- '.format(my_rank))
      print(pre_str+stack_str)
      sys.stdout.flush()

    if index<self.x.shape[1]:
      value = self.x[:,index,:]
    else:
      # this is a sentinnel
      value = self.x[:,0,:].detach().clone()

    return value

  def initializeVector(self,t,x):
    try:
      if t!=0.0: # don't change the initial condition
        for ten in x.tensors():
          ten[:] = 0.0
      value = self.getSequenceVector(t)
      x.addWeightTensors((value,))

      # if fast forward is available do an early evaluation of the
      # sequence preemptively
      if self.has_fastforward:
        start_timer = timer()
        with torch.no_grad():
          self.seq_x_reduced[t] = self.RNN_models.reduceX(value)
        stop_timer = timer()

        self.fastforward_time  +=  stop_timer-start_timer
        self.fastforward_calls += 1
    except:
      print('\n**** InitializeVector: Torchbraid Internal Exception ****\n')
      sys.stdout.flush()
      traceback.print_exc()


  def run(self,x,h_c):
    self.use_deriv = self.training

    comm          = self.mpi_comm
    num_ranks     = self.mpi_comm.Get_size()
    my_rank       = self.my_rank

    assert(x.shape[1]==self.local_num_steps)

    self.fastforward_time  = 0.0 # we will measure new fastorward time
    self.fastforward_calls = 0
    self.seq_x_reduced = dict()

    # TODO: We're using the GPU -> CPU -> CPU -> GPU model here - might be a better way to reorganize
    #   this with the same logic we're using for the rest of it.

    self.x = x.cpu()
    self.seq_shapes = [x[:,0,:].shape]

    with self.timer("run:precomm"):
      # receive data vector from the right
      recv_request = None
      if my_rank<num_ranks-1:
        neighbor_x = torch.zeros(x[:,0,:].shape)
        recv_request = comm.Irecv(neighbor_x.numpy(),source=my_rank+1,tag=22)

      # send data vector to the left
      send_request = None
      if my_rank>0:
        send_request = comm.Isend(np.ascontiguousarray(self.x[:,0,:].numpy()),dest=my_rank-1,tag=22)

      if recv_request:
        recv_request.Wait()
        self.x = torch.cat((self.x,neighbor_x.unsqueeze(1)), 1)

      if send_request:
        send_request.Wait()
    # end with timer

    self.x = self.x.to(x.device)

    comm.Barrier()
    with self.timer("run:runBraid"):
      y = self.runBraid(h_c)

    # TODO: We're using the GPU -> CPU -> CPU -> GPU model here - might be a better way to reorganize
    #   this with the same logic we're using for the rest of it.

    with self.timer("run:postcomm"):
      y_cpu = None
      if y!=None:
        y_cpu = [s.cpu() for s in y]
      y = comm.bcast(y_cpu,root=num_ranks-1)
      y = tuple([s.to(self.device) for s in y])

    # y is a tuple with the final h,c components
    return y
  # end forward

  def parameters(self):
    params = list(self.RNN_models.parameters())

    return params

  def eval(self,g0,tstart,tstop,level,done):
    """
    Method called by "my_step" in braid. This is
    required to propagate from tstart to tstop, with the initial
    condition x. The level is defined by braid
    """

    with self.timer("eval"):

      seq_x = g0.weightTensors()[0].to(self.x.device)

      # don't need derivatives or anything, just compute
      if not done or level>0:
        u = [g.to(self.x.device) for g in g0.tensors()]
        with torch.no_grad():
          y = self.computeStep(level,tstart,tstop,seq_x,u,self.has_fastforward)
      else:
        # setup the solution vector for derivatives
        u = tuple([t.detach().to(self.x.device) for t in g0.tensors()])
        for t in u:
          t.requires_grad = True

        with torch.enable_grad():
          y = self.computeStep(level,tstart,tstop,seq_x,u,allow_ff=False)

        # store the fine level solution for reuse later in backprop
        if level==0:
          self.backpropped[tstart,tstop] = (u,y)

      seq_x = self.getSequenceVector(tstop)

      g0.addWeightTensors((seq_x,))
      for i,t in enumerate(y):
        g0.replaceTensor(t,i)
  # end eval

  def getPrimalWithGrad(self,tstart,tstop,level,done):
    """
    Get the forward solution associated with this
    time step and also get its derivative. This is
    used by the BackwardApp in computation of the
    adjoint (backprop) state and parameter derivatives.
    Its intent is to abstract the forward solution
    so it can be stored internally instead of
    being recomputed.
    """

    # use the short cut precomputed (with derivatives)
    if level==0 and (tstart,tstop) in self.backpropped:
      with self.timer("getPrimalWithGrad-short"):
        u,y = self.backpropped[(tstart,tstop)]
      return y,u, self.RNN_models

    with self.timer("getPrimalWithGrad-long"):
      # extract the various vectors for this value from the fine level to linearize around
      b_u = self.getUVector(0,tstart)

      seq_x = self.getSequenceVector(tstart)
      u = tuple([v.detach() for v in b_u.tensors()])

      for t in u:
        t.requires_grad = True

      # evaluate the step
      with torch.enable_grad():
        y = self.computeStep(level,tstart,tstop,seq_x,u,allow_ff=self.has_fastforward)

    sys.stdout.flush()

    return y, u, self.RNN_models
  # end getPrimalWithGrad

# end ForwardBraidApp

##############################################################

class BackwardBraidApp(BraidApp):

  def __init__(self,fwd_app,timer_manager):
    # call parent constructor
    BraidApp.__init__(self,'BWRNN',fwd_app.getMPIComm(),
                          fwd_app.getMPIComm().Get_size()*fwd_app.local_num_steps,
                          fwd_app.Tf,
                          fwd_app.max_levels,
                          fwd_app.max_iters,spatial_ref_pair=None,require_storage=True)

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

  def initializeVector(self,t,x):
    # this is being really careful, can't
    # change the intial condition
    if t!=0.0:
      for ten in x.tensors():
        ten[:] = 0.0

  def timer(self,name):
    return self.timer_manager.timer("BckWD::"+name)

  def run(self,x):

    sys.stdout.flush()
    try:
      f = self.runBraid(x)

      my_params = self.fwd_app.parameters()
      self.grads = []
      for item in my_params:
        if item.grad is not None:
          self.grads += [ item.grad.detach().clone() ]
        else:
          self.grads += [ None ]

      # required otherwise we will re-add the gradients
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
    with self.timer("eval"):
      try:

        # we need to adjust the time step values to reverse with the adjoint
        # this is so that the renumbering used by the backward problem is properly adjusted
        t_y,t_x,rnn_models = self.fwd_app.getPrimalWithGrad(self.Tf-tstop,self.Tf-tstart,level,done)

        # play with the parameter gradients to make sure they are on apprpriately,
        # store the initial state so we can revert them later
        required_grad_state = []
        if done!=1:
          for p in rnn_models.parameters():
            required_grad_state += [p.requires_grad]
            p.requires_grad = False

        # perform adjoint computation
        t_w = w.tensors()
        s_w = torch.stack(t_w)
        with torch.enable_grad():
          s_y = torch.stack(t_y)
        s_w.requires_grad = False
        s_y.backward(s_w,retain_graph=True)

        # this little bit of pytorch magic ensures the gradient isn't
        # stored too long in this calculation (in particulcar setting
        # the grad to None after saving it and returning it to braid)
        for wv,xv in zip(t_w,t_x):
          wv.copy_(xv.grad.detach())
          xv.grad = None

        # revert the gradient state to where they started
        if done!=1:
          for p,s in zip(rnn_models.parameters(),required_grad_state):
            p.requires_grad = s
      except:
        print('\n**** Torchbraid Internal Exception ****\n')
        traceback.print_exc()
  # end eval

# end BackwardODENetApp
