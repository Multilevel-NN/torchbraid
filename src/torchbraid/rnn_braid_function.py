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

import numpy as np
import torch.autograd
import torchbraid.utils as utils
import traceback

from torch.nn.functional import pad
from mpi4py import MPI

class BraidFunction(torch.autograd.Function):

  @staticmethod
  def padForBatchChange(old_batch,temp_batch,ten,batch_dim):
    shape = ten.size()
    assert(len(shape)>batch_dim)

    padding = []
    for i in range(len(shape)-batch_dim-1):
      padding += [0,0]
    padding += [0,old_batch-temp_batch]
    return pad(ten,tuple(padding),'constant',0.0)

  @staticmethod
  def forward(ctx, fwd_app, bwd_app, num_input_tensors, x, *input_and_param_tensors):
    comm          = fwd_app.getMPIComm()
    my_rank       = fwd_app.getMPIComm().Get_rank()
    num_ranks     = fwd_app.getMPIComm().Get_size()

    fwd_app.setDevice(x.device)
    bwd_app.setDevice(x.device)

    # copy the input to all processors (ensure consistency)
    with fwd_app.timer("func:precomm"):
      sizes = tuple([input_and_param_tensors[i].size() for i in range(num_input_tensors)])
      shape = list(comm.bcast(sizes,root=0))

    old_shape = fwd_app.getShape()
    adjusting = old_shape is not None and old_shape!=shape

    # setup context
    ctx.fwd_app = fwd_app
    ctx.bwd_app = bwd_app
    ctx.num_input_tensors = num_input_tensors
    ctx.adjusting = adjusting
    ctx.save_for_backward(x, *input_and_param_tensors)
    ctx.device = x.device

    if adjusting:
      old_batch  = old_shape[0][1]
      temp_batch = shape[0][1]
      state = tuple([BraidFunction.padForBatchChange(old_batch,temp_batch,input_and_param_tensors[i],1) for i in range(num_input_tensors)])
      x = BraidFunction.padForBatchChange(old_batch,temp_batch,x,0)
      ctx.old_batch = old_batch
      ctx.temp_batch = temp_batch
    else:
      fwd_app.setShape(shape)
      bwd_app.setShape(shape)

      state = tuple([input_and_param_tensors[i] for i in range(num_input_tensors)])


    with fwd_app.timer("func:run"):
      result = fwd_app.run(x,state)
    if num_input_tensors==1:
      result = result[0]

    if adjusting:
      return result[:,0:temp_batch,:]
    else:
      return result

  @staticmethod
  def backward(ctx, *grad_state):
    comm          = ctx.bwd_app.getMPIComm()
    my_rank       = ctx.bwd_app.getMPIComm().Get_rank()
    num_ranks     = ctx.bwd_app.getMPIComm().Get_size()
    device        = ctx.device

    # copy the input to the final processor (where iter time integration begins)
    with ctx.bwd_app.timer("func:precomm"):
      if num_ranks>1:
        if my_rank==num_ranks-1: 
          grad_state = torch.stack(grad_state)
          req = comm.Irecv(grad_state.cpu().numpy(),source=0,tag=22)
          req.Wait()

        if my_rank==0:
          grad_state = torch.stack(grad_state)
          grad_state_cpu = grad_state.cpu()
          comm.Isend(grad_state_cpu.numpy(),dest=num_ranks-1,tag=22)
          grad_state = grad_state_cpu.to(device)

        grad_state = tuple([grad_state[i] for i in range(len(grad_state))])
      # end if num_ranks
    # end with

    with ctx.bwd_app.timer("func:run"):
      if my_rank==num_ranks-1:
        if ctx.adjusting:
          grad_state = tuple([BraidFunction.padForBatchChange(ctx.old_batch,ctx.temp_batch,t,1) for t in grad_state])
        result = ctx.bwd_app.run(grad_state)
      else:
        result = ctx.bwd_app.run(None)

    with ctx.bwd_app.timer("func:postrun"):
      # pack up the buffer, and then send it out
      buf_size = utils.buffer_size(ctx.bwd_app.grads)

      src_buf = utils.pack_buffer([g.cpu() if g is not None else None for g in ctx.bwd_app.grads])
      dst_buf = np.zeros(buf_size)

      req = comm.Iallreduce(src_buf,dst_buf,MPI.SUM)

      # grad_input follows the input to forward: fwd_app, bwd_app, Num_input_tensors, x, params
      grad_input = [None,None,None]

      if ctx.needs_input_grad[3]: grad_input += [ctx.fwd_app.x.grad]
      else: grad_input += [None] # x

      grad_input += ctx.num_input_tensors*[None]
      if result is not None:
        for i,r in enumerate(result):
          if ctx.needs_input_grad[4+i]:
            grad_input[4+i] = r

      # with for communication to complete
      MPI.Request.Wait(req)

      grads_cpu = [g.cpu() if g is not None else None for g in ctx.bwd_app.grads]
      utils.unpack_buffer(grads_cpu,dst_buf)
      ctx.bwd_app.grads = [g.to(device) if g is not None else None for g in grads_cpu]

      # setup the return value (perversely grad_input)
      for grad_needed,g in zip(ctx.needs_input_grad[5:],ctx.bwd_app.grads):
        if grad_needed:
          grad_input += [g]
        else:
          grad_input += [None]
    # end with timer

    return tuple(grad_input)
