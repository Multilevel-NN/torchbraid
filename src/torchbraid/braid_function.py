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

import torch.autograd
import torch.nn.functional as F

class BraidFunction(torch.autograd.Function):

  @staticmethod
  def padForBatchChange(old_batch,temp_batch,ten,batch_dim):
    shape = ten.size()
    assert(len(shape)>batch_dim)

    padding = []
    for i in range(len(shape)-batch_dim-1):
      padding += [0,0]
    padding += [0,old_batch-temp_batch]
    return F.pad(ten,tuple(padding),'constant',0.0)

  @staticmethod
  def forward(ctx, fwd_app, bwd_app, x, *params):
    comm          = fwd_app.getMPIComm()
    my_rank       = fwd_app.getMPIComm().Get_rank()
    num_ranks     = fwd_app.getMPIComm().Get_size()

    fwd_app.setDevice(x.device)
    bwd_app.setDevice(x.device)

    # copy the input to all processors (ensure consistency)
    if my_rank==0:
      shape = fwd_app.buildShapes(x)
    else: 
      shape = None
    shape = comm.bcast(shape,root=0)

    old_shape = fwd_app.getShape()
    adjusting = old_shape is not None and old_shape!=shape

    # if batch size is larger (expand)
    if old_shape is not None and shape[0][0] >  old_shape[0][0]:
      adjusting = False

    # setup context
    ctx.fwd_app = fwd_app
    ctx.bwd_app = bwd_app
    ctx.adjusting = adjusting
    ctx.save_for_backward(None, *params)

    if adjusting:
      old_batch  = old_shape[0][0]
      temp_batch = shape[0][0]
      x = BraidFunction.padForBatchChange(old_batch,temp_batch,x,0)
      ctx.old_batch = old_batch
      ctx.temp_batch = temp_batch
  
      shape = old_shape[0]
    else:
      fwd_app.setShape(shape)
      bwd_app.setShape(shape)

    if my_rank!=num_ranks-1:
      result = torch.zeros(shape[-1],device=x.device)
      fwd_app.run(x)
    else:
      result = fwd_app.run(x)

    # broadcast the output of the last layer
    comm.Bcast(result, root=num_ranks - 1)

    if adjusting:
      return result[0:temp_batch,:]
    else:
      return result

  @staticmethod
  def backward(ctx, grad_output):
    comm          = ctx.bwd_app.getMPIComm()
    my_rank       = ctx.bwd_app.getMPIComm().Get_rank()
    num_ranks     = ctx.bwd_app.getMPIComm().Get_size()

    # copy the input to the final processor (where time integration begins)
    if num_ranks>1:
      if my_rank==0:
        if ctx.fwd_app.use_cuda:
          torch.cuda.synchronize()
        comm.Isend(grad_output,dest=num_ranks-1)
      elif my_rank==num_ranks-1: 
        req = comm.Irecv(grad_output,source=0)
        req.Wait()

    if my_rank==num_ranks-1:
      if ctx.adjusting:
        grad_output = BraidFunction.padForBatchChange(ctx.old_batch,ctx.temp_batch,grad_output,0)
      result = ctx.bwd_app.run(grad_output)
    else:
      result = ctx.bwd_app.run(None)

    # grad_input follows the input to forward: fwd_app, bwd_app, x, params
    grad_input = (None,None) 
    grad_input += (result,)

    grads = ctx.bwd_app.grads

    # flatten the grads array
    grads = [g for sublist in grads for g in sublist]

    for grad_needed,param in zip(ctx.needs_input_grad[3:],grads):
      if grad_needed:
        grad_input += (param,)
      else:
        grad_input += (None,)

    return grad_input
