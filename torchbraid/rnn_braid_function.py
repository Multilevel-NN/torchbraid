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
import traceback

from mpi4py import MPI

class BraidFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, fwd_app, bwd_app, x, h,c, *params):

    # copy the input to all processors (ensure consistency)
    comm = fwd_app.getMPIComm()
    shape = comm.bcast((h.size(),c.size()),root=0)

    # prefix_rank = fwd_app.getMPIData().getRank()
    # print("Rank %d BraidFunction -> forward() - start" % prefix_rank)

    # setup context
    ctx.fwd_app = fwd_app
    ctx.bwd_app = bwd_app
    ctx.save_for_backward(x, h,c, *params)

    fwd_app.setShape(shape)
    bwd_app.setShape(shape)

    h_c = (h,c)

    result = fwd_app.run(x,h_c)

    return result

  @staticmethod
  def backward(ctx, grad_hn, grad_cn):
    comm          = ctx.bwd_app.getMPIComm()
    my_rank       = ctx.bwd_app.getMPIComm().Get_rank()
    num_ranks     = ctx.bwd_app.getMPIComm().Get_size()

    # copy the input to the final processor (where iter time integration begins)
    if num_ranks>1:
      if my_rank==num_ranks-1: 
        req_h = comm.Irecv(grad_hn.numpy(),source=0)
        req_c = comm.Irecv(grad_cn.numpy(),source=0)
        req_h.Wait()
        req_c.Wait()

      if my_rank==0:
        comm.Isend(grad_hn.numpy(),dest=num_ranks-1)
        comm.Isend(grad_cn.numpy(),dest=num_ranks-1)
    # end if num_ranks

    if my_rank==num_ranks-1:
      result = ctx.bwd_app.run((grad_hn,grad_cn))
    else:
      result = ctx.bwd_app.run(None)

    # grad_input follows the input to forward: fwd_app, bwd_app, x, params
    grad_input = (None,None) 

    if ctx.needs_input_grad[2]: grad_input += (ctx.fwd_app.x.grad,)
    else: grad_input += (None,) # x

    if result is not None:
      if ctx.needs_input_grad[3]: grad_input += (result[0],) # h
      else: grad_input += (None,) # h

      if ctx.needs_input_grad[4]: grad_input += (result[1],) # h
      else: grad_input += (None,) # h
    else: 
      grad_input += (None,) # h
      grad_input += (None,) # h

    try:
      grads = ctx.bwd_app.grads
      req = []
      bgrads = []
      for g in grads:
        # sum all gradients into the root
        b = torch.zeros(g.shape)
        req += [comm.Iallreduce(g.numpy(),b.numpy(),MPI.SUM)]
        bgrads += [b]
      # end for g

      MPI.Request.Waitall(req) 
      for g,b in zip(grads,bgrads):
        g.copy_(b) 
      # end for g,b

      # setup the returvn value (perversly grad_input)
      for grad_needed,param in zip(ctx.needs_input_grad[5:],grads):
        if grad_needed:
          grad_input += (param,)
        else:
          grad_input += (None,)
    except:
      traceback.print_exc()

    return grad_input
