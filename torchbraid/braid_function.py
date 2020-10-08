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

import torchbraid.utils as utils

class BraidFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, fwd_app, bwd_app, x, *params):
    comm          = fwd_app.getMPIComm()
    my_rank       = fwd_app.getMPIComm().Get_rank()
    num_ranks     = fwd_app.getMPIComm().Get_size()

    # copy the input to all processors (ensure consistency)
    shape = comm.bcast(x.size(),root=0)

    with fwd_app.timer_manager.timer("BraidFunction::forward::run"):
      # setup context
      ctx.fwd_app = fwd_app
      ctx.bwd_app = bwd_app
      ctx.save_for_backward(None, *params)

      fwd_app.setShape(shape)
      bwd_app.setShape(shape)

      if my_rank==0:
        result = fwd_app.run(x)
      else:
        result = fwd_app.run(None)

      if my_rank!=num_ranks-1:
        result = torch.zeros(shape)

    # broadcast the output of the last layer 
    comm.Bcast(result.numpy(),root=num_ranks-1)

    return result

  @staticmethod
  def backward(ctx, grad_output):
    comm          = ctx.bwd_app.getMPIComm()
    my_rank       = ctx.bwd_app.getMPIComm().Get_rank()
    num_ranks     = ctx.bwd_app.getMPIComm().Get_size()

    # copy the input to the final processor (where iter time integration begins)
    if num_ranks>1:
      if my_rank==0:
        comm.Send(grad_output.numpy(),dest=num_ranks-1)
      elif my_rank==num_ranks-1: 
        comm.Recv(grad_output.numpy(),source=0)

    if my_rank==num_ranks-1:
      result = ctx.bwd_app.run(grad_output)
    else:
      result = ctx.bwd_app.run(None)

    # send gradients to the right (braid doesn't maintain symmetry with the forward and
    # adjoint problems)
    # grad_input follows the input to forward: fwd_app, bwd_app, x, params
    grad_input = (None,None) 
    grad_input += (result,)

    grads = ctx.bwd_app.grads
    if my_rank<num_ranks-1:
      comm.send(grads[-1],dest=my_rank+1,tag=22)
    if my_rank>0:
      neighbor_model = comm.recv(source=my_rank-1,tag=22)
      grads.insert(0,neighbor_model)

    # flatten the grads array
    grads = [g for sublist in grads for g in sublist]

    for grad_needed,param in zip(ctx.needs_input_grad[3:],grads):
      if grad_needed:
        grad_input += (param,)
      else:
        grad_input += (None,)

    return grad_input
