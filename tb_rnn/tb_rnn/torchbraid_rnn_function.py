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

class BraidFunction(torch.autograd.Function):

  # Original Code
  # @staticmethod
  # def forward(ctx, fwd_app, bwd_app, x, *params):

  #   prefix_rank = fwd_app.getMPIData().getRank()

  #   print("Rank %d BraidFunction -> forward() - start" % prefix_rank)
  #   with fwd_app.timer_manager.timer("BraidFunction::forward::run"):
  #     # setup context
  #     ctx.fwd_app = fwd_app
  #     # ctx.bwd_app = bwd_app
  #     # ctx.save_for_backward(x, *params)

  #     result = fwd_app.run(x)

  #   # commented out 08/28 for testing

  #   # with fwd_app.timer_manager.timer("BraidFunction::forward::broadCastForwardResult"):
  #   #   result = BraidFunction.broadcastForwardResult(fwd_app.getMPIData(),result)

  #   print("Rank %d BraidFunction -> forward() - end" % prefix_rank)
  #   return result

  #############################
  ##### CRITICAL CHANGE!! #####
  #############################
  @staticmethod
  def forward(ctx, fwd_app, x, *params):

    # prefix_rank = fwd_app.getMPIData().getRank()
    # print("Rank %d BraidFunction -> forward() - start" % prefix_rank)
    with fwd_app.timer_manager.timer("BraidFunction::forward::run"):
      # setup context
      ctx.fwd_app = fwd_app
      # ctx.bwd_app = bwd_app
      # ctx.save_for_backward(x, *params)

      result = fwd_app.run(x)

    # commented out 08/28 for testing

    # with fwd_app.timer_manager.timer("BraidFunction::forward::broadCastForwardResult"):
    #   result = BraidFunction.broadcastForwardResult(fwd_app.getMPIData(),result)

    # print("Rank %d BraidFunction -> forward() - end" % prefix_rank)
    return result

  @staticmethod
  def backward(ctx, grad_output):
    # print("BraidFunction -> backward() - start")
    result = ctx.bwd_app.run(grad_output)

    # grad_input follows the input to forward: fwd_app, bwd_app, x, params
    grad_input = (None,None) 
    grad_input += (result,)

    comm          = ctx.bwd_app.getMPIComm()
    my_rank       = ctx.bwd_app.getMPIComm().Get_rank()
    num_ranks     = ctx.bwd_app.getMPIComm().Get_size()

    # send gradients to the right (braid doesn't maintain symmetry with the forward and
    # adjoint problems)
    with ctx.bwd_app.fwd_app.timer_manager.timer("BraidFunction::backward::propParameterDerivs"):
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
    # print("BraidFunction -> backward() - end")
    return grad_input

  @staticmethod
  def broadcastForwardResult(mpi_data,result):
    # prefix_rank = mpi_data.getRank()
    # print("Rank %d BraidFunction -> broadcastForwardResult() - start" % prefix_rank)
    build_seq_tag = 96        # this 
    comm          = mpi_data.getComm()
    my_rank       = mpi_data.getRank()
    num_ranks     = mpi_data.getSize()

    # short circuit for serial case
    if num_ranks==1:
      return result

    # broadcast the output of the last layer 
    result = comm.bcast(result,root=num_ranks-1)
    # print("Rank %d BraidFunction -> broadcastForwardResult() - end" % prefix_rank)

    return result
  # end broadcast
