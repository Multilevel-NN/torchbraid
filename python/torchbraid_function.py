import torch.autograd

class BraidFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, fwd_app, bwd_app, x, *params):
    # setup context
    ctx.fwd_app = fwd_app
    ctx.bwd_app = bwd_app
    ctx.save_for_backward(x, *params)

    result = fwd_app.run(x)
    return BraidFunction.broadcastForwardResult(fwd_app.getMPIData(),result,reverse=False)

  @staticmethod
  def backward(ctx, grad_output):
    result = ctx.bwd_app.run(grad_output)
    result = BraidFunction.broadcastForwardResult(ctx.bwd_app.getMPIData(),result,reverse=True)

    grad_input = (None,None)
    if ctx.needs_input_grad[2]:
      grad_input += (result,)

    comm          = ctx.bwd_app.getMPIData().getComm()
    my_rank       = ctx.bwd_app.getMPIData().getRank()
    num_ranks     = ctx.bwd_app.getMPIData().getSize()

    # send everything to the right (this helps with the adjoint method)
    grads = ctx.bwd_app.grads

    if len(grads)>0:

      if my_rank<num_ranks-1:
        comm.send(grads[-1],dest=my_rank+1,tag=22)
        comm.send(grads[-2],dest=my_rank+1,tag=23)

      if my_rank>0:
        neighbor_model_1 = comm.recv(source=my_rank-1,tag=22)
        neighbor_model_2 = comm.recv(source=my_rank-1,tag=23)
        grads.insert(0,neighbor_model_1)
        grads.insert(0,neighbor_model_2)


    for grad_needed,param in zip(ctx.needs_input_grad[3:],grads):
      if grad_needed:
        grad_input += (param,)
      else:
        grad_input += (None,)

    return grad_input

  @staticmethod
  def broadcastForwardResult(mpi_data,result,reverse):
    build_seq_tag = 96        # this 
    comm          = mpi_data.getComm()
    my_rank       = mpi_data.getRank()
    num_ranks     = mpi_data.getSize()

    # short circuit for serial case
    if num_ranks==1:
      return result

    # send the output of the last layer to the root (if revsere is false)
    if not reverse:
      if my_rank==num_ranks-1:
        for dest in range(my_rank):
          comm.send(result,dest=dest,tag=build_seq_tag)
      else:
        result = comm.recv(source=num_ranks-1,tag=build_seq_tag)
    else:
      # no work to do here if reverse is True
      return result

    return result
  # end broadcast
