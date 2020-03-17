import torch.autograd

class BraidFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, params, fwd_app, bwd_app):
    # setup context
    ctx.x       = x 
    ctx.params  = params 
    ctx.fwd_app = fwd_app
    ctx.bwd_app = bwd_app

    result = fwd_app.run(x)
    return BraidFunction.broadcastForwardResult(fwd_app.getMPIData(),result)

  @staticmethod
  def backward(ctx, grad_output):
    x       = ctx.x 
    params  = ctx.params 
    fwd_app = ctx.fwd_app
    bwd_app = ctx.bwd_app

    print('backward: %d' % fwd_app.getMPIData().getRank())
    result = bwd_app.run(grad_output)
    result = BraidFunction.broadcastForwardResult(bwd_app.getMPIData(),result)
    return result,None,None,None

  @staticmethod
  def broadcastForwardResult(mpi_data,result):
    build_seq_tag = 96        # this 
    comm          = mpi_data.getComm()
    my_rank       = mpi_data.getRank()
    num_ranks     = mpi_data.getSize()

    # short circuit for serial case
    if num_ranks==1:
      return result

    # send the output of the last layer to the root
    if my_rank==num_ranks-1:
      for dest in range(my_rank):
        comm.send(result,dest=dest,tag=build_seq_tag)
    else:
      result = comm.recv(source=num_ranks-1,tag=build_seq_tag)

    return result
  # end broadcast
