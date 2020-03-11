import torch.autograd

class BraidFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, fwd_app):
    print('forward: %d' % fwd_app.getMPIData().getRank())
    ctx.fwd_app = fwd_app
    result = fwd_app.run(x)
    result = BraidFunction.broadcastForwardResult(fwd_app.getMPIData(),result)
    print('done forward: %d' % fwd_app.getMPIData().getRank())
    return result

  @staticmethod
  def backward(ctx, grad_output):
    fwd_app = ctx.fwd_app
    print('backward: %d' % fwd_app.getMPIData().getRank())
    return grad_output.clone()

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
