import torch

def l2_reg(net,comm=None):
  """
  Compute an L2 regularization for the parameters in a neural network.
  For a distributed neural network communicate and reduce the value (not 
  the gradient) to the root)
  """

  l2_list = [ ]
  for p in net.parameters():
    l2 = torch.norm(p)**2
    l2_list += [l2]

  l2_list.sort()
  result = sum(l2_list)
  if comm is not None:
    # sum to the root
    rank = comm.Get_rank()
    l2_values = [ten.item() for ten in l2_list]

    values = comm.gather(l2_values)
    if rank==0:
      # flatten the values array
      l2_values = [ item for sublist in values for item in sublist]
      l2_values.sort()
      with torch.no_grad():
        result.fill_(sum(l2_values))

  return result
# l2 regularization
