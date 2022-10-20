import torch
import unittest
from mpi4py import MPI

class TestGpuDirect(unittest.TestCase):

  def test_gpu_direct(self):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ten = torch.ones(100, device='cuda') * (10+rank)
    ten2 = torch.empty(100, device='cuda')
    if rank == 0:
      ten3 = torch.empty(100, device='cuda') * (10+1)
    else:
      ten3 = torch.empty(100, device='cuda') * (10)

    comm.Barrier()

    if rank == 0:
      comm.Send(ten, dest=1, tag=88)
      comm.Recv(ten2, source=1, tag=89)
      torch.allclose(ten2, ten3)
    else:
      comm.Recv(ten2, source=0, tag=88)
      comm.Send(ten, dest=0, tag=89)
      torch.allclose(ten2, ten3)


if __name__ == '__main__':
  unittest.main()



