import torch
import unittest
from mpi4py import MPI
from torchbraid.test_fixtures import check_gpu_direct_mpi

def simple_gpu_direct():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  # do the quick check first
  chk = check_gpu_direct_mpi()

  if rank==1:
    comm.send(chk, dest=0, tag=90)

    # figure out if both processors passed the check
    passed_rank0 = comm.recv(source=0, tag=90)
    passed_rank1 = chk 
  else:
    comm.send(chk, dest=1, tag=90)

    # figure out if both processors passed the check
    passed_rank0 = chk 
    passed_rank1 = comm.recv(source=1, tag=90)

  return passed_rank0 and passed_rank1

def test_gpu_direct():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  if not simple_gpu_direct():
    if rank==0:
      print()
      print('FAILED: GPU aware MPI is NOT available - "MPIX_Query_cuda_support" test failed.')
    return

  passed = False

  try:

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
  except AssertionError as e:
    err_msg = f'Assertion Failure - {e}'
  except RuntimeError as e:
    err_msg = f'RuntimeError Failure - {e}'
  else:
    passed = True

  if not passed:
    print(f'FAILURE (p={rank}): {err_msg}')

  comm.Barrier()

  if rank==1:
    comm.send(passed, dest=0, tag=90)
  else:
    passed_rank0 = passed
    passed_rank1 = comm.recv(source=1, tag=90)

    if passed_rank0 and passed_rank1:
      print('PASSED: GPU aware MPI is available')
    else:
      print()
      print('FAILED: GPU aware MPI is NOT available')

if __name__ == '__main__':
    test_gpu_direct()
