"""MNIST training with the pure-PyTorch MGRIT backend (TorchMGRIT).

Mirrors mnist_script.py but replaces the XBraid-based LayerParallel block
with torchbraid.TorchMGRIT. Composition with serial layers works by
replication instead of rank-0 composition: every rank builds identical
opening/closing layers (same seed), every rank sees the same data
(shuffle=False), the parallel block broadcasts its output to all ranks, and
the input gradient is broadcast back — so the serial layers receive
identical gradients everywhere and identical optimizers keep them in sync.

Run:  mpirun -n 4 python mnist_torch_mgrit.py --percent-data 0.2 --epochs 2 --steps 32
"""
from __future__ import print_function

import argparse
import statistics as stats
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mpi4py import MPI
from torchvision import datasets, transforms

from torchbraid import TorchMGRIT
from torchbraid.utils import getDevice

from network_architecture import OpenFlatLayer, CloseLayer, StepLayer


class ParallelNet(nn.Module):
  def __init__(self, comm, device, channels, global_steps, Tf, cfactor,
               fwd_iters, bwd_iters, seed):
    super().__init__()
    # serial layers must be identical on every rank (replication pattern)
    torch.manual_seed(seed)
    self.open_nn = OpenFlatLayer(channels)
    # the parallel block's layer construction is rank-dependent (each rank
    # builds its own time interval; replica construction also draws RNG)
    self.parallel_nn = TorchMGRIT(comm, lambda: StepLayer(channels),
                                  global_steps, Tf, cfactor=cfactor,
                                  fwd_iters=fwd_iters, bwd_iters=bwd_iters,
                                  device=device)
    torch.manual_seed(seed + 1)   # resync the RNG for the closing layer
    self.close_nn = CloseLayer(channels)
    self.to(device)

  def forward(self, x):
    x = self.open_nn(x)
    x = self.parallel_nn(x)
    return self.close_nn(x)


def root_print(rank, s):
  if rank == 0:
    print(s, flush=True)


def train(rank, args, model, train_loader, optimizer, epoch, device):
  model.train()
  criterion = nn.CrossEntropyLoss()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
      root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))


def test(rank, model, test_loader, device):
  model.eval()
  correct, total, test_loss = 0, 0, 0.0
  criterion = nn.CrossEntropyLoss(reduction='sum')
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += criterion(output, target).item()
      pred = output.argmax(dim=1)
      correct += int((pred == target).sum())
      total += len(target)
  test_loss /= total
  root_print(rank, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, total, 100. * correct / total))
  return correct, total


def main():
  parser = argparse.ArgumentParser(description='MNIST with TorchMGRIT')
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--batch-size', type=int, default=50)
  parser.add_argument('--percent-data', type=float, default=0.05)
  parser.add_argument('--epochs', type=int, default=3)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--channels', type=int, default=3)
  parser.add_argument('--steps', type=int, default=32, help='global ODE steps')
  parser.add_argument('--Tf', type=float, default=1.0)
  parser.add_argument('--lp-cfactor', type=int, default=4)
  parser.add_argument('--lp-fwd-max-iters', type=int, default=2)
  parser.add_argument('--lp-bwd-max-iters', type=int, default=1)
  parser.add_argument('--log-interval', type=int, default=10)
  args = parser.parse_args()

  comm = MPI.COMM_WORLD
  rank, nprocs = comm.Get_rank(), comm.Get_size()
  device, host = getDevice(comm)

  # identical data order on every rank: fixed split, no shuffling
  torch.manual_seed(args.seed)
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
  dataset = datasets.MNIST('./digit-data', download=(rank == 0), transform=transform)
  comm.Barrier()
  train_size = int(50000 * args.percent_data)
  test_size = int(10000 * args.percent_data)
  train_set = torch.utils.data.Subset(dataset, range(train_size))
  test_set = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                             shuffle=False, drop_last=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                            shuffle=False, drop_last=True)

  model = ParallelNet(comm, device, args.channels, args.steps, args.Tf,
                      args.lp_cfactor, args.lp_fwd_max_iters,
                      args.lp_bwd_max_iters, args.seed)
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

  epoch_times = []
  for epoch in range(1, args.epochs + 1):
    start = timer()
    train(rank, args, model, train_loader, optimizer, epoch, device)
    epoch_times.append(timer() - start)
    test(rank, model, test_loader, device)

  # replication sanity check: serial-layer parameters must agree across ranks
  close_sq = sum(float((p**2).sum()) for p in model.close_nn.parameters())
  all_sq = comm.allgather(close_sq)
  if max(all_sq) - min(all_sq) > 0:
    root_print(rank, f'WARNING: closing layer desynced across ranks: {all_sq}')

  root_print(rank, 'TIME PER EPOCH: {:.2f} (1 std dev {:.2f})'.format(
      stats.mean(epoch_times),
      stats.stdev(epoch_times) if len(epoch_times) > 1 else 0.0))


if __name__ == '__main__':
  main()
