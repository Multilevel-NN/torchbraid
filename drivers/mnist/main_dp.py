# @HEADER
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
# @HEADER

# some helpful examples
# 
# BATCH_SIZE=50
# STEPS=12
# CHANNELS=8

# IN SERIAL
# python  main.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 # 2>&1 | tee serial.out
# mpirun -n 4 python  main.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 # 2>&1 | tee serial.out

from __future__ import print_function

import argparse
import matplotlib.pyplot as pyplot
import numpy as np
import statistics as stats
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mpi4py import MPI
from timeit import default_timer as timer
from torchvision import datasets, transforms

import torchbraid
import torchbraid.utils


# from sgd import SGD as SGD2

def root_print(rank, s):
  if rank == 0:
    print(s)


class Partition(object):
  def __init__(self, data, index):
    self.data = data
    self.index = index

  def __len__(self):
    return len(self.index)

  def __getitem__(self, index):
    data_idx = self.index[index]
    return self.data[data_idx]


class Partioner(object):
  def __init__(self, data, procs, seed=42):
    self.data = data
    data_len = len(self.data)
    indices = np.arange(0, data_len)
    np.random.seed(seed)
    np.random.shuffle(indices)
    split_indices = np.array([int(data_len / procs + 1)] * (data_len % procs) +
                             [int(data_len / procs)] * (procs - data_len % procs))
    self.partitions = [indices[np.sum(split_indices[:i]):np.sum(split_indices[:i + 1])].tolist()
                       for i in range(len(split_indices))]

  def get_partion(self, rank):
    return Partition(self.data, self.partitions[rank])


class OpenConvLayer(nn.Module):
  def __init__(self, channels):
    super(OpenConvLayer, self).__init__()
    ker_width = 3
    self.conv = nn.Conv2d(1, channels, ker_width, padding=1)

  def forward(self, x):
    return F.relu(self.conv(x))


class OpenFlatLayer(nn.Module):
  def __init__(self, channels):
    super(OpenFlatLayer, self).__init__()
    self.channels = channels

  def forward(self, x):
    # this bit of python magic simply replicates each image in the batch
    s = len(x.shape) * [1]
    s[1] = self.channels
    x = x.repeat(s)
    return x


class CloseLayer(nn.Module):
  def __init__(self, channels):
    super(CloseLayer, self).__init__()
    self.fc1 = nn.Linear(channels * 28 * 28, 32)
    self.fc2 = nn.Linear(32, 10)

  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


class StepLayer(nn.Module):
  def __init__(self, channels):
    super(StepLayer, self).__init__()
    ker_width = 3
    self.conv1 = nn.Conv2d(channels, channels, ker_width, padding=1)
    self.conv2 = nn.Conv2d(channels, channels, ker_width, padding=1)

  def forward(self, x):
    return F.relu(self.conv2(F.relu(self.conv1(x))))


class SerialNet(nn.Module):
  def __init__(self, channels=12, local_steps=8, Tf=1.0, serial_nn=None, open_nn=None, close_nn=None):
    super(SerialNet, self).__init__()

    if open_nn is None:
      self.open_nn = OpenFlatLayer(channels)
    else:
      self.open_nn = open_nn

    if serial_nn is None:
      step_layer = lambda: StepLayer(channels)
      numprocs = 1
      parallel_nn = torchbraid.LayerParallel(MPI.COMM_SELF, step_layer, numprocs * local_steps, Tf,
                                             max_fwd_levels=1, max_bwd_levels=1, max_iters=1)
      parallel_nn.setPrintLevel(0, True)
      self.serial_nn = parallel_nn.buildSequentialOnRoot()
    else:
      self.serial_nn = serial_nn

    if close_nn is None:
      self.close_nn = CloseLayer(channels)
    else:
      self.close_nn = close_nn

  def forward(self, x):
    x = self.open_nn(x)
    x = self.serial_nn(x)
    x = self.close_nn(x)
    return x


class ParallelNet(nn.Module):
  def __init__(self, channels=12, local_steps=8, Tf=1.0, max_levels=1, max_iters=1, fwd_max_iters=0, print_level=0,
               braid_print_level=0, cfactor=4, fine_fcf=False, skip_downcycle=True, fmg=False, relax_only_cg=0,
               user_mpi_buf=False, comm=MPI.COMM_WORLD):
    super(ParallelNet, self).__init__()

    step_layer = lambda: StepLayer(channels)

    numprocs = comm.Get_size()

    self.parallel_nn = torchbraid.LayerParallel(comm=comm, layer_blocks=step_layer,
                                                global_steps=local_steps * numprocs, Tf=Tf,
                                                max_fwd_levels=max_levels, max_bwd_levels=max_levels,
                                                max_iters=max_iters, user_mpi_buf=user_mpi_buf)
    if fwd_max_iters > 0:
      self.parallel_nn.setFwdMaxIters(fwd_max_iters)
    self.parallel_nn.setPrintLevel(print_level, True)
    self.parallel_nn.setPrintLevel(braid_print_level, False)
    self.parallel_nn.setCFactor(cfactor)
    self.parallel_nn.setSkipDowncycle(skip_downcycle)
    self.parallel_nn.setBwdRelaxOnlyCG(relax_only_cg)
    self.parallel_nn.setFwdRelaxOnlyCG(relax_only_cg)

    if fmg:
      self.parallel_nn.setFMG()
    self.parallel_nn.setNumRelax(1)  # FCF elsewehre
    if not fine_fcf:
      self.parallel_nn.setNumRelax(0, level=0)  # F-Relaxation on the fine grid
    else:
      self.parallel_nn.setNumRelax(1, level=0)  # F-Relaxation on the fine grid

    # this object ensures that only the LayerParallel code runs on ranks!=0
    compose = self.compose = self.parallel_nn.comp_op()

    # by passing this through 'compose' (mean composition: e.g. OpenFlatLayer o channels)
    # on processors not equal to 0, these will be None (there are no parameters to train there)
    self.open_nn = compose(OpenFlatLayer, channels)
    self.close_nn = compose(CloseLayer, channels)

  def saveSerialNet(self, name):
    serial_nn = self.parallel_nn.buildSequentialOnRoot()
    if MPI.COMM_WORLD.Get_rank() == 0:
      s_net = SerialNet(-1, -1, -1, serial_nn=serial_nn, open_nn=self.open_nn, close_nn=self.close_nn)
      s_net.eval()
      torch.save(s_net, name)

  def getDiagnostics(self):
    return self.parallel_nn.getDiagnostics()

  def forward(self, x):
    # by passing this through 'o' (mean composition: e.g. self.open_nn o x)
    # this makes sure this is run on only processor 0

    x = self.compose(self.open_nn, x)
    x = self.parallel_nn(x)
    x = self.compose(self.close_nn, x)

    return x


def train(args, model, train_loader, optimizer, epoch, compose, device, comm_dp, comm_lp):
  rank_lp = comm_lp.Get_rank()
  rank_dp = comm_dp.Get_rank()
  model.train()
  criterion = nn.CrossEntropyLoss()
  total_time = 0.0
  for batch_idx, (data, target) in enumerate(train_loader):
    start_time = timer()
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = compose(criterion, output, target)
    loss.backward()
    #get_grads(model=model, comm_dp=comm_dp, comm_lp=comm_lp)
    dp_avg_grads(model=model, comm_dp=comm_dp, comm_lp=comm_lp)
    #get_grads(model=model, comm_dp=comm_dp, comm_lp=comm_lp)

    stop_time = timer()
    optimizer.step()
    # optimizer.step(rank=rank_dp)

    total_time += stop_time - start_time
    if batch_idx % args.log_interval == 0:
      if rank_lp == 0:
        recv = comm_dp.gather([loss.item(), len(train_loader.dataset), len(data), len(train_loader)], root=0)
        if rank_dp == 0:
          losses = np.sum(np.array([item[0] for item in recv])) / len(recv)
          dataset_size = np.sum(np.array([item[1] for item in recv]))
          data_size = np.sum(np.array([item[2] for item in recv]))
          root_print(rank_dp, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
            epoch, batch_idx * data_size, dataset_size,
                   100. * batch_idx / len(train_loader), losses, total_time / (batch_idx + 1.0)))

  if rank_lp == 0:
    recv = comm_dp.gather([loss.item(), len(train_loader.dataset), len(data), len(train_loader)], root=0)
    if rank_dp == 0:
      losses = np.sum(np.array([item[0] for item in recv])) / len(recv)
      dataset_size = np.sum(np.array([item[1] for item in recv]))
      data_size = np.sum(np.array([item[2] for item in recv]))
      root_print(rank_dp, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
        epoch, (batch_idx + 1) * data_size, dataset_size,
               100. * (batch_idx+1) / len(train_loader), losses, total_time / (batch_idx + 1.0)))


def diagnose(rank, model, test_loader, epoch):
  model.parallel_nn.diagnostics(True)
  model.eval()
  test_loss = 0
  correct = 0
  criterion = nn.CrossEntropyLoss()

  itr = iter(test_loader)
  data, target = next(itr)

  # compute the model and print out the diagnostic information
  with torch.no_grad():
    output = model(data)

  diagnostic = model.getDiagnostics()

  if rank != 0:
    return

  features = np.array([diagnostic['step_in'][0]] + diagnostic['step_out'])
  params = np.array(diagnostic['params'])

  fig, axs = pyplot.subplots(2, 1)
  axs[0].plot(range(len(features)), features)
  axs[0].set_ylabel('Feature Norm')

  coords = [0.5 + i for i in range(len(features) - 1)]
  axs[1].set_xlim([0, len(features) - 1])
  axs[1].plot(coords, params, '*')
  axs[1].set_ylabel('Parameter Norms: {}/tstep'.format(params.shape[1]))
  axs[1].set_xlabel('Time Step')

  fig.suptitle('Values in Epoch {}'.format(epoch))

  # pyplot.show()
  pyplot.savefig('diagnose{:03d}.png'.format(epoch))


def test(model, test_loader, compose, device, comm_dp, comm_lp):
  rank_lp = comm_lp.Get_rank()
  rank_dp = comm_dp.Get_rank()
  model.eval()
  test_loss = 0
  correct = 0
  criterion = nn.CrossEntropyLoss()
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += compose(criterion, output, target).item()

      # Old version
      # output = MPI.COMM_WORLD.bcast(output, root=0)
      # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      # correct += pred.eq(target.view_as(pred)).sum().item()

      if rank_lp == 0:
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

  if rank_lp == 0:
    data = comm_dp.gather([test_loss, len(test_loader.dataset), correct], root=0)

    if rank_dp == 0:
      test_loss = np.sum(np.array([item[0] for item in data])) / len(data)
      dataset_size = np.sum(np.array([item[1] for item in data]))
      correct = np.sum(np.array([item[2] for item in data]))
      test_loss /= dataset_size

      root_print(rank_dp, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, dataset_size,
        100. * correct / dataset_size))


def compute_levels(num_steps, min_coarse_size, cfactor):
  from math import log, floor

  # we want to find $L$ such that ( max_L min_coarse_size*cfactor**L <= num_steps)
  levels = floor(log(float(num_steps) / min_coarse_size, cfactor)) + 1

  if levels < 1:
    levels = 1
  return levels


def split_communicator(comm: MPI.Comm, splitting: int):
  """
  Creates new communicators for data parallelism & layer parallelism by
  "splitting" the input communicator into two sub-communicators.
  :param comm: Communicator to be used as the basis for new communicators
  :param splitting: Splitting factor (number of processes for spatial parallelism)
  :return: Space and time communicator
  """

  # Determine color based on splitting factor
  # All processes with the same color will be assigned to the same communicator.
  rank = comm.Get_rank()
  x_color = rank // splitting
  t_color = rank % splitting

  # Split the communicator based on the color and key
  comm_dp = comm.Split(color=x_color, key=rank)
  comm_lp = comm.Split(color=t_color, key=rank)
  return comm_dp, comm_lp


def dp_avg_grads(model, comm_dp, comm_lp):
  for param in model.parameters():
    send_buf = param.grad.data
    recv_buf = torch.zeros_like(send_buf)
    comm_dp.Allreduce(send_buf, recv_buf, op=MPI.SUM)
    param.grad.data = recv_buf / float(comm_dp.Get_size())



def main():
  # Training settings
  parser = argparse.ArgumentParser(description='TORCHBRAID CIFAR10 Example')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 783253419)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--percent-data', type=float, default=1.0, metavar='N',
                      help='how much of the data to read in and use for training/testing')

  # artichtectural settings
  parser.add_argument('--steps', type=int, default=4, metavar='N',
                      help='Number of times steps in the resnet layer (default: 4)')
  parser.add_argument('--channels', type=int, default=4, metavar='N',
                      help='Number of channels in resnet layer (default: 4)')
  parser.add_argument('--digits', action='store_true', default=False,
                      help='Train with the MNIST digit recognition problem (default: False)')
  parser.add_argument('--serial-file', type=str, default=None,
                      help='Load the serial problem from file')
  parser.add_argument('--tf', type=float, default=1.0,
                      help='Final time')

  # algorithmic settings (gradient descent and batching
  parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                      help='input batch size for training (default: 50)')
  parser.add_argument('--epochs', type=int, default=2, metavar='N',
                      help='number of epochs to train (default: 2)')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')

  # algorithmic settings (parallel or serial)
  parser.add_argument('--force-lp', action='store_true', default=False,
                      help='Use layer parallel even if there is only 1 MPI rank')
  parser.add_argument('--lp-levels', type=int, default=3, metavar='N',
                      help='Layer parallel levels (default: 3)')
  parser.add_argument('--lp-iters', type=int, default=2, metavar='N',
                      help='Layer parallel iterations (default: 2)')
  parser.add_argument('--lp-fwd-iters', type=int, default=-1, metavar='N',
                      help='Layer parallel (forward) iterations (default: -1, default --lp-iters)')
  parser.add_argument('--lp-print', type=int, default=0, metavar='N',
                      help='Layer parallel internal print level (default: 0)')
  parser.add_argument('--lp-braid-print', type=int, default=0, metavar='N',
                      help='Layer parallel braid print level (default: 0)')
  parser.add_argument('--lp-cfactor', type=int, default=4, metavar='N',
                      help='Layer parallel coarsening factor (default: 4)')
  parser.add_argument('--lp-finefcf', action='store_true', default=False,
                      help='Layer parallel fine FCF on or off (default: False)')
  parser.add_argument('--lp-use-downcycle', action='store_true', default=False,
                      help='Layer parallel use downcycle on or off (default: False)')
  parser.add_argument('--lp-use-fmg', action='store_true', default=False,
                      help='Layer parallel use FMG for one cycle (default: False)')
  parser.add_argument('--lp-use-relaxonlycg', action='store_true', default=0,
                      help='Layer parallel use relaxation only on coarse grid (default: False)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--lp-user-mpi-buf', action='store_true', default=False,
                      help='Layer parallel use user-defined mpi buffers (default: False)')
  parser.add_argument('--warm-up', action='store_true', default=False,
                      help='Warm up for GPU timings (default: False)')
  parser.add_argument('--dp-size', type=int, default=1, metavar='N',
                      help='Data parallelism (used if value != 1)')

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  procs = comm.Get_size()
  args = parser.parse_args()

  if procs % args.dp_size == 0:
    comm_dp, comm_lp = split_communicator(comm=comm, splitting=args.dp_size)
    rank_dp = comm_dp.Get_rank()
    size_dp = comm_dp.Get_size()
    rank_lp = comm_lp.Get_rank()
    size_lp = comm_lp.Get_size()
  else:
    raise Exception('Please choose the dp size so that it can be divided evenly by all procs.')


  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device, host = torchbraid.utils.getDevice(comm=comm)
  if not use_cuda:
    device = torch.device("cuda" if use_cuda else "cpu")
  print(f'Run info rank: {rank}: Torch version: {torch.__version__} | Device: {device} | Host: {host}')

  # some logic to default to Serial if on one processor,
  # can be overriden by the user to run layer-parallel
  if args.force_lp:
    force_lp = True
  elif size_lp > 1:
    force_lp = True
  else:
    force_lp = False

  # torch.manual_seed(torchbraid.utils.seed_from_rank(args.seed,rank))
  torch.manual_seed(args.seed)

  if args.lp_levels == -1:
    min_coarse_size = 3
    args.lp_levels = compute_levels(args.steps, min_coarse_size, args.lp_cfactor)

  local_steps = int(args.steps / size_lp)
  if args.steps % size_lp != 0:
    root_print(rank, 'Steps must be an even multiple of the number of processors: %d %d' % (args.steps, size_lp))
    sys.exit(0)

  root_print(rank, 'MNIST ODENet:')

  # read in Digits MNIST or Fashion MNIST
  if args.digits:
    root_print(rank, '-- Using Digit MNIST')
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
    dataset = datasets.MNIST('./data', download=False, transform=transform)
  else:
    root_print(rank, '-- Using Fashion MNIST')
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST('./fashion-data', download=False, transform=transform)
  # if args.digits

  root_print(rank, f'-- procs_dp    = {size_dp}\n'
                   f'-- procs_lp    = {size_lp}\n'
                   f'-- channels    = {args.channels}\n'
                   f'-- tf          = {args.tf}\n'
                   f'-- steps       = {args.steps}')

  train_size = int(50000 * args.percent_data)
  test_size = int(10000 * args.percent_data)
  train_set = torch.utils.data.Subset(dataset, range(train_size))
  test_set = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))

  if args.batch_size % size_dp == 0:
    batch_size = int(args.batch_size / float(size_dp))
  else:
    raise Exception('Please choose the batch size so that it can be divided evenly by the dp size.')
  train_partition = Partioner(data=train_set, procs=size_dp).get_partion(rank=rank_dp)
  train_loader = torch.utils.data.DataLoader(train_partition,
                                             batch_size=batch_size,
                                             shuffle=True)
  test_partition = Partioner(data=test_set, procs=size_dp).get_partion(rank=rank_dp)
  test_loader = torch.utils.data.DataLoader(test_partition,
                                            batch_size=batch_size,
                                            shuffle=True)

  root_print(rank, '')

  if force_lp:
    root_print(rank, 'Using ParallelNet:')
    root_print(rank, f'-- max_levels     = {args.lp_levels}\n'
                     f'-- max_iters      = {args.lp_iters}\n'
                     f'-- fwd_iters      = {args.lp_fwd_iters}\n'
                     f'-- cfactor        = {args.lp_cfactor}\n'
                     f'-- fine fcf       = {args.lp_finefcf}\n'
                     f'-- skip down      = {not args.lp_use_downcycle}\n'
                     f'-- relax only cg  = {args.lp_use_relaxonlycg}\n'
                     f'-- fmg            = {args.lp_use_fmg}\n')

    model = ParallelNet(channels=args.channels,
                        local_steps=local_steps,
                        max_levels=args.lp_levels,
                        max_iters=args.lp_iters,
                        fwd_max_iters=args.lp_fwd_iters,
                        print_level=args.lp_print,
                        braid_print_level=args.lp_braid_print,
                        cfactor=args.lp_cfactor,
                        fine_fcf=args.lp_finefcf,
                        skip_downcycle=not args.lp_use_downcycle,
                        fmg=args.lp_use_fmg, Tf=args.tf,
                        relax_only_cg=args.lp_use_relaxonlycg,
                        user_mpi_buf=args.lp_user_mpi_buf,
                        comm=comm_lp).to(device)

    if args.serial_file is not None:
      model.saveSerialNet(args.serial_file)
    compose = model.compose

    model.parallel_nn.fwd_app.setTimerFile(
      f'b_fwd_s_{args.steps}_c_{args.channels}_bs_{args.batch_size}_p_{size_lp}')
    model.parallel_nn.bwd_app.setTimerFile(
      f'b_bwd_s_{args.steps}_c_{args.channels}_bs_{args.batch_size}_p_{size_lp}')
  else:
    root_print(rank, 'Using SerialNet:')
    root_print(rank, '-- serial file = {}\n'.format(args.serial_file))
    if args.serial_file is not None:
      print('loading model')
      model = torch.load(args.serial_file)
    else:
      model = SerialNet(channels=args.channels, local_steps=local_steps, Tf=args.tf).to(device)
    compose = lambda op, *p: op(*p)

  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
  # optimizer = SGD2(model.parameters(), lr=args.lr, momentum=0.9)

  epoch_times = []
  test_times = []

  # check out the initial conditions
  # if force_lp:
  #   diagnose(rank, model, test_loader,0)

  if args.warm_up:
    warm_up_timer = timer()
    train(args=args, model=model, train_loader=train_loader, optimizer=optimizer, epoch=0,
          compose=compose, device=device, comm_dp=comm_dp, comm_lp=comm_lp)
    if force_lp:
      model.parallel_nn.timer_manager.resetTimers()
      model.parallel_nn.fwd_app.resetBraidTimer()
      model.parallel_nn.bwd_app.resetBraidTimer()
    if use_cuda:
      torch.cuda.synchronize()
    epoch_times = []
    test_times = []
    root_print(rank, f'Warm up timer {timer() - warm_up_timer}')

  for epoch in range(1, args.epochs + 1):
    start_time = timer()
    train(args=args, model=model, train_loader=train_loader, optimizer=optimizer, epoch=epoch,
          compose=compose, device=device, comm_dp=comm_dp, comm_lp=comm_lp)
    end_time = timer()
    epoch_times += [end_time - start_time]

    start_time = timer()

    test(model=model, test_loader=test_loader, compose=compose, device=device, comm_dp=comm_dp, comm_lp=comm_lp)
    end_time = timer()
    test_times += [end_time - start_time]

    # print out some diagnostics
    # if force_lp:
    #  diagnose(rank, model, test_loader,epoch)

  if force_lp:
    timer_str = model.parallel_nn.getTimersString()
    root_print(rank, timer_str)

  root_print(rank,
             f'TIME PER EPOCH: {"{:.2f}".format(stats.mean(epoch_times))} '
             f'{("(1 std dev " + "{:.2f})".format(stats.mean(epoch_times))) if len(epoch_times) > 1 else ""}')
  root_print(rank,
             f'TIME PER TEST:  {"{:.2f}".format(stats.mean(test_times))} '
             f'{("(1 std dev " + "{:.2f})".format(stats.mean(test_times))) if len(test_times) > 1 else ""}')


if __name__ == '__main__':
  main()
