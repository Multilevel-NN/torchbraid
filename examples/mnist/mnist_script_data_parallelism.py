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


# Train network for fashion or digits MNIST for Layer + Data parallelism
# Network architecture is Opening Layer + ResNet + CloseLayer
#
#
# The following command runs the example with two processes in layer parallel and two in data parallel dimension
# $ mpirun -np 4 python3 mnist_script_data_parallelism.py --dp-size 2

from __future__ import print_function

import statistics as stats
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbraid
import torchbraid.utils
import torchbraid.utils.data_parallel
from torchvision import datasets, transforms

from network_architecture import parse_args, ParallelNet
from mpi4py import MPI

##
# Make sure datasets are present (if already downloaded, this is not repeated)
datasets.MNIST('./digit-data', download=True)
datasets.FashionMNIST('./fashion-data', download=True)

##
# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train(params, model, train_loader, optimizer, epoch, compose, device, comm_dp, comm_lp):
  rank_lp = comm_lp.Get_rank()
  rank_dp = comm_dp.Get_rank()

  train_times = []
  losses = []
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
    # Average gradients for data parallelism
    torchbraid.utils.data_parallel.average_gradients(model=model, comm_dp=comm_dp)
    stop_time = timer()
    optimizer.step()

    total_time += stop_time - start_time
    train_times.append(stop_time - start_time)
    losses.append(loss.item())
    if batch_idx % params.log_interval == 0:
      if rank_lp == 0:
        recv = comm_dp.gather([loss.item(), len(train_loader.dataset), len(data), len(train_loader)], root=0)
        if rank_dp == 0:
          loss_global = np.sum(np.array([item[0] for item in recv])) / len(recv)
          losses[-1] = loss_global  # Reset the last lost by a global loss
          dataset_size = np.sum(np.array([item[1] for item in recv]))
          data_size = np.sum(np.array([item[2] for item in recv]))
          root_print(rank_dp, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
            epoch, batch_idx * data_size, dataset_size,
                   100. * batch_idx / len(train_loader), loss_global, total_time / (batch_idx + 1.0)))

  if rank_lp == 0:
    recv = comm_dp.gather([loss.item(), len(train_loader.dataset), len(data), len(train_loader)], root=0)
    if rank_dp == 0:
      loss_global = np.sum(np.array([item[0] for item in recv])) / len(recv)
      losses[-1] = loss_global  # Reset the last lost by a global loss
      dataset_size = np.sum(np.array([item[1] for item in recv]))
      root_print(rank_dp, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
        epoch, (batch_idx + 1) * data_size, dataset_size,
               100. * (batch_idx + 1) / len(train_loader), loss_global, total_time / (batch_idx + 1.0)))
  return losses, train_times


##
# Evaluate model on validation data
# Return: number of correctly classified test items, total number of test items, loss on test data set
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
      return correct, dataset_size, test_loss

  return correct, len(test_loader.dataset), test_loss


##
# Parallel printing helper function  
def root_print(rank, s):
  if rank == 0:
    print(s)


def main():
  # Begin setting up run-time environment 
  # MPI information
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  procs = comm.Get_size()
  args = parse_args()

  if procs % args.dp_size == 0:
    comm_dp, comm_lp = torchbraid.utils.data_parallel.split_communicator(comm=comm, splitting=args.dp_size)
    rank_dp = comm_dp.Get_rank()
    size_dp = comm_dp.Get_size()
    rank_lp = comm_lp.Get_rank()
    size_lp = comm_lp.Get_size()
  else:
    raise Exception('Please choose the data parallel communicator size so that it can be divided evenly by all procs.')

  # Use device or CPU?
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device, host = torchbraid.utils.getDevice(comm=comm)
  if not use_cuda:
    device = torch.device("cuda" if use_cuda else "cpu")
  print(f'Run info rank: {rank}: Torch version: {torch.__version__} | Device: {device} | Host: {host}')

  # Set seed for reproducibility
  torch.manual_seed(args.seed)

  # Compute number of steps in ResNet per processor
  local_steps = int(args.steps / size_lp)

  # Read in Digits MNIST or Fashion MNIST
  root_print(rank, 'MNIST ODENet:')
  if args.dataset == 'digits':
    root_print(rank, '-- Using Digit MNIST')
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./digit-data', download=False, transform=transform)
  elif args.dataset == 'fashion':
    root_print(rank, '-- Using Fashion MNIST')
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST('./fashion-data', download=False, transform=transform)
  else:
    print(f'Unrecognized dataset {args.dataset}')
    return 0

  # Finish assembling training and test datasets
  if args.batch_size % size_dp == 0:
    batch_size = int(args.batch_size / float(size_dp))
  else:
    raise Exception('Please choose the batch size so that it can be divided evenly by the dp size.')

  train_size = int(50000 * args.percent_data)
  test_size = int(10000 * args.percent_data)
  train_set = torch.utils.data.Subset(dataset, range(train_size))
  test_set = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
  train_partition = torchbraid.utils.data_parallel.Partioner(data=train_set, procs=size_dp, seed=args.seed, batch_size=batch_size).get_partion(
    rank=rank_dp)
  test_partition = torchbraid.utils.data_parallel.Partioner(data=test_set, procs=size_dp, seed=args.seed, batch_size=batch_size).get_partion(
    rank=rank_dp)
  train_loader = torch.utils.data.DataLoader(train_partition, batch_size=batch_size, shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_partition, batch_size=batch_size, shuffle=False)

  # Diagnostic information
  root_print(rank, '-- procs_lp       = {}\n'
                   '-- procs_dp       = {}\n'
                   '-- channels       = {}\n'
                   '-- Tf             = {}\n'
                   '-- steps          = {}\n'
                   '-- max_levels     = {}\n'
                   '-- max_bwd_iters  = {}\n'
                   '-- max_fwd_iters  = {}\n'
                   '-- cfactor        = {}\n'
                   '-- fine fcf       = {}\n'
                   '-- skip down      = {}\n'.format(size_lp, size_dp, args.channels,
                                                     args.Tf, args.steps,
                                                     args.lp_max_levels,
                                                     args.lp_bwd_max_iters,
                                                     args.lp_fwd_max_iters,
                                                     args.lp_cfactor,
                                                     args.lp_fine_fcf,
                                                     not args.lp_use_downcycle))

  # Create layer-parallel network
  # Note this can be done on only one processor, but will be slow
  model = ParallelNet(channels=args.channels,
                      local_steps=local_steps,
                      max_levels=args.lp_max_levels,
                      bwd_max_iters=args.lp_bwd_max_iters,
                      fwd_max_iters=args.lp_fwd_max_iters,
                      print_level=args.lp_print_level,
                      braid_print_level=args.lp_braid_print_level,
                      cfactor=args.lp_cfactor,
                      fine_fcf=args.lp_fine_fcf,
                      skip_downcycle=not args.lp_use_downcycle,
                      fmg=False,
                      Tf=args.Tf,
                      relax_only_cg=False,
                      user_mpi_buf=args.lp_user_mpi_buf,
                      comm_lp=comm_lp).to(device)

  # Detailed XBraid timings are output to these files for the forward and backward phases
  model.parallel_nn.fwd_app.setTimerFile(
    f'b_fwd_s_{args.steps}_c_{args.channels}_bs_{args.batch_size}_p_{procs}')
  model.parallel_nn.bwd_app.setTimerFile(
    f'b_bwd_s_{args.steps}_c_{args.channels}_bs_{args.batch_size}_p_{procs}')

  # Declare optimizer  
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

  # For better timings (especially with GPUs) do a little warm up
  if args.warm_up:
    warm_up_timer = timer()
    train(params=args, model=model, train_loader=train_loader, optimizer=optimizer, epoch=0,
          compose=model.compose, device=device, comm_dp=comm_dp, comm_lp=comm_lp)
    model.parallel_nn.timer_manager.resetTimers()
    model.parallel_nn.fwd_app.resetBraidTimer()
    model.parallel_nn.bwd_app.resetBraidTimer()
    if use_cuda:
      torch.cuda.synchronize()
    root_print(rank, f'\nWarm up timer {timer() - warm_up_timer}\n')

  # Carry out parallel training
  batch_losses = []
  batch_times = []
  epoch_times = []
  test_times = []
  validat_correct_counts = []

  for epoch in range(1, args.epochs + 1):
    start_time = timer()
    [losses, train_times] = train(params=args, model=model, train_loader=train_loader, optimizer=optimizer,
                                  epoch=epoch, compose=model.compose, device=device, comm_dp=comm_dp, comm_lp=comm_lp)
    epoch_times += [timer() - start_time]
    batch_losses += losses
    batch_times += train_times

    start_time = timer()
    validat_correct, validat_size, validat_loss = test(model=model, test_loader=test_loader,
                                                       compose=model.compose, device=device, comm_dp=comm_dp,
                                                       comm_lp=comm_lp)
    test_times += [timer() - start_time]
    validat_correct_counts += [validat_correct]

  # Print out Braid internal timings, if desired
  # timer_str = model.parallel_nn.getTimersString()
  # root_print(rank, timer_str)

  root_print(rank,
             f'TIME PER EPOCH: {"{:.2f}".format(stats.mean(epoch_times))} '
             f'{("(1 std dev " + "{:.2f})".format(stats.mean(epoch_times))) if len(epoch_times) > 1 else ""}')
  root_print(rank,
             f'TIME PER TEST:  {"{:.2f}".format(stats.mean(test_times))} '
             f'{("(1 std dev " + "{:.2f})".format(stats.mean(test_times))) if len(test_times) > 1 else ""}')

  if args.serial_file is not None:
    # Model can be reloaded in serial format with: model = torch.load(filename)
    model.saveSerialNet(args.serial_file)

  # Plot the loss, validation 
  root_print(rank, f'\nMin batch time:   {"{:.3f}".format(np.min(batch_times))} ')
  root_print(rank, f'Mean batch time:  {"{:.3f}".format(stats.mean(batch_times))} ')

  if rank == 0:
    fig, ax1 = plt.subplots()
    plt.title('MNIST %s dataset' % (args.dataset), fontsize=15)
    ax1.plot(batch_losses, color='b', linewidth=2)
    ax1.grid(True, color='k', linestyle='-', linewidth=0.4)
    ax1.set_xlabel(r"Batch number", fontsize=13)
    ax1.set_ylabel(r"Loss", fontsize=13, color='b')

    ax2 = ax1.twinx()
    epoch_points = np.arange(1, len(validat_correct_counts) + 1) * len(train_loader)
    validation_percentage = np.array(validat_correct_counts) / validat_size
    ax2.plot(epoch_points, validation_percentage, color='r', linestyle='dashed', linewidth=2, marker='o')
    ax2.set_ylabel(r"Validation rate", fontsize=13, color='r')
    plt.savefig(f'mnist_layerparallel_dataparallel_training_{size_dp}.png', bbox_inches="tight")


if __name__ == '__main__':
  main()
