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


# Train network for UCI Human Action Recognition dataset with a GRU recurrent neural network
#
# Running the following should reproduce parallel output of UCI_HAR_GRU_notebook.ipynb
# $ mpirun -np 4 python3 UCI_HAR_GRU_script.py
#
# To reproduce serial UCI_HAR_GRU_notebook.ipynb output, do
# $ mpirun -np 4 python3 UCI_HAR_GRU_script.py --lp-max-levels 1
#
# Many options are available, see
# $ python3 UCI_HAR_GRU_script.py --help
#
# For example,
# $ mpirun -np 4 python3 UCI_HAR_GRU_script.py --hidden-size 50 --percent-data 0.25 --batch-size 50 --epochs 12
# trains on 25% of the UCI data with hidden dimension of 50 in the GRU cells, 50 sequences per batch over 12 epochs


from __future__ import print_function

import statistics as stats
from timeit import default_timer as timer
import os
from mpi4py import MPI

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbraid
import torchbraid.utils

from network_architecture import parse_args, ParallelNet
from data import download_UCI_Data, load_data, ParallelRNNDataLoader


##
# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train(rank, params, model, train_loader, optimizer, epoch, compose, device):
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
    stop_time = timer()
    optimizer.step()

    total_time += stop_time - start_time
    train_times.append(stop_time - start_time)
    losses.append(loss.item())
    if batch_idx % params.log_interval == 0:
      root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item(), total_time / (batch_idx + 1.0)))

  root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
           100. * (batch_idx + 1) / len(train_loader), loss.item(), total_time / (batch_idx + 1.0)))
  return losses, train_times

##
# Evaluate model on validation data
# Return: number of correctly classified test items, total number of test items, loss on test data set
def test(rank, model, test_loader, compose, device):
  model.eval()
  test_loss = 0
  correct = 0
  criterion = nn.CrossEntropyLoss()
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += compose(criterion, output, target).item()

      if rank == 0:
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  root_print(rank, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
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

  ##
  # Make sure datasets are present (if already downloaded, this is not repeated)
  example_path = os.path.dirname(os.path.realpath(__file__))
  if rank == 0:
    download_UCI_Data(example_path)

  # Wait for the dataset download if needed
  comm.Barrier()

  # Use device or CPU?
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device, host = torchbraid.utils.getDevice(comm=comm)
  if not use_cuda:
    device = torch.device("cuda" if use_cuda else "cpu")
  print(f'Run info rank: {rank}: Torch version: {torch.__version__} | Device: {device} | Host: {host}')

  # Set seed for reproducibility
  torch.manual_seed(args.seed)

  # Compute number of steps in network per processor
  local_steps = int(args.sequence_length / procs)

  train_set = torch.utils.data.TensorDataset(*load_data(True, example_path))
  test_set = torch.utils.data.TensorDataset(*load_data(False, example_path))

  # Finish assembling training and test datasets
  train_size = int(7352 * args.percent_data)
  test_size  = int(2947 * args.percent_data)

  train_set = torch.utils.data.Subset(train_set, range(train_size))
  test_set = torch.utils.data.Subset(test_set, range(test_size))

  train_loader = ParallelRNNDataLoader(comm, dataset=train_set, batch_size=args.batch_size, shuffle=False)
  test_loader = ParallelRNNDataLoader(comm, dataset=test_set, batch_size=args.batch_size, shuffle=False)

  # Diagnostic information
  root_print(rank, '-- procs       = {}\n'
                   '-- hidden_size = {}\n'
                   '-- local_steps = {}\n'
                   '-- max_levels     = {}\n'
                   '-- max_iters  = {}\n'
                   '-- max_fwd_iters  = {}\n'
                   '-- cfactor        = {}\n'
                   '-- skip down      = {}\n'.format(procs, args.hidden_size,
                                                     local_steps,
                                                     args.lp_max_levels,
                                                     args.lp_max_iters,
                                                     args.lp_fwd_max_iters,
                                                     args.lp_cfactor,
                                                     not args.lp_use_downcycle) )

  # Create layer-parallel network
  # Note this can be done on only one processor, but will be slow
  model = ParallelNet(hidden_size=args.hidden_size,
                      Tf=None, # None to have it be calculated "correctly" based on other arguments
                      num_layers=args.num_layers,
                      num_classes=args.num_classes,
                      num_steps=local_steps,
                      max_levels=args.lp_max_levels,
                      max_iters=args.lp_max_iters,
                      fwd_max_iters=args.lp_fwd_max_iters,
                      print_level=args.lp_print_level,
                      braid_print_level=args.lp_braid_print_level,
                      cfactor=args.lp_cfactor,
                      skip_downcycle=not args.lp_use_downcycle).to(device)

  # Detailed XBraid timings are output to these files for the forward and backward phases
  model.parallel_rnn.fwd_app.setTimerFile(
    'b_fwd_s_%d_c_%d_bs_%d_p_%d'%(local_steps, args.hidden_size, args.batch_size, procs) )
  model.parallel_rnn.bwd_app.setTimerFile(
    'b_bwd_s_%d_c_%d_bs_%d_p_%d'%(local_steps, args.hidden_size, args.batch_size, procs) )

  # Declare optimizer
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

  # For better timings (especially with GPUs) do a little warm up
  if args.warm_up:
    warm_up_timer = timer()
    train(rank=rank, params=args, model=model, train_loader=train_loader, optimizer=optimizer,
          epoch=0, compose=model.compose, device=device)

    model.parallel_rnn.timer_manager.resetTimers()
    model.parallel_rnn.fwd_app.resetBraidTimer()
    model.parallel_rnn.bwd_app.resetBraidTimer()
    if use_cuda:
      torch.cuda.synchronize()
    root_print(rank, f'\nWarm up timer {timer() - warm_up_timer}\n')

  # Carry out parallel training
  batch_losses = []; batch_times = []
  epoch_times = []; test_times = []
  validat_correct_counts = []

  for epoch in range(1, args.epochs + 1):
    start_time = timer()
    [losses, train_times] = train(rank=rank, params=args, model=model, train_loader=train_loader,
                                  optimizer=optimizer, epoch=epoch, compose=model.compose, device=device)
    epoch_times += [timer() - start_time]
    batch_losses += losses
    batch_times += train_times

    start_time = timer()
    validat_correct, validat_size, validat_loss = test(rank=rank, model=model, test_loader=test_loader,
                                                       compose=model.compose, device=device)
    test_times += [timer() - start_time]
    validat_correct_counts += [validat_correct]

  # Print out Braid internal timings, if desired
  #timer_str = model.parallel_rnn.getTimersString()ll *
  #root_print(rank, timer_str)

  root_print(rank,
            f'TIME PER EPOCH: {"{:.2f}".format(stats.mean(epoch_times))} '
            f'{("(1 std dev " + "{:.2f})".format(stats.mean(epoch_times))) if len(epoch_times) > 1 else ""}')
  root_print(rank,
            f'TIME PER TEST:  {"{:.2f}".format(stats.mean(test_times))} '
            f'{("(1 std dev " + "{:.2f})".format(stats.mean(test_times))) if len(test_times) > 1 else ""}')

  # plot the loss, validation
  root_print(rank, f'\nMin batch time:   {"{:.3f}".format(np.min(batch_times))} ')
  root_print(rank, f'Mean batch time:  {"{:.3f}".format(stats.mean(batch_times))} ')


  if rank == 0:
    fig, ax1 = plt.subplots()
    plt.title('Layer-parallel run with %d processors\n UCI HAR dataset'%(procs), fontsize=15)
    ax1.plot(batch_losses, color='b', linewidth=2)
    ax1.grid(True, color='k', linestyle='-', linewidth=0.4)
    ax1.set_xlabel(r"Batch number", fontsize=13)
    ax1.set_ylabel(r"Loss", fontsize=13, color='b')

    ax2 = ax1.twinx()
    epoch_points = np.arange(1, len(validat_correct_counts)+1) * len(train_loader)
    validation_percentage = np.array(validat_correct_counts) / validat_size
    ax2.plot( epoch_points, validation_percentage, color='r', linestyle='dashed', linewidth=2, marker='o')
    ax2.set_ylabel(r"Validation rate", fontsize=13, color='r')
    plt.savefig('UCI-HAR_layerparallel_training.png', bbox_inches="tight")

if __name__ == '__main__':
  main()
