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


# Train network for ModelNet10 dataset
# Network architecture is Opening Layer + ResNet + CloseLayer
#
# Run with the following
# $ mpirun -np 4 python3 ModelNet_script.py
#
# Run in serial with the following
# $ python3 ModelNet_script.py --lp-max-levels 1
#
# Many options are available, see
# $ python3 ModelNet_script.py --help
#
# For example, 
# $ mpirun -np 4 python3 ModelNet_script.py --steps 24 --channels 8 --percent-data 0.25 --batch-size 100 --epochs 5 
# trains on 25% of ModelNet10 with 24 steps in the ResNet, 8 channels, 100 models per batch over 5 epochs 

from __future__ import print_function

import statistics as stats
import os
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbraid
import torchbraid.utils
from torchvision import datasets, transforms

from network_architecture import parse_args, ParallelNet
from downloadModelNet import downloadModelNet, ModelNet, root_loader
from mpi4py import MPI


##
# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train(rank, params, model, train_loader, optimizer, epoch, compose, device):
  train_times = []
  losses = []
  model.train()
  criterion = nn.CrossEntropyLoss()
  total_time = 0.0
  processed = 0
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

    if rank==0:
      processed += len(target)

    if batch_idx % params.log_interval == 0 and rank==0:
      root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
        epoch, processed, len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item(), total_time / (batch_idx + 1.0)))

  if rank==0: 
    root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
      epoch, processed, len(train_loader.dataset),
             100. * (batch_idx + 1) / len(train_loader), loss.item(), total_time / (batch_idx + 1.0)))

  return losses, train_times


##
# Evaluate model on validation data
# Return: number of correctly classified test items, total number of test items, loss on test data set
def test(rank,epoch, model, test_loader, compose, device):
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

  root_print(rank, '\nTest Epoch {:6d}:  Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    epoch,test_loss, correct, len(test_loader.dataset),
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
  if rank==0:
    print(args)

  # Download the ModelNet10 dataset if it does not exist, then convert the models to voxels
  # This will take around 10 minutes (if nx=63)
  #if rank == 0:
  #  downloadModelNet(nx=args.numx)

  # Use device or CPU?
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device, host = torchbraid.utils.getDevice(comm=comm)
  device = torch.device("cuda" if use_cuda else "cpu")
  print(f'Run info rank: {rank}: Torch version: {torch.__version__} | Device: {device} | Host: {host}')

  # Set seed for reproducibility
  torch.manual_seed(args.seed)

  # Compute number of steps in ResNet per processor
  local_steps = int(args.steps / procs)

  # load ModelNet10
  root_print(rank, 'ModelNet10 ODENet:')
  # transform = transforms.Compose([transforms.ToTensor()])
  train_dataset = ModelNet()
  test_dataset = ModelNet(train=False)

  # Finish assembling training and test datasets
  train_indices = torch.randperm(len(train_dataset))[:int(args.percent_data * len(train_dataset))]
  test_indices = torch.randperm(len(test_dataset))[:int(args.percent_data * len(test_dataset))]
  train_set = torch.utils.data.Subset(train_dataset, train_indices)
  test_set = torch.utils.data.Subset(test_dataset, test_indices)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
  train_loader = root_loader(rank, train_loader, device)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
  test_loader = root_loader(rank, test_loader, device)

  # some logic to determine on which levels to apply the spatial coarsening
  if args.lp_sc_levels == -1:
    sc_levels = list(range(args.lp_max_levels))
  else:
    sc_levels = args.lp_sc_levels

  # Diagnostic information
  root_print(rank, '-- procs    = {}\n'
                   '-- channels = {}\n'
                   '-- Tf       = {}\n'
                   '-- steps    = {}\n'
                   '-- max_levels     = {}\n'
                   '-- max_bwd_iters  = {}\n'
                   '-- max_fwd_iters  = {}\n'
                   '-- cfactor        = {}\n'
                   '-- sc levels      = {}\n'
                   '-- sc full weight = {}\n'
                   '-- fine fcf       = {}\n'
                   '-- F-cycles       = {}\n'
                   '-- skip down      = {}\n'.format(procs, args.channels, 
                                                     args.Tf, args.steps,
                                                     args.lp_max_levels,
                                                     args.lp_bwd_max_iters,
                                                     args.lp_fwd_max_iters,
                                                     args.lp_cfactor,
                                                     sc_levels,
                                                     not args.lp_sc_injection,
                                                     args.lp_fine_fcf,
                                                     args.lp_fmg,
                                                     not args.lp_use_downcycle) )
  
  # Create layer-parallel network
  # Note this can be done on only one processor, but will be slow
  model = ParallelNet(
    channels=args.channels,
    local_steps=local_steps,
    max_levels=args.lp_max_levels,
    bwd_max_iters=args.lp_bwd_max_iters,
    fwd_max_iters=args.lp_fwd_max_iters,
    print_level=args.lp_print_level,
    braid_print_level=args.lp_braid_print_level,
    cfactor=args.lp_cfactor,
    fine_fcf=args.lp_fine_fcf,
    skip_downcycle=not args.lp_use_downcycle,
    fmg=args.lp_fmg, 
    Tf=args.Tf,
    relax_only_cg=False,
    user_mpi_buf=args.lp_user_mpi_buf,
    sc_levels=sc_levels,
    full_weighting=not args.lp_sc_injection,
    numx=args.numx
  ).to(device)

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
    train(rank=rank, params=args, model=model, train_loader=train_loader, optimizer=optimizer, epoch=0,
          compose=model.compose, device=device)
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

  sc_str = ','.join([f'{l}' for l in args.lp_sc_levels]) if args.lp_sc_levels is not None else None
  if args.filename is None:
    filename = f"nx{args.numx}_nt{args.steps}_ml{args.lp_max_levels}_sc{sc_str}"
  else:
    filename = args.filename

  os.makedirs('models',exist_ok=True)
  os.makedirs('models/training_history',exist_ok=True)

  for epoch in range(1, args.epochs + 1):
    start_time = timer()
    [losses, train_times] = train(rank=rank, params=args, model=model, train_loader=train_loader, optimizer=optimizer, epoch=epoch,
          compose=model.compose, device=device)
    epoch_times += [timer() - start_time]
    batch_losses += losses
    batch_times += train_times

    start_time = timer()
    validate_correct, validate_size, validate_loss = test(rank=rank, epoch=epoch, model=model, test_loader=test_loader, compose=model.compose, device=device)
    test_times += [timer() - start_time]
    validat_correct_counts += [validate_correct]


    # save checkpoint
    if rank == 0:
      contents = (args.channels, args.steps, model.state_dict())
      torch.save(contents, f"models/{filename}.pt")

      # write training history
      np.savetxt(f"models/training_history/{filename}_losses.csv", batch_losses, delimiter=",")
      np.savetxt(f"models/training_history/{filename}_accuracies.csv", np.array(validat_correct_counts)/len(test_loader.dataset), delimiter=",")

  # write training history to file
  #if rank == 0:
  #  np.savetxt(f"models/training_history/{filename}_losses.csv", batch_losses, delimiter=",")
  #  np.savetxt(f"models/training_history/{filename}_accuracies.csv", np.array(validat_correct_counts)/len(test_loader.dataset), delimiter=",")
  
  # Print out Braid internal timings, if desired
  #timer_str = model.parallel_nn.getTimersString()
  #root_print(rank, timer_str)

  root_print(rank,
             f'TIME PER EPOCH: {"{:.2f}".format(stats.mean(epoch_times))} '
             f'{("(1 std dev " + "{:.2f}".format(stats.mean(epoch_times))) if len(epoch_times) > 1 else ""}')
  root_print(rank,
             f'TIME PER TEST:  {"{:.2f}".format(stats.mean(test_times))} '
             f'{("(1 std dev " + "{:.2f}".format(stats.mean(test_times))) if len(test_times) > 1 else ""}')

  if args.serial_file is not None:
    # Model can be reloaded in serial format with: model = torch.load(filename)
    model.saveSerialNet(args.serial_file)

  # Plot the loss, validation 
  root_print(rank, f'\nMin batch time:   {"{:.3f}".format(np.min(batch_times))} ')
  root_print(rank, f'Mean batch time:  {"{:.3f}".format(stats.mean(batch_times))} ')

  if rank == 0:
    fig, ax1 = plt.subplots()
    plt.title('ModelNet10 dataset', fontsize=15)
    ax1.plot(batch_losses, color='C0', linewidth=2)
    ax1.grid(True, color='k', linestyle='-', linewidth=0.4)
    ax1.set_xlabel(r"Batch number", fontsize=13)
    ax1.set_ylabel(r"Loss", fontsize=13, color='C0')
    
    ax2 = ax1.twinx()
    epoch_points = np.arange(1, len(validat_correct_counts)+1) * len(train_loader)
    validation_percentage = np.array(validat_correct_counts) / validate_size
    ax2.plot( epoch_points, validation_percentage, color='C1', linestyle='dashed', linewidth=2, marker='o')
    ax2.set_ylabel(r"Validation rate", fontsize=13, color='C1')
    plt.savefig('modelnet_layerparallel_training.png', bbox_inches="tight")

if __name__ == '__main__':
  main()
