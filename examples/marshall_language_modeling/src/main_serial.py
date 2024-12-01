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

from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from network_architecture import parse_args
import data
from main import TextDataset, CosineWarmupScheduler

####################################################################################
####################################################################################


##
# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train(rank, params, model, train_loader, optimizer, epoch, device, scheduler):
  # note that we dont' call the optimizer directly, but use the scheduler instead
  train_times = []
  fwd_times = []
  bwd_times = []
  losses = []

  # Train the model
  model.train()

  criterion = nn.CrossEntropyLoss(ignore_index=-1)

  total_time = 0.0

  for batch_idx, (data, target) in enumerate(train_loader):
    start_time = timer()
    data, target = data.to(device), target.to(device)

    # Syncrhonize
    torch.cuda.synchronize()
    batch_fwd_pass_start = time.time()

    output = model(data)

    # Syncrhonize 
    torch.cuda.synchronize()
    batch_fwd_pass_end = time.time()
    
    loss = criterion(output.reshape(-1, output.shape[-1]), target.reshape(-1))

    optimizer.zero_grad(set_to_none=True)
    
    # Sync
    torch.cuda.synchronize()
    batch_bwd_pass_start = time.time()

    loss.backward()
    
    # Sync
    torch.cuda.synchronize()
    batch_bwd_pass_end = time.time()

    stop_time = timer()
    optimizer.step()
    scheduler.step()

    total_time += stop_time - start_time
    train_times.append(stop_time - start_time)
    fwd_times.append(batch_fwd_pass_end - batch_fwd_pass_start)
    bwd_times.append(batch_bwd_pass_end - batch_bwd_pass_start)

    losses.append(loss.item())

    if batch_idx % params.log_interval == 0:
      root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {:.2e}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item(), 
               float(scheduler.get_lr()[0])))
      root_print(rank, f'\t Some times: {fwd_times[-4:-1]=} {bwd_times[-4:-1]=} {train_times[-4:-1]=}')
    del loss, output

  # root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2e}'.format(
  #   epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
  #          100. * (batch_idx + 1) / len(train_loader), loss.item()))

  return losses, train_times, fwd_times, bwd_times


##
# Evaluate model on validation data
# Return: number of correctly classified test items, total number of test items, loss on test data set
def test(rank, model, test_loader, device):
  # Evaluate the model
  model.eval()
  test_loss = 0
  criterion = nn.CrossEntropyLoss(ignore_index=-1)

  with torch.no_grad():
    for (data, target) in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss = criterion(output.reshape(-1, output.shape[-1]), target.reshape(-1)).item()

  test_loss /= len(test_loader)
  # root_print(rank, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
  root_print(rank, 'Test set: Average loss: {:.4f}'.format(test_loss))
  return test_loss

def root_print(rank, s):
  if rank == 0:
    print(s, flush=True)

def main():
  # Load serial; no need for MPI and stuff
  rank = 0 # Dummy argument since I'm too lazy to redefine rootprint
  args = parse_args()
  print('Loading serial model')
  model = torch.load(f'serialnet_gpt_{args.steps}')
  print('Model loaded')
  model = model.to('cuda')

  # Use device or CPU?
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  # Set seed for reproducibility
  torch.manual_seed(args.seed)

  # Finish assembling training and test datasets
  root_print(rank, f'Loading {args.percent_data * 100}% of dataset')
  DATA_DIR = os.path.join('..', 'data')
  train_data, val_data, decode, vocabulary_size = \
    data.obtain_data(DATA_DIR, args.input_text, args.tokenization, args.percent_data)
  train_dataset = TextDataset(train_data, args.context_window) 
  train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, pin_memory=True)
  val_dataset = TextDataset(val_data, args.context_window) 
  val_loader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, pin_memory=True)

  # Diagnostic information
  root_print(rank, 	'-- Tf       = {}\n'
          '-- steps    = {}\n'
          '-- max_levels     = {}\n'
          '-- max_bwd_iters  = {}\n'
          '-- max_fwd_iters  = {}\n'
          '-- cfactor        = {}\n'
          '-- fine fcf       = {}\n'
          '-- skip down      = {}\n'.format(args.Tf, args.steps,
              args.lp_max_levels,
              args.lp_bwd_max_iters,
              args.lp_fwd_max_iters,
              args.lp_cfactor,
              args.lp_fine_fcf,
              not args.lp_use_downcycle))

          # Declare optimizer  
  # print(f'rank {rank}: len(list(model.parameters())) {len(list(model.parameters()))}')
  weight_decay=1e-1
  betas=(0.9, 0.95)
  eps=1e-09
  optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=betas, eps=eps, weight_decay=weight_decay)
  warmup = 2000
  scheduler = CosineWarmupScheduler(optimizer, warmup=int(warmup), max_iters=args.epochs*len(train_loader))
  root_print(rank, f'Training with {warmup=} and {args.lr=}')

  # Carry out parallel training
  batch_losses = [] 
  test_losses = []
  batch_times = []
  forward_times = []
  backward_times = []

  torch.manual_seed(0)
  for epoch in range(1, args.epochs + 1):
    torch.manual_seed(epoch)

    epoch_time_start = time.time()
    [losses, train_times, batch_f_times, batch_b_times] = train(rank=rank, params=args, model=model, train_loader=train_loader, optimizer=optimizer, epoch=epoch,
          device=device, scheduler=scheduler)

    batch_losses += losses
    batch_times += train_times
    forward_times += batch_f_times
    backward_times += batch_b_times

    # valid_loss = test(rank=rank, model=model, test_loader=val_loader, device=device)
    valid_loss = losses[-1] # Dummy variable

    test_losses.append(valid_loss)
    
    epoch_time_end = time.time()

    if epoch == args.endepochs:
        # Break early based on user command; epochs also set the LR so if we want to compare need to break
        break
    if rank == 0: root_print(rank, f'Epoch time: {epoch_time_end - epoch_time_start} seconds')


	# # Print out Braid internal timings, if desired
	# #timer_str = model.parallel_nn.getTimersString()
	# #root_print(rank, timer_str)

	# Note: the MNIST example is not meant to exhibit performance
	#root_print(rank,
	#           f'TIME PER EPOCH: {"{:.2f}".format(stats.mean(epoch_times))} '
	#           f'{("(1 std dev " + "{:.2f}".format(stats.mean(epoch_times))) if len(epoch_times) > 1 else ""}')
  if rank == 0:
    _, ax1 = plt.subplots()
    ax1.plot(batch_losses, color='b', linewidth=2)
    ax1.grid(True, color='k', linestyle='-', linewidth=0.4)
    ax1.set_xlabel(r"Batch number", fontsize=13)
    ax1.set_ylabel(r"Loss", fontsize=13, color='b')
    
    ax2 = ax1.twinx()
    epoch_points = np.arange(1, len(test_losses)+1) * len(train_loader)
    ax2.plot( epoch_points, test_losses, color='r', linestyle='dashed', linewidth=2, marker='o')
    ax2.set_ylabel(r"Validation rate", fontsize=13, color='r')
    plt.savefig(f'gpt_training_serial_{args.steps}.png', bbox_inches="tight")

    np.save(f'test_losses_serial_{args.steps}.npy', np.array(batch_losses))
    np.save(f'valid_losses_serial_{args.steps}.npy', np.array(test_losses))

    # Plot and save timings to get approximate 
    # Calculate means, ignoring the first few entries
    mean_batch = np.mean(batch_times[3:])
    mean_forward = np.mean(forward_times[3:])
    mean_backward = np.mean(backward_times[3:])

    # Create figure and axes
    _, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plotting
    axs[0].plot(batch_times[3:], label='Batch Times', color='blue', marker='o')
    axs[0].set_title(f'Mean Batch Time: {mean_batch:.2f}')
    axs[0].set_xlabel('Batches')
    axs[0].set_ylabel('Time')
    axs[0].legend()

    axs[1].plot(forward_times[3:], label='Forward Times', color='green', marker='o')
    axs[1].set_title(f'Mean Forward Time: {mean_forward:.2f}')
    axs[1].set_xlabel('Batches')
    axs[1].set_ylabel('Time')
    axs[1].legend()

    axs[2].plot(backward_times[3:], label='Backward Times', color='red', marker='o')
    axs[2].set_title(f'Mean Backward Time: {mean_backward:.2f}')
    axs[2].set_xlabel('Batches')
    axs[2].set_ylabel('Time')
    axs[2].legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'timing_data_plots_serial_{args.steps}.png')


if __name__ == '__main__':
  print('Starting main')
  main()
  print('Finished.')
