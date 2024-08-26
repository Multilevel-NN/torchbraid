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

from __future__ import print_function

import statistics as stats
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbraid
import torchbraid.utils
from torch.utils.data import Dataset, DataLoader
import sys

from network_architecture import parse_args, ParallelNet
from mpi4py import MPI

import data

# torch.set_default_dtype(torch.float64)

DATA_DIR = os.path.join('..', 'data')

def get_batch(data, context_window, batch_size, device):
  ix = torch.randint(len(data) - context_window, (batch_size,))
  x = torch.stack([data[i : i + context_window] for i in ix])
  y = torch.stack([data[i+1 : i+1 + context_window] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

def init_weights(module):
  if isinstance(module, nn.Linear):
    nn.init.normal_(module.weight, mean=0.0, std=0.02)

    if module.bias is not None:
      nn.init.zeros_(module.bias)

  elif isinstance(module, nn.Embedding):
    nn.init.normal_(module.weight, mean=0.0, std=0.02)

##
# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train(rank, params, model, optimizer, epoch, compose, device, criterion,
  train_loader, scheduler):
  train_times = []
  fwd_times = []
  bwd_times = []

  losses = []
  total_time = 0.0
  corr, tot = 0, 0

  model.train()

  for batch_idx, (data, target) in enumerate(train_loader):
    # print(batch_idx, data.shape, target.shape)
    start_time = timer()
    data, target = data.to(device), target.to(device)

    batch_fwd_pass_start = time.time()
    output = model(data)
    batch_fwd_pass_end = time.time()

    loss = compose(criterion, output.reshape(-1, output.shape[-1]), target.reshape(-1))
    #loss += .0001 * model.get_forward_reg()
    
    optimizer.zero_grad(set_to_none=True)

    batch_bwd_pass_start = time.time()
    loss.backward()
    batch_bwd_pass_end = time.time()

    stop_time = timer()
    optimizer.step()
    scheduler.step()

    total_time += stop_time - start_time
    train_times.append(stop_time - start_time)
    fwd_times.append(batch_fwd_pass_end - batch_fwd_pass_start)
    bwd_times.append(batch_bwd_pass_end - batch_bwd_pass_start)

    losses.append(loss.item())

    if batch_idx == 3500:
        break

    if batch_idx % params.log_interval == 0:
      root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} LR {:.2e}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item(), float(scheduler.get_lr()[0])))

    del loss, output
      #root_print(
      #   rank, f'\t Magnitude of data {torch.norm(data.float()):.3e} output {torch.norm(output):.3e} target {torch.norm(target.float()):.3e}'
      #)
      #root_print(
      #  rank, f'\t {model.get_forward_reg()=:.3e}'
      #)

  # root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} LR {:.2e}'.format(
  #   epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
  #          100. * (batch_idx + 1) / len(train_loader), loss.item(), float(scheduler.get_lr()[0])))


  return losses, train_times, fwd_times, bwd_times

##
# Evaluate model on validation data
# Return: number of correctly classified test items, total number of test items, loss on test data set
def test(rank, model, test_loader, compose, device):
  model.eval()
  test_loss = 0
  criterion = nn.CrossEntropyLoss(ignore_index=-1)

  with torch.no_grad():
    for (data, target) in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += compose(criterion, output.reshape(-1, output.shape[-1]), target.reshape(-1)).item()

  test_loss /= len(test_loader)
  model.train()

  # root_print(rank, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
  root_print(rank, f'Test set: Average loss: {test_loss :.8f}')

  return test_loss


##
# Parallel printing helper function  
def root_print(rank, s):
  if rank == 0:
    print(s, flush=True)

class TextDataset(Dataset):
    def __init__(self, data, context_window = 256):
        self.length = len(data) // context_window - 1

        self.data = data
        self.context_window = context_window

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx * self.context_window:(idx + 1) * self.context_window], \
                self.data[1 + idx * self.context_window:1 + (idx + 1) * self.context_window]


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

def main():
  # Begin setting up run-time environment 
  # MPI information
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  procs = comm.Get_size()
  args = parse_args()

  # Use device or CPU?
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device, host = torchbraid.utils.getDevice(comm=comm)
  if not use_cuda:
    device = torch.device("cuda" if use_cuda else "cpu")
  print(f'Run info rank: {rank}: Torch version: {torch.__version__} | Device: {device} | Host: {host}')

  # Set seed for reproducibility
  torch.manual_seed(0)#args.seed)

  # Compute number of steps in ResNet per processor
  local_steps = int(args.steps / procs)

  # Finish assembling training and test datasets
  root_print(rank, f'Loading dataset; {args.percent_data=}')
  train_data, val_data, decode, vocabulary_size = \
    data.obtain_data(DATA_DIR, args.input_text, args.tokenization, args.percent_data)

  # Create dataloader 
  train_dataset = TextDataset(train_data, args.context_window)
  train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, pin_memory=True)

  val_dataset   = TextDataset(val_data, args.context_window)
  val_loader    = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, pin_memory=True)

  # Diagnostic information
  root_print(rank, '-- procs    = {}\n'
                   '-- Tf       = {}\n'
                   '-- steps    = {}\n'
                   '-- max_levels     = {}\n'
                   '-- max_bwd_iters  = {}\n'
                   '-- max_fwd_iters  = {}\n'
                   '-- cfactor        = {}\n'
                   '-- fine fcf       = {}\n'
                   '-- skip down      = {}\n'.format(procs,
                                                     args.Tf, args.steps,
                                                     args.lp_max_levels,
                                                     args.lp_bwd_max_iters,
                                                     args.lp_fwd_max_iters,
                                                     args.lp_cfactor,
                                                     args.lp_fine_fcf,
                                                     not args.lp_use_downcycle) )
  root_print(rank, f'{args.model_dimension=} {args.num_heads=} {args.batch_size=}')
  # Create layer-parallel network
  # Note this can be done on only one processor, but will be slow
  model = ParallelNet(args.model_dimension, args.num_heads, vocabulary_size, args.context_window,
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
                  user_mpi_buf=args.lp_user_mpi_buf).to(device)

  if args.serial_file:
      model.saveSerialNet(f'serialnet_gpt_{args.steps}')
      print('Saved file')

      import sys
      sys.exit()

  # Declare optimizer  
  print(f'rank {rank}: len(list(model.parameters())) {len(list(model.parameters()))}')
  print(f'rank {rank}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-09, weight_decay=1e-1)

  # Choose shorter warm-up if using shakespeare
  warmup = 2000
  scheduler = CosineWarmupScheduler(optimizer, warmup=int(warmup), 
                                      max_iters=args.epochs*len(train_loader))
  criterion = nn.CrossEntropyLoss(ignore_index=-1)

  # Carry out parallel training
  batch_losses = [] 
  test_losses = []
  batch_times = []
  epoch_times = [] 
  forward_times = []
  backward_times = []
  test_times = []
  # validat_correct_counts = []

  torch.manual_seed(0)
  # epoch_time_start = time.time()
  for epoch in range(1, args.epochs+1):
    torch.manual_seed(epoch)

    start_time = timer()
    [losses, train_times, batch_f_times, batch_b_times] = train(rank=rank, params=args, model=model, optimizer=optimizer, epoch=epoch,
          compose=model.compose, device=device, criterion=criterion,
          train_loader=train_loader, scheduler=scheduler)
    epoch_times += [timer() - start_time]
    batch_losses += losses
    batch_times += train_times
    forward_times += batch_f_times
    backward_times += batch_b_times
    
    start_time = timer()
    # validat_loss = test(rank=rank, model=model, test_loader=val_loader, compose=model.compose, device=device)
    validat_loss = losses[-1]
    test_times += [timer() - start_time] 
    test_losses.append(validat_loss)

    if epoch == args.endepochs:
      # Don't need to run that many to see that it's off
      break
  

    # if epoch % 500 == 0 or epoch == args.epochs:
    #   start_time = timer()
    #   validat_loss = test(rank=rank, model=model, val_data=val_data, compose=model.compose, device=device, 
    #           context_window=args.context_window, batch_size=args.batch_size)
    #   test_times += [timer() - start_time]
  
    #   # epoch_time_end = time.time()
    #   # if rank == 0: root_print(rank, f'Epoch time: {epoch_time_end - epoch_time_start} seconds')
    #   # epoch_time_start = time.time()
  # model.saveSerialNet('serialnet_trained')

  if rank == 0:
    fig, ax1 = plt.subplots()
    ax1.plot(batch_losses, color='b', linewidth=2)
    ax1.grid(True, color='k', linestyle='-', linewidth=0.4)
    ax1.set_xlabel(r"Batch number", fontsize=13)
    ax1.set_ylabel(r"Loss", fontsize=13, color='b')
    
    ax2 = ax1.twinx()
    epoch_points = np.arange(1, len(test_losses)+1) * len(train_loader)
    # validation_percentage = np.array(validat_correct_counts) / validat_size
    ax2.plot( epoch_points, test_losses, color='r', linestyle='dashed', linewidth=2, marker='o')
    ax2.set_ylabel(r"Validation Loss", fontsize=13, color='r')
    plt.savefig(f'data_run_{procs=}_{args.lp_bwd_max_iters}_{args.lp_fwd_max_iters}_{args.lp_max_levels}.png', bbox_inches="tight")

    # Save to files
    import pickle

    with open(f'data_run_{procs=}_{args.lp_bwd_max_iters}_{args.lp_fwd_max_iters}_{args.lp_max_levels}', 'wb') as fp:
      pickle.dump([batch_losses, epoch_points, test_losses], fp)

    # )Plot and save timings to get approximate 
    # Calculate means, ignoring the first few entries
    mean_batch = np.mean(batch_times[3:])
    mean_forward = np.mean(forward_times[3:])
    mean_backward = np.mean(backward_times[3:])

    from datetime import datetime
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create figure and axes
    _, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plotting
    axs[0].plot(batch_times[3:], label='Batch Times', color='blue', marker='o')
    axs[0].set_title(f'Mean Batch Time: {mean_batch:.2f} (Generated at {current_time})')
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
    plt.savefig(f'timing_data_plots_{procs}_{args.steps}_{args.lp_max_levels}.png')

    # Convert lists to numpy arrays
    batch_times_array = np.array(batch_times)
    forward_times_array = np.array(forward_times)
    backward_times_array = np.array(backward_times)

    # Save arrays to a .npz file
    np.savez(f'times_data_{procs}_{args.steps}_{args.lp_max_levels}.npz', batch_times=batch_times_array, forward_times=forward_times_array, backward_times=backward_times_array)


  max_memory_allocated = torch.cuda.max_memory_allocated(device)
  print(f"Maximum CUDA memory allocated on {device}: {max_memory_allocated / (1024 ** 2):.2f} MB")
  if args.serial_file is not None:
    # Model can be reloaded in serial format with: model = torch.load(filename)
    model.saveSerialNet(args.serial_file)

if __name__ == '__main__':
  main()

