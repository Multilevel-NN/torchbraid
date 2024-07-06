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


import statistics as stats
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbraid
import torchbraid.utils
import sys

from network_architecture import ParallelNet
from mpi4py import MPI
from get_dataset import obtain_dataset
from torch.utils.data import DataLoader

####################################################################################
####################################################################################

def int_or_tuple(value):
  try:
    # Check if the value is an integer
    return int(value)
  except ValueError:
    # Check if the value is a tuple-like object
    try:
      elements = tuple(map(int, value.strip('()').split(',')))
      return elements
    except ValueError:
      raise argparse.ArgumentTypeError("Invalid value. Must be an integer or a tuple-like object.")
        
# Parse command line 
def parse_args():
  """
  Return back an args dictionary based on a standard parsing of the command line inputs
  """

  # Command line settings
  parser = argparse.ArgumentParser(description='Simple BERT training parser')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')

  # artichtectural settings
  parser.add_argument('--steps', type=int, default=32, metavar='N',
                      help='Number of times steps in the transformer layer (default: 32)')
  parser.add_argument('--Tf',type=float,default=1.0,
                      help='Final time for transformer layer-parallel part')
  parser.add_argument('--serial-file', type=str, default=None,
                      help='Save network to file in serial (not parallel) format')
  parser.add_argument('--seq-len', type=int, default=64,
                      help='Max sequence length')

  # algorithmic settings (batching)
  parser.add_argument('--percent-data', type=float, default=0.05, metavar='N',
                      help='how much of the data to read in and use for training/testing')
  parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                      help='input batch size for training (default: 32)')
  parser.add_argument('--epochs', type=int, default=3, metavar='N',
                      help='number of epochs to train (default: 3)')
  parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                      help='learning rate (default: 1e-4)')
  
  # algorithmic settings (layer-parallel)
  parser.add_argument('--lp-max-levels', type=int_or_tuple, default=(1,2), metavar='N',
                      help='Layer parallel max number of levels (default: (1,2) one forward, 2 backwards)')
  parser.add_argument('--lp-bwd-max-iters', type=int, default=1, metavar='N',
                      help='Layer parallel max backward iterations (default: 1)')
  parser.add_argument('--lp-fwd-max-iters', type=int, default=2, metavar='N',
                      help='Layer parallel max forward iterations (default: 2)')
  parser.add_argument('--lp-print-level', type=int, default=0, metavar='N',
                      help='Layer parallel internal print level (default: 0)')
  parser.add_argument('--lp-braid-print-level', type=int, default=0, metavar='N',
                      help='Layer parallel braid print level (default: 0)')
  parser.add_argument('--lp-cfactor', type=int, default=4, metavar='N',
                      help='Layer parallel coarsening factor (default: 4)')
  parser.add_argument('--lp-fine-fcf',action='store_true', default=False,
                      help='Layer parallel fine FCF for forward solve, on or off (default: False)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--lp-user-mpi-buf',action='store_true', default=False,
                      help='Layer parallel use user-defined mpi buffers (default: False)')
  parser.add_argument('--lp-use-downcycle', action='store_true', default=False,
                      help='Layer parallel use downcycle on or off (default: False)')

  # data parallelism
  parser.add_argument('--dp-size', type=int, default=1, metavar='N',
                      help='Data parallelism (used if value != 1)')

  ## save model
  parser.add_argument('--output_fn',type=str, default=None,#required=True,
                      help='Output filename (for model saving)')
  parser.add_argument('--models_dir',type=str, default=None,#required=True,
                      help='Models directory (for model saving)')

  ## additional arguments
  parser.add_argument('--model_dimension', type=int, default=128)
  parser.add_argument('--num_heads', type=int, default=1)
  parser.add_argument('--optimizer', type=str, default='SGD')#required=True)
  parser.add_argument('--momentum', type=float, default=.9)

  ##
  # Do some parameter checking
  rank  = MPI.COMM_WORLD.Get_rank()
  procs = MPI.COMM_WORLD.Get_size()
  args = parser.parse_args()

  if procs % args.dp_size != 0:
    root_print(rank, 1, 1, 'Data parallel size must be an even multiple of the number of processors: %d %d'
               % (procs, args.dp_size) )
    sys.exit(0)
  else:
    procs_lp = int(procs / args.dp_size)

  if args.steps % procs_lp != 0:
    root_print(rank, 1, 1, 'Steps must be an even multiple of the number of layer parallel processors: %d %d'
               % (args.steps, procs_lp) )
    sys.exit(0)

  return args


####################################################################################
####################################################################################


##
# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train(rank, params, model, train_loader, optimizer, epoch, compose, device, scheduler):
  # note that we dont' call the optimizer directly, but use the scheduler instead
  train_times = []
  fwd_times = []
  bwd_times = []
  losses = []

  # Train the model
  model.train()

  criterion = nn.CrossEntropyLoss(ignore_index=0)

  total_time = 0.0

  for batch_idx, batch_data in enumerate(train_loader):
    data, target, segment_label, is_next = batch_data['bert_input'], batch_data['bert_label'], batch_data['segment_label'], batch_data['is_next']

    start_time = timer()
    data, target, segment_label, is_next = data.to(device), target.to(device), segment_label.to(device), is_next.to(device)

    batch_fwd_pass_start = time.time()
    next_sent_output, mask_lm_output = model(data, segment_label)
    batch_fwd_pass_end = time.time()
    
    next_loss = compose(
      criterion, next_sent_output, is_next
    )
    mask_loss = compose(
      criterion, mask_lm_output.reshape(-1, mask_lm_output.shape[-1]), target.reshape(-1)
    )

    loss = next_loss + mask_loss
    scheduler.zero_grad()

    batch_bwd_pass_start = time.time()
    loss.backward()
    batch_bwd_pass_end = time.time()
    scheduler.step_and_update_lr()
    
    stop_time = timer()

    total_time += stop_time - start_time
    train_times.append(stop_time - start_time)
    fwd_times.append(batch_fwd_pass_end - batch_fwd_pass_start)
    bwd_times.append(batch_bwd_pass_end - batch_bwd_pass_start)
    losses.append(loss.item())

    if batch_idx % params.log_interval == 0:
      root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {:.2e}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item(), 
               scheduler.get_current_lr()))
      root_print(rank, f'\t Some times: {fwd_times[-4:-1]=} {bwd_times[-4:-1]=} {train_times[-4:-1]=}')

  root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2e}'.format(
    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
           100. * (batch_idx + 1) / len(train_loader), loss.item()))

  return losses, train_times, fwd_times, bwd_times


##
# Evaluate model on validation data
# Return: number of correctly classified test items, total number of test items, loss on test data set
def test(rank, model, test_loader, compose, device):
  # Evaluate the model
  model.eval()

  test_loss = 0
  criterion = nn.CrossEntropyLoss(ignore_index=0)

  with torch.no_grad():
    for _, batch_data in enumerate(test_loader):
      data, target, segment_label, is_next = batch_data['bert_input'], batch_data['bert_label'], batch_data['segment_label'], batch_data['is_next']
      
      data, target, segment_label, is_next = data.to(device), target.to(device), segment_label.to(device), is_next.to(device)
      next_sent_output, mask_lm_output = model(data, segment_label)

      next_loss = compose(
        criterion, next_sent_output, is_next
      ).item()
      mask_loss = compose(
        criterion, mask_lm_output.reshape(-1, mask_lm_output.shape[-1]), target.reshape(-1)
      ).item()
      test_loss += next_loss + mask_loss

  test_loss /= len(test_loader)

  # root_print(rank, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
  root_print(rank, 'Test set: Average loss: {:.4f}'.format(test_loss))
  return test_loss

##
# Parallel printing helper function  
def root_print(rank, s):
  if rank == 0:
    print(s, flush='True')

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
    
    def get_current_lr(self):
       return self.init_lr * self._get_lr_scale()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def main():
  # Begin setting up run-time environment 
  # Initialize MPI
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
  torch.manual_seed(args.seed)

  # Compute number of steps in ResNet per processor
  local_steps = int(args.steps / procs)

  # Finish assembling training and test datasets
  if args.percent_data <= 1:
    root_print(rank, f'Loading {int(args.percent_data * 100)}% of dataset')
  else:
    root_print(rank, f'Loading approx {args.percent_data} elements from each dataset')

  # Get dataloader
  sequence_length = args.seq_len
  ds, vocab_size = obtain_dataset(percent_data = args.percent_data, seq_len=sequence_length)
  train_size, test_size = int(len(ds) * 0.8), len(ds) - int(len(ds) * 0.8)  # 80/20 split by default
  train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])

  train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True
  )
  test_loader = DataLoader(
    test_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True
  )

  root_print(
    rank, f'Data processed. Proceeding to train.'
  )

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
																											not args.lp_use_downcycle))

	# Create layer-parallel network
	# Note this can be done on only one processor, but will be slow
  model = ParallelNet(vocab_size=vocab_size,
                  model_dimension=args.model_dimension, 
                  seq_len=sequence_length,
                  num_heads=args.num_heads,
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
    model.saveSerialNet(f'serialnet_bert_{args.steps}')
    print('Saved file')
    # Quit after saving? 
    import sys
    sys.exit()

    
	# Declare optimizer  
  # print(f'rank {rank}: len(list(model.parameters())) {len(list(model.parameters()))}')
  weight_decay=0.01
  betas=(0.9, 0.999)
  warmup_steps=10000
  optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=betas, weight_decay=weight_decay)
  optim_schedule = ScheduledOptim(
    optimizer, args.model_dimension, n_warmup_steps=warmup_steps
  )
  # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

  root_print(rank, f'Training with {warmup_steps=} and {args.lr=}')
	# Carry out parallel training
  batch_losses = [] 
  test_losses = []
  batch_times = []
  forward_times = []
  backward_times = []

  torch.manual_seed(0)
  for epoch in range(1, args.epochs + 1):
    epoch_time_start = time.time()
    [losses, train_times, batch_f_times, batch_b_times] = train(rank=rank, params=args, model=model, train_loader=train_loader, optimizer=optimizer, epoch=epoch,
          compose=model.compose, device=device, scheduler=optim_schedule)

    batch_losses += losses
    batch_times += train_times
    forward_times += batch_f_times
    backward_times += batch_b_times

    valid_loss = test(rank=rank, model=model, test_loader=test_loader, compose=model.compose, device=device)

    test_losses.append(valid_loss)
    
    epoch_time_end = time.time()
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
    ax2.set_ylabel(r"Validation loss", fontsize=13, color='r')
    plt.savefig(f'bert_layerparallel_training_{procs}_{args.steps}_{args.lp_fwd_max_iters}.png', bbox_inches="tight")

    # Save to file
    # Save the NumPy array to a file
    np.save(f'test_losses_f{procs}_{args.steps}_{args.lp_fwd_max_iters}.npy', np.array(batch_losses))
    np.save(f'valid_losses_f{procs}_{args.steps}_{args.lp_fwd_max_iters}.npy', np.array(test_losses))

    # )Plot and save timings to get approximate 
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
    plt.savefig(f'timing_data_plots_{procs}_{args.steps}_{args.lp_fwd_max_iters}.png')


if __name__ == '__main__':
  main()
  print('Finished.')
