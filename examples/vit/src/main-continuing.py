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


# ************************************************************************
# This is for continuing the training; we load the weights and training
# data from previous and continue! 
# ************************************************************************

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

# from network_architecture import ParallelNet
from network_architecture import ParallelNet
from mpi4py import MPI
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import get_cosine_schedule_with_warmup

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
def train(rank, params, model, train_loader, optimizer, epoch, compose, device, scheduler, model_resid=None):
  # note that we dont' call the optimizer directly, but use the scheduler instead
  train_times = []
  fwd_times = []
  bwd_times = []
  losses = []

  comm = MPI.COMM_WORLD
  procs = comm.Get_size()

  # Train the model
  model.train()

  criterion = nn.CrossEntropyLoss()

  total_time = 0.0

  for batch_idx, batch_data in enumerate(train_loader):
    optimizer.zero_grad()

    images, labels = batch_data

    start_time = timer()
    images, labels = images.to(device), labels.to(device)

    torch.cuda.synchronize()
    batch_fwd_pass_start = time.time()
    output = model(images)
    torch.cuda.synchronize()
    batch_fwd_pass_end = time.time()
    loss = compose(
      criterion, 
      output.reshape(-1, output.shape[-1]), 
      labels.view(-1)
    )
    
    torch.cuda.synchronize()
    batch_bwd_pass_start = time.time()
    loss.backward()
    torch.cuda.synchronize()
    batch_bwd_pass_end = time.time()

    # scheduler.step_and_update_lr() # Custom will auto step optimizer
    if (batch_idx + 1) % params.log_interval == 0:
      optimizer.step()  # Update weights
      scheduler.step()
      optimizer.zero_grad()  # Reset gradients for the next accumulation

    stop_time = timer()

    total_time += stop_time - start_time
    train_times.append(stop_time - start_time)
    fwd_times.append(batch_fwd_pass_end - batch_fwd_pass_start)
    bwd_times.append(batch_bwd_pass_end - batch_bwd_pass_start)
    losses.append(loss.item())

    # Save data; note that 
    if batch_idx % 1000 == 0:
      checkpoint = {
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict(),
          'images': images,
          'labels': labels,
      }
      torch.save(
        checkpoint, 
        f'vit-save-continue-{procs}/model_checkpoint_{rank}_{batch_idx=}'
      )

    # 
    if batch_idx % params.log_interval == 0:
      root_print(rank, f'Train Epoch: {epoch} {batch_idx} {losses[-1]} {scheduler.get_last_lr()}')
      root_print(rank, f'\t Some times: {fwd_times[-4:-1]=} {bwd_times[-4:-1]=} {train_times[-4:-1]=}')

  return losses, train_times, fwd_times, bwd_times

# Parallel printing helper function  
def root_print(rank, s):
  if rank == 0:
    print(s, flush='True')


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

  # Load MNIST
  # transform = transforms.Compose([
  #     transforms.Grayscale(3),  # Convert grayscale to 3 channels (required for ViT)
  #     transforms.Resize((224, 224)),  # Resize images to 224x224 (ViT input size)
  #     transforms.ToTensor(),  # Convert images to tensors
  #     # There might be better normalization 
  #     transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
  # ])
  
  # train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
  # train_loader = DataLoader(train_dataset, 
  #                           shuffle=False, pin_memory=True, drop_last=True,
  #                           batch_size=args.batch_size, 
  #                           )
  # Load ImageNet
  transform = transforms.Compose([
      transforms.Resize((224, 224)),  # Resize images to 224x224 (ViT input size)
      transforms.ToTensor(),  # Convert images to tensors
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
  ])
  
  # Load training and validation datasets
  train_dataset = datasets.ImageNet(root="~/", transform=transform)
  # test_dataset = datasets.ImageNet(root="~/", train=False, transform=transform, download=True)
  
  # Create data loaders
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
  # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  
  root_print(
    rank, f'Data processed. Proceeding to train. {len(train_loader)}'
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
  model = ParallelNet(
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


  model.parallel_nn.fwd_app.setBraidTimers(flag=1)
  model.parallel_nn.fwd_app.setTimerFile(
      f'timing_test_p_{procs}')

  betas=(0.9, 0.999)
  warmup_steps=10_000 #50000
  optimizer = optim.AdamW(
    model.parameters(), 
    lr=args.lr, 
    betas=betas, weight_decay=0.01
  )
  optim_schedule = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,  # Number of warmup steps
    num_training_steps=1000000   # Total number of training steps
)

  root_print(rank, f'Training with {warmup_steps=} and {args.lr=}')
	# Carry out parallel training
  batch_losses = [] 
  test_losses = []
  batch_times = []
  forward_times = []
  backward_times = []

  # Loading model 
  checkpoint = torch.load(f'vit-save-parallel-2/model_checkpoint_{rank}_batch_idx={6000}')
  model.load_state_dict(checkpoint['model_state_dict'])  # Adjust the key if necessary
  optimizer.load_state_dict(
    checkpoint['optimizer_state_dict']
  )
  root_print(rank, 'f{optim_schedule=}')
  optim_schedule.load_state_dict(
     checkpoint['scheduler_state_dict']
  )
  root_print(rank, 'f{optim_schedule=}')
  for epoch in range(1, args.epochs + 1):
    epoch_time_start = time.time()
    [losses, train_times, batch_f_times, batch_b_times] = train(rank=rank, params=args, model=model, train_loader=train_loader, optimizer=optimizer, epoch=epoch,
          compose=model.compose, device=device, scheduler=optim_schedule)
    checkpoint = {    'model_state': model.state_dict()}

    batch_losses += losses
    batch_times += train_times
    forward_times += batch_f_times
    backward_times += batch_b_times

    # valid_loss = test(rank=rank, model=model, test_loader=test_loader, compose=model.compose, device=device)

    # test_losses.append(valid_loss)
    
    epoch_time_end = time.time()
    if rank == 0: root_print(rank, f'Epoch time: {epoch_time_end - epoch_time_start} seconds')


if __name__ == '__main__':
  main()
  print('Finished.')
