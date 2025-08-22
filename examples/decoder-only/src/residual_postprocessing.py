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


import argparse
import sys
import time
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchbraid
import torchbraid.utils

from mpi4py import MPI
from network_architecture import ParallelNet, GPTConfig

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
    
    # Constant
    betas=(0.9, 0.95)
    warmup_iters=2_000
    block_size = args.seq_len   

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

    if args.lp_bwd_max_iters < 2 or args.lp_fwd_max_iters < 2: 
       root_print(rank, f'Iterations are low; are you sure?')

    # Create layer-parallel network
    config = GPTConfig(
       n_layer=args.steps,
       block_size=block_size,
      bias=False
    )

    model = ParallelNet(config,
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=betas)
    
    root_print(rank, f'Training with {warmup_iters=} and {args.lr=}')
    
    # Carry out parallel training
    torch.manual_seed(0)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for iter_num in range(2000, 10001, 200):
        root_print(rank, f'Processing iteration number {iter_num}')

        # Load stuff 
        checkpoint = torch.load(f'gpt-save-parallel-2/model_checkpoint_{rank}_iter_num={iter_num}', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])  # Adjust the key if necessary
        X = checkpoint['inputs']
        Y = checkpoint['labels']

        torch.cuda.synchronize()
        output = model(X)
        torch.cuda.synchronize()
        loss = model.compose(
            criterion, 
            output.reshape(-1, output.shape[-1]), 
            Y.view(-1)
        )
        torch.cuda.synchronize()
        loss.backward()
        torch.cuda.synchronize()
        
    # epoch_time_end = time.time()
    # if rank == 0: root_print(rank, f'Epoch time: {epoch_time_end - epoch_time_start} seconds')


if __name__ == '__main__':
  main()
  print('Finished.')
