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


# Train network for fashion or digits MNIST 
# Network architecture is Opening Layer + ResNet + CloseLayer
#
#
# Running the following should reproduce parallel output of mnist_notebook.ipynb
# $ mpirun -np 4 python3 mnist_script.py
#
# To reproduce serial mnist_notebook.ipynb output, do
# $ mpirun -np 4 python3 mnist_script.py --lp-max-levels 1
#
# Many options are available, see
# $ python3 mnist_script.py --help
#
# For example, 
# $ mpirun -np 4 python3 mnist_script.py --dataset fashion --steps 24 --channels 8 --percent-data 0.25 --batch-size 100 --epochs 5 
# trains on 25% of fashion MNIST with 24 steps in the ResNet, 8 channels, 100 images per batch over 5 epochs 

from __future__ import print_function

print('Importing modules')
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
# from torchvision import datasets, transforms
from transformers import AutoTokenizer#, AutoModelForSeq2SeqLM
import sys

from network_architecture import parse_args, ParallelNet
from mpi4py import MPI

print('Importing local files')
from data import obtain_data#*
# from model.transformer import Transformer
# from train import evaluate_bleu#*

# torch.set_default_dtype(torch.float64)

DATA_DIR = os.path.join('..', 'data')

##
# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train(rank, params, model, batch, optimizer, epoch, compose, device, criterion):
  train_times = []
  losses = []
  total_time = 0.0
  corr, tot = 0, 0

  # print(batch)
  data, target = batch['input_ids'], batch['labels']
  data, target = data.to(device), target.to(device)
  data, target_inputs, target_outputs = data[:, :-1], target[:, :-1], target[:, 1:]
  x = torch.stack((data, target_inputs))

  batch_fwd_time_start = time.time()
  output = model(x)
  output_embeddings = output
  loss = compose(criterion, output_embeddings.reshape(-1, output_embeddings.shape[-1]), target_outputs.reshape(-1))
  batch_fwd_time_end = time.time()

  optimizer.zero_grad(set_to_none=True)

  batch_bwd_time_start = time.time()
  loss.backward()
  batch_bwd_time_end = time.time()

  optimizer.step()
    
  if rank == 0:
    root_print(rank, f'Training batch fwd pass time: {batch_fwd_time_end - batch_fwd_time_start} seconds')
    root_print(rank, f'Training batch bwd pass time: {batch_bwd_time_end - batch_bwd_time_start} seconds')

  losses.append(loss.item())

  return losses, train_times


##
# Evaluate model on validation data
# Return: number of correctly classified test items, total number of test items, loss on test data set
def test(rank, model, val_data, compose, device, context_window, batch_size):
  model.eval()
  test_loss = 0
  criterion = nn.CrossEntropyLoss()
  eval_iters = 200

  with torch.no_grad():
    for k in range(eval_iters):
      torch.manual_seed(-(k+1))
      batch = get_batch(val_data, context_window, batch_size, device)
      data, target = batch
      output = model(data)
      test_loss += compose(criterion, output.reshape(-1, output.shape[-1]), target.reshape(-1)).item()

  test_loss /= eval_iters
  model.train()

  # root_print(rank, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
  root_print(rank, f'Test set: Average loss: {test_loss :.8f}')

  return test_loss


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

  if rank == 0: print('args', args)

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

  torch.manual_seed(0)

  # name_model = "Helsinki-NLP/opus-mt-en-de"
  root_print(rank, 'Loading tokenizer')
  tokenizer = AutoTokenizer.from_pretrained('marian-tokenizer')#name_model)
  pad_id = tokenizer.pad_token_id
  bos_id = pad_id
  eos_id = tokenizer.eos_token_id

  root_print(rank, 'Loading data')
  lang_src, lang_tgt, dss, dls = obtain_data(tokenizer, device, args.batch_size, args.dp_size, args.debug)
  ds, dl = dss[0], dls[0]
  root_print(rank, f"Number of samples: train, {len(dl['train'])}; test, {len(dl['test'])}.")
  source_length, target_length = 128, 128

  # model = Transformer(args.model_dimension, args.num_heads, args.dim_ff, args.num_encoder_layers, args.num_decoder_layers, tokenizer, pad_id, bos_id, eos_id, device)

  # Diagnostic information
  root_print(rank, '-- procs    = {}\n'
                   '-- channels = {}\n'
                   '-- Tf       = {}\n'
                   '-- steps    = {}\n'
                   '-- max_levels     = {}\n'
                   '-- max_bwd_iters  = {}\n'
                   '-- max_fwd_iters  = {}\n'
                   '-- cfactor        = {}\n'
                   '-- fine fcf       = {}\n'
                   '-- skip down      = {}\n'.format(procs, args.channels, 
                                                     args.Tf, args.steps,
                                                     args.lp_max_levels,
                                                     args.lp_bwd_max_iters,
                                                     args.lp_fwd_max_iters,
                                                     args.lp_cfactor,
                                                     args.lp_fine_fcf,
                                                     not args.lp_use_downcycle) )
  
  # Create layer-parallel network
  # Note this can be done on only one processor, but will be slow
  model = ParallelNet(args.model_dimension, args.num_heads, args.dim_ff, tokenizer, pad_id, bos_id, eos_id, device, args.batch_size, source_length, target_length,
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

  root_print(rank, f'Model: {model}')

  # torch.manual_seed(0)

  # if rank == 0:
  #   for p in model.parameters():
  #     root_print(rank, f'{p.shape} {p.ravel()[:10]}')
  # sys.exit()

  # Detailed XBraid timings are output to these files for the forward and backward phases
  model.parallel_nn.fwd_app.setTimerFile(
    f'b_fwd_s_{args.steps}_c_{args.channels}_bs_{args.batch_size}_p_{procs}')
  model.parallel_nn.bwd_app.setTimerFile(
    f'b_bwd_s_{args.steps}_c_{args.channels}_bs_{args.batch_size}_p_{procs}')

  # Declare optimizer  
  print(f'rank {rank}: len(list(model.parameters())) {len(list(model.parameters()))}')
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
  # if rank == 0: root_print(rank, optimizer)
  # sys.exit()

  # For better timings (especially with GPUs) do a little warm up
  if args.warm_up:
    warm_up_timer = timer()
    train(rank=rank, params=args, model=model, batch=next(iter(dl['train'])), optimizer=optimizer, epoch=0, compose=model.compose, device=device, criterion=criterion)
    model.parallel_nn.timer_manager.resetTimers()
    model.parallel_nn.fwd_app.resetBraidTimer()
    model.parallel_nn.bwd_app.resetBraidTimer()
    if use_cuda:
      torch.cuda.synchronize()
    root_print(rank, f'\nWarm up timer {timer() - warm_up_timer}\n')

  # Carry out parallel training
  # batch_losses = [] 
  batch_times = []
  epoch_times = [] 
  test_times = []
  # validat_correct_counts = []

  torch.manual_seed(0)
  # epoch_time_start = time.time()
  train_dl_iter = iter(dl['train'])

  for epoch in range(args.epochs+1):
    if rank == 0: root_print(rank, f'Epoch {epoch}')
    torch.manual_seed(epoch)

    batch = next(train_dl_iter, None)
    if batch is None: 
        train_dl_iter = iter(dl['train'])
        batch = next(train_dl_iter)

    if epoch > 0:
      # start_time = timer()
      [losses, train_times] = train(rank=rank, params=args, model=model, batch=batch, optimizer=optimizer, epoch=epoch,
            compose=model.compose, device=device, criterion=criterion)
      # epoch_times += [timer() - start_time]
      # batch_losses += losses
      # batch_times += train_times
     

    # if epoch % 500 == 0 or epoch == args.epochs:
    #   start_time = timer()
    #   validat_loss = test(rank=rank, model=model, val_data=val_data, compose=model.compose, device=device, 
    #           context_window=args.context_window, batch_size=args.batch_size)
    #   test_times += [timer() - start_time]
  
    #   # epoch_time_end = time.time()
    #   # if rank == 0: root_print(rank, f'Epoch time: {epoch_time_end - epoch_time_start} seconds')
    #   # epoch_time_start = time.time()

  # Print out Braid internal timings, if desired
  #timer_str = model.parallel_nn.getTimersString()
  #root_print(rank, timer_str)

  # Note: the MNIST example is not meant to exhibit performance
  #root_print(rank,
  #           f'TIME PER EPOCH: {"{:.2f}".format(stats.mean(epoch_times))} '
  #           f'{("(1 std dev " + "{:.2f}".format(stats.mean(epoch_times))) if len(epoch_times) > 1 else ""}')

  if args.serial_file is not None:
    # Model can be reloaded in serial format with: model = torch.load(filename)
    model.saveSerialNet(args.serial_file)

if __name__ == '__main__':
  main()

