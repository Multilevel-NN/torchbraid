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

import statistics as stats
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbraid
import torchbraid.utils
from torchvision import datasets, transforms
import sys

from network_architecture import parse_args, ParallelNet, SerialNet; print('split')
# from network_architecture_joint import parse_args, ParallelNet, SerialNet; print('joint')
# from network_architecture_semijoint import parse_args, ParallelNet, SerialNet; print('semijoint')
# from network_architecture_womasks import parse_args, ParallelNet, SerialNet; print('split-womasks')
from mpi4py import MPI

from cosine_warmup_scheduler import CosineWarmupScheduler
from data import obtain_data

# torch.set_default_dtype(torch.float64)

##
# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train_epoch(
  rank, params, model, training_data_loader, optimizer, batch_scheduler, 
  epoch, compose, device, target_vocabulary, debug,
):
  losses, accuracies, times = [], [], []
  model.train()
  criterion = nn.CrossEntropyLoss(ignore_index=target_vocabulary.pad_id)

  for batch_idx, (src, tgt) in enumerate(training_data_loader):
    start_time = timer()
    src, tgt = src.to(device), tgt.to(device)
    input_tgt, output_tgt = tgt[:, :-1], tgt[:, 1:]
    optimizer.zero_grad()

    batch_fwd_pass_start = time.time()
    output = model(src, input_tgt)
    loss = compose(
      criterion,
      output.reshape(-1, output.shape[-1]), output_tgt.reshape(-1),
    )
    batch_fwd_pass_end = time.time()

    # print(f'src={src}')
    # print(f'input_tgt={input_tgt}')
    # print(f'output_tgt={output_tgt}')
    # print(f'output={output}')

    # for p in model.parameters(): 
    #   print(p.ravel()[:3].tolist() + p.ravel()[-3:].tolist())

    # sys.exit()

    batch_bwd_pass_start = time.time()
    loss.clip(max=.1)
    loss.backward()
    batch_bwd_pass_end = time.time()

    stop_time = timer()
    optimizer.step()
    if batch_scheduler is not None: batch_scheduler.step()

    if 1:
      print(f'rank={rank}, Batch idx: {batch_idx}')
      print(f'rank={rank}, Batch fwd pass time: {batch_fwd_pass_end - batch_fwd_pass_start}')
      print(f'rank={rank}, Batch bwd pass time: {batch_bwd_pass_end - batch_bwd_pass_start}')
      if batch_idx == 11: sys.exit()
      if batch_idx ==  2: sys.exit()

    predictions = output.argmax(dim=-1)
    correct = (
        (predictions == output_tgt              ) \
      + (output_tgt  == target_vocabulary.pad_id)
    ).prod(axis=-1).sum().item()
    total = output_tgt.shape[0]
    # print(f'predictions={predictions}')
    # print(f'output_tgt={output_tgt}')
    # correct = (
    #     (predictions == output_tgt              ) \
    #   * (output_tgt  != target_vocabulary.pad_id)
    # ).sum().item()
    # total = (output_tgt != target_vocabulary.pad_id).sum().item()

    times.append(stop_time - start_time)
    losses    .append(loss.item()                       )
    accuracies.append(correct/total if total > 0 else 0.)

    # if debug and batch_idx == 1: break
    # root_print(rank, f'loss={loss}')

    if batch_idx == 5000: break

  mean_loss     = np.mean(losses    )
  mean_accuracy = np.mean(accuracies)

  return mean_loss, mean_accuracy, times

##
# Evaluate model on validation data
# Return: number of correctly classified test items, total number of test items, loss on test data set
def validate(
  rank, model, validation_data_loader, compose, device, target_vocabulary,
  debug,
):
  model.eval()
  losses = []
  correct, total = 0, 0
  criterion = nn.CrossEntropyLoss(ignore_index=target_vocabulary.pad_id)

  with torch.no_grad():
    for batch_idx, (src, tgt) in enumerate(validation_data_loader):
      src, tgt = src.to(device), tgt.to(device)
      input_tgt, output_tgt = tgt[:, :-1], tgt[:, 1:]
      output = model(src, input_tgt)
      loss = compose(
        criterion, 
        output.reshape(-1, output.shape[-1]), 
        output_tgt.reshape(-1),
      )

      # sys.exit()

      predictions = output.argmax(dim=-1)
      correct += (
        (predictions == output_tgt              ) \
      + (output_tgt  == target_vocabulary.pad_id)
      ).prod(axis=-1).sum().item()
      total += output_tgt.shape[0]
      # correct += (
      #     (predictions == output_tgt              ) \
      #   * (output_tgt  != target_vocabulary.pad_id)
      # ).sum().item()
      # total += (output_tgt != target_vocabulary.pad_id).sum().item()

      losses.append(loss.item())

      if debug and batch_idx == 1: break
      
      if batch_idx == 500: break

  loss = np.mean(losses)
  accuracy = correct/total

  return loss, accuracy

##
# Parallel printing helper function  
def root_print(rank, s):
  if rank == 0: print(s)

def main():
  # Begin setting up run-time environment 
  # MPI information
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  num_procs = comm.Get_size()
  args = parse_args()
  root_print(rank, f'args: {args}')

  # Use device or CPU?
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device, host = torchbraid.utils.getDevice(comm=comm)
  if not use_cuda:
    device = torch.device("cuda" if use_cuda else "cpu")
  print(f'Run info rank: {rank}: Torch version: {torch.__version__} | Device: {device} | Host: {host}')

  # Set seed for reproducibility
  torch.manual_seed(args.seed)

  # Compute number of steps in ResNet per processor
  local_steps = int(args.steps / num_procs)

  # Finish assembling training and test datasets
  root_print(rank, 'Loading data set...')
  training_data_path = '/users/msalvado/MLT/ML_PQ/data/en_gum-ud-train.conllu.txt'
  validation_data_path = '/users/msalvado/MLT/ML_PQ/data/en_gum-ud-dev.conllu.txt'
  (
    data_sets, data_loaders, source_vocabulary, target_vocabulary,
    source_max_length, target_max_length,
  ) = obtain_data(args.batch_size, device, args.debug)
  training_data_set   = data_sets['training'  ]
  validation_data_set = data_sets['validation']
  training_data_loader   = data_loaders['training'  ]
  validation_data_loader = data_loaders['validation']
  root_print(rank, f'# Training samples: {len(training_data_loader.dataset)},' \
                + f' # Validation samples: {len(validation_data_loader.dataset)}'
  )
  root_print(rank, f'# Training batches: {len(training_data_loader)},' \
                + f' # Validation batches: {len(validation_data_loader)}'
  )
  print(source_max_length, target_max_length)

  # Diagnostic information
  root_print(rank, '-- num_procs    = {}\n'
                   '-- Tf       = {}\n'
                   '-- steps    = {}\n'
                   '-- max_levels     = {}\n'
                   '-- max_bwd_iters  = {}\n'
                   '-- max_fwd_iters  = {}\n'
                   '-- cfactor        = {}\n'
                   '-- fine fcf       = {}\n'
                   '-- skip down      = {}\n'.format(
    num_procs,
    args.Tf, 
    args.steps,
    args.lp_max_levels,
    args.lp_bwd_max_iters,
    args.lp_fwd_max_iters,
    args.lp_cfactor,
    args.lp_fine_fcf,
    not args.lp_use_downcycle)
  )
  
  if not args.enforce_serial:
    root_print(rank, 'Building ParallelNet...')
    # Create layer-parallel network
    # Note this can be done on only one processor, but will be slow
    model = ParallelNet(
      args.model_dimension, args.num_heads, args.dim_ff, args.dropout, 
      args.batch_size, source_vocabulary, target_vocabulary, 
      source_max_length, target_max_length - 1, device, 
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
    ).to(device)

    # Detailed XBraid timings are output to these files for the forward and backward phases
    model.parallel_nn.fwd_app.setBraidTimers(flag=1)
    model.parallel_nn.fwd_app.setTimerFile(
      #f'b_fwd_s_{args.steps}_bs_{args.batch_size}_p_{num_procs}'
      '/users/msalvado/fwd'
    )
    model.parallel_nn.bwd_app.setBraidTimers(flag=1)
    model.parallel_nn.bwd_app.setTimerFile(
      #f'b_bwd_s_{args.steps}_bs_{args.batch_size}_p_{num_procs}'
      '/users/msalvado/bwd'
    )
    print('model.parallel_nn.bwd_app braid and timers initialized')

  else:
    assert num_procs == 1, 'If enforce_serial, num_procs must be 1'
    root_print(rank, 'Building SerialNet...')
    model = SerialNet(
      args.model_dimension, args.num_heads, args.dim_ff, args.dropout, 
      args.batch_size, source_vocabulary, target_vocabulary, 
      source_max_length, target_max_length - 1, device, 
      local_steps=local_steps, Tf=args.Tf,
    ).to(device)
    model.compose = lambda op, *p: op(*p)

  print(f'model={model}')

  # print(f'rank {rank}: len(list(model.parameters())) {len(list(model.parameters()))}')
  # Declare optimizer  
  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  ## Scheduler
  if args.scheduler is not None and args.scheduler != 'None':
    if args.scheduler == 'cosine-warmup':
      total_num_batches = args.epochs * len(training_data_loader)
      warmup = round(.05 * total_num_batches)
      root_print(
        rank, 
        f'cosine-warmup (W={warmup}, max_iters={total_num_batches})',
      )
      batch_scheduler = CosineWarmupScheduler(
        optimizer, warmup=warmup, max_iters=total_num_batches,
      )
      epoch_scheduler = None
    else:
      milestones, gamma = args.scheduler.split('-')
      milestones = [int(s) for s in milestones.split(',')]
      gamma = float(gamma)
      root_print(rank, f'milestones={milestones}, gamma={gamma}')
      epoch_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma,
      )
      batch_scheduler = None
  else: epoch_scheduler = None; batch_scheduler = None
  root_print(rank, f'epoch_scheduler={epoch_scheduler}')
  root_print(rank, f'batch_scheduler={batch_scheduler}')

  # load_save_path = f'../stored_models/n{num_procs}_seed{args.seed}.pt'
  # if args.load:
  #   checkpoint = torch.load(load_save_path)
  #   model    .load_state_dict(checkpoint.pop('model_state'    ))
  #   optimizer.load_state_dict(checkpoint.pop('optimizer_state'))
  #   first_epoch = checkpoint.pop('last_epoch')
  #   root_print(rank, f'Model loaded from {load_save_path}')
  # else: first_epoch = 0
  first_epoch = 0

  # Carry out parallel training
  training_losses  , training_accuracies  , training_times   = [], [], []
  validation_losses, validation_accuracies, validation_times = [], [], []

  root_print(rank, 'Starting training...')
  for epoch in range(first_epoch, args.epochs + 1):
    if epoch > first_epoch:
      start_time = timer()
      training_loss, training_accuracy, training_times = train_epoch(
        rank=rank, params=args, model=model, 
        training_data_loader=training_data_loader, optimizer=optimizer, 
        batch_scheduler=batch_scheduler, epoch=epoch, compose=model.compose, 
        device=device, target_vocabulary=target_vocabulary, debug=args.debug,
      )
      training_losses    .append(np.mean(training_loss    ))
      training_accuracies.append(np.mean(training_accuracy))
      training_times += [timer() - start_time]

      if epoch_scheduler is not None: epoch_scheduler.step()

      # if args.save: 
      #   checkpoint = {
      #     'model_state': model.state_dict(), 
      #     'optimizer_state': optimizer.state_dict(), 
      #     'last_epoch': epoch,
      #   }
        # torch.save(checkpoint, load_save_path)

      root_print(rank, f'Epoch {epoch}, Training time: {training_times[-1]}')
      root_print(
        rank, 
        f'Training loss: {training_losses[-1]}' \
      + f', training accuracy: {training_accuracies[-1]*100}%',
      )

    # validation_loss, validation_accuracy = validate(
    #   rank=rank, model=model, validation_data_loader=validation_data_loader,
    #   compose=model.compose, device=device, 
    #   target_vocabulary=target_vocabulary, debug=args.debug,
    # )
    # root_print(
    #   rank, 
    #   f'Validation loss: {validation_loss}' \
    # + f', validation accuracy: {validation_accuracy*100}%',
    # )

  root_print(rank, 'Training finished.')

if __name__ == '__main__':
  main()




