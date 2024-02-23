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

from network_architecture import parse_args, ParallelNet
from mpi4py import MPI

import input_pipeline
import preprocessing

# torch.set_default_dtype(torch.float64)

##
# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train(rank, params, model, train_loader, optimizer, epoch, compose, device):
  train_times = []
  losses = []
  model.train()
  criterion = nn.CrossEntropyLoss(ignore_index=model.pad_id_tgt)
  total_time = 0.0
  corr, tot = 0, 0
  for batch_idx, (data, target) in enumerate(train_loader):
    start_time = timer()
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    batch_fwd_pass_start = time.time()
    output = model(data)
    # print(output.reshape(-1, output.shape[-1]).shape, target.reshape(-1).shape)
    # import sys; sys.exit()
    loss = compose(criterion, output.reshape(-1, output.shape[-1]), target.reshape(-1))
    batch_fwd_pass_end = time.time()

    batch_bwd_pass_start = time.time()
    loss.backward()
    batch_bwd_pass_end = time.time()

    stop_time = timer()
    optimizer.step()
    
    # if rank == 0: 
    #   root_print(rank, f'rank{rank}, batch_idx {batch_idx}, data {data}, target {target}, loss {loss}')
    #   for p in model.parameters(): root_print(rank, f'{p.shape}, {p.ravel()[:10]}')
    #   sys.exit()

    if rank == 0:
        root_print(rank, f'Batch idx: {batch_idx}')
        root_print(rank, f'Batch fwd pass time: {batch_fwd_pass_end - batch_fwd_pass_start}')
        root_print(rank, f'Batch bwd pass time: {batch_bwd_pass_end - batch_bwd_pass_start}')
    if batch_idx == 11: import sys; sys.exit()

    # print(output.shape, target.shape, output.argmax(dim=-1).shape)
    preds = output.argmax(dim=-1)
    corr += ((preds == target) * (target != model.pad_id_tgt)).sum().item()
    tot += (target != model.pad_id_tgt).sum().item()

    total_time += stop_time - start_time
    train_times.append(stop_time - start_time)
    losses.append(loss.item())

  return losses, train_times


##
# Evaluate model on validation data
# Return: number of correctly classified test items, total number of test items, loss on test data set
def test(rank, model, test_loader, compose, device):
  model.eval()
  test_loss = 0
  # correct = 0
  # criterion = nn.CrossEntropyLoss()

  corr, tot = 0, 0
  criterion = nn.CrossEntropyLoss(ignore_index=model.pad_id_tgt)

  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      # print(rank, output.shape)
      test_loss += compose(criterion, output.reshape(-1, output.shape[-1]), target.reshape(-1)).item()

      if rank == 0:
        # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        # correct += pred.eq(target.view_as(pred)).sum().item()
        pred = output.argmax(dim=-1)
        corr += ((pred == target) * (target != model.pad_id_tgt)).sum().item()
        tot += (target != model.pad_id_tgt).sum().item()

  test_loss /= len(test_loader.dataset)

  # root_print(rank, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
  root_print(rank, 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
    test_loss, corr, tot, corr/tot if tot > 0 else 0.))#correct, len(test_loader.dataset),
    #100. * correct / len(test_loader.dataset)))
  return corr, tot, test_loss#correct, len(test_loader.dataset), test_loss


##
# Parallel printing helper function  
def root_print(rank, s):
  if rank == 0:
    print(s)


def obtain_ds_dl(data_path_train, data_path_dev, batch_size, max_len):
  train = data_path_train
  dev = data_path_dev

  vocabs = input_pipeline.create_vocabs(train)

  attributes_input = [input_pipeline.CoNLLAttributes.FORM]
  attributes_target = [input_pipeline.CoNLLAttributes.XPOS]

  train_ds, train_dl = preprocessing.obtain_dataset(
    filename=train, 
    vocabs=vocabs, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=batch_size, 
    bucket_size=max_len,
    seed=0,
  )
  eval_ds, eval_dl = preprocessing.obtain_dataset(
    filename=dev, 
    vocabs=vocabs, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=batch_size,#64,#187, 
    bucket_size=max_len,
    seed=0,
  )

  return train_ds, eval_ds, train_dl, eval_dl, vocabs

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
  root_print(rank, 'Loading dataset')
  data_path_train = '/users/msalvado/MLT/ML_PQ/data/en_gum-ud-train.conllu.txt'
  data_path_dev = '/users/msalvado/MLT/ML_PQ/data/en_gum-ud-dev.conllu.txt'
  train_ds, eval_ds, train_dl, eval_dl, vocabs = obtain_ds_dl(data_path_train, data_path_dev, args.batch_size, max_len=2048)
  train_set, test_set, train_loader, test_loader = train_ds, eval_ds, train_dl, eval_dl
  # root_print(rank, f'{len(train_loader)}, {next(iter(train_loader))}')
  # sys.exit()


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
  model = ParallelNet(args.model_dimension, args.num_heads,
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

  # if rank == 0:
  #   for p in model.parameters():
  #     root_print(rank, f'{p.dtype}, {p.shape}, {p.ravel()[:10]}')
  # sys.exit()

  model.pad_id_src = vocabs['forms']['<p>']
  model.pad_id_tgt = vocabs['xpos']['<p>']

  # Detailed XBraid timings are output to these files for the forward and backward phases
  model.parallel_nn.fwd_app.setTimerFile(
    f'b_fwd_s_{args.steps}_c_{args.channels}_bs_{args.batch_size}_p_{procs}')
  model.parallel_nn.bwd_app.setTimerFile(
    f'b_bwd_s_{args.steps}_c_{args.channels}_bs_{args.batch_size}_p_{procs}')

  # Declare optimizer  
  print(f'rank {rank}: len(list(model.parameters())) {len(list(model.parameters()))}')
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
  # if rank == 0: root_print(rank, optimizer)
  # sys.exit()

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

  torch.manual_seed(0)
  epoch_time_start = time.time()
  for epoch in range(1, args.epochs + 1):
    start_time = timer()
    [losses, train_times] = train(rank=rank, params=args, model=model, train_loader=train_loader, optimizer=optimizer, epoch=epoch,
          compose=model.compose, device=device)
    epoch_times += [timer() - start_time]
    batch_losses += losses
    batch_times += train_times

    # start_time = timer()
    # validat_correct, validat_size, validat_loss = test(rank=rank, model=model, test_loader=test_loader, compose=model.compose, device=device)
    # test_times += [timer() - start_time]
    # validat_correct_counts += [validat_correct]
  
    # epoch_time_end = time.time()
    # if rank == 0: root_print(rank, f'Epoch time: {epoch_time_end - epoch_time_start} seconds')
    # epoch_time_start = time.time()

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

