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

from network_architecture import parse_args, ParallelNet, SerialNet
from mpi4py import MPI

import input_pipeline
import preprocessing

from cosine_warmup_scheduler import CosineWarmupScheduler

# torch.set_default_dtype(torch.float64)

##
# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train(rank, params, model, train_loader, optimizer, batch_scheduler, gradient_accumulation, epoch, compose, device):
  train_times = []
  losses = []
  model.train()
  criterion = nn.CrossEntropyLoss(ignore_index=model.pad_id_tgt, reduction='sum')
  # total_time = 0.0
  corr, tot = 0, 0
  accuracies = []
  # optimizer.zero_grad()
  # accumulated_loss = []
  # extended_batch_nonignored_elements = 0
  for extended_batch_idx, (data, target) in enumerate(train_loader):
    # start_time = timer()
    data, target = data.to(device), target.to(device)
    num_extended_batch_nonpad_elements = (target != model.pad_id_tgt).sum().item()
    optimizer.zero_grad()
    extended_batch_loss = 0.
    grads = []

    for mini_batch_idx in range(gradient_accumulation):
      mini_batch_size = data.shape[0] // gradient_accumulation
      mini_batch_data = data[
        mini_batch_idx * mini_batch_size : (mini_batch_idx+1) * mini_batch_size,
      ]
      mini_batch_target = target[
        mini_batch_idx * mini_batch_size : (mini_batch_idx+1) * mini_batch_size,
      ]
      batch_fwd_pass_start = time.time()
      mini_batch_output = model(mini_batch_data)
      # print(rank, mini_batch_size, mini_batch_idx, mini_batch_data.shape, mini_batch_target.shape, mini_batch_output.shape)
      # mini_batch_loss = compose(criterion, mini_batch_output.transpose(1, 2), mini_batch_target) / num_extended_batch_nonpad_elements# if rank == 0 else mini_batch_output.new(1)
      mini_batch_loss = compose(criterion, mini_batch_output.reshape(-1, mini_batch_output.shape[-1]), mini_batch_target.reshape(-1)) / num_extended_batch_nonpad_elements
      batch_fwd_pass_end = time.time()

      extended_batch_loss += mini_batch_loss

      optimizer.zero_grad()
      mini_batch_loss.backward()
      grads.append([p.grad for p in model.parameters()])

      # torch.save(mini_batch_output, f'ga_{gradient_accumulation}_output{mini_batch_idx}.pt')
      # torch.save(mini_batch_target, f'ga_{gradient_accumulation}_target{mini_batch_idx}.pt')

      # loss = compose(criterion, output.reshape(-1, output.shape[-1]), target.reshape(-1))
      # mini_batch_loss = compose(criterion, mini_batch_output.transpose(1, 2), mini_batch_target)
      # mini_batch_loss = criterion(mini_batch_output.transpose(1, 2), mini_batch_target)
      # mini_batch_loss /= num_extended_batch_nonpad_elements
      # mini_batch_loss = compose(lambda x: x / num_extended_batch_nonpad_elements, mini_batch_loss)

      # loss /= gradient_accumulation
      # accumulated_loss.append(loss.item())

      # extended_batch_loss += mini_batch_loss.item()

      # batch_bwd_pass_start = time.time()
      # mini_batch_loss.backward()
      # batch_bwd_pass_end = time.time()

      # stop_time = timer()

      # print('mini_batch_data'  , mini_batch_data  )
      # print('mini_batch_target', mini_batch_target)
      # print('mini_batch_output', mini_batch_output)
      # print('mini_batch_loss'  , mini_batch_loss  )

      # if (batch_idx + 1) % gradient_accumulation == 0:
      #   print('updating...')
      #   optimizer.step()
      #   optimizer.zero_grad()
      #   losses.append(np.sum(accumulated_loss))
      #   print(accumulated_loss, losses)
      #   exit()
      #   accumulated_loss = []

      # print(output.shape, target.shape, output.argmax(dim=-1).shape)
      mini_batch_preds = mini_batch_output.argmax(dim=-1)
      corr += (
        (mini_batch_preds == mini_batch_target) * (mini_batch_target != model.pad_id_tgt)
      ).sum().item()
      tot += (mini_batch_target != model.pad_id_tgt).sum().item()

    for i, p in enumerate(model.parameters()): p.grad = sum(grad[i] for grad in grads)
    optimizer.step()

    if batch_scheduler is not None: batch_scheduler.step()

    # total_time += stop_time - start_time
    # train_times.append(stop_time - start_time)
    # losses.append(loss.item())
    losses    .append(extended_batch_loss.item())
    accuracies.append(corr/tot if tot > 0 else 0.)
    corr, tot = 0, 0

    # print('extended_batch_loss', extended_batch_loss.item())

    # if extended_batch_idx == 10: exit()

    # if 0: ## not updated regarding grad accum
    #   # print(rank, f'Batch idx: {batch_idx}')
    #   print(f'rank: {rank}, batch fwd pass time: {batch_fwd_pass_end - batch_fwd_pass_start}')
    #   print(f'rank: {rank}, batch bwd pass time: {batch_bwd_pass_end - batch_bwd_pass_start}')
    # # if batch_idx == 11: import sys; sys.exit()

    # break#!
    
  # return losses, train_times
  return losses, accuracies

##
# Evaluate model on validation data
# Return: number of correctly classified test items, total number of test items, loss on test data set
def test(rank, model, test_loader, gradient_accumulation, compose, device):
  # if rank != 0: return 0., 0.
  model.eval()
  loss = 0.
  # correct = 0
  # criterion = nn.CrossEntropyLoss()

  corr, tot = 0, 0
  criterion = nn.CrossEntropyLoss(ignore_index=model.pad_id_tgt, reduction='sum')

  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
  
      for mini_batch_idx in range(gradient_accumulation):
        mini_batch_size = data.shape[0] // gradient_accumulation
        mini_batch_data = data[
          mini_batch_idx * mini_batch_size : (mini_batch_idx+1) * mini_batch_size,
        ]
        mini_batch_target = target[
          mini_batch_idx * mini_batch_size : (mini_batch_idx+1) * mini_batch_size,
        ]
        mini_batch_output = model(mini_batch_data)
  
        mini_batch_loss = compose(
          criterion, 
          mini_batch_output.reshape(-1, mini_batch_output.shape[-1]), mini_batch_target.reshape(-1),
        )
        # mini_batch_loss = compose(criterion, mini_batch_output.transpose(1, 2), mini_batch_target)
        # mini_batch_loss /= num_extended_batch_nonpad_elements
        loss += mini_batch_loss.item()
  
        mini_batch_preds = mini_batch_output.argmax(dim=-1)
        corr += (
          (mini_batch_preds == mini_batch_target) * (mini_batch_target != model.pad_id_tgt)
        ).sum().item()
        tot += (mini_batch_target != model.pad_id_tgt).sum().item()
  
      # break#!

  # test_loss /= len(test_loader.dataset)
  loss /= tot
  accuracy = corr/tot if tot > 0 else 0.
  
  # root_print(rank, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
  # root_print(rank, 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
  #   test_loss, corr, tot, corr/tot if tot > 0 else 0.))#correct, len(test_loader.dataset),
  #   #100. * correct / len(test_loader.dataset)))
  # return corr, tot, test_loss#correct, len(test_loader.dataset), test_loss
  return loss, accuracy

##
# Parallel printing helper function  
def root_print(rank, s):
  if rank == 0: print(s)

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
    # seed=0,
  )
  eval_ds, eval_dl = preprocessing.obtain_dataset(
    filename=dev, 
    vocabs=vocabs, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=batch_size,#64,#187, 
    bucket_size=max_len,
    # seed=0,
  )

  return train_ds, eval_ds, train_dl, eval_dl, vocabs

# class SeedSetter:
#   def __init__(self, initial_seed):
#     self.initial_seed = initial_seed
#     self.ctr = 0
#   def set_seed(self, update_ctr=True):
#     seed = 0#self.initial_seed + 10000*self.ctr
#     print(f'Setting seed {seed}')
#     torch.manual_seed(seed)
#     if update_ctr: self.ctr += 1

def main():
  # Begin setting up run-time environment 
  # MPI information
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  procs = comm.Get_size()
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
  # seed_setter = SeedSetter(args.seed)
  # seed_setter.set_seed()

  # Compute number of steps in ResNet per processor
  local_steps = int(args.steps / procs)

  # Finish assembling training and test datasets
  root_print(rank, 'Loading dataset')
  data_path_train = '/users/msalvado/MLT/ML_PQ/data/en_gum-ud-train.conllu.txt'
  data_path_dev = '/users/msalvado/MLT/ML_PQ/data/en_gum-ud-dev.conllu.txt'
  train_ds, eval_ds, train_dl, eval_dl, vocabs = obtain_ds_dl(
    data_path_train, data_path_dev, args.batch_size*args.gradient_accumulation, max_len=2048,
  )
  train_set, test_set, train_loader, test_loader = train_ds, eval_ds, train_dl, eval_dl
  # root_print(rank, f'{len(train_loader)}, {next(iter(train_loader))}')
  root_print(rank, f'len(train_loader)={len(train_loader)}, len(test_loader)={len(test_loader)}')
  # sys.exit()

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
  
  if not args.enforce_serial:
    root_print(rank, 'Building ParallelNet')
    # Create layer-parallel network
    # Note this can be done on only one processor, but will be slow
    model = ParallelNet(args.model_dimension, args.num_heads, #seed_setter,
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

    # Detailed XBraid timings are output to these files for the forward and backward phases
    model.parallel_nn.fwd_app.setTimerFile(
      f'b_fwd_s_{args.steps}_bs_{args.batch_size}_p_{procs}')
    model.parallel_nn.bwd_app.setTimerFile(
      f'b_bwd_s_{args.steps}_bs_{args.batch_size}_p_{procs}')

  else:
    root_print(rank, 'Building SerialNet')
    model = SerialNet(args.model_dimension, args.num_heads, #seed_setter, 
            local_steps=local_steps, Tf=args.Tf).to(device)
    model.compose = lambda op, *p: op(*p)

  # root_print(rank, 'Printing model')
  # root_print(rank, model)
  # for p in model.parameters(): 
  # if rank == 0: print(p)
  # print(rank, 'Weights printed')
  # if rank == 1: 
  #   for batch in train_dl:
  #     print(batch)
  # exit()

  model.pad_id_src = vocabs['forms']['<p>']
  model.pad_id_tgt = vocabs['xpos' ]['<p>']

  print(f'rank {rank}: len(list(model.parameters())) {len(list(model.parameters()))}')
  # Declare optimizer  
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
  # if rank == 0: root_print(rank, optimizer)
  # sys.exit()
  # print(rank, optimizer)
  # exit()
  # if args.scheduler is not None:
  #   lr_changes = args.scheduler.split('--')
  #   lr_changes = [( int(lr_change.split('-')[0]), float(lr_change.split('-')[1]) ) for lr_change in lr_changes]
  # else: lr_changes = None
  #   step_size, gamma = args.scheduler.split('-')
  #   step_size, gamma = int(step_size), float(gamma)
  #   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
  # else: scheduler = None
  if args.scheduler is not None and args.scheduler != 'None':
    if args.scheduler == 'cosine-warmup':
      total_num_batches = args.epochs * len(train_loader)
      warmup = round(.05 * total_num_batches)
      root_print(rank, f'cosine-warmup (W={warmup}, max_iters={total_num_batches})')
      batch_scheduler = CosineWarmupScheduler(optimizer, warmup=warmup, max_iters=total_num_batches)
      epoch_scheduler = None
    else:
      milestones, gamma = args.scheduler.split('-')
      milestones, gamma = [int(s) for s in milestones.split(',')], float(gamma)
      root_print(rank, f'milestones={milestones}, gamma={gamma}')
      epoch_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
      batch_scheduler = None
  else: epoch_scheduler = None; batch_scheduler = None
  root_print(rank, f'epoch_scheduler={epoch_scheduler}')
  root_print(rank, f'batch_scheduler={batch_scheduler}')

  load_save_path = f'../stored_models/n{procs}_bs{args.batch_size}_ga{args.gradient_accumulation}_{args.lp_fwd_max_iters}-{args.lp_bwd_max_iters}its_sch{args.scheduler}.pt'
  if args.load:
    checkpoint = torch.load(load_save_path)
    model    .load_state_dict(checkpoint.pop('model_state'    ))
    optimizer.load_state_dict(checkpoint.pop('optimizer_state'))
    first_epoch = checkpoint.pop('last_epoch') + 1
  else: first_epoch = 1
  
  # Carry out parallel training
  batch_losses = [] 
  batch_times = []
  epoch_times = [] 
  test_times = []
  validat_correct_counts = []

  # torch.manual_seed(0)
  # seed_setter.set_seed()
  for epoch in range(first_epoch, args.epochs + 1):
    # if lr_changes:
    #   next_epoch, gamma = lr_changes[0]
    #   if next_epoch == epoch:
    #     root_print(rank, f'Multiplying LR by {gamma} at epoch {epoch}')
    #     for param_group in optimizer.param_groups:
    #       param_group['lr'] *= gamma
    #     _ = lr_changes.pop(0)

    if epoch > 0:
      start_time = timer()
      # [losses, train_times] = train(rank=rank, params=args, model=model, train_loader=train_loader, optimizer=optimizer, gradient_accumulation=args.gradient_accumulation, epoch=epoch, compose=model.compose, device=device)
      [training_losses, training_accuracies] = train(rank=rank, params=args, model=model, train_loader=train_loader, optimizer=optimizer, batch_scheduler=batch_scheduler, gradient_accumulation=args.gradient_accumulation, epoch=epoch, compose=model.compose, device=device)
      epoch_times += [timer() - start_time]
      # batch_losses += losses
      # batch_times += train_times

      root_print(rank, f'Epoch {epoch}, Training time: {epoch_times[-1]}')
      root_print(rank, f'Training loss: {np.mean(training_losses)}, training accuracy: {np.mean(training_accuracies)*100}%')

      if epoch_scheduler is not None: epoch_scheduler.step()

      if args.save: 
        checkpoint = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'last_epoch': epoch}
        torch.save(checkpoint, load_save_path)

    # start_time = timer()
    validation_loss, validation_accuracy = test(rank=rank, model=model, test_loader=test_loader, gradient_accumulation=args.gradient_accumulation, compose=model.compose, device=device)
    # test_times += [timer() - start_time]
    # validat_correct_counts += [validat_correct]
    root_print(rank, f'Validation loss: {validation_loss}, validation accuracy: {validation_accuracy*100}%')
  
if __name__ == '__main__':
  main()

