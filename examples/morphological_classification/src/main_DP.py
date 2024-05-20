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
import torchbraid.utils.data_parallel
from torchvision import datasets, transforms
import sys

from network_architecture import parse_args, ParallelNet
from mpi4py import MPI

import input_pipeline
import preprocessing

from cosine_warmup_scheduler import CosineWarmupScheduler

# torch.set_default_dtype(torch.float64)

##
# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train(
  rank, params, model, training_data_loader, optimizer, batch_scheduler, 
  epoch, compose, device, comm_lp, comm_dp,
):
  rank_lp = comm_lp.Get_rank()
  rank_dp = comm_dp.Get_rank()

  losses, accuracies, times = [], [], []
  model.train()
  criterion = nn.CrossEntropyLoss(ignore_index=model.pad_id_tgt)

  for batch_idx, (data, target) in enumerate(training_data_loader):
    start_time = timer()
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    batch_fwd_pass_start = time.time()
    output = model(data)
    loss = compose(criterion, output.reshape(-1, output.shape[-1]), target.reshape(-1))
    batch_fwd_pass_end = time.time()

    batch_bwd_pass_start = time.time()
    loss.backward()
    torchbraid.utils.data_parallel.average_gradients(model=model, comm_dp=comm_dp)
    batch_bwd_pass_end = time.time()

    stop_time = timer()
    optimizer.step()
    if batch_scheduler is not None: batch_scheduler.step()

    gather_things_start = time.time()
    if rank_lp == 0:
      recv = comm_dp.gather([loss.item(), 1, 1, 1], root=0)
      if rank_dp == 0:
        loss_global = np.sum(np.array([item[0] for item in recv])) / len(recv)
        losses.append(loss_global)  # Reset the last lost by a global loss
        dataset_size = np.sum(np.array([item[1] for item in recv]))
    gather_things_end = time.time()

    if 1:
      print(f'rank={rank}, lp-rank={rank_lp}, dp-rank={rank_dp}, Batch idx: {batch_idx}')
      print(f'rank={rank}, lp-rank={rank_lp}, dp-rank={rank_dp}, Batch fwd pass time: {batch_fwd_pass_end - batch_fwd_pass_start}')
      print(f'rank={rank}, lp-rank={rank_lp}, dp-rank={rank_dp}, Batch bwd pass time: {batch_bwd_pass_end - batch_bwd_pass_start}')
      if batch_idx == 11: import sys; sys.exit()

    predictions = output.argmax(dim=-1)
    correct = (
      (predictions == target) * (target != model.pad_id_tgt)
    ).sum().item()
    total = (target != model.pad_id_tgt).sum().item()

    times.append(stop_time - start_time)
    losses    .append(loss.item()                )
    accuracies.append(correct/total if total > 0 else 0.)

  mean_loss     = np.mean(losses    )
  mean_accuracy = np.mean(accuracies)

  return mean_loss, mean_accuracy, times

##
# Evaluate model on validation data
# Return: number of correctly classified test items, total number of test items, loss on test data set
def validate(rank, model, validation_data_loader, compose, device):
  model.eval()
  losses = []
  correct, total = 0, 0
  criterion = nn.CrossEntropyLoss(ignore_index=model.pad_id_tgt)

  with torch.no_grad():
    for data, target in validation_data_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      loss = compose(
        criterion,
        output.reshape(-1, output.shape[-1]),
        target.reshape(-1),
      )

      predictions = output.argmax(dim=-1)
      correct += (
        (predictions == target) * (target != model.pad_id_tgt)
      ).sum().item()
      total += (target != model.pad_id_tgt).sum().item()

      losses.append(loss.item())

  loss = np.mean(losses)
  accuracy = correct/total

  return loss, accuracy

##
# Parallel printing helper function  
def root_print(rank, s):
  if rank == 0: print(s)

def obtain_ds_dl(
  training_data_path, validation_data_path, batch_size, max_len,
):
  vocabs = input_pipeline.create_vocabs(training_data_path)

  attributes_input = [input_pipeline.CoNLLAttributes.FORM]
  attributes_target = [input_pipeline.CoNLLAttributes.XPOS]

  training_data_set, training_data_loader = preprocessing.obtain_dataset(
    filename=training_data_path, 
    vocabs=vocabs, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=batch_size, 
    bucket_size=max_len,
    # seed=0,
  )
  validation_data_set, validation_data_loader = preprocessing.obtain_dataset(
    filename=validation_data_path, 
    vocabs=vocabs, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=batch_size,#64,#187, 
    bucket_size=max_len,
    # seed=0,
  )

  return (
    training_data_set, validation_data_set, training_data_loader,
    validation_data_loader, vocabs,
  )

def split_data_set_by_dp_rank(data_loaders, size_dp):
  for j, data_loader in enumerate(data_loaders):
    assert data_loader.batch_size % size_dp == 0
    batch_size_per_partition = data_loader.batch_size // size_dp
    
    # partitions = [[] for _ in range(size_dp)]
    # for k, entry in enumerate(data_loader.dataset):
    #   dp_rank = k % size_dp
    #   partitions[dp_rank].append(entry)
    partitions = []
    for k in range(size_dp):
      partition = data_loader.dataset[k::size_dp]
      partitions.append(partition)
  
    data_loaders = [
      torch.utils.data.DataLoader(
        dataset=partition, 
        batch_size=batch_size_per_partition,
        shuffle = (j == 0),
        drop_last=data_loader.drop_last,
      ) for partition in partitions
    ]
    # print(f'drop_last={data_loader.drop_last}')

    yield data_loaders

def main():
  # Begin setting up run-time environment 
  # MPI information
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  num_procs = comm.Get_size()
  args = parse_args()
  root_print(rank, f'args: {args}')

  root_print(rank, f'comm: {comm}'          )
  root_print(rank, f'rank: {rank}'          )
  root_print(rank, f'num_procs: {num_procs}')
  root_print(rank, f'args: {args}'          )

  # MPI information for layer and data parallel
  if num_procs % args.dp_size != 0: 
    raise Exception('Please choose the data parallel communicator size so that it can be divided evenly by all num_procs.')
  comm_dp, comm_lp = torchbraid.utils.data_parallel.split_communicator(comm=comm, splitting=args.dp_size)
  rank_dp = comm_dp.Get_rank()
  size_dp = comm_dp.Get_size()
  rank_lp = comm_lp.Get_rank()
  size_lp = comm_lp.Get_size()
  print(f'rank {rank}, rank_dp {rank_dp}, size_dp {size_dp}, rank_lp {rank_lp}, size_lp {size_lp}')

  # Use device or CPU?
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device, host = torchbraid.utils.getDevice(comm=comm)
  if not use_cuda:
    device = torch.device("cuda" if use_cuda else "cpu")
  print(f'Run info rank: {rank}: Torch version: {torch.__version__} | Device: {device} | Host: {host}')

  # Set seed for reproducibility
  torch.manual_seed(args.seed)

  # Compute number of steps in ResNet per processor
  local_steps = int(args.steps / size_lp)#num_procs)

  # Finish assembling training and test datasets
  root_print(rank, 'Loading data set...')
  training_data_path = '/users/msalvado/MLT/ML_PQ/data/en_gum-ud-train.conllu.txt'
  validation_data_path = '/users/msalvado/MLT/ML_PQ/data/en_gum-ud-dev.conllu.txt'
  (
    training_data_set, validation_data_set, whole_training_data_loader, 
    whole_validation_data_loader, vocabs,
  ) = obtain_ds_dl(
    training_data_path, validation_data_path, args.batch_size, max_len=2048,
  )
  root_print(rank, f'# Training samples: {len(whole_training_data_loader.dataset)},' \
                + f' # Validation samples: {len(whole_validation_data_loader.dataset)}'
  )
  root_print(rank, f'# Training batches: {len(whole_training_data_loader)},' \
                + f' # Validation batches: {len(whole_validation_data_loader)}'
  )

  root_print(rank, f'Partitioning data over {size_dp} partitions...')
  t0 = timer()
  training_data_loaders, validation_data_loaders = split_data_set_by_dp_rank((whole_training_data_loader, whole_validation_data_loader), size_dp)
  t1 = timer(); print(rank_dp, t1 - t0, 'seconds')
  training_data_loader, validation_data_loader = \
    training_data_loaders[rank_dp], validation_data_loaders[rank_dp]
  # assert len(training_data_loader) == len(whole_training_data_loader) and len(validation_data_loader) == len(whole_validation_data_loader), f'len(training_data_loader) {len(training_data_loader)}, len(whole_training_data_loader) {len(whole_training_data_loader)}, len(validation_data_loader) {len(validation_data_loader)}, len(whole_validation_data_loader) {len(whole_validation_data_loader)}'
  root_print(rank, f'# Training samples per rank: {len(training_data_loader.dataset)},' \
                + f' # Validation samples per rank: {len(validation_data_loader.dataset)}'
  )
  root_print(rank, f'# Training batches per rank: {len(training_data_loader)},' \
                + f' # Validation batches per rank: {len(validation_data_loader)}'
  )

  # Diagnostic information
  root_print(rank, '-- num_procs    = {}\n'
                   '-- Tf       = {}\n'
                   '-- steps    = {}\n'
                   '-- max_levels     = {}\n'
                   '-- max_bwd_iters  = {}\n'
                   '-- max_fwd_iters  = {}\n'
                   '-- cfactor        = {}\n'
                   '-- fine fcf       = {}\n'
                   '-- skip down      = {}\n'.format(num_procs,
                                                     args.Tf, args.steps,
                                                     args.lp_max_levels,
                                                     args.lp_bwd_max_iters,
                                                     args.lp_fwd_max_iters,
                                                     args.lp_cfactor,
                                                     args.lp_fine_fcf,
                                                     not args.lp_use_downcycle) )
  
  if not args.enforce_serial:
    root_print(rank, 'Building ParallelNet...')
    # Create layer-parallel network
    # Note this can be done on only one processor, but will be slow
    model = ParallelNet(
      args.model_dimension, args.num_heads,
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
      comm_lp=comm_lp,
      comm_dp=comm_dp,
    ).to(device)

    # Detailed XBraid timings are output to these files for the forward and backward phases
    model.parallel_nn.fwd_app.setTimerFile(
      f'b_fwd_s_{args.steps}_bs_{args.batch_size}_p_{num_procs}')
    model.parallel_nn.bwd_app.setTimerFile(
      f'b_bwd_s_{args.steps}_bs_{args.batch_size}_p_{num_procs}')

  else: raise Exception('''Can't do Serial (enforce_serial) with Data-parallelism)''')

  model.pad_id_src = vocabs['forms']['<p>']
  model.pad_id_tgt = vocabs['xpos' ]['<p>']

  # print(f'rank {rank}: len(list(model.parameters())) {len(list(model.parameters()))}')
  # Declare optimizer  
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

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

  load_save_path = f'../stored_models/n{num_procs}_seed{args.seed}_dpsize{size_dp}.pt'
  if args.load:
    checkpoint = torch.load(load_save_path)
    model    .load_state_dict(checkpoint.pop('model_state'    ))
    optimizer.load_state_dict(checkpoint.pop('optimizer_state'))
    first_epoch = checkpoint.pop('last_epoch')
    root_print(rank, f'Model loaded from {load_save_path}')
  else: first_epoch = 0

  # Carry out parallel training
  training_losses  , training_accuracies  , training_times   = [], [], []
  validation_losses, validation_accuracies, validation_times = [], [], []

  root_print(rank, 'Starting training...')
  for epoch in range(args.epochs + 1):
    if epoch > 0:
      start_time = timer()
      training_loss, training_accuracy, training_times = train(
        rank=rank, params=args, model=model, 
        training_data_loader=training_data_loader, optimizer=optimizer, 
        batch_scheduler=batch_scheduler, epoch=epoch, compose=model.compose,
        device=device, comm_lp=comm_lp, comm_dp=comm_dp,
      )
      training_losses    .append(np.mean(training_loss    ))
      training_accuracies.append(np.mean(training_accuracy))
      training_times += [timer() - start_time]

      if epoch_scheduler is not None: epoch_scheduler.step()

      if args.save: 
        checkpoint = {
          'model_state': model.state_dict(), 
          'optimizer_state': optimizer.state_dict(), 
          'last_epoch': epoch,
        }
        torch.save(checkpoint, load_save_path)

      root_print(rank, f'Epoch {epoch}, Training time: {training_times[-1]}')
      root_print(
        rank, 
        f'Training loss: {training_losses[-1]}' \
      + f', training accuracy: {training_accuracies[-1]*100}%',
      )

    # validation_loss, validation_accuracy = validate(
    #   rank=rank, model=model, validation_data_loader=validation_data_loader,
    #   compose=model.compose, device=device, comm_lp=comm_lp, comm_dp=comm_dp,
    # )
    # root_print(
    #   rank, 
    #   f'Validation loss: {validation_loss}' \
    # + f', validation accuracy: {validation_accuracy*100}%',
    # )

  root_print(rank, 'Training finished.')
  
if __name__ == '__main__':
  main()




