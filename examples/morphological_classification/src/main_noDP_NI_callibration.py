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

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbraid
import torchbraid.utils
from torchvision import datasets, transforms

from network_architecture import parse_args, ParallelNet, SerialNet
from mpi4py import MPI

import input_pipeline
import preprocessing

from cosine_warmup_scheduler import CosineWarmupScheduler

# torch.set_default_dtype(torch.float64)

##
# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train_epoch(
  rank, params, model, training_data_loader, optimizer, batch_scheduler, 
  epoch, compose, device, debug=False,
):
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
    batch_bwd_pass_end = time.time()

    stop_time = timer()
    optimizer.step()
    if batch_scheduler is not None: batch_scheduler.step()

    if 0:
      print(f'rank={rank}, Batch idx: {batch_idx}')
      print(f'rank={rank}, Batch fwd pass time: {batch_fwd_pass_end - batch_fwd_pass_start}')
      print(f'rank={rank}, Batch bwd pass time: {batch_bwd_pass_end - batch_bwd_pass_start}')
      if batch_idx == 11: import sys; sys.exit()

    predictions = output.argmax(dim=-1)
    correct = (
      (predictions == target) * (target != model.pad_id_tgt)
    ).sum().item()
    total = (target != model.pad_id_tgt).sum().item()

    times.append(stop_time - start_time)
    losses    .append(loss.item()                )
    accuracies.append(correct/total if total > 0 else 0.)

    if debug and batch_idx == 1: break

  mean_loss     = np.mean(losses    )
  mean_accuracy = np.mean(accuracies)

  return mean_loss, mean_accuracy, times

##
# Evaluate model on validation data
# Return: number of correctly classified test items, total number of test items, loss on test data set
def validate(rank, model, validation_data_loader, compose, device, debug=False):
  model.eval()
  losses = []
  correct, total = 0, 0
  criterion = nn.CrossEntropyLoss(ignore_index=model.pad_id_tgt)

  with torch.no_grad():
    for batch_idx, (data, target) in enumerate(validation_data_loader):
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
    
      if debug and batch_idx == 1: break
      
  loss = np.mean(losses)
  accuracy = correct/total

  return loss, accuracy

##
# Parallel printing helper function  
def root_print(rank, *s):
  if rank == 0: print(*s)

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

def get_model(rank, args, local_steps, num_procs, device, vocabs):
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
    ).to(device)

    # Detailed XBraid timings are output to these files for the forward and backward phases
    model.parallel_nn.fwd_app.setTimerFile(
      f'b_fwd_s_{args.steps}_bs_{args.batch_size}_p_{num_procs}')
    model.parallel_nn.bwd_app.setTimerFile(
      f'b_bwd_s_{args.steps}_bs_{args.batch_size}_p_{num_procs}')

  elif num_procs == 1:
    root_print(rank, 'Building SerialNet...')
    model = SerialNet(
      args.model_dimension, args.num_heads, 
      local_steps=local_steps, Tf=args.Tf,
    ).to(device)
    model.compose = lambda op, *p: op(*p)

  else: raise Exception('If enforce_serial, num_procs must be 1')

  model.pad_id_src = vocabs['forms']['<p>']
  model.pad_id_tgt = vocabs['xpos' ]['<p>']

  return model

def get_optimizer(model, args):  # Declare optimizer  
  optimizer = optim.SGD(
    model.parameters(), 
    lr=args.lr, 
    momentum=args.momentum,
  )
  return optimizer

def interpolate_weights(
  coarse_model, fine_model, cf, rank, num_procs, comm, 
  interpolation_mode='constant', interpolate_momentum=False, 
  coarse_optimizer=None, fine_optimizer=None,
):
  assert interpolation_mode in ['constant', 'linear']

  if interpolate_momentum: 
    assert coarse_optimizer is not None and fine_optimizer is not None, \
    'if interpolate-momentum, coarse & fine optimizers must be provided'

  text = 'Interpolating weights ' \
       + ('w/ ' if interpolate_momentum else 'w/o ') \
       + 'momentum...'
  root_print(rank, text)
  root_print(rank, f'Interpolation-mode={interpolation_mode}')

  if num_procs == 1:
    root_print(rank, 'Code shortcut: #Nodes=1')
    
    def interpolate_weights_from_layers(
      coarse_layers, interpolation_coefficient, mode,
    ):
      if   mode == 'constant': 
        coarse_layer, = coarse_layers
        return coarse_layer.parameters()

      elif mode == 'linear':
        coarse_layer_on_left, coarse_layer_on_right = coarse_layers

        if coarse_layer_on_right is None:
          coarse_layer_on_right = coarse_layer_on_left

        assert len(list(coarse_layer_on_left .parameters())) == \
               len(list(coarse_layer_on_right.parameters()))  # delete for memory efficiency?

        for (coarse_layer_on_left_parameter, coarse_layer_on_right_parameter) \
          in zip(coarse_layer_on_left .parameters(), 
                 coarse_layer_on_right.parameters()):
          yield (1 - interpolation_coefficient) * \
                  coarse_layer_on_left_parameter \
              +    interpolation_coefficient    * \
                  coarse_layer_on_right_parameter

    def interpolate_momentum_from_layers(
      coarse_layers, interpolation_coefficient, mode,
    ):
      if   mode == 'constant': 
        coarse_layer, = coarse_layers
        for coarse_layer_parameter in coarse_layer.parameters():
          coarse_layer_parameter_momentum = \
            coarse_optimizer.state[coarse_layer_parameter] \
                            .get('momentum_buffer')
          yield coarse_layer_parameter_momentum

      elif mode == 'linear':
        coarse_layer_on_left, coarse_layer_on_right = coarse_layers

        if coarse_layer_on_right is None:
          coarse_layer_on_right = coarse_layer_on_left

        assert len(list(coarse_layer_on_left .parameters())) == \
               len(list(coarse_layer_on_right.parameters()))  # delete for memory efficiency?

        for (coarse_layer_on_left_parameter, coarse_layer_on_right_parameter) \
          in zip(coarse_layer_on_left .parameters(), 
                 coarse_layer_on_right.parameters()):
          coarse_layer_on_left_parameter_momentum  = \
            coarse_optimizer.state[coarse_layer_on_left_parameter ] \
                            .get('momentum_buffer')
          coarse_layer_on_right_parameter_momentum = \
            coarse_optimizer.state[coarse_layer_on_right_parameter] \
                            .get('momentum_buffer')
          yield (1 - interpolation_coefficient) * \
                  coarse_layer_on_left_parameter_momentum \
              +    interpolation_coefficient    * \
                  coarse_layer_on_right_parameter_momentum

    # def clone(x): return x.clone() if x is not None else None

    def replace_layer_weights(
      old_parameters, new_parameters, new_momentums=None,
    ):
      for (old_parameter, new_parameter) in zip(
        old_parameters, new_parameters,
      ): old_parameter[:] = new_parameter[:]

        if new_momentums is not None:
          fine_optimizer.state[old_parameter]['momentum_buffer'] = \
            next(new_momentums)

      if new_momentums is not None: assert next(new_momentums, None) is None

    with torch.no_grad():
      for (coarse_model_child, fine_model_child) in zip(
        coarse_model.children(), fine_model.children(),
      ):
        if isinstance(fine_model_child, torchbraid.LayerParallel):
          assert isinstance(coarse_model_child, torchbraid.LayerParallel)

          N_coarse = len(coarse_model_child.layer_models)

          for k in range(N_coarse):
            coarse_layer_on_left  = coarse_model_child.layer_models[k]
            coarse_layer_on_right = [*coarse_model_child.layer_models, None] \
                                    [k+1]
            for j in range(cf):
              fine_layer_idx = cf*k + j
              if fine_layer_idx >= len(fine_model_child.layer_models): return
              fine_layer = fine_model_child.layer_models[fine_layer_idx]
              interpolation_coefficient = j/cf

              lp_new_parameters = interpolate_weights_from_layers(
                layers=(coarse_layer_on_left, coarse_layer_on_right), 
                interpolation_coefficient=interpolation_coefficient,
                mode=interpolation_mode,
              )
              lp_new_momentums = interpolate_momentum_from_layers(
                layers=(coarse_layer_on_left, coarse_layer_on_right), 
                interpolation_coefficient=interpolation_coefficient,
                mode=interpolation_mode,
              ) if interpolate_momentum else None

              replace_layer_weights(
                old_parameters=fine_layer.parameters(),
                new_parameters=lp_new_parameters,
                new_momentums=lp_new_momentums,
              )

        else:
          open_close_new_momentums = (
            coarse_optimizer.state[open_close_parameter].get('momentum_buffer') \
            for open_close_parameter in coarse_model_child.parameters()
          ) if interpolate_momentum else None
          replace_layer_weights(
            old_parameters=fine_model_child.parameters(),
            new_parameters=coarse_model_child.parameters(),
            new_momentums=open_close_new_momentums,
          )

    # else: raise Exception('Unknown interpolation-mode')

  else: raise Exception('Remaining to implement: linear interp. w/ momentum w/ #Nodes > 1')

  # rank_coarse_model_lp_parameters                = []
  # rank_coarse_model_open_close_layers_parameters = []

  # for child in coarse_model.children():
  #   if isinstance(child, torchbraid.LayerParallel):
  #     for layer in child.layer_models:
  #       rank_coarse_model_lp_parameters.append(
  #         list(layer.parameters())
  #       )
  #   else:
  #     rank_coarse_model_open_close_layers_parameters.append(
  #       list(child.parameters())
  #     )

  # state_tag = 'state'
  # send_parameters_pattern = 'send parameters to rank=(\d+)'

  # with torch.no_grad():
  #   if rank == 0:
  #     k, coarse_model_lp_parameters, lp_new_parameters, \
  #       next_rank_to_provide_params = 0, [], None, 0
  #     coarse_optimizer_state = coarse_optimizer.state \
  #                              if interpolate_momentum else None
  #   else:
  #     rank_turn = False

  #     while not rank_turn:
  #       message = comm.recv(source=MPI.ANY_SOURCE)
  #       tag, content = message

  #       if re.match(send_parameters_pattern, tag):
  #         message_source = int(re.search(send_parameters_pattern, tag)[1])
  #         coarse_optimizer_state = coarse_optimizer.state \
  #                                  if interpolate_momentum else None
  #         message = (rank_coarse_model_lp_parameters, coarse_optimizer_state)
  #         comm.send(message, dest=message_source)

  #       elif tag == state_tag:
  #         k, coarse_model_lp_parameters, lp_new_parameters, \
  #           next_rank_to_provide_params = content
  #         rank_turn = True

  #       else: raise Exception(f'Unknown request: {tag}')

  #   for child in fine_model.children():
  #     if isinstance(child, torchbraid.LayerParallel):
  #       for layer in child.layer_models:
  #         if k % cf == 0:
  #           if not coarse_model_lp_parameters:
  #             if rank == next_rank_to_provide_params:
  #               coarse_model_lp_parameters = \
  #                 rank_coarse_model_lp_parameters
  #               # print(f'LP1aI: {len(list(layer.parameters()))} {len(lp_new_parameters)}')

  #             else:
  #               tag = send_parameters_pattern.replace('(\d+)', f'{rank}')
  #               content = None
  #               message_to_send = (tag, content)
  #               comm.send(message_to_send, dest=next_rank_to_provide_params)
  #               message_received = \
  #                 comm.recv(source=next_rank_to_provide_params)
  #               coarse_model_lp_parameters, coarse_optimizer_state = \
  #                 message_received
  #               # print(f'LP1aII: {len(list(layer.parameters()))} {len(lp_new_parameters)}')
          
  #             next_rank_to_provide_params += 1

  #           else: 
  #             # print(f'LP1b: {len(list(layer.parameters()))} {len(lp_new_parameters)}')
  #             pass

  #           lp_new_parameters = coarse_model_lp_parameters.pop(0)

  #         else:
  #           # print(f'LP2: {len(list(layer.parameters()))} {len(lp_new_parameters)}')
  #           pass

  #         for j, (old_parameter, new_parameter) in enumerate(zip(
  #           layer.parameters(), lp_new_parameters,
  #         )): 
  #           # print(f'j={j}')
  #           assert old_parameter.shape == new_parameter.shape, \
  #                  f'{old_parameter.shape}, {new_parameter.shape}'
  #           old_parameter[:] = new_parameter[:]

  #         k += 1

  #     else:  # Open/Close layer
  #       open_close_new_parameters = \
  #         rank_coarse_model_open_close_layers_parameters.pop(0)
  #       # print(f'OC: {len(list(child.parameters()))} {len(open_close_new_parameters)}')
      
  #       for (old_parameter, new_parameter) in zip(
  #         child.parameters(), open_close_new_parameters,
  #       ): 
  #         assert old_parameter.shape == new_parameter.shape, \
  #                f'{old_parameter.shape}, {new_parameter.shape}'
  #         old_parameter[:] = new_parameter[:]

  #   ## Send start signal to next rank
  #   if rank != (num_procs - 1):
  #     content = k, coarse_model_lp_parameters, lp_new_parameters, \
  #               next_rank_to_provide_params
  #     message = (state_tag, content)
  #     comm.send(message, dest=(rank + 1))

  #   else:
  #     assert k % cf == 0, f'k={k}, cf={cf}'
  #     assert not coarse_model_lp_parameters
  #   assert not rank_coarse_model_open_close_layers_parameters

  # comm.Barrier()  # (Maybe unnecessary) To ~synchronize them (in order to avoid any future timeout)

  root_print(rank, '-> Done.')

def main():
  ## MPI information
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  num_procs = comm.Get_size()
  args = parse_args()
  root_print(rank, f'args: {args}')

  ## DEVICE
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device, host = torchbraid.utils.getDevice(comm=comm)
  if not use_cuda:
    device = torch.device("cuda" if use_cuda else "cpu")
  print(
    f'Run info rank: {rank}: Torch version: {torch.__version__}' \
  + f' | Device: {device} | Host: {host}'
  )

  torch.manual_seed(args.seed)  # set seed for reproducibility
  local_steps = int(args.steps / num_procs)  # compute number of steps in ResNet per processor

  ## DATA
  root_print(rank, 'Loading data set...')
  training_data_path = '/users/msalvado/MLT/ML_PQ/data/en_gum-ud-train.conllu.txt'
  validation_data_path = '/users/msalvado/MLT/ML_PQ/data/en_gum-ud-dev.conllu.txt'
  (
    training_data_set, validation_data_set, training_data_loader, 
    validation_data_loader, vocabs,
  ) = obtain_ds_dl(
    training_data_path, validation_data_path, args.batch_size, max_len=2048,
  )
  root_print(rank, f'# Training samples: {len(training_data_loader.dataset)},' \
                + f' # Validation samples: {len(validation_data_loader.dataset)}'
  )
  root_print(rank, f'# Training batches: {len(training_data_loader)},' \
                + f' # Validation batches: {len(validation_data_loader)}'
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
  
  ## How should scheduler work in nested iteration?
  ## --> No scheduler when applying nested iteration (ATM)
  # ## (Scheduler)
  # if 0 and (args.scheduler is not None and args.scheduler != 'None'):
  #   if args.scheduler == 'cosine-warmup':
  #     total_num_batches = args.epochs * len(training_data_loader)
  #     warmup = round(.05 * total_num_batches)
  #     root_print(
  #       rank, 
  #       f'cosine-warmup (W={warmup}, max_iters={total_num_batches})',
  #     )
  #     batch_scheduler = CosineWarmupScheduler(
  #       optimizer, warmup=warmup, max_iters=total_num_batches,
  #     )
  #     epoch_scheduler = None
  #   else:
  #     milestones, gamma = args.scheduler.split('-')
  #     milestones = [int(s) for s in milestones.split(',')]
  #     gamma = float(gamma)
  #     root_print(rank, f'milestones={milestones}, gamma={gamma}')
  #     epoch_scheduler = torch.optim.lr_scheduler.MultiStepLR(
  #       optimizer, milestones, gamma,
  #     )
  #     batch_scheduler = None
  # else: epoch_scheduler = None; batch_scheduler = None
  epoch_scheduler = None; batch_scheduler = None
  root_print(rank, f'epoch_scheduler={epoch_scheduler}')
  root_print(rank, f'batch_scheduler={batch_scheduler}')

  # datetime = dt.datetime.now().strftime('%Y%m%d%H%M%S')
  # load_save_path = f'../stored_models/n{num_procs}_{datetime}.pt'
  load_save_path = f'../stored_models/n{num_procs}_Tf{args.Tf}.pt'
  if args.load:
    checkpoint = torch.load(load_save_path)
    model    .load_state_dict(checkpoint.pop('model_state'    ))
    optimizer.load_state_dict(checkpoint.pop('optimizer_state'))
    # first_epoch = checkpoint.pop('last_epoch')
    root_print(rank, f'Model loaded from {load_save_path}')
  # else: first_epoch = 0

  assert args.ni_num_levels > 0, '#NI-levels must be positive'
  ni_starting_level = -(args.ni_num_levels - 1)

  # assert args.ni_starting_level <= 0, 'Levels ids must be non-positive'
  root_print(
    rank, 
    f'Applying nested iteration from level={ni_starting_level}' \
  + f' to level=0 (both included)'
  )

  # Carry out parallel training
  training_losses  , training_accuracies  , training_times   = [], [], []
  validation_losses, validation_accuracies, validation_times = [], [], []

  root_print(rank, 'Starting training...')

  previous_model = None
  patience = 10
  previous_model_best_accuracy = -1.
  for level in range(ni_starting_level, 1):
    level_global_steps =  args.steps // args.ni_cfactor**(-level)
    level_local_steps  = local_steps // args.ni_cfactor**(-level)
    model = get_model(
      rank, args, level_local_steps, num_procs, device, vocabs,
    )
    optimizer = get_optimizer(model, args)

    # root_print(rank, 'before')
    # for p in model.parameters():
    #   root_print(rank, p.ravel()[:3].tolist() + p.ravel()[-3:].tolist())
    # if level == 0:
    #   for p in previous_model.parameters(): 
    #     root_print(rank, previous_optimizer.state[p])
    #   root_print(rank, 'f')
    #   for p in model.parameters(): 
    #     root_print(rank, optimizer.state[p])

    if previous_model is not None: 
      interpolate_weights(
        previous_model, model, args.ni_cfactor, rank, num_procs, comm,
        previous_optimizer, optimizer, True, args.ni_interpolation,
      )

    # root_print(rank, 'after')
    # # for p in model.parameters():
    # #   root_print(rank, p.ravel()[:3].tolist() + p.ravel()[-3:].tolist())
    # if level == 0:
    #   for p in model.parameters():
    #     root_print(rank, optimizer.state[p])

    root_print(
      rank, f'Level={level}, ' \
          + f'#steps={     level_global_steps}, ' \
          + f'local_steps={level_local_steps }')

    next_validation_accuracy_goal = -1.
    too_slow_improvement_ctr = 0
    epoch = 0
    current_model_best_accuracy = -1.

    while too_slow_improvement_ctr < patience or \
          validation_accuracy < previous_model_best_accuracy:
      if epoch > 0:
        start_time = timer()
        training_loss, training_accuracy, training_times = train_epoch(
          rank=rank, params=args, model=model, 
          training_data_loader=training_data_loader, optimizer=optimizer, 
          batch_scheduler=batch_scheduler, epoch=epoch, compose=model.compose, 
          device=device, debug=args.debug,
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

      validation_loss, validation_accuracy = validate(
        rank=rank, model=model, validation_data_loader=validation_data_loader,
        compose=model.compose, device=device, debug=args.debug,
      )
      root_print(
        rank, 
        f'Validation loss: {validation_loss}' \
      + f', validation accuracy: {validation_accuracy*100}%',
      )

      validation_accuracy = comm.bcast(validation_accuracy, root=0)

      if validation_accuracy >= next_validation_accuracy_goal:
        too_slow_improvement_ctr = 0
        next_validation_accuracy_goal = validation_accuracy + .005
      else:
        too_slow_improvement_ctr += 1

      current_model_best_accuracy = max(
        current_model_best_accuracy, validation_accuracy,
      )

      epoch += 1

    previous_model = model
    previous_optimizer = optimizer
    previous_model_best_accuracy = current_model_best_accuracy

  root_print(rank, 'Training finished.')

if __name__ == '__main__':
  main()




