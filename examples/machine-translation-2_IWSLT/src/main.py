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


from timeit import default_timer as timer

import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torchbraid
import torchbraid.utils

from network_architecture import parse_args, ParallelNet, SerialNet
from mpi4py import MPI

from bleu       import corpus_bleu
from data       import get_data
from generation import generate
from optimizer  import get_optimizer
from _utils     import LabelSmoothingDistribution

# torch.set_default_dtype(torch.float64)

def bwd(loss, optimizer, do_step):
  loss.backward()
  if do_step: optimizer.step(); optimizer.zero_grad()

def fwd(batch, model, criterion, label_smoother, compose, device, rank, gradient_accumulation):
  src, tgt = batch
  src, tgt = src.to(device), tgt.to(device)

  input_tgt, output_tgt = tgt[:, :-1], tgt[:, 1:]
  output_tgt_distribution = label_smoother(output_tgt.reshape(-1, 1))

  output = model(src, input_tgt)
  loss = compose(criterion, output.reshape(-1, output.shape[-1]), 
                            output_tgt_distribution)
  loss /= gradient_accumulation
  return output, loss, src, input_tgt, output_tgt

##
# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train_epoch(
  rank, params, model, optimizer, criterion, label_smoother, 
  training_data_loader, epoch, compose, device, target_vocabulary, debug, 
  scale, gradient_accumulation_ctr, gradient_accumulation,
):
  model.train()
  mean_loss = None
  training_time = 0

  for batch_idx, batch in enumerate(training_data_loader):
    batch_fwd_pass_start = time.time()
    output, loss, src, input_tgt, output_tgt = fwd(
               batch, model, criterion, label_smoother, compose, device, rank, gradient_accumulation)
    batch_fwd_pass_end = time.time()

    gradient_accumulation_ctr += 1

    batch_bwd_pass_start = time.time()
    bwd(loss, optimizer, gradient_accumulation_ctr%gradient_accumulation == 0)
    batch_bwd_pass_end = time.time()

    batch_fwd_pass_time = batch_fwd_pass_end - batch_fwd_pass_start
    batch_bwd_pass_time = batch_bwd_pass_end - batch_bwd_pass_start

    if scale:
      print(f'rank={rank}, Batch idx: {batch_idx}')
      print(f'rank={rank}, Batch fwd pass time: {batch_fwd_pass_time}')
      print(f'rank={rank}, Batch bwd pass time: {batch_bwd_pass_time}')
      if batch_idx == 11: sys.exit()

    mean_loss = mean_loss + loss if mean_loss is not None else loss
    training_time += batch_fwd_pass_time + batch_bwd_pass_time

    if batch_idx == 2000: break

  mean_loss /= len(training_data_loader)
  mean_loss = mean_loss.item()

  return mean_loss, training_time, gradient_accumulation_ctr

def extend_sentences(originals, references, candidates, src_vocab, tgt_vocab, 
                                                             src, tgt, preds):
  originals .extend([ src_vocab.decode(x)  for x in  src  ])
  references.extend([[tgt_vocab.decode(x)] for x in  tgt  ])
  candidates.extend([ tgt_vocab.decode(x)  for x in  preds])

##
# Evaluate model on validation data
# Return: number of correctly classified test items, total number of test items, loss on test data set
def validate(
  rank, model, criterion, label_smoother, data_loader, compose, src_vocab, 
  tgt_vocab, device, target_vocabulary, debug,
):
  model.eval()
  mean_loss = None
  originals, references, candidates = [], [], []

  with torch.no_grad():
    for batch_idx, batch in enumerate(data_loader):
      output, loss, src, _, output_tgt = fwd(
        batch, model, criterion, label_smoother, compose, device, rank, 1,
      )
      mean_loss = mean_loss + loss if mean_loss is not None else loss
      preds = generate(model, src, output_tgt.shape[1]) 
      extend_sentences(originals, references, candidates, 
                       src_vocab, tgt_vocab, src, output_tgt, preds)

  mean_loss /= len(data_loader)
  mean_loss = mean_loss.item()

  bleu_score = corpus_bleu(candidates=candidates, references=references)

  return mean_loss, bleu_score

##
# Parallel printing helper function  
def root_print(rank, s):
  if rank == 0: print(s)

def main():
  print(f'Torch-version: {torch.__version__}')
  print(f'Sys path     : {sys.path}')
  print(f'Torch file   : {torch.__file__}')

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
  if not use_cuda: device = torch.device("cuda" if use_cuda else "cpu")
  print(f'Run info rank: {rank}: Torch version: {torch.__version__} '
        f'| Device: {device} | Host: {host}')

  # Set seed for reproducibility
  torch.manual_seed(args.seed)

  # Compute number of steps in ResNet per processor
  local_steps = int(args.steps / num_procs)

  # Finish assembling training and test datasets
  root_print(rank, 'Loading data set...')
  datasets, data_loaders, vocabs = get_data(device, args.debug, 1, False, 
                         args.tokenization, args.vocab_size, args.scale,
                         batch_size=args.batch_size, drop_last=args.drop_last)
  source_vocabulary, target_vocabulary = vocabs['de'], vocabs['en']

  max_sequence_length = max(max(de_tokens.shape[1], en_tokens.shape[1] - 1) 
                        for (de_tokens, en_tokens) in data_loaders['train'])

  training_dataset   = datasets[  'train'   ]
  validation_dataset = datasets['validation']
  training_data_loader   = data_loaders[  'train'   ]
  validation_data_loader = data_loaders['validation']
  root_print(rank, f'#  Training  samples: {len(training_data_loader.dataset)}, '
                   f'# Validation samples: {len(validation_data_loader.dataset)}'
  )
  root_print(rank, f'#  Training  batches: {len(training_data_loader)}, '
                   f'# Validation batches: {len(validation_data_loader)}'
  )
  root_print(rank, #f'Source max sequence length: {source_max_sequence_length}, '
                   #f'Target max sequence length: {target_max_sequence_length}, '
                   f'Max sequence length: {max_sequence_length}')

  # Diagnostic information
  root_print(rank, f'-- num_procs      = {num_procs                }\n'
                   f'-- Tf             = {args.Tf                  }\n'
                   f'-- steps          = {args.steps               }\n'
                   f'-- max_levels     = {args.lp_max_levels       }\n'
                   f'-- max_bwd_iters  = {args.lp_bwd_max_iters    }\n'
                   f'-- max_fwd_iters  = {args.lp_fwd_max_iters    }\n'
                   f'-- cfactor        = {args.lp_cfactor          }\n'
                   f'-- fine fcf       = {args.lp_fine_fcf         }\n'
                   f'-- skip down      = {not args.lp_use_downcycle}\n')
  
  if not args.enforce_serial:
    root_print(rank, 'Building ParallelNet...')
    # Create layer-parallel network
    # Note this can be done on only one processor, but will be slow
    model = ParallelNet(
      args.d_model, args.nhead, args.dim_feedforward, args.dropout,
      source_vocabulary, target_vocabulary, args.batch_size, 
      max_sequence_length, device, args.split_decoder,
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
      args.d_model, args.nhead, args.dim_feedforward, args.dropout, 
      source_vocabulary, target_vocabulary, args.batch_size, 
      max_sequence_length, device, args.split_decoder, 
      local_steps=local_steps, Tf=args.Tf,
    ).to(device)
    model.compose = lambda op, *p: op(*p)

  print(f'Model: {model}')
  # print(f'rank {rank}: len(list(model.parameters())) {len(list(model.parameters()))}')
  # Declare optimizer  
  optimizer = get_optimizer(model, args.num_warmup_steps)
  print(f'Optimizer: {optimizer}')
  criterion = nn.KLDivLoss(reduction='batchmean')
  label_smoother = LabelSmoothingDistribution(.1, target_vocabulary.pad_id, 
                                              len(target_vocabulary), device)
  # Carry out parallel training
  training_losses  , training_bleus  , training_times   = [], [], []
  validation_losses, validation_bleus, validation_times = [], [], []
  gradient_accumulation_ctr = 0

  root_print(rank, 'Starting training...')
  for epoch in range(args.epochs+1):
    if epoch > 0:
      epoch_mean_loss, epoch_training_time, gradient_accumulation_ctr = \
       train_epoch(
        rank=rank, params=args, model=model, optimizer=optimizer,
        criterion=criterion, label_smoother=label_smoother, 
        training_data_loader=training_data_loader, epoch=epoch, 
        compose=model.compose, device=device, 
        target_vocabulary=target_vocabulary, debug=args.debug, 
        scale=args.scale, gradient_accumulation_ctr=gradient_accumulation_ctr,
        gradient_accumulation=args.gradient_accumulation,
      )
      training_losses.append(epoch_mean_loss)
      training_times .append(epoch_training_time)

      root_print(rank, f'Epoch {epoch}, Training time: {training_times[-1]}')
      root_print(rank, f'Training loss: {training_losses[-1]}')#, '
                       # f'training bleu: {training_bleus[-1]}')

    if not args.scale:
      validation_loss, validation_bleu = validate(
        rank=rank, model=model, criterion=criterion, 
        label_smoother=label_smoother, src_vocab=source_vocabulary, 
        tgt_vocab=target_vocabulary, data_loader=validation_data_loader, 
        compose=model.compose, device=device, 
        target_vocabulary=target_vocabulary, debug=args.debug,
      )
      root_print(rank, f'Validation loss: {validation_loss}, '
                       f'validation bleu: {validation_bleu}')

  root_print(rank, 'Training finished.')

if __name__ == '__main__':
  main()




