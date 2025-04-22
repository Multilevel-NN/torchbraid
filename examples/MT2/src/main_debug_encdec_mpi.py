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

## Run examples
# ...: 
#   srun python3 -u main.py --batch-size 8 --epochs 1000000 --d_model 512 --dropout .1 --gradient_accumulation 16 --initialize_parameters --num_warmup_steps 8000 --tokenization unigram --vocab_size 8000 --steps 6 --Tf 6. --lp-max-levels 2 --lp-cfactor 3 --lp-fwd-max-iters 3 --lp-bwd-max-iters 2 --seed 0 --num_training_batches 20000 --load
# 20240102: 
#   srun python3 -u main.py --lp-fwd-max-iters 3 --lp-bwd-max-iters 2 --max_lr 5e-4 --num_warmup_steps 8000

from timeit import default_timer as timer

import datetime as dt
import numpy as np
import os
import re
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

from MacTmp import CPU_Temp

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # when debugging with MPS

print(f'Python version info: {sys.version_info}')

# torch.set_default_dtype(torch.float64)

def bwd(loss, optimizer, do_step, model, grads):
  loss.backward()

  # print(f'{grads=}')

  if isinstance(model, ParallelNet):
    if not grads:
      for i, p in enumerate(model.parallel_nn.parameters()):
        grads.append(p.grad.clone())
    else:
      for i, p in enumerate(model.parallel_nn.parameters()):
        # print(f'{p.grad=}')
        grads[i] += p.grad.clone()
        p.grad = grads[i]  # grads[i]
        # print(f'{p.grad=}')
        # sys.exit()

  if do_step:
    optimizer.step()
    optimizer.zero_grad()

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
  scale, gradient_accumulation_ctr, gradient_accumulation, num_batches,
):
  try:
    while float(CPU_Temp().split()[0]) > 75:
      time.sleep(5)
  except:
    pass

  model.train()
  mean_loss = None
  training_time = 0.
  grads = []
  
  for batch_idx, batch in enumerate(training_data_loader):
    # print(batch)
    # Before proceeding, get some new masks
    if not isinstance(model, SerialNet) and \
       gradient_accumulation_ctr%gradient_accumulation == 0:
      model.new_mask()
      # model.new_mask(batch[0].shape)
      # print(batch[0].shape, batch[1].shape)
      # model.new_mask(batch[0].shape)

    batch_fwd_pass_start = time.time()
    output, loss, src, input_tgt, output_tgt = fwd(
               batch, model, criterion, label_smoother, compose, device, rank, gradient_accumulation)
    batch_fwd_pass_end = time.time()

    gradient_accumulation_ctr += 1

    do_opt_step = (gradient_accumulation_ctr % gradient_accumulation == 0)
    batch_bwd_pass_start = time.time()
    bwd(loss, optimizer, do_opt_step, model, grads)
    batch_bwd_pass_end = time.time()
    if do_opt_step:
      grads = []

    batch_fwd_pass_time = batch_fwd_pass_end - batch_fwd_pass_start
    batch_bwd_pass_time = batch_bwd_pass_end - batch_bwd_pass_start

    if scale:
      print(f'rank={rank}, Batch idx: {batch_idx}')
      print(f'rank={rank}, Batch fwd pass time: {batch_fwd_pass_time}')
      print(f'rank={rank}, Batch bwd pass time: {batch_bwd_pass_time}')
      if batch_idx == 11: sys.exit()

    mean_loss = mean_loss + loss if mean_loss is not None else loss
    training_time += batch_fwd_pass_time + batch_bwd_pass_time

    root_print(rank, next(model.parameters()).grad)
    # optimizer.zero_grad()

    if batch_idx == num_batches - 1:
      break
    # if debug: break  # debug

  mean_loss /= num_batches#len(training_data_loader)
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
  tgt_vocab, device, target_vocabulary, debug, gradient_accumulation,
):
  model.eval()
  mean_loss = None
  originals, references, candidates = [], [], []

  with torch.no_grad():
    for batch_idx, batch in enumerate(data_loader):
      output, loss, src, _, output_tgt = fwd(
        batch, model, criterion, label_smoother, compose, device, rank, 
        gradient_accumulation,
      )
      mean_loss = mean_loss + loss if mean_loss is not None else loss
      preds = generate(model, src, output_tgt.shape[1], debug) 
      extend_sentences(originals, references, candidates, 
                       src_vocab, tgt_vocab, src, output_tgt, preds)
      if debug: break  # debug

  mean_loss /= len(data_loader)
  mean_loss = mean_loss.item()

  bleu_score = corpus_bleu(candidates=candidates, references=references)
  idx = np.random.randint(0, len(originals))
  root_print(rank, f'Original  : {originals [idx]}')
  root_print(rank, f'Candidate : {candidates[idx]}')
  root_print(rank, f'References: {references[idx][0]}')

  # print(f'rank: {rank}, loss: {mean_loss}, bleu: {bleu_score}')

  return mean_loss, bleu_score

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

  root_print(rank, f'Torch-version: {torch.__version__}')
  root_print(rank, f'Sys path     : {sys.path}')
  root_print(rank, f'Torch file   : {torch.__file__}')

  args = parse_args()
  root_print(rank, f'args: {args}')

  datetime = dt.datetime.now().strftime('%Y%m%d%H%M%S')
  # run_id = f'{datetime}_{args.lp_fwd_max_iters=}_{args.lp_bwd_max_iters=}_{args.seed=}_{str(np.random.randint(0, 1000000000)).zfill(9)}'
  run_id = f'{datetime}_{str(np.random.randint(0, 1000000000)).zfill(9)}'
  model_id = f'n{num_procs}_f{args.lp_fwd_max_iters}_b{args.lp_bwd_max_iters}' \
             f'_lr{args.max_lr}_w{args.num_warmup_steps}'
  root_print(rank, f'run_id: {run_id}')
  root_print(rank, f'model_id: {model_id}')

  # Use device or CPU?
  device, host = torchbraid.utils.getDevice(comm=comm)

  if args.no_gpu or not torch.cuda.is_available():
    device = torch.device(
      f'mps:{rank}' if not args.no_gpu and torch.backends.mps.is_available() else 
      'cpu'
    )
  root_print(rank, f'Run info rank: {rank}: Torch version: {torch.__version__} '
        f'| Device: {device} | Host: {host}')

  # print(f'{rank=}, {device=}')

  # Set seed for reproducibility
  torch.manual_seed(args.seed)

  # Compute number of steps in ResNet per processor
  local_steps = int(args.steps/num_procs)

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
      serial_fwd=args.serial_fwd,
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

  if args.initialize_parameters:
    print('Initializing parameters...')
    torch.manual_seed(args.seed + 10)  # remove eventually
    for parameter in model.parameters():
      # root_print(rank, parameter)
      if parameter.ndim > 1: nn.init.xavier_uniform_(parameter)
#       print(parameter.ravel()[:5].tolist(), parameter.ravel()[-5:].tolist())
#     sys.exit()

  print(f'Model: {model}')
  # print(f'rank {rank}: len(list(model.parameters())) {len(list(model.parameters()))}')
  # Declare optimizer  
  optimizer = get_optimizer(model, args.num_warmup_steps, args.max_lr)#torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=5e-4)
  print(f'Optimizer: {optimizer}')
  criterion = nn.KLDivLoss(reduction='batchmean')
  label_smoother = LabelSmoothingDistribution(.1, target_vocabulary.pad_id, 
                                              len(target_vocabulary), device)
  
  # Carry out parallel training
  training_losses  , training_bleus  , training_times   = [], [], []
  validation_losses, validation_bleus, validation_times = [], [], []
  gradient_accumulation_ctr = 0
  best_bleu = float('-inf')

  # for p in model.parameters():
  #   print(rank, p.shape)
  # sys.exit()

  tensors_id = f'{args.enforce_serial}_{args.lp_fwd_max_iters}_{args.num_training_batches}_rank{rank}'
  torch.save(
    [
      p for p in model.parameters()
    ],
    f'grads/firstparams_{tensors_id}.pt'
  )

  # parameters_partition = {
  #   2: (90+4, 90),
  #   3: (60+4, 66, 54),
  # }
  x = torch.load(f'grads/firstparams_True_{args.lp_fwd_max_iters}_{args.num_training_batches}_rank0.pt')
  assert len(x) == 90 + 90 + 4
  if args.enforce_serial:
    for p, px in zip(model.parameters(), x):
      assert p.shape == px.shape, f'{p.shape=}, {px.shape=}'
      p.data = px.data
  else:
    model_parameters = model.parameters()

    if num_procs == 2:
      for i in range(90):
        p = next(model_parameters)
        px = x[90*rank + i]
        assert p.shape == px.shape, f'{p.shape=}, {px.shape=}'
        p.data = px.data

    elif num_procs == 3:
      if rank == 0:
        for i in range(60):
          p = next(model_parameters)
          px = x[i]
          assert p.shape == px.shape, f'{p.shape=}, {px.shape=}'
          p.data = px.data

      elif rank == 1:
        for i in range(66):
          p = next(model_parameters)
          px = x[60 + i]
          assert p.shape == px.shape, f'{p.shape=}, {px.shape=}'
          p.data = px.data

      elif rank == 2:
        for i in range(54):
          p = next(model_parameters)
          px = x[60 + 66 + i]
          assert p.shape == px.shape, f'{p.shape=}, {px.shape=}'
          p.data = px.data
  
    if rank == 0:
      for i in range(180, 184):
        p = next(model_parameters)
        px = x[i]
        assert p.shape == px.shape, f'{p.shape=}, {px.shape=}, {i=}'
        p.data = px.data

    assert next(model_parameters, None) is None

  # torch.save(
  #   [
  #     p for p in model.parameters()
  #   ],
  #   f'oldparams_{args.enforce_serial}_{args.lp_fwd_max_iters}_{args.num_training_batches}.pt'
  # )

  epoch = 0

  epoch_mean_loss, epoch_training_time, gradient_accumulation_ctr = \
   train_epoch(
    rank=rank, params=args, model=model, optimizer=optimizer,
    criterion=criterion, label_smoother=label_smoother, 
    training_data_loader=training_data_loader, epoch=epoch, 
    compose=model.compose, device=device, 
    target_vocabulary=target_vocabulary, debug=args.debug, 
    scale=args.scale, gradient_accumulation_ctr=gradient_accumulation_ctr,
    gradient_accumulation=args.gradient_accumulation,
    num_batches=args.num_training_batches,
  )
  training_losses.append(epoch_mean_loss)
  training_times .append(epoch_training_time)

  root_print(rank, f'Epoch {epoch}, Training time: {training_times[-1]}')
  root_print(rank, f'Training loss: {training_losses[-1]}')#, '
                   # f'training bleu: {training_bleus[-1]}')

  # if args.enforce_serial:
  #   print(f'{model.serial_nn}')
  # else:
  #   print(f'{model.parallel_nn.local_layers=}')
  # print(f'{model.open_nn=}')
  # print(f'{model.close_nn=}')

  # torch.save(
  #   training_losses[-1],
  #   f'loss_{args.enforce_serial}_{args.lp_fwd_max_iters}_{args.num_training_batches}.pt'
  # )
  # torch.save(
  #   [
  #     p for p in model.parameters()
  #   ],
  #   f'newparams_{args.enforce_serial}_{args.lp_fwd_max_iters}_{args.num_training_batches}.pt'
  # )
  torch.save(
    [
      (nm, p.grad) for (nm, p) in model.named_parameters()
    ],
    f'grads/grads_{tensors_id}_encdec.pt'
  )

  root_print(rank, 'Training finished.')


if __name__ == '__main__':
  main()
