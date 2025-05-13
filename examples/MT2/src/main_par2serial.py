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

import copy
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

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # when debugging with MPS

print(f'Python version info: {sys.version_info}')

# torch.set_default_dtype(torch.float64)

# def bwd(loss, optimizer, do_step):
#   loss.backward()
#   if do_step: optimizer.step(); optimizer.zero_grad()

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
  gradient_accumulation_ctr = 0
# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train_epoch(
  rank, params, model, optimizer, criterion, label_smoother, 
  training_data_loader, epoch, compose, device, target_vocabulary, debug, 
  scale, gradient_accumulation_ctr, gradient_accumulation, num_batches,
):
  gradient_accumulation_ctr = 0
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
  
  # if False:#True:#not args.enforce_serial:
  #   root_print(rank, 'Building ParallelNet...')
  #   # Create layer-parallel network
  #   # Note this can be done on only one processor, but will be slow
  #   model = ParallelNet(
  #     args.d_model, args.nhead, args.dim_feedforward, args.dropout,
  #     source_vocabulary, target_vocabulary, args.batch_size, 
  #     max_sequence_length, device, args.split_decoder,
  #     local_steps=local_steps,
  #     max_levels=args.lp_max_levels,
  #     serial_fwd=args.serial_fwd,
  #     bwd_max_iters=args.lp_bwd_max_iters,
  #     fwd_max_iters=args.lp_fwd_max_iters,
  #     print_level=args.lp_print_level,
  #     braid_print_level=args.lp_braid_print_level,
  #     cfactor=args.lp_cfactor,
  #     fine_fcf=args.lp_fine_fcf,
  #     skip_downcycle=not args.lp_use_downcycle,
  #     fmg=False, 
  #     Tf=args.Tf,
  #     relax_only_cg=False,
  #     user_mpi_buf=args.lp_user_mpi_buf,
  #   ).to(device)

  #   # Detailed XBraid timings are output to these files for the forward and backward phases
  #   # model.parallel_nn.fwd_app.setBraidTimers(flag=1)
  #   model.parallel_nn.fwd_app.setTimerFile(
  #     #f'b_fwd_s_{args.steps}_bs_{args.batch_size}_p_{num_procs}'
  #     '/users/msalvado/fwd'
  #   )
  #   # model.parallel_nn.bwd_app.setBraidTimers(flag=1)
  #   model.parallel_nn.bwd_app.setTimerFile(
  #     #f'b_bwd_s_{args.steps}_bs_{args.batch_size}_p_{num_procs}'
  #     '/users/msalvado/bwd'
  #   )
  #   print('model.parallel_nn.bwd_app braid and timers initialized')

  #   # model.parallel_nn.setFwdResidualCompute(False)
  #   # model.parallel_nn.setBwdResidualCompute(False)

  # else:
  if 1:
    # assert num_procs == 1, 'If enforce_serial, num_procs must be 1'
    # assert (
    #   #     args.lp_fwd_max_iters == 4 
    #   # and args.lp_bwd_max_iters == 2 
    #   and args.max_lr == 5e-4 
    #   and args.dropout == .1 
    #   and args.steps == 6
    #   and args.Tf == 6
    #   and args.batch_size == 8
    #   and args.lp_max_levels == 2
    #   and args.lp_cfactor == 3
    #   and args.d_model == 512
    #   and args.nhead == 8
    #   and args.dim_feedforward == 2048
    #   # and args.gradient_accumulation == 16
    #   and args.num_warmup_steps == 8000
    #   and args.debug == False
    #   and args.scale == False
    #   and args.tokenization == 'unigram'
    #   and args.vocab_size == 8000
    #   and args.num_training_batches == 20000
    #   # and args.serial_fwd == False
    # )
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

  print(f'{args.load=}')

  # print('PARAMETERS BEFORE')
  # for p in model.parameters():
  #   print(f'{rank=}', p.ravel()[:10], p.ravel()[:-10])
  # print()

  # for p in model.parameters():
  #   p.data = p.data.fill_(3)

  if True:
    ## Local debug:
    # model_copy_rank0 = f'../stored_models/id20250122105605_513206685_n2_f20_b20_lr0.0005_w8000_rank0_cp1.pt'
    # model_copy_rank1 = f'../stored_models/id20250122105605_064996318_n2_f20_b20_lr0.0005_w8000_rank1_cp1.pt'
    # dummy_serial_copy = f'../stored_models/id20250122105450_095036526_n1_f2_b1_lr0.0005_w8000_rank0_cp1.pt'
    ## JZ:
    # model_copy_rank0 = f'../stored_models/id20250107122246_439261231_n2_f4_b2_lr0.0005_w8000_rank0_cp1.pt'
    # model_copy_rank1 = f'../stored_models/id20250107122246_395644323_n2_f4_b2_lr0.0005_w8000_rank1_cp1.pt'
    # dummy_serial_copy = f'../stored_models/dummy_serial.pt'
    assert args.load_model_rank0_nm and args.load_model_rank1_nm

    model_copy_rank0 = f'../stored_models/{args.load_model_rank0_nm}'
    model_copy_rank1 = f'../stored_models/{args.load_model_rank1_nm}'


    checkpoint0 = torch.load(model_copy_rank0, map_location=device)#torch.load(model_copy_rank0)
    checkpoint1 = torch.load(model_copy_rank1, map_location=device)#torch.load(model_copy_rank0)
    # checkpointS = torch.load(dummy_serial_copy)
    model_state0 = checkpoint0['model_state']
    model_state1 = checkpoint1['model_state']
    model_stateS = copy.deepcopy(model.state_dict())  # checkpointS['model_state']
    # for k in model_stateS.keys():
    #   print(k)
    # sys.exit()
    updated = []
    layer_ids, layer_idx_pattern = set(), '\.(\d+)\.layer\.layer\.'
    layer_idx = 0
    for j, model_state in enumerate([model_state0, model_state1]):
      assert layer_ids == set(range(len(layer_ids))), f'{layer_ids}'
      layer_shift = len(layer_ids)
      serial_k = None
      for k, v in model_state.items():
        serial_k = k.replace('parallel_nn.local_layers', 'serial_nn').replace('.layer.', '.layer.layer.')
        # assert serial_k in model_stateS, f'{serial_k} not in model_stateS'
        if 'dropout' not in serial_k:  # 1 or serial_k in model_stateS:
          if '.layer.layer.' in serial_k:
            layer_idx = int(re.search(layer_idx_pattern, serial_k)[1])
            serial_k = serial_k.replace(f'.{layer_idx}.layer.layer.', f'.{layer_idx + layer_shift}.layer.layer.')
            layer_ids.add(layer_idx + layer_shift)
          model_stateS[serial_k] = v
          updated.append(serial_k)
        # print(j, layer_idx, serial_k)
    model.load_state_dict(model_stateS)
    checkpoint = checkpoint0

    optimizer = get_optimizer(model, args.num_warmup_steps, args.max_lr)
    optimizer_state0 = checkpoint0['optimizer_state']
    optimizer_state1 = checkpoint1['optimizer_state']
    optimizer_state1['state'] = {
      k + len(optimizer_state0['state']) - 4: v
      for (k, v) in optimizer_state1['state'].items()
    }
    l = len(optimizer_state0['state'])
    for j in range(4):
      k = l - 4 + j
      optimizer_state0['state'][180 + j] = optimizer_state0['state'][k]
    optimizer_state0['state'].update(optimizer_state1['state'])
    # optimizer.optimizer.load_state_dict(checkpoint['optimizer_state'])  # optimizer.load_state_dict(checkpoint['optimizer_state'])
    optimizer_state0['param_groups'][0]['params'] = list(range(184))
    optimizer.optimizer.load_state_dict(optimizer_state0)

    optimizer.current_step_number = checkpoint['optimizer_csn']

    training_losses   = checkpoint['training_losses']
    training_bleus    = checkpoint['training_bleus' ]
    training_times    = checkpoint['training_times' ]
    validation_losses = checkpoint['validation_losses']
    validation_bleus  = checkpoint['validation_bleus' ]
    validation_times  = checkpoint['validation_times' ]
    gradient_accumulation_ctr = checkpoint['gradient_accumulation_ctr']
    best_bleu = checkpoint['best_bleu']
    training_data_loader   = checkpoint['training_data_loader']
    validation_data_loader = checkpoint['validation_data_loader']
    rng_state = checkpoint['rng_state'].cpu()
    torch.set_rng_state(rng_state)

    print(f'Models {args.load_model_rank0_nm} and {args.load_model_rank1_nm} and optimizer loaded successfully')

  # print(f'S: PARAMETERS AFTER')
  # for p in model.parameters():
  #   print(p.ravel()[:10], p.ravel()[:-10])

  # sys.exit()

  root_print(rank, 'Starting training...')
  for epoch in range(0, args.epochs+1):
  #   print(list(model.parameters())[-1].flatten()[:-10])

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
        num_batches=args.num_training_batches,
      )
      training_losses.append(epoch_mean_loss)
      training_times .append(epoch_training_time)

      root_print(rank, f'Epoch {epoch}, Training time: {training_times[-1]}')
      root_print(rank, f'Training loss: {training_losses[-1]}')#, '
                       # f'training bleu: {training_bleus[-1]}')

    if not args.scale:
      t0 = time.time()
      validation_loss, validation_bleu = validate(
        rank=rank, model=model, criterion=criterion, 
        label_smoother=label_smoother, src_vocab=source_vocabulary, 
        tgt_vocab=target_vocabulary, data_loader=validation_data_loader, 
        compose=model.compose, device=device, 
        target_vocabulary=target_vocabulary, debug=args.debug, 
        gradient_accumulation=args.gradient_accumulation,
      )
      t1 = time.time()
      root_print(rank, f'Validation time: {t1 - t0} seconds')
      root_print(rank, f'Validation loss: {validation_loss}, '
                       f'validation bleu: {validation_bleu}')

      validation_loss, validation_bleu = comm.bcast(
        [validation_loss, validation_bleu], 
        root=0,
      )

      # validation_losses.append(...)

      if validation_bleu >= best_bleu:
        best_bleu = validation_bleu
        
        checkpoint = {    'model_state':     model.state_dict(),
                      'optimizer_state': optimizer.optimizer.state_dict(),#optimizer.state_dict(),
                      'optimizer_csn'  : optimizer.current_step_number,
                      'training_losses': training_losses,
                      'training_bleus' : training_bleus,
                      'training_times' : training_times,
                    'validation_losses': validation_losses,
                    'validation_bleus' : validation_bleus,
                    'validation_times' : validation_times,
            'gradient_accumulation_ctr': gradient_accumulation_ctr,
                            'best_bleu': best_bleu,
                 'training_data_loader': training_data_loader,
               'validation_data_loader': validation_data_loader,
                            'rng_state': torch.get_rng_state(),
        }
        torch.save(checkpoint, f'../stored_models/id{run_id}_{model_id}_rank{rank}_cp1.pt')
        torch.save(checkpoint, f'../stored_models/id{run_id}_{model_id}_rank{rank}_cp2.pt')

      # if epoch == 2: sys.exit()
      if args.debug: break

  root_print(rank, 'Training finished.')

if __name__ == '__main__':
  main()




