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
import inspect
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
  parser = argparse.ArgumentParser(description='Simple GPT training parser')
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
  parser.add_argument('--buffer', action='store_true', default=False, 
                    help='Enable buffer layers (default: False)')
  
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
    gradient_accumulation_steps = args.log_interval
    betas=(0.9, 0.95)
    warmup_iters=2_000
    block_size = args.seq_len 
    max_iters = 600000
    lr_decay_iters = 600000 # make equal to max_iters usually
    eval_iters = 200 # Kind of small...
    eval_interval = 1000 

    # Simple data grabbing thing
    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap('train.bin', dtype=np.uint16, mode='r')
        else:
            data = np.memmap('val.bin', dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (args.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
  
    min_lr = args.lr / 10
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return args.lr * (it + 1) / (warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (args.lr - min_lr)
    
    root_print(
        rank, f'Data processed. Proceeding to train. Using {block_size=}'
    )

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

    # Create layer-parallel network
    config = GPTConfig(
        n_layer=args.steps,
        block_size=block_size,
        bias=False,
        buffer_layers=args.buffer
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
    
    # Once parallel model is created 
    if args.serial_file:
        model.saveSerialNet(f'serialnet_{args.steps}_{args.Tf}')
        import sys
        sys.exit()

    def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optim groups. Any parameters that are 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(rank, f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(rank, f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(rank, f"using fused AdamW: {use_fused}")

        return optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=betas)
    optimizer = configure_optimizers(model, weight_decay=0.01, learning_rate=args.lr, betas=betas, device_type=device)

    root_print(rank, f'Training with {warmup_iters=} and {args.lr=}')

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)

                output = model(X)
                loss = model.compose(
                    criterion, 
                    output.reshape(-1, output.shape[-1]), 
                    Y.view(-1)
                ) 
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    # Carry out parallel training
    torch.manual_seed(0)
    iter_num = 0
    all_loss = []
    criterion = nn.CrossEntropyLoss()

    # Define the saving logic
    def should_save(iter_num):
        if iter_num <= 5000:
            return iter_num % 200 == 0
        elif iter_num <= 10000:
            return iter_num % 500 == 0
        else:
            return iter_num % 1000 == 0

    accumulated_grads = {name: torch.zeros_like(param, device=param.device) 
                     for name, param in model.named_parameters() if param.requires_grad}
    accumulate_flag = False


    model.train()
    while True:
        if iter_num % eval_interval == 0:
            root_print(rank, 'Evaluating loss on validation.')
            losses = estimate_loss()
            root_print(rank, f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        X, Y = get_batch('train')

        start_time = timer()
        torch.cuda.synchronize()
        batch_fwd_pass_start = time.time()
        output = model(X)
        torch.cuda.synchronize()
        batch_fwd_pass_end = time.time()
        loss = model.compose(
            criterion, 
            output.reshape(-1, output.shape[-1]), 
            Y.view(-1)
        ) / gradient_accumulation_steps 
        torch.cuda.synchronize()
        batch_bwd_pass_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        batch_bwd_pass_end = time.time()

        # Accumulate gradients manually
        for name, param in model.named_parameters():
            if param.requires_grad:
                accumulated_grads[name] += param.grad
                param.grad.zero_()  # Explicitly reset param.grad to zero; just in case

        if accumulate_flag:  
            # Set gradients to accumulated values
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.grad = accumulated_grads[name]

            optimizer.step()  # Update weights
            optimizer.zero_grad(set_to_none=True)  # Reset gradients for the next accumulation

            # Reset accumulated gradients
            accumulated_grads = {name: torch.zeros_like(param, device=param.device) 
                                for name, param in model.named_parameters() if param.requires_grad}

        all_loss.append(loss.item())

        # Calculate timings
        forward_time = batch_fwd_pass_end - batch_fwd_pass_start
        backward_time = batch_bwd_pass_end - batch_bwd_pass_start
        total_time = timer() - start_time

        # if should_save(iter_num=iter_num): 
        #     checkpoint = {
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'inputs': X,
        #         'labels': Y,
        #     }

        #     if args.lp_max_levels[1] > 1:
        #         torch.save(
        #         checkpoint, 
        #             f'full-gpt-saves/parallel-model_checkpoint_{rank}_{iter_num=}'
        #         )
        #     else:
        #         torch.save(
        #             checkpoint, 
        #             f'full-gpt-saves/serial-model_checkpoint_{rank}_{iter_num=}'
        #         )

        iter_num += 1
        
        if iter_num % args.log_interval == 0:
            accumulate_flag = True
        else:
            accumulate_flag = False

        if iter_num % args.log_interval == 0:
            root_print(rank, f"{iter_num} Loss: {np.sum(np.array(all_loss[-args.log_interval:])):.3e} Forward time: {forward_time:.3e} Backward Time: {backward_time:.3e} Total time: {total_time:.3e}")        
        if iter_num > max_iters:
            break

        
    # epoch_time_end = time.time()
    # if rank == 0: root_print(rank, f'Epoch time: {epoch_time_end - epoch_time_start} seconds')


if __name__ == '__main__':
  main()
  print('Finished.')
