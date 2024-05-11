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


import statistics as stats
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbraid
import torchbraid.utils
import sys

# from network_architecture import ParallelNet
from mpi4py import MPI
from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizerFast

####################################################################################
####################################################################################

# Parse command line 
def parse_args():
  """
  Return back an args dictionary based on a standard parsing of the command line inputs
  """

  # Command line settings
  parser = argparse.ArgumentParser(description='Simple BERT training parser')
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

  # algorithmic settings (batching)
  parser.add_argument('--percent-data', type=float, default=0.05, metavar='N',
                      help='how much of the data to read in and use for training/testing')
  parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                      help='input batch size for training (default: 32)')
  parser.add_argument('--epochs', type=int, default=3, metavar='N',
                      help='number of epochs to train (default: 3)')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')

  # algorithmic settings (layer-parallel)
  parser.add_argument('--lp-max-levels', type=int, default=3, metavar='N',
                      help='Layer parallel max number of levels (default: 3)')
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

  ##
  # Compute number of parallel-in-time multigrid levels 
  def compute_levels(num_steps, min_coarse_size, cfactor):
    from math import log, floor
    # Find L such that ( max_L min_coarse_size*cfactor**L <= num_steps)
    levels = floor(log(float(num_steps) / min_coarse_size, cfactor)) + 1

    if levels < 1:
      levels = 1
    return levels

  if args.lp_max_levels < 1:
    min_coarse_size = 3
    args.lp_max_levels = compute_levels(args.steps, min_coarse_size, args.lp_cfactor)

  if args.steps % procs_lp != 0:
    root_print(rank, 1, 1, 'Steps must be an even multiple of the number of layer parallel processors: %d %d'
               % (args.steps, procs_lp) )
    sys.exit(0)

  return args


####################################################################################
####################################################################################


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

    # if rank == 0:
        # root_print(rank, f'Batch idx: {batch_idx}')
    #     root_print(rank, f'Batch fwd pass time: {batch_fwd_pass_end - batch_fwd_pass_start}')
    #     root_print(rank, f'Batch bwd pass time: {batch_bwd_pass_end - batch_bwd_pass_start}')
    if batch_idx == 10:
      # print(data, target)
      # print(output)
      break

    # print(output.shape, target.shape, output.argmax(dim=-1).shape)
    preds = output.argmax(dim=-1)
    corr += ((preds == target) * (target != model.pad_id_tgt)).sum().item()
    tot += (target != model.pad_id_tgt).sum().item()

    total_time += stop_time - start_time
    train_times.append(stop_time - start_time)
    losses.append(loss.item())

    if batch_idx % params.log_interval == 0:
      root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))

  root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
           100. * (batch_idx + 1) / len(train_loader), loss.item()))

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

  # Finish assembling training and test datasets
  root_print(rank, f'Loading {int(args.percent_data * 100)}% of dataset')

  # We will just use the bookcorpus set + wikipedia simple for smaller sizes
  bookcorpus_train = load_dataset('bookcorpus', split=f'train[:{int(args.percent_data * 100)}%]')
  wiki_train = load_dataset("wikipedia", "20220301.simple", split=f'train[:{int(args.percent_data * 100)}%]')
  wiki_train = wiki_train.remove_columns([col for col in wiki_train.column_names if col != "text"]) # Only keep text
  assert bookcorpus_train.features.type == wiki_train.features.type
  raw_datasets = concatenate_datasets([bookcorpus_train, wiki_train])

  # Load pretrained 
  tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
  root_print(rank, f"The max length for the tokenizer is: {tokenizer.model_max_length}")

  def group_texts(examples):
    tokenized_inputs = tokenizer(
        examples["text"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
    )
    return tokenized_inputs

  # preprocess dataset
  tokenized_datasets = raw_datasets.map(group_texts, batched=True, remove_columns=["text"])
  root_print(rank, tokenized_datasets)

	# Diagnostic information
	# root_print(rank, '-- procs    = {}\n'
	# 									'-- channels = {}\n'
	# 									'-- Tf       = {}\n'
	# 									'-- steps    = {}\n'
	# 									'-- max_levels     = {}\n'
	# 									'-- max_bwd_iters  = {}\n'
	# 									'-- max_fwd_iters  = {}\n'
	# 									'-- cfactor        = {}\n'
	# 									'-- fine fcf       = {}\n'
	# 									'-- skip down      = {}\n'.format(procs, args.channels, 
	# 																										args.Tf, args.steps,
	# 																										args.lp_max_levels,
	# 																										args.lp_bwd_max_iters,
	# 																										args.lp_fwd_max_iters,
	# 																										args.lp_cfactor,
	# 																										args.lp_fine_fcf,
	# 																										not args.lp_use_downcycle) )


	# # Create layer-parallel network
	# # Note this can be done on only one processor, but will be slow
	# model = ParallelNet(args.model_dimension, args.num_heads,
	# 								local_steps=local_steps,
	# 								max_levels=args.lp_max_levels,
	# 								bwd_max_iters=args.lp_bwd_max_iters,
	# 								fwd_max_iters=args.lp_fwd_max_iters,
	# 								print_level=args.lp_print_level,
	# 								braid_print_level=args.lp_braid_print_level,
	# 								cfactor=args.lp_cfactor,
	# 								fine_fcf=args.lp_fine_fcf,
	# 								skip_downcycle=not args.lp_use_downcycle,
	# 								fmg=False, 
	# 								Tf=args.Tf,
	# 								relax_only_cg=False,
	# 								user_mpi_buf=args.lp_user_mpi_buf).to(device)

	# # if rank == 0:
	# #   for p in model.parameters():
	# #     root_print(rank, f'{p.dtype}, {p.shape}, {p.ravel()[:10]}')
	# # sys.exit()

	# model.pad_id_src = vocabs['forms']['<p>']
	# model.pad_id_tgt = vocabs['xpos']['<p>']

	# # Detailed XBraid timings are output to these files for the forward and backward phases
	# model.parallel_nn.fwd_app.setTimerFile(
	# 	f'b_fwd_s_{args.steps}_c_{args.channels}_bs_{args.batch_size}_p_{procs}')
	# model.parallel_nn.bwd_app.setTimerFile(
	# 	f'b_bwd_s_{args.steps}_c_{args.channels}_bs_{args.batch_size}_p_{procs}')

	# # Declare optimizer  
	# print(f'rank {rank}: len(list(model.parameters())) {len(list(model.parameters()))}')
	# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
	# # if rank == 0: root_print(rank, optimizer)
	# # sys.exit()

	# # Carry out parallel training
	# batch_losses = [] 
	# batch_times = []
	# epoch_times = [] 
	# test_times = []
	# validat_correct_counts = []

	# torch.manual_seed(0)
	# epoch_time_start = time.time()
	# for epoch in range(1, args.epochs + 1):
	# 	start_time = timer()
	# 	[losses, train_times] = train(rank=rank, params=args, model=model, train_loader=train_loader, optimizer=optimizer, epoch=epoch,
	# 				compose=model.compose, device=device)
	# 	epoch_times += [timer() - start_time]
	# 	batch_losses += losses
	# 	batch_times += train_times

	# 	start_time = timer()
	# 	validat_correct, validat_size, validat_loss = test(rank=rank, model=model, test_loader=test_loader, compose=model.compose, device=device)
	# 	test_times += [timer() - start_time]
	# 	validat_correct_counts += [validat_correct]

	# 	# epoch_time_end = time.time()
	# 	# if rank == 0: root_print(rank, f'Epoch time: s{epoch_time_end - epoch_time_start} seconds')
	# 	# epoch_time_start = time.time()

	# # Print out Braid internal timings, if desired
	# #timer_str = model.parallel_nn.getTimersString()
	# #root_print(rank, timer_str)

	# # Note: the MNIST example is not meant to exhibit performance
	# #root_print(rank,
	# #           f'TIME PER EPOCH: {"{:.2f}".format(stats.mean(epoch_times))} '
	# #           f'{("(1 std dev " + "{:.2f}".format(stats.mean(epoch_times))) if len(epoch_times) > 1 else ""}')
	# if rank == 0:
	# 	fig, ax1 = plt.subplots()
	# 	ax1.plot(batch_losses, color='b', linewidth=2)
	# 	ax1.grid(True, color='k', linestyle='-', linewidth=0.4)
	# 	ax1.set_xlabel(r"Batch number", fontsize=13)
	# 	ax1.set_ylabel(r"Loss", fontsize=13, color='b')
		
	# 	ax2 = ax1.twinx()
	# 	epoch_points = np.arange(1, len(validat_correct_counts)+1) * len(train_loader)
	# 	validation_percentage = np.array(validat_correct_counts) / validat_size
	# 	ax2.plot( epoch_points, validation_percentage, color='r', linestyle='dashed', linewidth=2, marker='o')
	# 	ax2.set_ylabel(r"Validation rate", fontsize=13, color='r')
	# 	plt.savefig('mnist_layerparallel_training.png', bbox_inches="tight")

	# 	# Save to files
	# 	import pickle

	# 	with open(f'data_run_{args.lp_bwd_max_iters}_{args.lp_fwd_max_iters}_{args.lp_max_levels}', 'wb') as fp:
	# 		pickle.dump([batch_losses, epoch_points, validation_percentage], fp)

	# if args.serial_file is not None:
	# 	# Model can be reloaded in serial format with: model = torch.load(filename)
	# 	model.saveSerialNet(args.serial_file)

if __name__ == '__main__':
  main()

