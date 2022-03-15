#@HEADER
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
#@HEADER

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import unittest
import os
import subprocess

import faulthandler
faulthandler.enable()

import sys
import argparse
import torch
import torchbraid
import torchbraid.utils
import torch.nn                 as nn
import torch.nn.functional      as F
import torch.optim              as optim
import torch.optim.lr_scheduler as lr_scheduler
import statistics               as stats
import numpy                    as np

from torchvision import datasets, transforms
from timeit      import default_timer as timer
from mpi4py      import MPI
from utils       import root_print, load_data, ParallelRNNDataLoader, SerialNet, ParallelNet, ImplicitSerialNet
from math        import log, floor

#from matplotlib  import pyplot

def git_rev():
  path = os.path.dirname(os.path.abspath(__file__))
  return subprocess.check_output(['git', 'rev-parse', 'HEAD'],cwd=path).decode('ascii').strip()

def train(rank,log_interval,model,train_loader,optimizer,epoch,compose):
  torch.enable_grad()

  model.train()
  criterion = nn.CrossEntropyLoss()
  total_time = 0.0
  total_time_fp = 0.0
  total_time_bp = 0.0
  total_time_cm = 0.0

  switched = False
  format_str = 'Train Epoch: {:2d} [{:4d}/{:4d}]\tLoss: {:.6f}\tTime Per Batch {:.6f} - F{:2d}/{:.2e}, B{:2d}/{:.2e}'
  total_data = 0
  for batch_idx,(data,target) in enumerate(train_loader):

    start_time = timer()

    # compute forward
    optimizer.zero_grad()
    start_time_fp = timer()
    output = model(data)
    total_time_fp += timer()-start_time_fp

    fwd_itr, fwd_res = model.getFwdStats()

    # compute loss
    start_time_cm = timer()
    loss = compose(criterion,output,target)
    total_time_cm += timer()-start_time_cm

    # compute gradient
    start_time_bp = timer()
    loss.backward()
    total_time_bp += timer()-start_time_bp

    bwd_itr, bwd_res = model.getBwdStats()

    # take step
    stop_time = timer()
    optimizer.step()

    total_data += len(data)
    total_time += stop_time-start_time
    if batch_idx % log_interval == 0:
        root_print(rank,format_str.format(
                        epoch, 
                        total_data,
                        len(train_loader.dataset), 
                        loss.item(),
                        total_time/(batch_idx+1.0),
                        fwd_itr,fwd_res,
                        bwd_itr,bwd_res
                        ))

  root_print(rank,(format_str+', fp={:.6f}, cm={:.6f}, bp={:.6f}').format(
                  epoch, 
                  len(train_loader.dataset),
                  len(train_loader.dataset), 
                  loss.item(),
                  total_time/(batch_idx+1.0),
                  fwd_itr,fwd_res,
                  bwd_itr,bwd_res,
                  total_time_fp/len(train_loader.dataset),
                  total_time_cm/len(train_loader.dataset),
                  total_time_bp/len(train_loader.dataset)
                  ))

def test(rank,model,test_loader,epoch,compose,prefix=''):
  model.eval()
  correct = 0
  criterion = nn.CrossEntropyLoss()
  total_time = 0.0
  with torch.no_grad():
    for data,target in test_loader:
      start_time = timer()

      # evaluate inference
      output = model(data)

      # compute the number of solutions
      pred = compose(output.argmax, dim=1, keepdim=True)
      stop_time = timer()

      total_time += stop_time-start_time

      # only accumulate the correct values on the root
      if rank==0:
        correct += pred.eq(target.view_as(pred)).sum().item()

  root_print(rank,'{}Test set epoch {:2d}: Accuracy: {}/{} ({:.0f}%)\tTime Per Batch {:.6f}'.format(
      prefix,
      epoch,
      correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset),
      total_time/len(test_loader.dataset)))

  return 100. * correct / len(test_loader.dataset)


def compute_levels(num_steps,min_coarse_size,cfactor):
  # we want to find $L$ such that ( max_L min_coarse_size*cfactor**L <= num_steps)
  levels = floor(log(float(num_steps)/min_coarse_size,cfactor)) + 1
  if levels < 1:
    levels = 1
  return levels
# end compute levels

def buildParallelNet(config,
                     rank,
                     input_size,
                     hidden_size,
                     num_layers,
                     num_classes,
                     num_steps,
                     seed):
  root_print(rank,'Using ParallelNet:')
  root_print(rank,'-- max_levels = {}\n'
                  '-- max_iters  = {}\n'
                  '-- fwd_iters  = {}\n'
                  '-- cfactor    = {}\n'
                  '-- fwd0 relax = {}\n'
                  '-- fwd relax  = {}\n'
                  '-- fwd tol    = {}\n'
                  '-- bwd tol    = {}\n'
                  '-- skip down  = {}\n'.format(config['lp_levels'],
                                                config['lp_iters'],
                                                config['lp_fwd_iters'],
                                                config['lp_cfactor'],
                                                config['lp_fwd_finerelax'],
                                                config['lp_fwd_relax'],
                                                config['lp_fwd_tol'],
                                                config['lp_fwd_tol'],
                                                not config['lp_use_downcycle']))
  # ATTENTION: Modified a ParallelNet for RNN
  model = ParallelNet(input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      num_classes=num_classes,
                      num_steps=num_steps,
                      max_levels=config['lp_levels'],
                      max_iters=config['lp_iters'],
                      fwd_max_iters=config['lp_fwd_iters'],
                      print_level=config['lp_print'],
                      cfactor=config['lp_cfactor'],
                      skip_downcycle=not config['lp_use_downcycle'],
                      fmg=False,
                      seed=seed,
                      Tf=config['tf'])

  # set forward relaxation
  model.setFwdAbsTol(config['lp_fwd_tol'])
  model.setBwdAbsTol(config['lp_bwd_tol'])
  model.setFwdNumRelax(config['lp_fwd_relax'])
  model.setFwdNumRelax(config['lp_fwd_finerelax'],level=0) # specialize on fine relaxation

  return model


def main():
  comm  = MPI.COMM_WORLD
  rank  = comm.Get_rank()
  procs = comm.Get_size()

  root_print(rank,'TORCHBRAID REV: %s' % torchbraid.utils.git_rev())
  root_print(rank,'DRIVER REV:     %s' % git_rev())
  root_print(rank,'')

  # Training settings
  parser = argparse.ArgumentParser(description='TORCHBRAID UCI-HAR Example')
  parser.add_argument('--seed', type=int, default=783214419, metavar='S',
                      help='random seed (default: 783214419)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--percent-data', type=float, default=1.0, metavar='N',
                      help='how much of the data to read in and use for training/testing')
  parser.add_argument('--ensemble-size', type=int, default=1, metavar='N',
                      help='how many samples to use')

  # artichtectural settings
  parser.add_argument('--sequence-length', type=int, default=128, metavar='N',
                      help='number of time steps in each sequence (default: 128)')
  parser.add_argument('--tf',type=float,default=128.0,
                      help='Final time (default=128.0, to match sequence length)')
  parser.add_argument('--input-size', type=int, default=9, metavar='N',
                      help='size of input at each time step (default: 9)')
  parser.add_argument('--hidden-size', type=int, default=100, metavar='N',
                      help='number of units in each hidden layer (default: 100)')
  parser.add_argument('--num-layers', type=int, default=2, metavar='N',
                      help='number of hidden layers (default: 2)')
  parser.add_argument('--num-classes', type=int, default=6, metavar='N',
                      help='number of classes (default: 6)')
  parser.add_argument('--implicit-serial', action='store_true', default=False,
                      help='Implicit Serial (default: False)')

  # algorithmic settings (gradient descent and batching)
  parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                      help='input batch size for training (default: 100)')
  parser.add_argument('--batch-bug', action='store_true', default=False,
                      help='Work around for the batching bug...temporary (default: False)')
  parser.add_argument('--epochs', type=int, default=7, metavar='N',
                      help='number of epochs to train (default: 7)')
  parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                      help='learning rate (default: 0.001)')
  parser.add_argument('--use-sgd', action='store_true', default=False,
                      help='Use SGD (default: False)')


  # algorithmic settings (parallel or serial)
  parser.add_argument('--force-lp', action='store_true', default=False,
                      help='Use layer parallel even if there is only 1 MPI rank (default off)')
  parser.add_argument('--lp-levels', type=int, default=1, metavar='N',
                      help='Layer parallel levels (default: 3)')
  parser.add_argument('--lp-iters', type=int, default=1, metavar='N',
                      help='Layer parallel iterations (default: 1)')
  parser.add_argument('--lp-fwd-iters', type=int, default=4, metavar='N',
                      help='Layer parallel (forward) iterations (default: 4, value of -1 implies uses --lp-iters)')
  parser.add_argument('--lp-print', type=int, default=0, metavar='N',
                      help='Layer parallel print level (default: 0, 3 means a lot)')
  parser.add_argument('--lp-cfactor', type=int, default=4, metavar='N',
                      help='Layer parallel coarsening factor (default: 4)')
  parser.add_argument('--lp-use-downcycle',action='store_true', default=False, 
                      help='Layer parallel use downcycle on or off (default: False)')
  parser.add_argument('--lp-fwd-finerelax', type=int, default=1, metavar='N',
                      help='Forward fine relaxation (default: 1, F-relax)')
  parser.add_argument('--lp-fwd-relax', type=int, default=3, metavar='N',
                      help='Forward relaxation (default: 3, FCF-relax)')
  parser.add_argument('--lp-fwd-tol', type=float, default=1e-16, metavar='N',
                      help='Forward parallel-in-time absolute tolerance (default: 1e-16, rely on iteration counts)')
  parser.add_argument('--lp-bwd-tol', type=float, default=1e-16, metavar='N',
                      help='Backward parallel-in-time absolute tolerance (default: 1e-16, rely on iteration counts)')

  args = parser.parse_args()

  # ATTENTION: Added a new variable num_steps (sequence_len should be defined before here)
  # In order to evenly distribute sub-sequences across the given processors,
  # the number of processors (procs) must be one of the factors of sequence length (e.g., when sequence length = 28 (MNIST), the feasible number of processors are 1, 2, 4, 7, 14 and 28)
  if args.sequence_length % procs != 0:
    root_print(rank,'The number of processors must be a factor of sequence length')
    sys.exit(0)

  root_print(rank,'MPI: procs = %d' % procs)
  root_print(rank,'INPUT: {}'.format(args))
  root_print(rank,'')

  if args.ensemble_size==1:
    torch.manual_seed(args.seed)

    run_training(args)
  else:
    root_print(rank,'ENSEMBLE RUNS BEGUN')
    root_print(rank,'====================================================')

    stdout = sys.stdout
    f_out = open('ensemble.log','w') 
    for samp in range(args.ensemble_size):
      seed = args.seed
      if rank==0:
        seed = torch.seed()
      args.seed = comm.bcast(seed,root=0)

      root_print(rank,'  ENSEMBLE %d' % (samp+1))
      root_print(rank,'  -------------------------------------') 

      sys.stdout.flush()
      sys.stdout = f_out

      parallel_accur,serial_accur, epoch_time = run_training(args)

      sys.stdout.flush()
      sys.stdout = stdout

      if rank==0:
        print('  AVG EPOCH TIME: {:0.6f}s'.format(stats.mean(epoch_time)))
        print('  ACCURACY:       parallel = {:.0f}%,   serial = {:.0f}%\n'.format(parallel_accur[-1],serial_accur[-1]))
    # end for samp

    root_print(rank,'====================================================')
    root_print(rank,'ENSEMBLE RUNS Completed')

def run_training(args):

  comm  = MPI.COMM_WORLD
  rank  = comm.Get_rank()
  procs = comm.Get_size()

  root_print(rank,'USING SEED: {}'.format(args.seed))
  root_print(rank,'')

  # some logic to default to Serial if on one processor,
  # can be overriden by the user to run layer-parallel
  if args.force_lp:
    force_lp = True
  elif procs>1:
    force_lp = True
  else:
    force_lp = False

  num_steps = int(args.sequence_length / procs) # number of time steps (over sequence) for each process

  root_print(rank,'Loading UCI HAR Dataset:')

  # Load UCI HAR train dataset
  ###########################################

  #  Using a script for loading UCI HAR
  dir_path     = os.path.dirname(os.path.realpath(__file__))
  uci_har_path = dir_path+'/UCI HAR Dataset'

  x_train_data, y_train_data = load_data(train=True,path=uci_har_path)
  if args.batch_bug:
    x_train_data = x_train_data[:7000]
    y_train_data = y_train_data[:7000]
  if args.percent_data < 1.0:
    x_train_data = x_train_data[::int(1./args.percent_data)]
    y_train_data = y_train_data[::int(1./args.percent_data)]

  train_set = TensorDataset(x_train_data,y_train_data)

  # build loaders
  train_loader_parallel = ParallelRNNDataLoader(comm,dataset=train_set,batch_size=args.batch_size,shuffle=True)
  train_loader_serial = train_loader_parallel.getSerialDataLoader()

  # Load UCI HAR test dataset
  ###########################################
  x_test_data, y_test_data = load_data(train=False,path=uci_har_path)
  if args.batch_bug:
    x_test_data = x_test_data[:2900]
    y_test_data = y_test_data[:2900]
  if args.percent_data < 1.0:
    x_test_data = x_test_data[::int(1./args.percent_data)]
    y_test_data = y_test_data[::int(1./args.percent_data)]

  test_set = TensorDataset(x_test_data,y_test_data)

  # build loaders
  test_loader_parallel = ParallelRNNDataLoader(comm,dataset=test_set,batch_size=args.batch_size,shuffle=False)
  test_loader_serial   = test_loader_parallel.getSerialDataLoader()

  ############################################

  sequence_length = args.sequence_length

  root_print(rank,'')

  if force_lp:
    assert(not args.implicit_serial)

    train_loader = train_loader_parallel
    test_loader  = test_loader_parallel

    if args.lp_levels==-1:
      min_coarse_size = 3
      args.lp_levels = compute_levels(args.sequence_length,min_coarse_size,args.lp_cfactor)
   
    config = vars(args)
    model = buildParallelNet(config,
                             rank,
                             input_size=args.input_size,
                             hidden_size=args.hidden_size,
                             num_layers=args.num_layers,
                             num_classes=args.num_classes,
                             num_steps=num_steps,
                             seed=args.seed)

    compose = model.compose
    serial_compose = lambda op,*p,**k: op(*p,**k)
  elif args.implicit_serial:

    root_print(rank,'Using Implicit SerialNet:')

    train_loader = train_loader_serial
    test_loader  = test_loader_serial

    model = ImplicitSerialNet(input_size=args.input_size,
                              hidden_size=args.hidden_size,
                              num_layers=args.num_layers,
                              num_classes=args.num_classes,
                              seed=args.seed)

    # dummy compose operator
    compose = lambda op,*p,**k: op(*p,**k)
  else:
    root_print(rank,'Using SerialNet:')
                                                  
    train_loader = train_loader_serial
    test_loader  = test_loader_serial

    model = SerialNet(input_size=args.input_size,
                      hidden_size=args.hidden_size,
                      num_layers=args.num_layers,
                      num_classes=args.num_classes,
                      num_steps=num_steps,
                      seed=args.seed)

    # dummy compose operator
    compose = lambda op,*p,**k: op(*p,**k)

  if not args.use_sgd:
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
  else:
    optimizer = optim.SGD(model.parameters(),lr=args.lr)

  epoch_times = []
  test_times = []

  parallel_test = []
  serial_test = []
  for epoch in range(1, args.epochs + 1):
    # train
    start_time = timer()
    train(rank,args.log_interval,model,train_loader,optimizer,epoch,compose)
    end_time = timer()
    epoch_times += [end_time-start_time]

    # test
    start_time = timer()
    parallel_test += [test(rank,model,test_loader,epoch,compose,prefix='\nPARALLEL: ')]
    end_time = timer()
    test_times += [end_time-start_time]

    if (force_lp and not args.implicit_serial) and rank==0:
      serial_test += [test(rank,model.getSerialModel(),test_loader_serial,epoch,serial_compose,prefix='SERIAL:   ')]
    else:
      serial_test += [parallel_test[-1]]
    comm.barrier()

    # end an empty line
    root_print(rank,'')
  # end for epoch

  if force_lp:
    timers = model.parallel_rnn.getTimersString()
    root_print(rank,timers)

  root_print(rank,'TIME PER EPOCH: %.2e' % stats.mean(epoch_times))
  root_print(rank,'TIME PER TEST:  %.2e' % stats.mean(test_times))

  return parallel_test, serial_test, epoch_times

if __name__ == '__main__':
  main()
# end main 
