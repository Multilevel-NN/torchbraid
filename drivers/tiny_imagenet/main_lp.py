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

###
#
# ----- Example Script -----

# export FWD_ITER=3
# export BWD_ITER=2
# export DIFF=0.0001
# export ACT=relu
# 
# mpirun -n 4 python ./main_lp.py --steps 32 --lp-fwd-cfactor 4 --lp-bwd-cfactor 4 --epochs=8 --seed 2069923971 --lp-fwd-iters ${FWD_ITER} --lp-fwd-levels -1 --lp-bwd-levels -1 --lp-iters ${BWD_ITER} --batch-size 100 --tf 5.0 --log-interval 5 --diff-scale ${DIFF} --activation ${ACT} --samp-ratio 0.1 --channels 8
#
# ----- Output -----
# Using Tiny ImageNet...
# 
# Namespace(seed=2069923971, log_interval=5, steps=32, channels=8, tf=5.0, diff_scale=0.0001, activation='relu', batch_size=100, epochs=8, samp_ratio=0.1, lr=0.01, lp_fwd_levels=2, lp_bwd_levels=2, lp_iters=2, lp_fwd_iters=2, lp_print=0, lp_braid_print=0, lp_fwd_cfactor=4, lp_bwd_cfactor=4, lp_fwd_nrelax_coarse=1, lp_bwd_nrelax_coarse=1, lp_fwd_finefcf=False, lp_bwd_finefcf=False, lp_fwd_finalrelax=False, lp_use_downcycle=False, lp_use_fmg=False, lp_bwd_relaxonlycg=0, lp_fwd_relaxonlycg=0, lp_use_crelax_wt=1.0)
# 
# Training setup:  Batch size:  100  Sample ratio:  0.1  Epochs:  8
# Train Epoch:  1 [   100/  8800]	Loss: 5.374e+00	Time Per Batch 1.002954/1.048842 - F 2/6.47e+03, B 2/2.76e-05
# Train Epoch:  1 [   600/  8800]	Loss: 3.100e+00	Time Per Batch 0.984198/1.075165 - F 2/1.43e+03, B 2/1.18e-05
# Train Epoch:  1 [  1100/  8800]	Loss: 2.920e+00	Time Per Batch 0.995636/1.064048 - F 2/1.25e+03, B 2/8.23e-06
# Train Epoch:  1 [  1600/  8800]	Loss: 3.079e+00	Time Per Batch 0.997011/1.057520 - F 2/1.35e+03, B 2/1.17e-05

from __future__ import print_function
import numpy as np

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import statistics as stats

from torchvision import datasets, transforms
from timeit import default_timer as timer

from utils import parse_args, buildNet, ParallelNet, getComm, git_rev, getDevice


def root_print(rank, s):
  if rank == 0:
    print(s)
    sys.stdout.flush()


def train(rank, args, model, train_loader, optimizer, epoch, compose, device):
  log_interval = args.log_interval
  torch.enable_grad()

  model.train()
  criterion = nn.CrossEntropyLoss()
  total_time = 0.0
  total_time_fp = 0.0
  total_time_bp = 0.0
  total_time_cm = 0.0

  switched = False
  format_str = 'Train Epoch: {:2d} [{:6d}/{:6d}]\tLoss: {:.3e}\tTime Per Batch {:.6f}/{:.6f} - F{:2d}/{:.2e}, B{:2d}/{:.2e}'
  total_data = 0

  cumulative_start_time = timer()
  for batch_idx, (data, target) in enumerate(train_loader):
    data = data.to(device)
    target = target.long()
    target = target.to(device)

    start_time = timer()

    # compute forward
    optimizer.zero_grad()
    start_time_fp = timer()
    output = model(data)
    total_time_fp += timer() - start_time_fp

    fwd_itr, fwd_res = model.getFwdStats()

    # compute loss
    start_time_cm = timer()
    if len(output.shape) == 1:
      output = torch.unsqueeze(output, 0)
    loss = compose(criterion, output, target)
    total_time_cm += timer() - start_time_cm

    # compute gradient
    start_time_bp = timer()
    loss.backward()
    total_time_bp += timer() - start_time_bp

    bwd_itr, bwd_res = model.getBwdStats()

    # take step
    stop_time = timer()
    optimizer.step()

    total_data += len(data)
    total_time += stop_time - start_time

    cumulative_stop_time = timer()
    cumulative_time = (cumulative_stop_time - cumulative_start_time)

    if batch_idx % log_interval == 0:
      root_print(rank, format_str.format(
        epoch,
        total_data,
        len(train_loader.dataset),
        loss.item(),
        total_time / (batch_idx + 1.0),
        cumulative_time / (batch_idx + 1.0),
        fwd_itr, fwd_res,
        bwd_itr, bwd_res
      ))

    # on GPU's let the first batch be a warm up
    if batch_idx==0:
      total_time = 0.0
      cumulative_time = 0.0

  root_print(rank, (format_str + ', fp={:.6f}, cm={:.6f}, bp={:.6f}').format(
    epoch,
    len(train_loader.dataset),
    len(train_loader.dataset),
    loss.item(),
    total_time / (batch_idx + 1.0),
    cumulative_time / (batch_idx + 1.0),
    fwd_itr, fwd_res,
    bwd_itr, bwd_res,
    total_time_fp / len(train_loader.dataset),
    total_time_cm / len(train_loader.dataset),
    total_time_bp / len(train_loader.dataset),
  ))


def test(rank, args, model, test_loader, epoch, compose, device, prefix=''):
  log_interval = args.log_interval
  model.eval()
  correct = 0
  criterion = nn.CrossEntropyLoss()
  total_time = 0.0
  root_print(rank, f'EPOCH={epoch} TEST')

  total_data = 0
  with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
      start_time = timer()
      data = data.to(device)
      target = target.long()
      target = target.to(device)

      # evaluate inference
      output = model(data)

      if len(output.shape) == 1:
        output = torch.unsqueeze(output, 0)

      # compute the number of solutions
      pred = compose(torch.argmax, output, dim=1, keepdim=True)
      stop_time = timer()

      total_time += stop_time - start_time
      total_data += len(data)

      # only accumulate the correct values on the root
      if rank == 0:
        correct += pred.eq(target.view_as(pred)).sum().item()

      if batch_idx % log_interval == 0:
        format_str = 'Test Epoch: {:2d} [{:6d}/{:6d}]'
        root_print(rank, format_str.format(
          epoch,
          total_data,
          len(test_loader.dataset)))

  root_print(rank, '{}Test set epoch {:2d}: Accuracy: {}/{} ({:.0f}%)\tTime Per Batch {:.6f}'.format(
    prefix,
    epoch,
    correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset),
    total_time / len(test_loader.dataset)))

  return 100. * correct / len(test_loader.dataset)


class FakeIter:
  def __init__(self, cnt, elmt):
    self.cnt = cnt
    self.elmt = elmt

  def __iter__(self):
    return self

  def __next__(self):
    if self.cnt > 0:
      self.cnt -= 1
      return self.elmt
    else:
      raise StopIteration


class FakeLoader:
  def __init__(self, cnt, device):
    self.cnt = cnt

    elmt = torch.tensor((), device=device)
    self.elmt = (elmt, elmt)

    self.dataset = cnt * [None]

  def __iter__(self):
    return FakeIter(self.cnt, self.elmt)


def parallel_loader(rank, loader, device):
  if rank == 0:
    return loader
  batches = len(loader)
  return FakeLoader(batches, device)


def main():
  ##
  # Parse command line args (function defined above)
  args = parse_args(mgopt_on=False)
  comm = getComm()
  procs = comm.Get_size()
  rank = comm.Get_rank()

  root_print(rank, 'TORCHBRAID REV: %s' % git_rev())

  my_device, my_host = getDevice(comm)

  global_steps = args.steps

  ##
  # Load Tiny ImageNet
  if rank == 0:
    print('Using Tiny ImageNet...\n')
    print(args)
  ##

  # Load datasets
  traindir = './tiny-imagenet-200/new_train'
  valdir = './tiny-imagenet-200/new_test'
  #normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
  #                                 std=[0.2023, 0.1994, 0.2010])
  normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5])
  train_dataset = datasets.ImageFolder(traindir,
                                       transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.RandomCrop(224,padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize, transforms.RandomErasing(0.5)]))
  test_dataset = datasets.ImageFolder(valdir, transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize]))

  # Trim datasets 
  train_size = int(88000 * args.samp_ratio)
  test_size = int(22000 * args.samp_ratio)
  #
  train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
  test_dataset = torch.utils.data.Subset(test_dataset, range(test_size))

  # Create data loaders
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size, shuffle=True,
                                             pin_memory=True)
  train_loader = parallel_loader(rank, train_loader, my_device)
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=args.batch_size, shuffle=False,
                                            pin_memory=True)
  test_loader = parallel_loader(rank, test_loader, my_device)
  if rank == 0:
    print("\nTraining setup:  Batch size:  " + str(args.batch_size) + "  Sample ratio:  " + str(
      args.samp_ratio) + "  Epochs:  " + str(args.epochs))

  ##
  # Define ParNet parameters for each nested iteration level, starting from fine to coarse
  network = {'channels': args.channels,
             'global_steps': global_steps,
             'max_iters': args.lp_iters,
             'print_level': args.lp_print,
             'Tf': args.tf,
             'max_fwd_levels': args.lp_fwd_levels,
             'max_bwd_levels': args.lp_bwd_levels,
             'max_fwd_iters': args.lp_fwd_iters,
             'print_level': args.lp_print,
             'braid_print_level': args.lp_braid_print,
             'fwd_cfactor': args.lp_fwd_cfactor,
             'bwd_cfactor': args.lp_bwd_cfactor,
             'fine_fwd_fcf': args.lp_fwd_finefcf,
             'fine_bwd_fcf': args.lp_bwd_finefcf,
             'fwd_nrelax': args.lp_fwd_nrelax_coarse,
             'bwd_nrelax': args.lp_bwd_nrelax_coarse,
             'skip_downcycle': not args.lp_use_downcycle,
             'fmg': args.lp_use_fmg,
             'fwd_relax_only_cg': args.lp_fwd_relaxonlycg,
             'bwd_relax_only_cg': args.lp_bwd_relaxonlycg,
             'CWt': args.lp_use_crelax_wt,
             'fwd_finalrelax': args.lp_fwd_finalrelax,
             'diff_scale': args.diff_scale,
             'activation': args.activation
             }

  ##
  # Initialize MG/Opt solver with nested iteration 
  epochs = args.epochs
  log_interval = args.log_interval

  model = buildNet(not args.use_serial, **network)

  # Activate Braid timers and specify output files
  if not args.use_serial:
    model.parallel_nn.fwd_app.setBraidTimers(flag=1)
    model.parallel_nn.bwd_app.setBraidTimers(flag=1)
    model.parallel_nn.fwd_app.setTimerFile(
      f'b_fwd_s_{args.steps}_c_{args.channels}_bs_{args.batch_size}_p_{procs}')
    model.parallel_nn.bwd_app.setTimerFile(
      f'b_bwd_s_{args.steps}_c_{args.channels}_bs_{args.batch_size}_p_{procs}')

  if args.opt == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9 )#, weight_decay=0.0001)
  else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # , weight_decay=0.0001)
  compose = model.compose

  epoch_times = []
  test_times = []

  scheduler = None

  if args.lr_scheduler:
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.1, verbose=(rank == 0))
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

  if args.load_model:
    root_print(rank, f'Loading from \"{args.model_dir}\"')
    model.loadParams(rank, args.model_dir)
    optimizer.load_state_dict(torch.load(f'{args.model_dir}/optimizer.{rank}.mdl'))
    if args.lr_scheduler:
      scheduler.load_state_dict(torch.load(f'{args.model_dir}/scheduler.{rank}.mdl'))

  model = model.to(my_device)

  if rank == 0:
    print('===============MODEL=============\n')
  print(model)

  if rank == 0:
    print('===============OPTIMIZER=============\n')
    print(optimizer)

  if rank == 0:
    print('===============SCHEDULER=============\n')
    print(scheduler)

  # Warm up gpu's (There has to be a cheaper way than calling a complete train call.
  # We just need one small run to "start" the gpu's.
  if str(my_device).startswith('cuda'):
    warm_up_timer = timer()
    train(rank=rank, args=args, model=model, train_loader=train_loader, optimizer=optimizer, epoch=0,
          compose=compose, device=my_device)
    if not args.use_serial:
      model.parallel_nn.timer_manager.resetTimers()
      model.parallel_nn.fwd_app.resetBraidTimer()
      model.parallel_nn.bwd_app.resetBraidTimer()
    torch.cuda.synchronize()
    epoch_times = []
    test_times = []
    root_print(rank, f'Warm up timer {timer() - warm_up_timer}')

  print(f'Run info rank: {rank}: Torch version: {torch.__version__} | Device: {my_device} | Host: {my_host}')

  epoch = 0
  start_time = timer()
  test_result = test(rank, args, model, test_loader, epoch, compose, my_device)
  end_time = timer()
  test_times += [end_time - start_time]

  for epoch in range(1, args.epochs + 1):
    start_time = timer()
    train(rank, args, model, train_loader, optimizer, epoch, compose, my_device)
    end_time = timer()
    epoch_times += [end_time - start_time]

    start_time = timer()
    test_result = test(rank, args, model, test_loader, epoch, compose, my_device)
    end_time = timer()
    test_times += [end_time - start_time]

    if scheduler is not None:
      scheduler.step()

    # output the serial and parallel models
    if args.save_model:
      root_print(rank, f'Saving to \"{args.model_dir}\"')
      torch.save(optimizer.state_dict(), f'{args.model_dir}/optimizer.{rank}.mdl')
      if args.lr_scheduler:
        torch.save(scheduler.state_dict(), f'{args.model_dir}/scheduler.{rank}.mdl')
      model.saveParams(rank, args.model_dir)
    # end args

  if not args.use_serial:
    timer_str = model.parallel_nn.getTimersString()
    root_print(rank, timer_str)

  root_print(rank,
             f'TIME PER EPOCH: {"{:.2f}".format(stats.mean(epoch_times))} '
             f'{("(1 std dev " + "{:.2f})".format(stats.mean(epoch_times))) if len(epoch_times) > 1 else ""}')
  root_print(rank,
             f'TIME PER TEST:  {"{:.2f}".format(stats.mean(test_times))} '
             f'{("(1 std dev " + "{:.2f})".format(stats.mean(test_times))) if len(test_times) > 1 else ""}')


if __name__ == '__main__':
  main()
