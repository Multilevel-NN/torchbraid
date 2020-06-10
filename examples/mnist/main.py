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
# 3. Neither the name of the Corporation nor the names of the
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
#  NOTICE:
#
# For five (5) years from  the United States Government is granted 
# for itself and others acting on its behalf a paid-up, nonexclusive, 
# irrevocable worldwide license in this data to reproduce, prepare derivative 
# work, and perform publicly and display publicly, by or on behalf of the 
# Government. There is provision for the possible extension of the term of
# this license. Subsequent to that period or any extension granted, the 
# United States Government is granted for itself and others acting on its
# behalf a paid-up, nonexclusive, irrevocable worldwide license in this data
# to reproduce, prepare derivative works, distribute copies to the public,
# perform publicly and display publicly, and to permit others to do so. The
# specific term of the license can be identified by inquiry made to National
# Technology and Engineering Solutions of Sandia, LLC or DOE.
#
# NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF 
# ENERGY, NOR NATIONAL TECHNOLOGY AND ENGINEERING SOLUTIONS OF SANDIA, LLC, 
# NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR 
# ASSUMES ANY LEGAL RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR 
# USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, 
# OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
# 
# Any licensee of this software has the obligation and responsibility to 
# abide by the applicable export control laws, regulations, and general 
# prohibitions relating to the export of technical data. Failure to obtain 
# an export control license or other authority from the Government may 
# result in criminal liability under U.S. laws.
#
# Questions? Contact Eric C. Cyr (eccyr@sandia.gov)
# 
# ************************************************************************
#@HEADER

from __future__ import print_function
import sys
import argparse
import torch
import torchbraid
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import statistics as stats

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from timeit import default_timer as timer

from mpi4py import MPI

def root_print(rank,s):
  if rank==0:
    print(s)

class OpenLayer(nn.Module):
  def __init__(self,channels):
    super(OpenLayer, self).__init__()
    ker_width = 3
    self.conv = nn.Conv2d(1,channels,ker_width,padding=1)

  def forward(self, x):
    return F.relu(self.conv(x))
# end layer

class CloseLayer(nn.Module):
  def __init__(self,channels):
    super(CloseLayer, self).__init__()
    ker_width = 3

    self.dropout1 = nn.Dropout2d(0.25)
    self.dropout2 = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(channels*14*14, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output
# end layer

class StepLayer(nn.Module):
  def __init__(self,channels):
    super(StepLayer, self).__init__()
    ker_width = 3
    self.conv1 = nn.Conv2d(channels,channels,ker_width,padding=1)
    self.conv2 = nn.Conv2d(channels,channels,ker_width,padding=1)

  def forward(self, x):
    return F.relu(self.conv2(F.relu(self.conv1(x))))
# end layer

class SerialNet(nn.Module):
  def __init__(self,channels=12,local_steps=8,Tf=1.0):
    super(SerialNet, self).__init__()

    step_layer = lambda: StepLayer(channels)
    
    self.open_nn = OpenLayer(channels)
    self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD,step_layer,local_steps,Tf,max_levels=1,max_iters=1)
    self.parallel_nn.setPrintLevel(0)
    
    self.serial_nn   = self.parallel_nn.buildSequentialOnRoot()
    self.close_nn = CloseLayer(channels)
 
  def forward(self, x):
    x = self.open_nn(x)
    x = self.serial_nn(x)
    x = self.close_nn(x)
    return x
# end SerialNet 

class ParallelNet(nn.Module):
  def __init__(self,channels=12,local_steps=8,Tf=1.0,max_levels=1,max_iters=1,print_level=0):
    super(ParallelNet, self).__init__()

    step_layer = lambda: StepLayer(channels)
    
    self.open_nn = OpenLayer(channels)
    self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD,step_layer,local_steps,Tf,max_levels=max_levels,max_iters=max_iters)
    self.parallel_nn.setPrintLevel(print_level)
    self.parallel_nn.setCFactor(4)
    self.close_nn = CloseLayer(channels)
 
  def forward(self, x):
    x = self.open_nn(x)
    x = self.parallel_nn(x)
    x = self.close_nn(x)
    return x
# end ParallelNet 

def train(rank, args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            root_print(rank,'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(rank, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    root_print(rank,'\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def forward_backward_perf(rank,model,train_loader):
  model.train()
  model.zero_grad()

  data, target = list(train_loader)[0]

  start_fore = timer()
  output = model(data)
  end_fore = timer()

  loss = F.nll_loss(output, target)

  start_back = timer()
  loss.backward()
  end_back = timer()

  root_print(rank,'TIME FORWARD:  %.2e' % (end_fore-start_fore))
  root_print(rank,'TIME BACKWARD: %.2e' % (end_back-start_back))
# end  forward_backward_perf

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    # artichtectural settings
    parser.add_argument('--steps', type=int, default=4, metavar='N',
                        help='Number of times steps in the resnet layer (default: 4)')
    parser.add_argument('--channels', type=int, default=4, metavar='N',
                        help='Number of channels in resnet layer (default: 4)')

    # algorithmic settings (gradient descent and batching
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--train-batches', type=int, default=int(50000/50), metavar='N',
                        help='input batch size for training (default: %d)' % (50000/50))
    parser.add_argument('--test-batches', type=int, default=int(10000/50), metavar='N',
                        help='input batch size for training (default: %d)' % (10000/50))
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    # algorithmic settings (parallel or serial)
    parser.add_argument('--force-lp', action='store_true', default=False,
                        help='Use layer parallel even if there is only 1 MPI rank')
    parser.add_argument('--lp-levels', type=int, default=3, metavar='N',
                        help='Layer parallel levels (default: 3)')
    parser.add_argument('--lp-iters', type=int, default=2, metavar='N',
                        help='Layer parallel iterations (default: 2)')
    parser.add_argument('--lp-print', type=int, default=0, metavar='N',
                        help='Layer parallel print level default: 0)')

    rank  = MPI.COMM_WORLD.Get_rank()
    procs = MPI.COMM_WORLD.Get_size()
    args = parser.parse_args()

    # some logic to default to Serial if on one processor,
    # can be overriden by the user to run layer-parallel
    if args.force_lp:
      force_lp = True
    elif procs>1:
      force_lp = True
    else:
      force_lp = False

    torch.manual_seed(args.seed)

    local_steps = int(args.steps/procs)
    if args.steps % procs!=0:
      root_print(rank,'Steps must be an even multiple of the number of processors: %d %d' % (args.steps,procs) )
      sys.exit(0)

    dataset = datasets.MNIST('./data', download=False,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    train_set = torch.utils.data.Subset(dataset,range(args.batch_size*args.train_batches))
    test_set  = torch.utils.data.Subset(dataset,range(args.batch_size*args.train_batches,args.batch_size*(args.train_batches+args.test_batches)))
 

    kwargs = { }

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size, 
                                               shuffle=True, 
                                               **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_set,
                                               batch_size=args.batch_size, 
                                               shuffle=True, 
                                               **kwargs)

    if force_lp :
      root_print(rank,'Using ParallelNet')
      model = ParallelNet(channels=args.channels,
                          local_steps=local_steps,
                          max_levels=args.lp_levels,
                          max_iters=args.lp_iters,
                          print_level=args.lp_print)
    else:
      root_print(rank,'Using Serial')
      model = SerialNet(channels=args.channels,local_steps=local_steps)

    forward_backward_perf(rank,model,train_loader)

    if force_lp:
      timer_str = model.parallel_nn.getTimersString()
      if rank==0:
        print(timer_str)
    # eend force_lp

    model = None
    return 

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    epoch_times = []
    test_times = []
    for epoch in range(1, args.epochs + 1):
        start_time = timer()
        train(rank,args, model, train_loader, optimizer, epoch)
        end_time = timer()
        epoch_times += [end_time-start_time]

        start_time = timer()
        test(rank,model, test_loader)
        end_time = timer()
        test_times += [end_time-start_time]

        scheduler.step()

    root_print(rank,'TIME PER EPOCH: %.2e (1 std dev %.2e)' % (stats.mean(epoch_times),stats.stdev(epoch_times)))
    root_print(rank,'TIME PER TEST:  %.2e (1 std dev %.2e)' % (stats.mean(test_times), stats.stdev(test_times)))

if __name__ == '__main__':
    main()
