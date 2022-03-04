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

# some helpful examples
# 
# BATCH_SIZE=50
# STEPS=12
# CHANNELS=8

# IN SERIAL
# python  main.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 # 2>&1 | tee serial.out
# mpirun -n 4 python  main.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 # 2>&1 | tee serial.out

from __future__ import print_function
import sys
import argparse
import torch
import torchbraid
import torchbraid.utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import statistics as stats

import numpy as np
import matplotlib.pyplot as pyplot

from torchvision import datasets, transforms

from timeit import default_timer as timer

from mpi4py import MPI

def root_print(rank,s):
  if rank==0:
    print(s)

class OpenConvLayer(nn.Module):
  def __init__(self,channels):
    super(OpenConvLayer, self).__init__()
    ker_width = 3
    self.conv = nn.Conv2d(1,channels,ker_width,padding=1)

  def forward(self, x):
    return F.relu(self.conv(x))
# end layer

class OpenFlatLayer(nn.Module):
  def __init__(self,channels):
    super(OpenFlatLayer, self).__init__()
    self.channels = channels

  def forward(self, x):
    # this bit of python magic simply replicates each image in the batch
    s = len(x.shape)*[1]
    s[1] = self.channels
    x = x.repeat(s)
    return x
# end layer

class CloseLayer(nn.Module):
  def __init__(self,channels):
    super(CloseLayer, self).__init__()
    self.fc1 = nn.Linear(channels*28*28, 32)
    self.fc2 = nn.Linear(32, 10)

  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)
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
  def __init__(self,channels=12,local_steps=8,Tf=1.0,serial_nn=None,open_nn=None,close_nn=None):
    super(SerialNet, self).__init__()

    
    if open_nn is None:
      self.open_nn = OpenFlatLayer(channels)
    else:
      self.open_nn = open_nn

    if serial_nn is None:
      step_layer = lambda: StepLayer(channels)
      numprocs = 1
      parallel_nn = torchbraid.LayerParallel(MPI.COMM_SELF,step_layer,numprocs*local_steps,Tf,max_levels=1,max_iters=1)
      parallel_nn.setPrintLevel(0)
    
      self.serial_nn   = parallel_nn.buildSequentialOnRoot()
    else:
      self.serial_nn = serial_nn

    if close_nn is None:
      self.close_nn = CloseLayer(channels)
    else:
      self.close_nn = close_nn

  def forward(self, x):
    x = self.open_nn(x)
    x = self.serial_nn(x)
    x = self.close_nn(x)
    return x
# end SerialNet 

class ParallelNet(nn.Module):
  def __init__(self,channels=12,local_steps=8,Tf=1.0,max_levels=1,max_iters=1,fwd_max_iters=0,print_level=0,braid_print_level=0,cfactor=4,fine_fcf=False,skip_downcycle=True,fmg=False):
    super(ParallelNet, self).__init__()

    step_layer = lambda: StepLayer(channels)

    numprocs = MPI.COMM_WORLD.Get_size()

    self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD,step_layer,local_steps*numprocs,Tf,max_levels=max_levels,max_iters=max_iters)
    if fwd_max_iters>0:
      print('fwd_amx_iters',fwd_max_iters)
      self.parallel_nn.setFwdMaxIters(fwd_max_iters)
    self.parallel_nn.setPrintLevel(print_level,True)
    self.parallel_nn.setPrintLevel(braid_print_level,False)
    self.parallel_nn.setCFactor(cfactor)
    self.parallel_nn.setSkipDowncycle(skip_downcycle)

    if fmg:
      self.parallel_nn.setFMG()
    self.parallel_nn.setNumRelax(1)         # FCF elsewehre
    if not fine_fcf:
      self.parallel_nn.setNumRelax(0,level=0) # F-Relaxation on the fine grid
    else:
      self.parallel_nn.setNumRelax(1,level=0) # F-Relaxation on the fine grid

    # this object ensures that only the LayerParallel code runs on ranks!=0
    compose = self.compose = self.parallel_nn.comp_op()
    
    # by passing this through 'compose' (mean composition: e.g. OpenFlatLayer o channels) 
    # on processors not equal to 0, these will be None (there are no parameters to train there)
    self.open_nn = compose(OpenFlatLayer,channels)
    self.close_nn = compose(CloseLayer,channels)

  def saveSerialNet(self,name):
    serial_nn = self.parallel_nn.buildSequentialOnRoot()
    if MPI.COMM_WORLD.Get_rank()==0:
      s_net = SerialNet(-1,-1,-1,serial_nn=serial_nn,open_nn=self.open_nn,close_nn=self.close_nn)
      s_net.eval()
      torch.save(s_net,name)

  def getDiagnostics(self):
    return self.parallel_nn.getDiagnostics()
 
  def forward(self, x):
    # by passing this through 'o' (mean composition: e.g. self.open_nn o x) 
    # this makes sure this is run on only processor 0

    x = self.compose(self.open_nn,x)
    x = self.parallel_nn(x)
    x = self.compose(self.close_nn,x)

    return x
# end ParallelNet 

def train(rank, args, model, train_loader, optimizer, epoch,compose):
  model.train()
  criterion = nn.CrossEntropyLoss()
  total_time = 0.0
  for batch_idx, (data, target) in enumerate(train_loader):
    start_time = timer()
    optimizer.zero_grad()
    output = model(data)
    loss = compose(criterion,output,target)
    loss.backward()
    stop_time = timer()
    optimizer.step()

    total_time += stop_time-start_time
    if batch_idx % args.log_interval == 0:
      root_print(rank,'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item(),total_time/(batch_idx+1.0)))

  root_print(rank,'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
    epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
    100. * (batch_idx+1) / len(train_loader), loss.item(),total_time/(batch_idx+1.0)))

def diagnose(rank, model, test_loader,epoch):
  model.parallel_nn.diagnostics(True)
  model.eval()
  test_loss = 0
  correct = 0
  criterion = nn.CrossEntropyLoss()

  itr = iter(test_loader)
  data,target = next(itr)

  # compute the model and print out the diagnostic information 
  with torch.no_grad():
    output = model(data)

  diagnostic = model.getDiagnostics()

  if rank!=0:
    return

  features = np.array([diagnostic['step_in'][0]]+diagnostic['step_out'])
  params = np.array(diagnostic['params'])

  fig,axs = pyplot.subplots(2,1)
  axs[0].plot(range(len(features)),features)
  axs[0].set_ylabel('Feature Norm')

  coords = [0.5+i for i in range(len(features)-1)]
  axs[1].set_xlim([0,len(features)-1])
  axs[1].plot(coords,params,'*')
  axs[1].set_ylabel('Parameter Norms: {}/tstep'.format(params.shape[1]))
  axs[1].set_xlabel('Time Step')

  fig.suptitle('Values in Epoch {}'.format(epoch))

  #pyplot.show()
  pyplot.savefig('diagnose{:03d}.png'.format(epoch))


def test(rank, model, test_loader,compose):
  model.eval()
  test_loss = 0
  correct = 0
  criterion = nn.CrossEntropyLoss()
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data, target
      output = model(data)
      test_loss += compose(criterion,output,target).item()
       
      output = MPI.COMM_WORLD.bcast(output,root=0)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  root_print(rank,'\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

def compute_levels(num_steps,min_coarse_size,cfactor): 
  from math import log, floor 
  
  # we want to find $L$ such that ( max_L min_coarse_size*cfactor**L <= num_steps)
  levels =  floor(log(float(num_steps)/min_coarse_size,cfactor))+1 

  if levels<1:
    levels = 1
  return levels
# end compute levels

def main():
  # Training settings
  parser = argparse.ArgumentParser(description='TORCHBRAID CIFAR10 Example')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 783253419)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--percent-data', type=float, default=1.0, metavar='N',
                      help='how much of the data to read in and use for training/testing')

  # artichtectural settings
  parser.add_argument('--steps', type=int, default=4, metavar='N',
                      help='Number of times steps in the resnet layer (default: 4)')
  parser.add_argument('--channels', type=int, default=4, metavar='N',
                      help='Number of channels in resnet layer (default: 4)')
  parser.add_argument('--digits',action='store_true', default=False, 
                      help='Train with the MNIST digit recognition problem (default: False)')
  parser.add_argument('--serial-file',type=str,default=None,
                      help='Load the serial problem from file')
  parser.add_argument('--tf',type=float,default=1.0,
                      help='Final time')

  # algorithmic settings (gradient descent and batching
  parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                      help='input batch size for training (default: 50)')
  parser.add_argument('--epochs', type=int, default=2, metavar='N',
                      help='number of epochs to train (default: 2)')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')

  # algorithmic settings (parallel or serial)
  parser.add_argument('--force-lp', action='store_true', default=False,
                      help='Use layer parallel even if there is only 1 MPI rank')
  parser.add_argument('--lp-levels', type=int, default=3, metavar='N',
                      help='Layer parallel levels (default: 3)')
  parser.add_argument('--lp-iters', type=int, default=2, metavar='N',
                      help='Layer parallel iterations (default: 2)')
  parser.add_argument('--lp-fwd-iters', type=int, default=-1, metavar='N',
                      help='Layer parallel (forward) iterations (default: -1, default --lp-iters)')
  parser.add_argument('--lp-print', type=int, default=0, metavar='N',
                      help='Layer parallel internal print level (default: 0)')
  parser.add_argument('--lp-braid-print', type=int, default=0, metavar='N',
                      help='Layer parallel braid print level (default: 0)')
  parser.add_argument('--lp-cfactor', type=int, default=4, metavar='N',
                      help='Layer parallel coarsening factor (default: 4)')
  parser.add_argument('--lp-finefcf',action='store_true', default=False, 
                      help='Layer parallel fine FCF on or off (default: False)')
  parser.add_argument('--lp-use-downcycle',action='store_true', default=False, 
                      help='Layer parallel use downcycle on or off (default: False)')
  parser.add_argument('--lp-use-fmg',action='store_true', default=False, 
                      help='Layer parallel use FMG for one cycle (default: False)')

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

  #torch.manual_seed(torchbraid.utils.seed_from_rank(args.seed,rank))
  torch.manual_seed(args.seed)

  if args.lp_levels==-1:
    min_coarse_size = 3
    args.lp_levels = compute_levels(args.steps,min_coarse_size,args.lp_cfactor)

  local_steps = int(args.steps/procs)
  if args.steps % procs!=0:
    root_print(rank,'Steps must be an even multiple of the number of processors: %d %d' % (args.steps,procs) )
    sys.exit(0)

  root_print(rank,'MNIST ODENet:')

  # read in Digits MNIST or Fashion MNIST
  if args.digits:
    root_print(rank,'-- Using Digit MNIST')
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                   ])
    dataset = datasets.MNIST('./data', download=False,transform=transform)
  else:
    root_print(rank,'-- Using Fashion MNIST')
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST('./fashion-data', download=False,transform=transform)
  # if args.digits

  root_print(rank,'-- procs    = {}\n'
                  '-- channels = {}\n'
                  '-- tf       = {}\n'
                  '-- steps    = {}'.format(procs,args.channels,args.tf,args.steps))

  train_size = int(50000 * args.percent_data)
  test_size  = int(10000 * args.percent_data)
  train_set = torch.utils.data.Subset(dataset,range(train_size))
  test_set  = torch.utils.data.Subset(dataset,range(train_size,train_size+test_size))
  train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size,shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False)

  root_print(rank,'')

  if force_lp :
    root_print(rank,'Using ParallelNet:')
    root_print(rank,'-- max_levels = {}\n'
                    '-- max_iters  = {}\n'
                    '-- fwd_iters  = {}\n'
                    '-- cfactor    = {}\n'
                    '-- fine fcf   = {}\n'
                    '-- skip down  = {}\n'
                    '-- fmg        = {}\n'.format(args.lp_levels,
                                                  args.lp_iters,
                                                  args.lp_fwd_iters,
                                                  args.lp_cfactor,
                                                  args.lp_finefcf,
                                                  not args.lp_use_downcycle,
                                                  args.lp_use_fmg))
    model = ParallelNet(channels=args.channels,
                        local_steps=local_steps,
                        max_levels=args.lp_levels,
                        max_iters=args.lp_iters,
                        fwd_max_iters=args.lp_fwd_iters,
                        print_level=args.lp_print,
                        braid_print_level=args.lp_braid_print,
                        cfactor=args.lp_cfactor,
                        fine_fcf=args.lp_finefcf,
                        skip_downcycle=not args.lp_use_downcycle,
                        fmg=args.lp_use_fmg,Tf=args.tf)


    if args.serial_file is not None:
      model.saveSerialNet(args.serial_file)
    compose = model.compose
  else:
    root_print(rank,'Using SerialNet:')
    root_print(rank,'-- serial file = {}\n'.format(args.serial_file))
    if args.serial_file is not None:
      print('loading model')
      model = torch.load(args.serial_file)
    else:
      model = SerialNet(channels=args.channels,local_steps=local_steps,Tf=args.tf)
    compose = lambda op,*p: op(*p)

  optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)

  epoch_times = []
  test_times = []

  # check out the initial conditions
  #if force_lp:
    #diagnose(rank, model, test_loader,0)

  for epoch in range(1, args.epochs + 1):
    start_time = timer()
    train(rank,args, model, train_loader, optimizer, epoch,compose)
    end_time = timer()
    epoch_times += [end_time-start_time]

    start_time = timer()
    test(rank,model, test_loader,compose)
    end_time = timer()
    test_times += [end_time-start_time]

    # print out some diagnostics
    #if force_lp:
    #  diagnose(rank, model, test_loader,epoch)

  #if force_lp:
  #  timer_str = model.parallel_nn.getTimersString()
  #  root_print(rank,timer_str)

  root_print(rank,'TIME PER EPOCH: %.2e (1 std dev %.2e)' % (stats.mean(epoch_times),stats.stdev(epoch_times)))
  root_print(rank,'TIME PER TEST:  %.2e (1 std dev %.2e)' % (stats.mean(test_times), stats.stdev(test_times)))

if __name__ == '__main__':
  main()
