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

import input_pipeline
import preprocessing
from models import PositionalEncoding, PE_Alternative

def root_print(rank,s):
  if rank==0:
    print(s)

class OpenLayer(nn.Module):
  def __init__(self,encoding):
    super(OpenLayer, self).__init__()
    self.encoding = encoding
    self.emb = nn.Embedding(15514, 128)
    self.dropout = nn.Dropout(p=.1)
    self.posenc = PositionalEncoding(128) if encoding == 'Torch'\
      else PE_Alternative(128) if encoding == 'Alternative'\
      else Exception('encoding unknown')

  def forward(self, x):
    # this bit of python magic simply replicates each image in the batch
    x = self.emb(x) 
    x = self.dropout(x)
    x = self.posenc(x)
    return x
# end layer

class CloseLayer(nn.Module):
  def __init__(self):
    super(CloseLayer, self).__init__()
    self.ln3 = nn.LayerNorm(128)
    self.fc3 = nn.Linear(128, 49)

  def forward(self, x):
    x = self.ln3(x)
    x = self.fc3(x)
    return x
# end layer

class StepLayer(nn.Module):
  def __init__(self):
    super(StepLayer, self).__init__()
    self.fc1 = nn.Linear(128, 128)
    self.fc2 = nn.Linear(128, 128)
    self.att = nn.MultiheadAttention(
      embed_dim=128, 
      num_heads=1, 
      dropout=.3, 
      batch_first=True
    )
    self.ln1 = nn.LayerNorm(128)
    self.ln2 = nn.LayerNorm(128)

    self.mask = None

  def forward(self, x):
    # mask = d_extra['mask']

    # ContinuousBlock - dxdtEncoder1DBlock
    x0 = x
    x = self.ln1(x)     # also try to remove layernorm
    x, _ = self.att(x, x, x, self.mask)
    x1 = x
    x = x + x0

    x = self.ln2(x)
    # MLPBlock
    x = self.fc1(x)
    x = nn.ELU()(x)
    x = self.fc2(x)
    
    x = x + x1
    return x
# end layer

class SerialNet(nn.Module):
  def __init__(self,encoding,local_steps=8,Tf=1.0,serial_nn=None,open_nn=None,close_nn=None):
    super(SerialNet, self).__init__()
    
    if open_nn is None:
      self.open_nn = OpenLayer(encoding)
    else:
      self.open_nn = open_nn

    if serial_nn is None:
      step_layer = lambda: StepLayer()
      numprocs = 1
      parallel_nn = torchbraid.LayerParallel(MPI.COMM_SELF,step_layer,numprocs*local_steps,Tf,
        #max_levels=1,
        max_iters=1)
      parallel_nn.setPrintLevel(0)
      
      self.serial_nn = parallel_nn.buildSequentialOnRoot()
    else:
      self.serial_nn = serial_nn

    if close_nn is None:
      self.close_nn = CloseLayer()
    else:
      self.close_nn = close_nn

  def forward(self, x):
    mask = (x == 0)
    x = self.open_nn(x)
    self.serial_nn.mask = mask
    x = self.serial_nn(x)
    x = self.close_nn(x)
    return x
# end SerialNet 

class ParallelNet(nn.Module):
  def __init__(self,encoding,local_steps=8,Tf=1.0,max_levels=1,max_iters=1,fwd_max_iters=0,print_level=0,
    braid_print_level=0,cfactor=4,fine_fcf=False,skip_downcycle=True,fmg=False,relax_only_cg=0):
    super(ParallelNet, self).__init__()

    step_layer = lambda: StepLayer()

    numprocs = MPI.COMM_WORLD.Get_size()

    self.parallel_nn = torchbraid.LayerParallel(
      MPI.COMM_WORLD,
      step_layer,           #[step_layer  for i in range(numprocs)] ?
      local_steps*numprocs, #[local_steps for i in range(numprocs)] ?
      Tf,
      max_fwd_levels=max_levels,
      max_bwd_levels=max_levels,
      max_iters=max_iters)

    if fwd_max_iters>0:
      print('fwd_max_iters',fwd_max_iters)
      self.parallel_nn.setFwdMaxIters(fwd_max_iters)
    self.parallel_nn.setPrintLevel(print_level,True)
    self.parallel_nn.setPrintLevel(braid_print_level,False)
    self.parallel_nn.setCFactor(cfactor)
    self.parallel_nn.setSkipDowncycle(skip_downcycle)
    # self.parallel_nn.setRelaxOnlyCG(relax_only_cg)

    if fmg:
      self.parallel_nn.setFMG()
    self.parallel_nn.setNumRelax(1)         # FCF elsewehre
    if not fine_fcf:
      self.parallel_nn.setNumRelax(0,level=0) # F-Relaxation on the fine grid
    else:
      self.parallel_nn.setNumRelax(1,level=0) # F-Relaxation on the fine grid

    # this object ensures that only the LayerParallel code runs on ranks!=0
    compose = self.compose = self.parallel_nn.comp_op()
    
    # by passing this through 'compose' (mean composition: e.g. OpenLayer o channels) 
    # on processors not equal to 0, these will be None (there are no parameters to train there)
    self.open_nn = compose(OpenLayer,encoding)
    self.close_nn = compose(CloseLayer)

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
    mask = (x == 0)
    x = self.compose(self.open_nn,x)
    self.parallel_nn.mask = mask
    x = self.parallel_nn(x)
    x = self.compose(self.close_nn,x)
    return x
# end ParallelNet 

def train(rank, args, model, train_loader, optimizer, epoch, compose, device):
  model.train()
  criterion = nn.CrossEntropyLoss(ignore_index=0)
  total_time = 0.0
  batch_epochs = 10
  batch_epochs = np.inf#args.epochs
  batch_ctr = 0
  for batch_idx, (data, target) in enumerate(train_loader):
    # root_print(rank, 'checkpoint 4')
    # root_print(rank, f'data.shape {data.shape}')
    if data.shape[0] != args.batch_size:
      break
    #root_print(rank, f'train data.shape {data.shape}')
    start_time = timer()
    optimizer.zero_grad()
    data = data.to(device)
    output = model(data).cpu()
    loss = compose(
      criterion,
      output.reshape(-1, output.shape[-1]),
      target.reshape(-1)
    )
    loss.backward()
    stop_time = timer()
    optimizer.step()

    total_time += stop_time-start_time
    if batch_idx % args.log_interval == 0:
      # root_print(rank,'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
      root_print(rank,'Batch-epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
          batch_ctr, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item(),total_time/(batch_idx+1.0)))

    batch_ctr += 1
    if batch_ctr >= batch_epochs:
      break
    elif batch_ctr%(batch_epochs//4) == 0:
      for g in optimizer.param_groups:
        g['lr'] *= .5
      root_print(rank, 'change lr *= .5')

  # root_print(rank,'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
  #   epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
  #   100. * (batch_idx+1) / len(train_loader), loss.item(),total_time/(batch_idx+1.0)))

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


def test(rank, args, model, test_loader, compose, device):
  model.eval()
  test_loss = 0
  correct, total = 0, 0
  criterion = nn.CrossEntropyLoss()
  with torch.no_grad():
    for data, target in test_loader:#test_loader
      if data.shape[0] != 64:#187
        break
      #root_print(rank, f'test data.shape {data.shape}')
      data = data.to(device)
      output = model(data).cpu()
      test_loss += compose(
        criterion,
        output.reshape(-1, output.shape[-1]),
        target.reshape(-1)
      ).item()

      output = MPI.COMM_WORLD.bcast(output,root=0)
      pred = output.reshape(-1, output.shape[-1]).argmax(dim=-1, keepdim=False)  # get the index of the max log-probability
      correct += ((pred == target.reshape(-1))*(target.reshape(-1) != 0)).sum().item()
      total += ((target.reshape(-1) != 0).sum()).item()

  accuracy = correct/total

  test_loss /= len(test_loader.dataset)

  # root_print(rank,'\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
  #     test_loss, correct, len(test_loader.dataset),
  #     100. * correct / len(test_loader.dataset)))
  root_print(rank,'\nTest set: Average loss: {:.4f}, Accuracy: {}\n'.format(
      test_loss, accuracy))

def compute_levels(num_steps,min_coarse_size,cfactor): 
  from math import log, floor 
  
  # we want to find $L$ such that ( max_L min_coarse_size*cfactor**L <= num_steps)
  levels =  floor(log(float(num_steps)/min_coarse_size,cfactor))+1 

  if levels<1:
    levels = 1
  return levels
# end compute levels


def obtain_ds_dl(data_path_train, data_path_dev, batch_size, max_len):
  train = data_path_train
  dev = data_path_dev

  vocabs = input_pipeline.create_vocabs(train)

  attributes_input = [input_pipeline.CoNLLAttributes.FORM]
  attributes_target = [input_pipeline.CoNLLAttributes.XPOS]

  train_ds, train_dl = preprocessing.obtain_dataset(
    filename=train, 
    vocabs=vocabs, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=batch_size, 
    bucket_size=max_len,
  )
  eval_ds, eval_dl = preprocessing.obtain_dataset(
    filename=dev, 
    vocabs=vocabs, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=64,#187, 
    bucket_size=max_len,
  )

  return train_ds, eval_ds, train_dl, eval_dl

def main():
  # Training settings
  parser = argparse.ArgumentParser(description='TORCHBRAID CIFAR10 Example')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 783253419)')
  parser.add_argument('--log-interval', type=int, default=1,#10, metavar='N',
                      metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--percent-data', type=float, default=1.0, metavar='N',
                      help='how much of the data to read in and use for training/testing')

  # artichtectural settings
  parser.add_argument('--steps', type=int, default=4, metavar='N',
                      help='Number of times steps in the resnet layer (default: 4)')
  parser.add_argument('--encoding', type=str, default='Torch',
                      help='which positional encoding will be used for the attention')
  parser.add_argument('--digits',action='store_true', default=False, 
                      help='Train with the MNIST digit recognition problem (default: False)')
  parser.add_argument('--serial-file',type=str,default=None,
                      help='Load the serial problem from file')
  parser.add_argument('--tf',type=float,default=1.0,
                      help='Final time')

  # algorithmic settings (gradient descent and batching
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
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
  parser.add_argument('--lp-use-relaxonlycg',action='store_true', default=0, 
                      help='Layer parallel use relaxation only on coarse grid (default: False)')

  rank  = MPI.COMM_WORLD.Get_rank()
  procs = MPI.COMM_WORLD.Get_size()
  args = parser.parse_args()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  root_print(rank, f'device {device}')
  # root_print(rank, 'checkpoint 1')

  # some logic to default to Serial if on one processor,
  # can be overriden by the user to run layer-parallel
  if args.force_lp:
    force_lp = True
  elif procs>1:
    force_lp = True
  else:
    force_lp = False

  force_lp = True 
  # REMAINING TO DEBUG SERIAL w/ gpu: CUDA OUT OF MEMORY

  root_print(rank, f'force_lp {force_lp}')

  #torch.manual_seed(torchbraid.utils.seed_from_rank(args.seed,rank))
  torch.manual_seed(args.seed)

  if args.lp_levels==-1:
    min_coarse_size = 3
    args.lp_levels = compute_levels(args.steps,min_coarse_size,args.lp_cfactor)

  local_steps = int(args.steps/procs)
  if args.steps % procs!=0:
    root_print(rank,'Steps must be an even multiple of the number of processors: %d %d' % (args.steps,procs) )
    sys.exit(0)

  root_print(rank,'Transformer ODENet:')

  root_print(rank,'-- procs    = {}\n'
                  '-- encoding = {}\n'
                  '-- tf       = {}\n'
                  '-- steps    = {}'.format(procs,args.encoding,args.tf,args.steps))

  # train_size = int(50000 * args.percent_data)
  # test_size  = int(10000 * args.percent_data)
  # train_set = torch.utils.data.Subset(dataset,range(train_size))
  # test_set  = torch.utils.data.Subset(dataset,range(train_size,train_size+test_size))
  # train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size,shuffle=False)
  # test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False)
  data_path_train = '/users/msalvado/MLT/ML_PQ/data/en_gum-ud-train.conllu.txt'
  data_path_dev = '/users/msalvado/MLT/ML_PQ/data/en_gum-ud-dev.conllu.txt'
  train_ds, eval_ds, train_dl, eval_dl = obtain_ds_dl(data_path_train, data_path_dev, args.batch_size, max_len=2048)
  train_set, test_set, train_loader, test_loader = train_ds, eval_ds, train_dl, eval_dl

  root_print(rank,'')

  if force_lp :
    root_print(rank,'Using ParallelNet:')
    root_print(rank,'-- max_levels     = {}\n'
                    '-- max_iters      = {}\n'
                    '-- fwd_iters      = {}\n'
                    '-- cfactor        = {}\n'
                    '-- fine fcf       = {}\n'
                    '-- skip down      = {}\n'
                    '-- relax only cg  = {}\n'
                    '-- fmg            = {}\n'.format(args.lp_levels,
                                                  args.lp_iters,
                                                  args.lp_fwd_iters,
                                                  args.lp_cfactor,
                                                  args.lp_finefcf,
                                                  not args.lp_use_downcycle,
                                                  args.lp_use_relaxonlycg,
                                                  args.lp_use_fmg))

    model = ParallelNet(encoding=args.encoding,
                        local_steps=local_steps,
                        max_levels=args.lp_levels,
                        max_iters=args.lp_iters,
                        fwd_max_iters=args.lp_fwd_iters,
                        print_level=args.lp_print,
                        braid_print_level=args.lp_braid_print,
                        cfactor=args.lp_cfactor,
                        fine_fcf=args.lp_finefcf,
                        skip_downcycle=not args.lp_use_downcycle,
                        fmg=args.lp_use_fmg,Tf=args.tf,
                        relax_only_cg=args.lp_use_relaxonlycg)


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
      model = SerialNet(encoding=args.encoding,local_steps=local_steps,Tf=args.tf)
    compose = lambda op,*p: op(*p)

  optimizer = optim.Adam(model.parameters(), lr=args.lr)#optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)

  epoch_times = []
  test_times = []

  # check out the initial conditions
  #if force_lp:
    #diagnose(rank, model, test_loader,0)

  # root_print(rank, 'checkpoint 3')

  model = model.to(device)

  for epoch in range(1, args.epochs + 1):
    start_time = timer()
    root_print(rank, f'epoch {epoch}')
    train(rank, args, model, train_loader, optimizer, epoch, compose, device)
    end_time = timer()
    epoch_times += [end_time-start_time]
    
    start_time = timer()
    test(rank, args, model, test_loader, compose, device)
    end_time = timer()
    test_times += [end_time-start_time]
    
    # print out some diagnostics
    #if force_lp:
    #  diagnose(rank, model, test_loader,epoch)

  #if force_lp:
  #  timer_str = model.parallel_nn.getTimersString()
  #  root_print(rank,timer_str)

  # root_print(rank,'TIME PER EPOCH: %.2e (1 std dev %.2e)' % (stats.mean(epoch_times),stats.stdev(epoch_times)))
  # root_print(rank,'TIME PER TEST:  %.2e (1 std dev %.2e)' % (stats.mean(test_times), stats.stdev(test_times)))

if __name__ == '__main__':
  main()
