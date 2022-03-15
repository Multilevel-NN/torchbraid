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

import faulthandler
faulthandler.enable()

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

from torchvision import datasets, transforms

from timeit import default_timer as timer

from mpi4py import MPI
from math import pow

dt_default = lambda level,tstart,tstop,fine_dt: np.sqrt(np.sqrt((tstop-tstart)/fine_dt))
dt_sqrt    = lambda level,tstart,tstop,fine_dt: np.sqrt((tstop-tstart)/fine_dt)
dt_unit    = lambda level,tstart,tstop,fine_dt: (tstop-tstart)/fine_dt
dt_one     = lambda level,tstart,tstop,fine_dt: 1.0
dt_cfactor = lambda level,tstart,tstop,fine_dt: ((tstop-tstart)/fine_dt)**(1.0/args.cfactor)
dt_eight = lambda level,tstart,tstop,fine_dt: ((tstop-tstart)/fine_dt)**(1.0/8.0)

def root_print(rank,s):
  if rank==0:
    print(s)
    sys.stdout.flush()

# This is all instrumented for torchscript...of course after the effort, it doesn't
# seem to matter on the CPUs
def imp_gru_cell_fast(dt : float, x_red_r : torch.Tensor, x_red_z : torch.Tensor, x_red_n : torch.Tensor, h : torch.Tensor,
        lin_rh_W : torch.Tensor,
        lin_zh_W : torch.Tensor,
        lin_nr_W : torch.Tensor, lin_nr_b : torch.Tensor) -> torch.Tensor:

  r   =    torch.sigmoid(x_red_r +     F.linear(h,lin_rh_W))
  n   =    torch.   tanh(x_red_n + r * F.linear(h,lin_nr_W,lin_nr_b))
  dtz = dt*torch.sigmoid(x_red_z +     F.linear(h,lin_zh_W))

  return torch.div(torch.addcmul(h,dtz,n),1.0+dtz)

# This is all instrumented for torchscript...of course after the effort, it doesn't
# seem to matter on the CPUs
def imp_gru_cell(dt : float, x : torch.Tensor, h : torch.Tensor,
        lin_rx_W : torch.Tensor, lin_rx_b : torch.Tensor, lin_rh_W : torch.Tensor,
        lin_zx_W : torch.Tensor, lin_zx_b : torch.Tensor, lin_zh_W : torch.Tensor,
        lin_nx_W : torch.Tensor, lin_nx_b : torch.Tensor, lin_nr_W : torch.Tensor, lin_nr_b : torch.Tensor) -> torch.Tensor:

  r   =    torch.sigmoid(F.linear(x,lin_rx_W,lin_rx_b) +     F.linear(h,lin_rh_W))
  n   =    torch.   tanh(F.linear(x,lin_nx_W,lin_nx_b) + r * F.linear(h,lin_nr_W,lin_nr_b))
  dtz = dt*torch.sigmoid(F.linear(x,lin_zx_W,lin_zx_b) +     F.linear(h,lin_zh_W))

  return torch.div(torch.addcmul(h,dtz,n),1.0+dtz)

class ImplicitGRUBlock(nn.Module):
  def __init__(self, input_size, hidden_size, seed=20):
    super(ImplicitGRUBlock, self).__init__()

    torch.manual_seed(seed)

    #

    self.lin_rx = [None,None]
    self.lin_rh = [None,None]
    self.lin_rx[0] = nn.Linear(input_size,hidden_size,True)
    self.lin_rh[0] = nn.Linear(hidden_size,hidden_size,False)

    self.lin_zx = [None,None]
    self.lin_zh = [None,None]
    self.lin_zx[0] = nn.Linear(input_size,hidden_size,True)
    self.lin_zh[0] = nn.Linear(hidden_size,hidden_size,False)

    self.lin_nx = [None,None]
    self.lin_nr = [None,None]
    self.lin_nx[0] = nn.Linear(input_size,hidden_size,True)
    self.lin_nr[0] = nn.Linear(hidden_size,hidden_size,True)

    #

    self.lin_rx[1] = nn.Linear(hidden_size,hidden_size,True)
    self.lin_rh[1] = nn.Linear(hidden_size,hidden_size,False)

    self.lin_zx[1] = nn.Linear(hidden_size,hidden_size,True)
    self.lin_zh[1] = nn.Linear(hidden_size,hidden_size,False)

    self.lin_nx[1] = nn.Linear(hidden_size,hidden_size,True)
    self.lin_nr[1] = nn.Linear(hidden_size,hidden_size,True)

    # record the layers so that they are handled by backprop correctly
    layers =  self.lin_rx+self.lin_rh + \
              self.lin_zx+self.lin_zh + \
              self.lin_nx+self.lin_nr
    self.lin_layers = nn.ModuleList(layers)

  def reduceX(self, x):
    x_red_r = self.lin_rx[0](x)
    x_red_z = self.lin_zx[0](x)
    x_red_n = self.lin_nx[0](x)

    return (x_red_r,x_red_z,x_red_n)

  def fastForward(self, level,tstart,tstop,x_red, h_prev):
    dt = tstop-tstart

    h_prev = h_prev[0]
    h0 = imp_gru_cell_fast(dt, *x_red,h_prev[0],
                           self.lin_rh[0].weight,
                           self.lin_zh[0].weight,
                           self.lin_nr[0].weight, self.lin_nr[0].bias)
    h1 = imp_gru_cell(dt,h0,h_prev[1],
                      self.lin_rx[1].weight, self.lin_rx[1].bias, self.lin_rh[1].weight,
                      self.lin_zx[1].weight, self.lin_zx[1].bias, self.lin_zh[1].weight,
                      self.lin_nx[1].weight, self.lin_nx[1].bias, self.lin_nr[1].weight, self.lin_nr[1].bias)

    return torch.stack((h0,h1)),


  def forward(self, level,tstart,tstop,x, h_prev):
    dt = tstop-tstart

    h_prev = h_prev[0]
    h0 = imp_gru_cell(dt, x,h_prev[0],
                      self.lin_rx[0].weight, self.lin_rx[0].bias, self.lin_rh[0].weight,
                      self.lin_zx[0].weight, self.lin_zx[0].bias, self.lin_zh[0].weight,
                      self.lin_nx[0].weight, self.lin_nx[0].bias, self.lin_nr[0].weight, self.lin_nr[0].bias)
    h1 = imp_gru_cell(dt,h0,h_prev[1],
                      self.lin_rx[1].weight, self.lin_rx[1].bias, self.lin_rh[1].weight,
                      self.lin_zx[1].weight, self.lin_zx[1].bias, self.lin_zh[1].weight,
                      self.lin_nx[1].weight, self.lin_nx[1].bias, self.lin_nr[1].weight, self.lin_nr[1].bias)

    return torch.stack((h0,h1)),


def RNN_build_block(input_size, hidden_size, num_layers,seed=20):
  b = ImplicitGRUBlock(input_size, hidden_size, seed=seed)
  return b

def load_data(train,path):
  """
  Load the data from the UCI HAR data set.
  train: if true the training data is loaded, if false
         the test data is loaded.
  path: path to the data set directory.
  returns: An x,y pair where x is a rank 3 
           array (samples,seq length,data size)
  """

  if train:
    type_str = 'train'
  else:
    type_str = 'test'

  num_classes = 6
  d_path = path+'/'+type_str 
  i_path = d_path+'/Inertial Signals/'

  # load label data
  y = np.loadtxt('%s/y_%s.txt' % (d_path,type_str))

  # give upu on the pythonic way
  y_data = torch.zeros(y.shape[0],dtype=torch.long)
  for i,yi in enumerate(y):
    y_data[i] = int(yi-1)
  
  # load feature data
  body_x = np.loadtxt('%s/body_acc_x_%s.txt' % (i_path,type_str))
  body_y = np.loadtxt('%s/body_acc_y_%s.txt' % (i_path,type_str))
  body_z = np.loadtxt('%s/body_acc_z_%s.txt' % (i_path,type_str))

  gyro_x = np.loadtxt('%s/body_gyro_x_%s.txt' % (i_path,type_str))
  gyro_y = np.loadtxt('%s/body_gyro_y_%s.txt' % (i_path,type_str))
  gyro_z = np.loadtxt('%s/body_gyro_z_%s.txt' % (i_path,type_str))

  totl_x = np.loadtxt('%s/total_acc_x_%s.txt' % (i_path,type_str))
  totl_y = np.loadtxt('%s/total_acc_y_%s.txt' % (i_path,type_str))
  totl_z = np.loadtxt('%s/total_acc_z_%s.txt' % (i_path,type_str))

  x_data = np.stack([body_x,body_y,body_z,
                gyro_x,gyro_y,gyro_z,
                totl_x,totl_y,totl_z],axis=2)

  return torch.Tensor(x_data),y_data
  
def preprocess_dataset_parallel(comm,dataset):
  """
  Take in a serial loader on the root rank=0 for the training set, and then distribute
  it in parallel by forcing a split on the sequences (assumed to be dimension 1). The shuffling
  of the data is as constructed in the original loader (e.g. the data is copied)

  comm: Parallel communicator to use
  train_loader: Loader to pull data from
  
  returns a data loader that is distributed in paralled
  """

  rank = comm.Get_rank()
  num_procs = comm.Get_size()
  x_block = [[] for n in range(num_procs)]
  y_block = [[] for n in range(num_procs)]
  if rank == 0:
    sz = len(dataset)
    for i in range(sz):
      x,y = dataset[i]
      x_split = torch.chunk(x, num_procs, dim=0)
      y_split = num_procs*[y]

      for p,(x_in,y_in) in enumerate(zip(x_split,y_split)):
        x_block[p].append(x_in)
        y_block[p].append(y_in)
  # end if rank

    for p,(x,y) in enumerate(zip(x_block,y_block)):
      x_block[p] = torch.stack(x)
      y_block[p] = torch.stack(y)

  x_local = comm.scatter(x_block,root=0)
  y_local = comm.scatter(y_block,root=0)

  local_dataset = TensorDataset(x_local, y_local)

  return local_dataset

class ParallelRNNDataLoader(torch.utils.data.DataLoader):
  def __init__(self,comm,dataset,batch_size,shuffle=False):
    self.dataset = dataset 
    self.shuffle = shuffle

    if comm.Get_rank()==0:
      # build a gneerator to build one master initial seed
      serial_generator = torch.Generator()
      self.initial_seed = serial_generator.initial_seed()

      # build the serial loader for limited use
      if shuffle==True:
        sampler = torch.utils.data.sampler.RandomSampler(dataset,generator=serial_generator)
        self.serial_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,sampler=sampler)
      else:
        self.serial_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False)
    else:
      self.serial_loader = None
      self.initial_seed = None
    # if rank==0

    # distribute the initial seed
    self.initial_seed = comm.bcast(self.initial_seed,root=0)

    # break up sequences
    self.parallel_dataset = preprocess_dataset_parallel(comm,dataset)

    # now setup the serial generator
    parallel_generator = torch.Generator()
    parallel_generator.manual_seed(self.initial_seed)

    # now setup the parallel loader
    if shuffle==True:
      sampler = torch.utils.data.sampler.RandomSampler(self.parallel_dataset,generator=parallel_generator)
      torch.utils.data.DataLoader.__init__(self,self.parallel_dataset,batch_size=batch_size,sampler=sampler)
    else:
      torch.utils.data.DataLoader.__init__(self,self.parallel_dataset,batch_size=batch_size,shuffle=False)

  def getSerialDataLoader(self):
    return self.serial_loader

class CloseLayer(nn.Module):
  def __init__(self,hidden_size,num_classes,seed=20):
    super(CloseLayer, self).__init__()
    torch.manual_seed(seed+37)
    self.fc = nn.Linear(hidden_size,num_classes)
    # self.fc = nn.Linear(hidden_size,6)

  def forward(self, x):
    x = self.fc(x)
    return x
# end layer

class SerialNet(nn.Module):
  def __init__(self,input_size=9,hidden_size=20,num_layers=2,num_classes=6,num_steps=8,close_rnn=None,seed=20):
               
    super(SerialNet, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = hidden_size

    torch.manual_seed(seed)
    self.serial_rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    if close_rnn is None:
      self.close_rnn = CloseLayer(hidden_size,num_classes,seed)
    else:
      self.close_rnn = close_rnn

  def getFwdStats(self):
    return 1, 0.0

  def getBwdStats(self):
    return 1, 0.0

  def forward(self, x):
    q, _ = self.serial_rnn(x)
    return self.close_rnn(torch.squeeze(q[:,-1:,]))
# end SerialNet 

class ImplicitSerialNet(nn.Module):
  def __init__(self,input_size=9,hidden_size=20,num_layers=2,num_classes=6,seed=20,
               rnn_model=None,close_rnn=None):
    super(ImplicitSerialNet, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = hidden_size

    if rnn_model is None:
      rnn_model = RNN_build_block(input_size,hidden_size,num_layers,seed=seed)
    self.serial_rnn = torchbraid.RNN_Serial(rnn_model,num_layers,hidden_size,dt=1.0)

    if close_rnn is None:
      self.close_rnn = CloseLayer(hidden_size,num_classes,seed)
    else:
      self.close_rnn = close_rnn

  def getFwdStats(self):
    return 1, 0.0

  def getBwdStats(self):
    return 1, 0.0

  def forward(self, x):
    h = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    hn = self.serial_rnn(x,h)
    return self.close_rnn(hn[-1,:,:])
# end ImplicitSerialNet 

class ParallelNet(nn.Module):
  def __init__(self,input_size=9,hidden_size=20,num_layers=2,num_classes=6,
               num_steps=8,max_levels=1,max_iters=1,fwd_max_iters=0,
               print_level=0,cfactor=4,
               skip_downcycle=True,fmg=False,
               seed=20,
               Tf=None):

    super(ParallelNet, self).__init__()

    self.RNN_model = RNN_build_block(input_size,hidden_size,num_layers,seed=seed)

    if Tf==None:
      Tf = float(num_steps)*MPI.COMM_WORLD.Get_size() # when using an implicit method with GRU, 
    self.Tf = Tf
    self.dt = Tf/float(num_steps*MPI.COMM_WORLD.Get_size())

    self.parallel_rnn = torchbraid.RNN_Parallel(MPI.COMM_WORLD,
                                                self.RNN_model,
                                                num_steps,hidden_size,num_layers,
                                                Tf,
                                                model_compute_steps=True,
                                                max_levels=max_levels,max_iters=max_iters)

    if fwd_max_iters>0:
      self.parallel_rnn.setFwdMaxIters(fwd_max_iters)

    self.parallel_rnn.setPrintLevel(print_level)

    cfactor_dict = dict()
    cfactor_dict[-1] = cfactor  # defaults to user on other levels
    self.parallel_rnn.setCFactor(cfactor_dict)
    self.parallel_rnn.setSkipDowncycle(skip_downcycle)

    if fmg:
      self.parallel_rnn.setFMG()

    self.parallel_rnn.setNumRelax(1)            # FCF on all levels, by default
    self.parallel_rnn.setFwdNumRelax(1,level=0) # F-Relaxation on the fine grid (by default)
    self.parallel_rnn.setBwdNumRelax(0,level=0) # F-Relaxation on the fine grid (by default)

    # this object ensures that only the RNN_Parallel code runs on ranks!=0
    compose = self.compose = self.parallel_rnn.comp_op()

    self.close_rnn = compose(CloseLayer,hidden_size,num_classes,seed=seed)

    self.hidden_size = hidden_size
    self.num_layers = num_layers 

  def setFwdAbsTol(self,tol):
    """
    Set the forward absolute tolerance to be used as a stopping   
    critieria for the parallel-in-time algorithm.
    """

    self.parallel_rnn.setFwdAbsTol(tol)

  def setBwdAbsTol(self,tol):
    """
    Set the forward absolute tolerance to be used as a stopping   
    critieria for the parallel-in-time algorithm.
    """

    self.parallel_rnn.setBwdAbsTol(tol)

  def setFwdNumRelax(self,sweeps,level=None):
    """
    Set the number of relaxation sweeps. If sweeps is
    greater than 0, FCF is used. If its equal to 0, only
    F relaxation is used.
    """

    # F-Relaxation on the fine grid
    if level==None:
      self.parallel_rnn.setFwdNumRelax(sweeps)
    else:
      self.parallel_rnn.setFwdNumRelax(sweeps,level) 

  def setBwdNumRelax(self,sweeps,level=None):
    """
    Set the number of relaxation sweeps. If sweeps is
    greater than 0, FCF is used. If its equal to 0, only
    F relaxation is used.
    """

    # F-Relaxation on the fine grid
    if level==None:
      self.parallel_rnn.setBwdNumRelax(sweeps)
    else:
      self.parallel_rnn.setBwdNumRelax(sweeps,level) 

  def forward(self, x):
    h = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    hn = self.parallel_rnn(x,h)

    x = self.compose(self.close_rnn,hn[-1,:,:])

    return x  

  def getSerialModel(self):

    return ImplicitSerialNet(hidden_size=self.hidden_size,num_layers=self.num_layers,
                             close_rnn=self.close_rnn,
			     rnn_model=self.RNN_model)

  def getFwdStats(self):
    return self.parallel_rnn.getFwdStats()

  def getBwdStats(self):
    return self.parallel_rnn.getBwdStats()
# end ParallelNet 
