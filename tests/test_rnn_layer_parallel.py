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

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import sys
import numpy as np
import statistics as stats

import torchbraid

import faulthandler
faulthandler.enable()

from mpi4py import MPI

class RNN_BasicBlock(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    super(RNN_BasicBlock, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    torch.manual_seed(20)
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

  def forward(self, x, h_prev, c_prev):
    h0 = h_prev
    c0 = c_prev
    _, (hn, cn) = self.lstm(x, (h0, c0))
    return _, (hn, cn)

def RNN_build_block_with_dim(input_size, hidden_size, num_layers):
  b = RNN_BasicBlock(input_size, hidden_size, num_layers) # channels = hidden_size
  return b

class RNN_SerialNet(nn.Module):
  def __init__(self,basic_block):
    super(RNN_SerialNet, self).__init__()
    self.net = basic_block()

  def forward(self,x):
    hidden_size = self.net.hidden_size
    num_layers  = self.net.num_layers

    hn = torch.zeros(num_layers, x.size(0), hidden_size)
    cn = torch.zeros(num_layers, x.size(0), hidden_size)
    
    result, _ = self.net(x,hn,cn)
    return result[:,-1,:].unsqueeze(1)

class RNN_ParallelNet(nn.Module):
  def __init__(self,*args,**kwargs):
    super(RNN_ParallelNet, self).__init__()
    self.net = torchbraid.RNN_Parallel(*args,**kwargs)

  def forward(self,x):
    (result, _) = self.net(x)
    return result.data[-1]

def preprocess_input_data_serial_test(num_blocks, num_batch, batch_size, channels, sequence_length, input_size):
  torch.manual_seed(20)
  x = torch.randn(num_batch,batch_size,channels,sequence_length,input_size)

  data_all = []
  x_block_all = []
  for i in range(len(x)):
    image = x[i].reshape(-1, sequence_length, input_size)
    images_split = torch.chunk(image, num_blocks, dim=1)
    seq_split = []
    for blk in images_split:
      seq_split.append(blk)
    x_block_all.append(seq_split)
    data_all.append(image)

  return data_all, x_block_all

def preprocess_distribute_input_data_parallel(rank,num_procs,num_batch,batch_size,channels,sequence_length,input_size,comm):
  if rank == 0:
    torch.manual_seed(20)
    x = torch.randn(num_batch,batch_size,channels,sequence_length,input_size)

    # x_block_all[total_images][total_blocks]
    x_block_all = []
    for i in range(len(x)):
      image = x[i].reshape(-1,sequence_length,input_size)
      data_split = torch.chunk(image, num_procs, dim=1)
      seq_split = []
      for blk in data_split:
        seq_split.append(blk)
      x_block_all.append(seq_split)

    x_block = []
    for image_id in range(len(x_block_all)):
      x_block.append(x_block_all[image_id][rank])

    for block_id in range(1,num_procs):
      x_block_tmp = []
      for image_id in range(len(x_block_all)):
        x_block_tmp.append(x_block_all[image_id][block_id])
      comm.send(x_block_tmp,dest=block_id,tag=20)

    return x_block
  
  else:
    x_block = comm.recv(source=0,tag=20)

    return x_block

class TestRNNLayerParallel(unittest.TestCase):
  def test_forward(self):
    self.forwardProp()

  def test_backward(self):
    self.backwardProp()

  def copyParameterGradToRoot(self,m):
    comm     = m.getMPIComm()
    my_rank  = m.getMPIComm().Get_rank()
    num_proc = m.getMPIComm().Get_size()
 
    params = [p.grad for p in list(m.parameters())]

    if len(params)==0:
      return params

    if my_rank==0:
      for i in range(1,num_proc):
        remote_p = comm.recv(source=i,tag=77)
        params.extend(remote_p)

      return params
    else:
      comm.send(params,dest=0,tag=77)
      return None
  # end copyParametersToRoot

  def forwardProp(self, sequence_length = 28, # total number of time steps for each sequence
                        input_size = 28, # input size for each time step in a sequence
                        hidden_size = 20,
                        num_layers = 2,
                        batch_size = 1):

    comm      = MPI.COMM_WORLD
    num_procs = comm.Get_size()
    my_rank   = comm.Get_rank()
      
    Tf              = 2.0
    channels        = 1
    images          = 10
    image_size      = 28
    print_level     = 0
    nrelax          = 1
    cfactor         = 2
    num_batch = int(images / batch_size)

    if my_rank==0:
      with torch.no_grad(): 

        basic_block = lambda: RNN_build_block_with_dim(input_size, hidden_size, num_layers)
        serial_rnn = basic_block()
        num_blocks = 2 # equivalent to the num_procs variable used for parallel implementation
        image_all, x_block_all = preprocess_input_data_serial_test(num_blocks,num_batch,batch_size,channels,sequence_length,input_size)
    
        # for i in range(len(image_all)):
        for i in range(1):
    
          # Serial ver. 1
          ###########################################
          y_serial_hn = torch.zeros(num_layers, image_all[i].size(0), hidden_size)
          y_serial_cn = torch.zeros(num_layers, image_all[i].size(0), hidden_size)
    
          _, (y_serial_hn, y_serial_cn) = serial_rnn(image_all[i],y_serial_hn,y_serial_cn)
    
          # Serial ver.2
          ###########################################
          for j in range(num_blocks):
            if j == 0:
              y_serial_prev_hn = torch.zeros(num_layers, x_block_all[i][j].size(0), hidden_size)
              y_serial_prev_cn = torch.zeros(num_layers, x_block_all[i][j].size(0), hidden_size)
    
            _, (y_serial_prev_hn, y_serial_prev_cn) = serial_rnn(x_block_all[i][j],y_serial_prev_hn,y_serial_prev_cn)
    
    # compute serial solution 

    # wait for serial processor
    comm.barrier()

    basic_block_parallel = lambda: RNN_build_block_with_dim(input_size, hidden_size, num_layers)
    num_procs = comm.Get_size()

    # preprocess and distribute input data
    ###########################################
    x_block = preprocess_distribute_input_data_parallel(my_rank,num_procs,num_batch,batch_size,channels,sequence_length,input_size,comm)
  
    max_levels = 1 # for testing parallel rnn
    max_iters = 1 # for testing parallel rnn
    num_steps = x_block[0].shape[1]
    # RNN_parallel.py -> RNN_Parallel() class
    parallel_nn = torchbraid.RNN_Parallel(comm,basic_block_parallel,num_steps,hidden_size,num_layers,Tf,max_levels=max_levels,max_iters=max_iters)
  
    parallel_nn.setPrintLevel(print_level)
    parallel_nn.setSkipDowncycle(True)
    parallel_nn.setCFactor(cfactor)
    parallel_nn.setNumRelax(nrelax)
    #parallel_nn.setNumRelax(nrelax,level=0)
  
    # for i in range(len(x_block)):
    for i in range(1):
  
      y_parallel = parallel_nn(x_block[i])
  
      (y_parallel_hn, y_parallel_cn) = y_parallel
  
      comm.barrier()
  
      # send the final inference step to root
      if my_rank == comm.Get_size()-1:
        comm.send(y_parallel_hn,0)
        comm.send(y_parallel_cn,0)

      if my_rank==0:
        # recieve the final inference step
        parallel_hn = comm.recv(source=comm.Get_size()-1)
        parallel_cn = comm.recv(source=comm.Get_size()-1)
        self.assertTrue(torch.norm(y_serial_cn.data[0]-parallel_cn.data[0])/torch.norm(y_serial_cn.data[0])<1e-6)
        self.assertTrue(torch.norm(y_serial_cn.data[1]-parallel_cn.data[1])/torch.norm(y_serial_cn.data[1])<1e-6)
        self.assertTrue(torch.norm(y_serial_hn.data[0]-parallel_hn.data[0])/torch.norm(y_serial_hn.data[0])<1e-6)
        self.assertTrue(torch.norm(y_serial_hn.data[1]-parallel_hn.data[1])/torch.norm(y_serial_hn.data[1])<1e-6)
  # forwardProp

  def backwardProp(self, sequence_length = 28, # total number of time steps for each sequence
                         input_size = 28, # input size for each time step in a sequence
                         hidden_size = 20,
                         num_layers = 2,
                         batch_size = 1):
    comm      = MPI.COMM_WORLD
    num_procs = comm.Get_size()
    my_rank   = comm.Get_rank()
      
    Tf              = 2.0
    channels        = 1
    images          = 10
    image_size      = 28
    print_level     = 0
    nrelax          = 1
    cfactor         = 2
    num_batch = int(images / batch_size)

    if my_rank==0:

      basic_block = lambda: RNN_build_block_with_dim(input_size, hidden_size, num_layers)
      serial_rnn = RNN_SerialNet(basic_block)
      num_blocks = 2 # equivalent to the num_procs variable used for parallel implementation
      image_all, x_block_all = preprocess_input_data_serial_test(num_blocks,num_batch,batch_size,channels,sequence_length,input_size)
  
      # Serial
      ###########################################
      i = 0
      x = image_all[i]
      x.requires_grad = True

      y_serial = serial_rnn(image_all[i])

      # compute the adjoint
      w0 = torch.randn(y_serial.shape) # adjoint initial cond
      y_serial.backward(w0)

      #for serial_param in list(serial_rnn.parameters()):
      #    print('param_grad = ',serial_param.grad.shape,serial_param.grad)

    # compute serial solution 

    # wait for serial processor
    comm.barrier()

    basic_block_parallel = lambda: RNN_build_block_with_dim(input_size, hidden_size, num_layers)
    num_procs = comm.Get_size()
    
    # preprocess and distribute input data
    ###########################################
    x_block = preprocess_distribute_input_data_parallel(my_rank,num_procs,num_batch,batch_size,channels,sequence_length,input_size,comm)
  
    max_levels = 1 # for testing parallel rnn
    max_iters = 1 # for testing parallel rnn
    num_steps = x_block[0].shape[1]
    # RNN_parallel.py -> RNN_Parallel() class
    parallel_nn = RNN_ParallelNet(comm,basic_block_parallel,num_steps,hidden_size,num_layers,Tf,max_levels=max_levels,max_iters=max_iters)
  
    parallel_nn.net.setPrintLevel(print_level)
    parallel_nn.net.setSkipDowncycle(True)
    parallel_nn.net.setCFactor(cfactor)
    parallel_nn.net.setNumRelax(nrelax)
  
    i = 0 # each image
    y_parallel = parallel_nn(x_block[i])
  
    comm.barrier()

    # send the final inference step to root
    if my_rank == comm.Get_size()-1:
      comm.send(y_parallel,0)

    if my_rank==0:
      # recieve the final inference step
      y_parallel = comm.recv(source=comm.Get_size()-1)
      self.assertTrue(torch.norm(y_serial-y_parallel)/torch.norm(y_serial)<1e-6)
  # forwardProp

if __name__ == '__main__':
  unittest.main()
