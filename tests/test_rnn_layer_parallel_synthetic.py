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

class LSTMBlock(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    super(LSTMBlock, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    torch.manual_seed(20)
    lstm_cells = num_layers*[None]
    lstm_cells[0] = nn.LSTMCell(input_size, hidden_size)
    for i in range(num_layers-1):
      lstm_cells[i+1] = nn.LSTMCell(hidden_size, hidden_size)

    self.lstm_cells = nn.ModuleList(lstm_cells)

  def forward(self, x, h_prev, c_prev):
    h_cur = h_prev[0] 
    c_cur = c_prev[0]
    x_cur = x

    hn = self.num_layers*[None]
    cn = self.num_layers*[None]
    for i in range(self.num_layers):
      hn[i], cn[i] = self.lstm_cells[i](x_cur, (h_prev[i], c_prev[i]))
      x_cur = hn[i]

    return (torch.stack(hn), torch.stack(cn))

def RNN_build_block_with_dim(input_size, hidden_size, num_layers):
  b = LSTMBlock(input_size, hidden_size, num_layers) # channels = hidden_size
  return b

def deltaW(dt,size):
  return torch.normal(mean=0.0,std=np.sqrt(dt),size=size)

def generate_data(num_batch,batch_size,channels,sequence_length,input_size):
  Y = torch.zeros(num_batch,batch_size,channels,sequence_length,input_size)

  T_final = 1.0
  dt = T_final/(sequence_length-1)
  Mu = 0.1
  Sigma = 0.2

  shape = Y[:,:,:,0,:].shape
  Y[:,:,:,0,:] = torch.normal(mean=0.1,std=0.02,size=shape)

  for n in range(sequence_length-1):
    Yn = Y[:,:,:,n,:]
    Y[:,:,:,n+1,:] = Yn + dt*Mu*Yn+Sigma*Yn*deltaW(dt,shape)

  return Y

def preprocess_synthetic_image_sequences_serial(num_blocks,num_batch,batch_size,channels,sequence_length,input_size):
  torch.manual_seed(20)
  x = generate_data(num_batch,batch_size,channels,sequence_length,input_size)

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

def preprocess_distribute_synthetic_image_sequences_parallel(rank,num_procs,num_batch,batch_size,channels,sequence_length,input_size,comm):
  if rank == 0:
    torch.manual_seed(20)
    x = generate_data(num_batch,batch_size,channels,sequence_length,input_size)

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
  def test_forward_exact(self):
    self.forwardProp(max_levels=1,max_iters=1)

  def test_forward_approx(self):
    self.forwardProp(max_levels=3,max_iters=20,tol=1e-4)

  def test_backward_exact(self):
    self.backwardProp()

  def test_backward_approx(self):
    self.backwardProp(max_levels=3,max_iters=40,sequence_length=128,tol=5e-4)

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

  def forwardProp(self, 
                  max_levels = 1, # for testing parallel rnn
                  max_iters = 1, # for testing parallel rnn
                  sequence_length = 128, # total number of time steps for each sequence
                  input_size = 1, # input size for each time step in a sequence
                  hidden_size = 20,
                  num_layers = 2,
                  batch_size = 100,
                  tol=1e-6):

    comm      = MPI.COMM_WORLD
    num_procs = comm.Get_size()
    my_rank   = comm.Get_rank()
      
    Tf              = 2.0
    channels        = 1
    images          = 1000
    # image_size      = 28
    print_level     = 1
    nrelax          = 3
    cfactor         = 2
    num_batch = int(images / batch_size)

    if my_rank==0:
      with torch.no_grad(): 

        torch.manual_seed(20)
        serial_rnn = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
        num_blocks = 2 # equivalent to the num_procs variable used for parallel implementation
        image_all, x_block_all = preprocess_synthetic_image_sequences_serial(num_blocks,num_batch,batch_size,channels,sequence_length,input_size)

        # for i in range(len(image_all)):
        for i in range(1):
    
          y_serial_hn = torch.zeros(num_layers, image_all[i].size(0), hidden_size)
          y_serial_cn = torch.zeros(num_layers, image_all[i].size(0), hidden_size)
    
          _, (y_serial_hn, y_serial_cn) = serial_rnn(image_all[i],(y_serial_hn,y_serial_cn))
    # compute serial solution 

    # wait for serial processor
    comm.barrier()

    basic_block_parallel = lambda: RNN_build_block_with_dim(input_size, hidden_size, num_layers)
    num_procs = comm.Get_size()

    # preprocess and distribute input data
    ###########################################
    x_block = preprocess_distribute_synthetic_image_sequences_parallel(my_rank,num_procs,num_batch,batch_size,channels,sequence_length,input_size,comm)
  
    num_steps = x_block[0].shape[1]
    # RNN_parallel.py -> RNN_Parallel() class
    parallel_rnn = torchbraid.RNN_Parallel(comm,basic_block_parallel(),num_steps,hidden_size,num_layers,Tf,max_levels=max_levels,max_iters=max_iters)
  
    parallel_rnn.setPrintLevel(print_level)
    parallel_rnn.setSkipDowncycle(True)
    parallel_rnn.setCFactor(cfactor)
    parallel_rnn.setNumRelax(nrelax)

    # for i in range(len(x_block)):
    for i in range(1):
  
      y_parallel_hn,y_parallel_cn = parallel_rnn(x_block[i])

      comm.barrier()
  
      # send the final inference step to root
      if my_rank == comm.Get_size()-1 and comm.Get_size()>1:
        comm.send(y_parallel_hn,0)
        comm.send(y_parallel_cn,0)

      if my_rank==0:
        # recieve the final inference step
        if comm.Get_size()>1:
          parallel_hn = comm.recv(source=comm.Get_size()-1)
          parallel_cn = comm.recv(source=comm.Get_size()-1)
        else:
          parallel_hn = y_parallel_hn
          parallel_cn = y_parallel_cn

        print('cn values = ',torch.norm(y_serial_cn-parallel_cn).item()/torch.norm(y_serial_cn).item())
        print('hn values = ',torch.norm(y_serial_hn-parallel_hn).item()/torch.norm(y_serial_hn).item())
        self.assertTrue(torch.norm(y_serial_cn-parallel_cn).item()/torch.norm(y_serial_cn).item()<tol,'check cn')
        self.assertTrue(torch.norm(y_serial_hn-parallel_hn).item()/torch.norm(y_serial_hn).item()<tol,'check hn')
  # forwardProp

  def backwardProp(self, 
                   max_levels = 1, # for testing parallel rnn
                   max_iters = 1, # for testing parallel rnn
                   sequence_length = 128, # total number of time steps for each sequence
                   input_size = 1, # input size for each time step in a sequence
                   hidden_size = 20,
                   num_layers = 2,
                   batch_size = 100,
                   tol=1e-6):

    comm      = MPI.COMM_WORLD
    num_procs = comm.Get_size()
    my_rank   = comm.Get_rank()
      
    Tf              = 2.0
    channels        = 1
    images          = 1000
    # image_size      = 28
    print_level     = 1
    nrelax          = 3
    cfactor         = 4
    num_batch = int(images / batch_size)

    # wait for serial processor
    comm.barrier()

    num_procs = comm.Get_size()

    # preprocess and distribute input data
    x_block = preprocess_distribute_synthetic_image_sequences_parallel(my_rank,num_procs,num_batch,batch_size,channels,sequence_length,input_size,comm)
  
    num_steps = x_block[0].shape[1]
  
    basic_block_parallel = lambda: RNN_build_block_with_dim(input_size, hidden_size, num_layers)
    parallel_rnn = torchbraid.RNN_Parallel(comm,basic_block_parallel(),num_steps,hidden_size,num_layers,Tf,max_levels=max_levels,max_iters=max_iters)
  
    parallel_rnn.setPrintLevel(print_level)
    parallel_rnn.setSkipDowncycle(True)
    parallel_rnn.setCFactor(cfactor)
    parallel_rnn.setNumRelax(nrelax)

    h_0 = torch.zeros(num_layers, x_block[0].size(0), hidden_size,requires_grad=True)
    c_0 = torch.zeros(num_layers, x_block[0].size(0), hidden_size,requires_grad=True)
  
    with torch.enable_grad(): 
      y_parallel_hn,y_parallel_cn = parallel_rnn(x_block[0],(h_0,c_0))

  
    comm.barrier()

    w_h = torch.zeros(y_parallel_hn.shape)
    w_c = torch.zeros(y_parallel_hn.shape)
    torch.manual_seed(20)
    w_h[-1,:,:] = torch.randn(y_parallel_hn[-1,:,:].shape)
  
    y_parallel_hn.backward(w_h)

    # compute serial solution 
    #############################################

    if my_rank==0:

      torch.manual_seed(20)
      serial_rnn = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
      image_all, x_block_all = preprocess_synthetic_image_sequences_serial(num_procs,num_batch,batch_size,channels,sequence_length,input_size)

      i = 0
      y_serial_hn_0 = torch.zeros(num_layers, image_all[i].size(0), hidden_size,requires_grad=True)
      y_serial_cn_0 = torch.zeros(num_layers, image_all[i].size(0), hidden_size,requires_grad=True)

      with torch.enable_grad(): 
        q, (y_serial_hn, y_serial_cn) = serial_rnn(image_all[i],(y_serial_hn_0,y_serial_cn_0))

      print('\n\n')
      print('fore in: ',
              (torch.norm(y_serial_hn_0).item(),torch.norm(y_serial_cn_0).item()),
            'out: ',
              (torch.norm(y_serial_hn).item(),torch.norm(y_serial_cn).item()))

      w_q = torch.zeros(q.shape)
      w_q[:,-1,:] = w_h[-1,:,:].detach().clone()

      q.backward(w_q)

      print('back in: ',
              torch.norm(w_q).item(),
            'out: ',
              (torch.norm(y_serial_hn_0.grad).item(),torch.norm(y_serial_cn_0.grad).item()))
      print('')
    # end if my_rank

    # now check the answers
    #############################################
  
    # send the final inference step to root
    if my_rank == comm.Get_size()-1 and comm.Get_size()>1:
      comm.send(y_parallel_hn,0)
      comm.send(y_parallel_cn,0)

    if my_rank==0:
      # recieve the final inference step
      if comm.Get_size()>1:
        parallel_hn = comm.recv(source=comm.Get_size()-1)
        parallel_cn = comm.recv(source=comm.Get_size()-1)
      else:
        parallel_hn = y_parallel_hn
        parallel_cn = y_parallel_cn

      print(torch.norm(y_serial_cn-parallel_cn).item()/torch.norm(y_serial_cn).item(),'forward cn')
      print(torch.norm(y_serial_hn-parallel_hn).item()/torch.norm(y_serial_hn).item(),'forward hn')

      #self.assertTrue(torch.norm(y_serial_cn-parallel_cn)/torch.norm(y_serial_cn)<tol,'cn value')
      #self.assertTrue(torch.norm(y_serial_hn-parallel_hn)/torch.norm(y_serial_hn)<tol,'rn value')

      print('back hn',torch.norm(h_0.grad).item(),torch.norm(y_serial_hn_0.grad).item())
      print('back cn',torch.norm(c_0.grad).item(),torch.norm(y_serial_cn_0.grad).item())
      #self.assertTrue(torch.norm(h_0.grad-y_serial_hn_0.grad).item()<tol)
      #self.assertTrue(torch.norm(c_0.grad-y_serial_cn_0.grad).item()<tol)

      root_grads = [p.grad for p in serial_rnn.parameters()]
    else:
      root_grads = None

    ref_grads = comm.bcast(root_grads,root=0)
    for pa_grad,pb in zip(ref_grads,parallel_rnn.parameters()):
      print('grad values = ',torch.norm(pa_grad-pb.grad).item()/torch.norm(pa_grad).item(),torch.norm(pa_grad).item())
      #self.assertTrue(torch.norm(pa_grad-pb.grad).item()/torch.norm(pa_grad).item()<tol,'param grad')
  # forwardProp

if __name__ == '__main__':
  unittest.main()
