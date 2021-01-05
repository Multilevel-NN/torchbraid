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

def print_backward(self,grad_input,grad_output):
  print('Inside ' + self.__class__.__name__ + ' backward')
  print('  Inside class:' + self.__class__.__name__)
  print('')

  print('  grad_input: ', type(grad_input),len(grad_input))
  for i,g in enumerate(grad_input):
    print('    grad_input[%d]: ' % i, type(g),g.size())
    print('        ', g)

  print('  grad_output: ', type(grad_output),len(grad_output))
  for i,g in enumerate(grad_output):
    print('    grad_output[%d]: ' % i, type(g),g.size())
    print('        ', g)

class RNN_SerialNet(nn.Module):

  def __init__(self,basic_block):
    super(RNN_SerialNet, self).__init__()
    self.net = basic_block()
    # self.net.register_backward_hook(print_backward)

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
    return result[-1,:,:].unsqueeze(0)

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
    if MPI.COMM_WORLD.Get_size()==1: 
      self.backwardProp()

  def test_backward_lstm(self):
    if MPI.COMM_WORLD.Get_size()==1: 
      self.backwardProp_lstm()

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

  def backwardProp_lstm(self,sequence_length = 28, # total number of time steps for each sequence
                        input_size = 28, # input size for each time step in a sequence
                        hidden_size = 20,
                        num_layers = 4,
                        batch_size = 1):
    """
    The goal of this test is simply to ensure that the behavior of pytorch is the same with
    regard to LSTM models.
    """
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
  
    image_all, x_block_all = preprocess_input_data_serial_test(2,num_batch,batch_size,channels,sequence_length,input_size)
    i = 0
    x_a = image_all[i].detach().clone()
    x_b = image_all[i].detach().clone()
  
    # forward
    ###########################################
    x_a.requires_grad = False 
    x_b.requires_grad = False
  
    h0_a = torch.zeros(num_layers, x_a.size(0), hidden_size,requires_grad=False)
    c0_a = torch.zeros(num_layers, x_a.size(0), hidden_size,requires_grad=False)
  
    # a simulation
    ######################################3
    serial_rnn_a = LSTMBlock(input_size, hidden_size, num_layers)
  
    h_a = h0_a
    c_a = c0_a
    hs = (sequence_length+1)*[None]
    cs = (sequence_length+1)*[None]
    hs[0] = h_a
    cs[0] = c_a
    for i in range(sequence_length):
      with torch.no_grad():
        h_a,c_a = serial_rnn_a(x_a[:,i,:],h_a,c_a)
  
      hs[i+1] = h_a
      cs[i+1] = c_a
  
    wh_a = torch.zeros(h_a.shape)
    wc_a = torch.zeros(c_a.shape)
    wh_a[-1,:,:] = torch.randn(h_a[-1,:,:].shape)
  
    wh_a_perm = wh_a
  
    for i in range(sequence_length):
      j = sequence_length-i-1
      hi = hs[j]
      ci = cs[j]
  
      hi.requires_grad = True
      ci.requires_grad = True
  
      with torch.enable_grad():
        ho,co = serial_rnn_a(x_a[:,j,:],hi,ci)
  
      ho.backward(wh_a,retain_graph=True)
      co.backward(wc_a)
  
      wh_a = hi.grad.detach().clone()
      wc_a = ci.grad.detach().clone()
    # end for i
  
    # b simulation
    ######################################3
    torch.manual_seed(20)
    serial_rnn_b = torch.nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
  
    h0_b = torch.zeros(num_layers, x_b.size(0), hidden_size,requires_grad=True)
    c0_b = torch.zeros(num_layers, x_b.size(0), hidden_size,requires_grad=True)
  
    y_b, (h_b,c_b) = serial_rnn_b(x_b,(h0_b,c0_b))
  
    #print('h error = ',torch.norm(h_b-h_a).item(),torch.norm(h_b).item(),h_b.shape)
    #print('c error = ',torch.norm(c_b-c_a).item(),torch.norm(c_b).item(),c_b.shape)

    self.assertTrue(torch.norm(h_b-h_a).item()<1e-6)
    self.assertTrue(torch.norm(c_b-c_a).item()<1e-6)
  
    w_b = torch.zeros(y_b.shape)
    w_b[:,-1,:] = wh_a_perm[-1,:,:]
  
    y_b.backward(w_b)
  
    #print('h grad error = ',torch.norm(h0_b.grad-wh_a).item(),torch.norm(h0_b.grad).item())
    #print('c grad error = ',torch.norm(c0_b.grad-wc_a).item(),torch.norm(c0_b.grad).item())
    self.assertTrue(torch.norm(h0_b.grad-wh_a).item()<1e-6)
    self.assertTrue(torch.norm(c0_b.grad-wc_a).item()<1e-6)
  
    #print('lengths = ',len(list(serial_rnn_b.parameters())),len(list(serial_rnn_a.parameters())))
    for pa,pb in zip(serial_rnn_a.parameters(),serial_rnn_b.parameters()):
    #  print('grad error = ',torch.norm(pa.grad-pb.grad).item(),torch.norm(pb.grad).item())
      self.assertTrue(torch.norm(pa.grad-pb.grad).item()<1e-6)
  # end lstm

  def backwardProp(self, sequence_length = 1, # total number of time steps for each sequence
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
      x_serial = image_all[i]
      x_serial.requires_grad = True

      y_serial = serial_rnn(x_serial)

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
    x_parallel = x_block[i]
    x_parallel.requires_grad = True
    y_parallel = parallel_nn(x_parallel)

    print('serial x versus parallel x = ',torch.norm(x_serial-x_parallel).item())
    print('   ',x_parallel.size(),x_serial.size())
    print('w0 = ',w0.shape)
    print('yp = ',y_parallel.shape)
    y_parallel.backward(w0)
  
    comm.barrier()

    # send the final inference step to root
    if my_rank == comm.Get_size()-1:
      comm.send(y_parallel,0)

    if my_rank==0:

      print('y out error = ',torch.norm(y_serial-y_parallel).item())

      # recieve the final inference step
      y_parallel = comm.recv(source=comm.Get_size()-1)
      self.assertTrue(torch.norm(y_serial-y_parallel)/torch.norm(y_serial)<1e-6)
  # forwardProp

if __name__ == '__main__':
  unittest.main()
