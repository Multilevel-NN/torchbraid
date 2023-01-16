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

def getDevice(comm):
  my_host    = torch.device('cpu')
  if torch.cuda.is_available() and torch.cuda.device_count()>=comm.Get_size():
    if comm.Get_rank()==0:
      print('Using GPU Device')
    my_device  = torch.device(f'cuda:{comm.Get_rank()}')
    torch.cuda.set_device(my_device)
  elif torch.cuda.is_available() and torch.cuda.device_count()<comm.Get_size():
    if comm.Get_rank()==0:
      print('GPUs are not used, because MPI ranks are more than the device count, using CPU')
    my_device = my_host
  else:
    if comm.Get_rank()==0:
      print('No GPUs to be used, CPU only')
    my_device = my_host
  return my_device,my_host
# end getDevice

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

  def forward(self, level, tstart, tstop, x, h):
    h_prev = h[0]
    c_prev = h[1]
    x_cur = x

    dt = tstop-tstart

    hn = self.num_layers*[None]
    cn = self.num_layers*[None]
    for i in range(self.num_layers):
      hn[i], cn[i] = self.lstm_cells[i](x_cur, (h_prev[i], c_prev[i]))
      x_cur = hn[i]

    # handle implicitness on coarse levels
    if level>0:
      for i in range(self.num_layers):
        hn[i] = (h_prev[i]+dt*hn[i])/(1.0+dt)
        cn[i] = (c_prev[i]+dt*cn[i])/(1.0+dt)

    return (torch.stack(hn), torch.stack(cn))

def RNN_build_block_with_dim(input_size, hidden_size, num_layers):
  b = LSTMBlock(input_size, hidden_size, num_layers) # channels = hidden_size
  return b

def preprocess_input_data_serial_test(num_blocks, num_batch, batch_size, channels, sequence_length, input_size, device):
  torch.manual_seed(20)
  x = torch.randn(num_batch,batch_size,channels,sequence_length,input_size)
  x = x.to(device)

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

def preprocess_distribute_input_data_parallel(rank,num_procs,num_batch,batch_size,channels,sequence_length,input_size,comm,device):
  if rank == 0:
    torch.manual_seed(20)
    x = torch.randn(num_batch,batch_size,channels,sequence_length,input_size)
    x = x.to(device)

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
        x_block_tmp.append(x_block_all[image_id][block_id].cpu())
      comm.send(x_block_tmp,dest=block_id,tag=20)

    return x_block
  else:
    x_block = comm.recv(source=0,tag=20)
    x_block = [b.to(device) for b in x_block]

    return x_block
# end preprocess_distr

class TestRNNLayerParallel(unittest.TestCase):
  def test_forward_exact(self):
    self.forwardProp(max_levels=1,max_iters=1,sequence_length=28)

  def test_forward_approx(self):
    self.forwardProp(max_levels=3,max_iters=20)

  def test_backward_exact(self):
    self.backwardProp()

  def test_backward_exact_multiple(self):
    self.backwardProp(applications=8)

  def test_backward_approx(self):
    self.backwardProp(max_levels=3,max_iters=20,sequence_length=27,tol=1e-5)

  # TODO: dead code?
  # def copyParameterGradToRoot(self,m):
  #   comm     = m.getMPIComm()
  #   my_rank  = m.getMPIComm().Get_rank()
  #   num_proc = m.getMPIComm().Get_size()

  #   params = [p.grad for p in list(m.parameters())]

  #   if len(params)==0:
  #     return params

  #   if my_rank==0:
  #     for i in range(1,num_proc):
  #       remote_p = comm.recv(source=i,tag=77)
  #       remote_p = [p.to(device) for p in remote_p]
  #       params.extend(remote_p)

  #     return params
  #   else:
  #     params_cpu = [p.cpu() for p in params]
  #     comm.send(params_cpu,dest=0,tag=77)
  #     return None
  # # end copyParametersToRoot

  def forwardProp(self,
                  max_levels = 1, # for testing parallel rnn
                  max_iters = 1, # for testing parallel rnn
                  sequence_length = 28, # total number of time steps for each sequence
                  input_size = 28, # input size for each time step in a sequence
                  hidden_size = 20,
                  num_layers = 2,
                  batch_size = 1):

    comm      = MPI.COMM_WORLD
    num_procs = comm.Get_size()
    my_rank   = comm.Get_rank()

    my_device,my_host = getDevice(MPI.COMM_WORLD)

    Tf              = float(sequence_length)
    channels        = 1
    images          = 10
    image_size      = 28
    print_level     = 0
    nrelax          = 1
    cfactor         = 2
    num_batch = int(images / batch_size)

    if my_rank==0:
      with torch.no_grad():

        torch.manual_seed(20)
        serial_rnn = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
        serial_rnn = serial_rnn.to(my_device)
        num_blocks = 2 # equivalent to the num_procs variable used for parallel implementation
        image_all, x_block_all = preprocess_input_data_serial_test(num_blocks,num_batch,batch_size,channels,sequence_length,input_size,my_device)

        for i in range(1):

          y_serial_hn = torch.zeros(num_layers, image_all[i].size(0), hidden_size, device=my_device)
          y_serial_cn = torch.zeros(num_layers, image_all[i].size(0), hidden_size, device=my_device)

          _, (y_serial_hn, y_serial_cn) = serial_rnn(image_all[i],(y_serial_hn,y_serial_cn))
          y_serial_hn = y_serial_hn.cpu()
          y_serial_cn = y_serial_cn.cpu()
    # compute serial solution

    # wait for serial processor
    comm.barrier()

    basic_block_parallel = lambda: RNN_build_block_with_dim(input_size, hidden_size, num_layers)
    num_procs = comm.Get_size()

    # preprocess and distribute input data
    ###########################################
    x_block = preprocess_distribute_input_data_parallel(my_rank,num_procs,num_batch,batch_size,channels,sequence_length,input_size,comm,my_device)

    num_steps = x_block[0].shape[1]
    # RNN_parallel.py -> RNN_Parallel() class
    parallel_rnn = torchbraid.RNN_Parallel(comm,
                                           basic_block_parallel(),
                                           num_steps,
                                           hidden_size,
                                           num_layers,
                                           Tf,
                                           max_fwd_levels=max_levels,
                                           max_bwd_levels=max_levels,
                                           max_iters=max_iters)
    parallel_rnn.to(my_device)

    parallel_rnn.setPrintLevel(print_level)
    parallel_rnn.setSkipDowncycle(True)
    parallel_rnn.setCFactor(cfactor)
    parallel_rnn.setNumRelax(nrelax)

    for i in range(1):

      y_parallel_hn,y_parallel_cn = parallel_rnn(x_block[i])

      comm.barrier()

      # send the final inference step to root
      if comm.Get_size()>1 and my_rank == comm.Get_size()-1:
        comm.send(y_parallel_hn.cpu(),0)
        comm.send(y_parallel_cn.cpu(),0)

      if my_rank==0:
        # recieve the final inference step
        if comm.Get_size()>1:
          parallel_hn = comm.recv(source=comm.Get_size()-1)
          parallel_cn = comm.recv(source=comm.Get_size()-1)
        else:
          parallel_hn = y_parallel_hn.cpu()
          parallel_cn = y_parallel_cn.cpu()

        print('cn values = ',torch.norm(y_serial_cn-parallel_cn).item()/torch.norm(y_serial_cn).item())
        print('hn values = ',torch.norm(y_serial_hn-parallel_hn).item()/torch.norm(y_serial_hn).item())
        self.assertTrue(torch.norm(y_serial_cn-parallel_cn).item()/torch.norm(y_serial_cn).item()<1e-6,'check cn')
        self.assertTrue(torch.norm(y_serial_hn-parallel_hn).item()/torch.norm(y_serial_hn).item()<1e-6,'check hn')
  # forwardProp

  def backwardProp(self,
                   max_levels = 1, # for testing parallel rnn
                   max_iters = 1, # for testing parallel rnn
                   sequence_length = 6, # total number of time steps for each sequence
                   input_size = 28, # input size for each time step in a sequence
                   hidden_size = 20,
                   num_layers = 1,
                   batch_size = 1,
                   tol=1e-6,
                   applications=1):

    comm      = MPI.COMM_WORLD
    num_procs = comm.Get_size()
    my_rank   = comm.Get_rank()

    my_device,my_host = getDevice(MPI.COMM_WORLD)

    Tf              = float(sequence_length)
    channels        = 1
    images          = 10
    image_size      = 28
    print_level     = 0
    nrelax          = 1
    cfactor         = 2
    num_batch = int(images / batch_size)

    # wait for serial processor
    comm.barrier()

    num_procs = comm.Get_size()

    # preprocess and distribute input data
    x_block = preprocess_distribute_input_data_parallel(my_rank,num_procs,num_batch,batch_size,channels,sequence_length,input_size,comm,my_device)

    num_steps = x_block[0].shape[1]

    basic_block_parallel = lambda: RNN_build_block_with_dim(input_size, hidden_size, num_layers)
    parallel_rnn = torchbraid.RNN_Parallel(comm,
                                           basic_block_parallel(),
                                           num_steps,
                                           hidden_size,
                                           num_layers,
                                           Tf,
                                           max_fwd_levels=max_levels,
                                           max_bwd_levels=max_levels,
                                           max_iters=max_iters)
    parallel_rnn.to(my_device)
    parallel_rnn.setPrintLevel(print_level)
    parallel_rnn.setSkipDowncycle(True)
    parallel_rnn.setCFactor(cfactor)
    parallel_rnn.setNumRelax(nrelax)

    torch.manual_seed(20)
    rand_w = torch.randn([1,x_block[0].size(0),hidden_size],device=my_device)

    for i in range(applications):
      h_0 = torch.zeros(num_layers, x_block[i].size(0), hidden_size,requires_grad=True,device=my_device)
      c_0 = torch.zeros(num_layers, x_block[i].size(0), hidden_size,requires_grad=True,device=my_device)

      with torch.enable_grad():
        y_parallel_hn,y_parallel_cn = parallel_rnn(x_block[i],(h_0,c_0))

      comm.barrier()

      w_h = torch.zeros(y_parallel_hn.shape,device=my_device)
      w_c = torch.zeros(y_parallel_hn.shape,device=my_device)

      w_h[-1,:,:] = rand_w

      y_parallel_hn.backward(w_h)

      if i<applications-1:
        with torch.no_grad():
          for p in parallel_rnn.parameters():
            p += p.grad
        parallel_rnn.zero_grad()

    # compute serial solution
    #############################################

    if my_rank==0:

      torch.manual_seed(20)
      serial_rnn = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
      serial_rnn.to(my_device)
      image_all, x_block_all = preprocess_input_data_serial_test(num_procs,num_batch,batch_size,channels,sequence_length,input_size,my_device)

      for i in range(applications):
        y_serial_hn_0 = torch.zeros(num_layers, image_all[i].size(0), hidden_size,requires_grad=True,device=my_device)
        y_serial_cn_0 = torch.zeros(num_layers, image_all[i].size(0), hidden_size,requires_grad=True,device=my_device)

        with torch.enable_grad():
          q, (y_serial_hn, y_serial_cn) = serial_rnn(image_all[i],(y_serial_hn_0,y_serial_cn_0))

        w_q = torch.zeros(q.shape,device=my_device)
        w_q[:,-1,:] = rand_w.detach().clone()

        q.backward(w_q)

        if i<applications-1:
          with torch.no_grad():
            for p in serial_rnn.parameters():
              p += p.grad
          serial_rnn.zero_grad()
    # end if my_rank

    # now check the answers
    #############################################

    # send the final inference step to root
    if comm.Get_size()>1 and my_rank == comm.Get_size()-1:
      comm.send(y_parallel_hn.cpu(),0)
      comm.send(y_parallel_cn.cpu(),0)

    if my_rank==0:
      y_serial_cn = y_serial_cn.cpu()
      y_serial_hn = y_serial_hn.cpu()

      if comm.Get_size()>1:
        # recieve the final inference step
        parallel_hn = comm.recv(source=comm.Get_size()-1)
        parallel_cn = comm.recv(source=comm.Get_size()-1)
      else:
        parallel_hn = y_parallel_hn.cpu()
        parallel_cn = y_parallel_cn.cpu()

      print('\n\n')
      print(torch.norm(y_serial_cn-parallel_cn).item()/torch.norm(y_serial_cn).item(),'forward cn')
      print(torch.norm(y_serial_hn-parallel_hn).item()/torch.norm(y_serial_hn).item(),'forward hn')
      sys.stdout.flush()

      self.assertTrue(torch.norm(y_serial_cn-parallel_cn)/torch.norm(y_serial_cn)<tol,'cn value')
      self.assertTrue(torch.norm(y_serial_hn-parallel_hn)/torch.norm(y_serial_hn)<tol,'hn value')

      print(torch.norm(h_0.grad.cpu()-y_serial_hn_0.grad.cpu()).item(),'back soln hn')
      print(torch.norm(c_0.grad.cpu()-y_serial_cn_0.grad.cpu()).item(),'back soln cn')
      self.assertTrue(torch.norm(h_0.grad.cpu()-y_serial_hn_0.grad.cpu()).item()<tol)
      self.assertTrue(torch.norm(c_0.grad.cpu()-y_serial_cn_0.grad.cpu()).item()<tol)

      root_grads = [p.grad for p in serial_rnn.parameters()]
      root_grads = [r.cpu() for r in root_grads]
    else:
      root_grads = None

    ref_grads = comm.bcast(root_grads,root=0)
    for pa_grad,pb in zip(ref_grads,parallel_rnn.parameters()):
      if torch.norm(pa_grad).item()==0.0:
        print(my_rank,torch.norm(pa_grad.cpu()-pb.grad.cpu()).item().item(),'param grad')
        self.assertTrue(torch.norm(pa_grad.cpu()-pb.grad.cpu()).item()<1e1*tol,'param grad')
      else:
        print(my_rank,torch.norm(pa_grad.cpu()-pb.grad.cpu()).item()/torch.norm(pa_grad.cpu()).item(),'param grad')
        self.assertTrue(torch.norm(pa_grad.cpu()-pb.grad.cpu()).item()/torch.norm(pa_grad.cpu()).item()<1e1*tol,'param grad')
  # forwardProp

if __name__ == '__main__':
  unittest.main()
