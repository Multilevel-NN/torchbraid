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
from torchbraid.utils import getDevice

import faulthandler
faulthandler.enable()

from mpi4py import MPI

def imp_gru_cell_fast(dt : float, x_red_r : torch.Tensor, x_red_z : torch.Tensor, x_red_n : torch.Tensor, 
                      h : torch.Tensor, lin_rh_W : torch.Tensor, lin_zh_W : torch.Tensor,
                      lin_nr_W : torch.Tensor, lin_nr_b : torch.Tensor) -> torch.Tensor:

  r   =    torch.sigmoid(x_red_r +     F.linear(h,lin_rh_W))
  n   =    torch.   tanh(x_red_n + r * F.linear(h,lin_nr_W, lin_nr_b))
  # dt*(1-z):
  dtz = dt*(1 - torch.sigmoid(x_red_z +     F.linear(h,lin_zh_W)))

  return torch.div(torch.addcmul(h,dtz,n),1.0+dtz)

def imp_gru_cell(dt : float, x : torch.Tensor, h : torch.Tensor,
                 lin_rx_W : torch.Tensor, lin_rx_b : torch.Tensor, lin_rh_W : torch.Tensor,
                 lin_zx_W : torch.Tensor, lin_zx_b : torch.Tensor, lin_zh_W : torch.Tensor,
                 lin_nx_W : torch.Tensor, lin_nx_b : torch.Tensor, lin_nr_W : torch.Tensor, 
                 lin_nr_b : torch.Tensor) -> torch.Tensor:

  r   =      torch.sigmoid(F.linear(x, lin_rx_W, lin_rx_b) +     F.linear(h, lin_rh_W))
  n   =      torch.   tanh(F.linear(x, lin_nx_W, lin_nx_b) + r * F.linear(h, lin_nr_W,lin_nr_b))
  # dt*(1-z):
  dtz = dt * (1 - torch.sigmoid(F.linear(x, lin_zx_W, lin_zx_b) + F.linear(h, lin_zh_W)))

  return torch.div(torch.addcmul(h, dtz, n), 1.0 + dtz)


class ImplicitGRUBlock(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(ImplicitGRUBlock, self).__init__()

    #

    self.lin_rx = [None,None]
    self.lin_rh = [None,None]
    self.lin_rx[0] = nn.Linear(input_size, hidden_size, True)
    self.lin_rh[0] = nn.Linear(hidden_size, hidden_size, False)

    self.lin_zx = [None,None]
    self.lin_zh = [None,None]
    self.lin_zx[0] = nn.Linear(input_size, hidden_size, True)
    self.lin_zh[0] = nn.Linear(hidden_size, hidden_size, False)

    self.lin_nx = [None,None]
    self.lin_nr = [None,None]
    self.lin_nx[0] = nn.Linear(input_size, hidden_size, True)
    self.lin_nr[0] = nn.Linear(hidden_size, hidden_size, True)

    #

    self.lin_rx[1] = nn.Linear(hidden_size, hidden_size, True)
    self.lin_rh[1] = nn.Linear(hidden_size, hidden_size, False)

    self.lin_zx[1] = nn.Linear(hidden_size, hidden_size, True)
    self.lin_zh[1] = nn.Linear(hidden_size, hidden_size, False)

    self.lin_nx[1] = nn.Linear(hidden_size, hidden_size, True)
    self.lin_nr[1] = nn.Linear(hidden_size, hidden_size, True)

    # record the layers so that they are handled by backprop correctly
    layers =  self.lin_rx + self.lin_rh + \
              self.lin_zx + self.lin_zh + \
              self.lin_nx + self.lin_nr
    self.lin_layers = nn.ModuleList(layers)

  def reduceX(self, x):
    x_red_r = self.lin_rx[0](x)
    x_red_z = self.lin_zx[0](x)
    x_red_n = self.lin_nx[0](x)

    return (x_red_r, x_red_z, x_red_n)

  # def fastForward(self, level, tstart, tstop, x_red, h_prev):
  #   dt = tstop-tstart

  #   h_prev = h_prev[0]
  #   h0 = imp_gru_cell_fast(dt, *x_red, h_prev[0],
  #                          self.lin_rh[0].weight,
  #                          self.lin_zh[0].weight,
  #                          self.lin_nr[0].weight, self.lin_nr[0].bias)
  #   h1 = imp_gru_cell(dt, h0, h_prev[1],
  #                     self.lin_rx[1].weight, self.lin_rx[1].bias, self.lin_rh[1].weight,
  #                     self.lin_zx[1].weight, self.lin_zx[1].bias, self.lin_zh[1].weight,
  #                     self.lin_nx[1].weight, self.lin_nx[1].bias, self.lin_nr[1].weight, self.lin_nr[1].bias)

  #   # Note: we return a tuple with a single element
  #   return (torch.stack((h0, h1)),)


  def forward(self, level, tstart, tstop, x, h_prev):
    dt = tstop-tstart

    h_prev = h_prev[0]
    h0 = imp_gru_cell(dt, x, h_prev[0],
                      self.lin_rx[0].weight, self.lin_rx[0].bias, self.lin_rh[0].weight,
                      self.lin_zx[0].weight, self.lin_zx[0].bias, self.lin_zh[0].weight,
                      self.lin_nx[0].weight, self.lin_nx[0].bias, self.lin_nr[0].weight, self.lin_nr[0].bias)
    h1 = imp_gru_cell(dt, h0, h_prev[1],
                      self.lin_rx[1].weight, self.lin_rx[1].bias, self.lin_rh[1].weight,
                      self.lin_zx[1].weight, self.lin_zx[1].bias, self.lin_zh[1].weight,
                      self.lin_nx[1].weight, self.lin_nx[1].bias, self.lin_nr[1].weight, self.lin_nr[1].bias)

    # Note: we return a tuple with a single element
    return (torch.stack((h0, h1)),)

def generate_fake_data(dataset_size, sequence_length, input_size, hidden_size):
  torch.manual_seed(20)
  x = torch.randn(dataset_size, sequence_length, input_size)
  y = torch.randn(dataset_size, hidden_size)
  return x,y

def test_args(comm):
  args = dict()
  args['num_data'] = 10
  args['input_size'] = 28
  args['sequence_length'] = 32
  args['print_level'] = 0
  args['nrelax'] = 1
  args['cfactor'] = 2
  args['Tf'] = float(args['sequence_length'])
  args['hidden_size'] = 20
  args['num_layers'] = 2
  args['batch_size'] = 1
  args['max_iters'] = 1
  args['max_levels'] = 1
  args['skip_downcycle'] = True
  args['local_steps'] = int(args['sequence_length']/comm.Get_size())
  args['dt'] = args['Tf'] / args['sequence_length']
  args['seed'] = 42
  return args
    
def test_args_small(comm):
  args = dict()
  args['num_data'] = 5
  args['input_size'] = 2
  args['sequence_length'] = 4
  args['print_level'] = 0
  args['nrelax'] = 1
  args['cfactor'] = 2
  args['Tf'] = float(args['sequence_length'])
  args['hidden_size'] = 3
  args['num_layers'] = 2
  args['batch_size'] = 1
  args['max_iters'] = 1
  args['max_levels'] = 1
  args['skip_downcycle'] = True
  args['local_steps'] = int(args['sequence_length']/comm.Get_size())
  args['dt'] = args['Tf'] / args['sequence_length']
  args['seed'] = 42
  return args

def get_parallel_gru(gru_model, args):
  parallel_gru = torchbraid.GRU_Parallel(comm = MPI.COMM_WORLD,
                                         basic_block = gru_model,
                                         num_steps = args['local_steps'],
                                         hidden_size = args['hidden_size'],
                                         num_layers = args['num_layers'],
                                         Tf = args['Tf'],
                                         max_fwd_levels = args['max_levels'],
                                         max_bwd_levels = args['max_levels'],
                                         max_iters = args['max_iters'])
  parallel_gru.setPrintLevel(args['print_level'])
  cfactor_dict = dict()
  cfactor_dict[-1] = args['cfactor']
  parallel_gru.setCFactor(cfactor_dict)
  parallel_gru.setSkipDowncycle(args['skip_downcycle'])
  parallel_gru.setNumRelax(1) # FCF on all levels
  parallel_gru.setFwdNumRelax(1, level=0) # F-Relax on fine grid
  parallel_gru.setBwdNumRelax(0, level=0) # F-Relax on fine grid
  return parallel_gru 

def distribute_input_data(x_global, y_global, comm):
  rank = comm.Get_rank()
  num_procs = comm.Get_size()
  x_block = [[] for _ in range(num_procs)]
  y_block = [[] for _ in range(num_procs)]
  if rank == 0:
    sz = len(x_global)
    for i in range(sz):
      x = x_global[i]
      y = y_global[i]
      x_split = torch.chunk(x, num_procs, dim = 0) 
      y_split = num_procs*[y]

      for p, (x_in, y_in) in enumerate(zip(x_split, y_split)):
        x_block[p].append(x_in)
        y_block[p].append(y_in)
      
    for p, (x, y) in enumerate(zip(x_block, y_block)):
      x_block[p] = torch.stack(x)
      y_block[p] = torch.stack(y)

  comm.barrier()
  
  x_local = comm.scatter(x_block, root=0)
  y_local = comm.scatter(y_block, root=0)
  return x_local, y_local


class TestGRULayerParallel(unittest.TestCase):
  def test_gru_serial_forward(self):
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    
    if my_rank == 0:
      self.gru_serial_forward_device('cpu')

      my_device, my_host = getDevice(comm) 
      if my_device != 'cpu':
        self.gru_serial_forward_device(my_device) 
    comm.barrier()
    
  def gru_serial_forward_device(self, device):
    """
    Tests the gru_serial implementation for functionality
    This should only be run on one processor
    """
    comm = MPI.COMM_WORLD
    args = test_args(comm)
    torch.manual_seed(args['seed'])
    gru_model = ImplicitGRUBlock(args['input_size'], args['hidden_size']).to(device)
    serial_gru = torchbraid.GRU_Serial(gru_model, args['num_layers'], args['hidden_size'], args['dt']).to(device)

    x, y = generate_fake_data(args['num_data'], args['sequence_length'], args['input_size'], args['hidden_size'], device)
    h = torch.zeros(args['num_layers'], x.size(0), args['hidden_size']).to(device)

    # Test the forward with an initial hidden state
    yhat_1 = serial_gru(x, h)
    self.assertTrue(yhat_1.shape[0] == args['num_layers'])
    self.assertTrue(yhat_1.shape[1] == args['num_data'])
    self.assertTrue(yhat_1.shape[2] == args['hidden_size'])

    # Test the forward without an initial hidden state
    yhat_2 = serial_gru(x)
    self.assertTrue(yhat_2.shape[0] == args['num_layers'])
    self.assertTrue(yhat_2.shape[1] == args['num_data'])
    self.assertTrue(yhat_2.shape[2] == args['hidden_size'])

    # Check that the outputs match
    self.assertEqual(torch.norm(yhat_1 - yhat_2), 0)

    # Check the output is as expected:
    h = torch.zeros(args['num_layers'], x.size(0), args['hidden_size']).to(device)
    for i in range(args['sequence_length']):
      h = gru_model(0,0.0,args['dt'],x[:,i,:], h)

    self.assertEqual(torch.norm(yhat_2 - h[0]), 0)
    
  def test_gru_parallel_forward(self):
    self.gru_parallel_forward_device('cpu')

    my_device, my_host = getDevice(MPI.COMM_WORLD) 
    if my_device != 'cpu':
      self.gru_parallel_forward_device(my_device) 

  def gru_parallel_forward_device(self, device):
    "Tests the gru_serial implementation for functionality"
    comm = MPI.COMM_WORLD
    args = test_args_small(comm)
    rank = comm.Get_rank()
    n_procs = comm.Get_size()
    torch.manual_seed(args['seed'])
    gru_model = ImplicitGRUBlock(args['input_size'], args['hidden_size']).to(device)

    # Set up the parallel GRU
    parallel_gru = get_parallel_gru(gru_model, args).to(device)

    if rank == 0:
      x_global, y_global = generate_fake_data(args['num_data'], args['sequence_length'], args['input_size'], args['hidden_size'])
    else:
      x_global, y_global = (None, None)

    x, y = distribute_input_data(x_global, y_global, comm)
    x = x.to(device)
    y = y.to(device)

    h = torch.zeros(args['num_layers'], x.size(0), args['hidden_size']).to(device)

    if rank == 0:
        print(f'x_global = \n{x_global}')
    print(f'rank {rank}, x = \n{x}')

    # Test the forward with an initial hidden state
    yhat_1 = parallel_gru(x, h)
    self.assertTrue(yhat_1.shape[0] == args['num_layers'])
    self.assertTrue(yhat_1.shape[1] == args['num_data'])
    self.assertTrue(yhat_1.shape[2] == args['hidden_size'])


    # Test the forward without an initial hidden state
    yhat_2 = parallel_gru(x)
    self.assertTrue(yhat_2.shape[0] == args['num_layers'])
    self.assertTrue(yhat_2.shape[1] == args['num_data'])
    self.assertTrue(yhat_2.shape[2] == args['hidden_size'])

    # Check that the outputs match
    self.assertEqual(torch.norm(yhat_1 - yhat_2), 0)

    # Run the serial case and compare results
    if rank == 0:
      # Compare to serial version
      serial_gru = torchbraid.GRU_Serial(gru_model, args['num_layers'], args['hidden_size'], args['dt']).to(device)
      yhat_serial = serial_gru(x_global)

      self.assertEqual(torch.norm(yhat_2 - yhat_serial), 0)
        
  # def test_forward_exact(self):
  #   self.forwardProp(max_levels=1,max_iters=1,sequence_length=28)

  # def test_forward_approx(self):
  #   self.forwardProp(max_levels=3,max_iters=20)

  # def test_backward_exact(self):
  #   self.backwardProp()

  # def test_backward_exact_multiple(self):
  #   self.backwardProp(applications=8)

  # def test_backward_approx(self):
  #   self.backwardProp(max_levels=3,max_iters=20,sequence_length=27,tol=1e-5)

  # # TODO: dead code?
  # # def copyParameterGradToRoot(self,m):
  # #   comm     = m.getMPIComm()
  # #   my_rank  = m.getMPIComm().Get_rank()
  # #   num_proc = m.getMPIComm().Get_size()

  # #   params = [p.grad for p in list(m.parameters())]

  # #   if len(params)==0:
  # #     return params

  # #   if my_rank==0:
  # #     for i in range(1,num_proc):
  # #       remote_p = comm.recv(source=i,tag=77)
  # #       remote_p = [p.to(device) for p in remote_p]
  # #       params.extend(remote_p)

  # #     return params
  # #   else:
  # #     params_cpu = [p.cpu() for p in params]
  # #     comm.send(params_cpu,dest=0,tag=77)
  # #     return None
  # # # end copyParametersToRoot

  # def forwardProp(self,
  #                 max_levels = 1, # for testing parallel rnn
  #                 max_iters = 1, # for testing parallel rnn
  #                 sequence_length = 28, # total number of time steps for each sequence
  #                 input_size = 28, # input size for each time step in a sequence
  #                 hidden_size = 20,
  #                 num_layers = 2,
  #                 batch_size = 1):

  #   comm      = MPI.COMM_WORLD
  #   num_procs = comm.Get_size()
  #   my_rank   = comm.Get_rank()

  #   my_device,my_host = getDevice(MPI.COMM_WORLD)

  #   Tf              = float(sequence_length)
  #   channels        = 1
  #   images          = 10
  #   image_size      = 28
  #   print_level     = 0
  #   nrelax          = 1
  #   cfactor         = 2
  #   num_batch = int(images / batch_size)

  #   if my_rank==0:
  #     with torch.no_grad():

  #       torch.manual_seed(20)
  #       serial_rnn = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
  #       serial_rnn = serial_rnn.to(my_device)
  #       num_blocks = 2 # equivalent to the num_procs variable used for parallel implementation
  #       image_all, x_block_all = preprocess_input_data_serial_test(num_blocks,num_batch,batch_size,channels,sequence_length,input_size,my_device)

  #       for i in range(1):

  #         y_serial_hn = torch.zeros(num_layers, image_all[i].size(0), hidden_size, device=my_device)
  #         y_serial_cn = torch.zeros(num_layers, image_all[i].size(0), hidden_size, device=my_device)

  #         _, (y_serial_hn, y_serial_cn) = serial_rnn(image_all[i],(y_serial_hn,y_serial_cn))
  #         y_serial_hn = y_serial_hn.cpu()
  #         y_serial_cn = y_serial_cn.cpu()
  #   # compute serial solution

  #   # wait for serial processor
  #   comm.barrier()

  #   basic_block_parallel = ImplicitGRUBlock(input_size, hidden_size)
  #   num_procs = comm.Get_size()

  #   # preprocess and distribute input data
  #   ###########################################
  #   x_block = preprocess_distribute_input_data_parallel(my_rank,num_procs,num_batch,batch_size,channels,sequence_length,input_size,comm,my_device)

  #   num_steps = x_block[0].shape[1]
  #   # GRU_parallel.py -> GRU_Parallel() class
  #   parallel_rnn = torchbraid.GRU_Parallel(comm,
  #                                          basic_block_parallel,
  #                                          num_steps,
  #                                          hidden_size,
  #                                          num_layers,
  #                                          Tf,
  #                                          max_fwd_levels=max_levels,
  #                                          max_bwd_levels=max_levels,
  #                                          max_iters=max_iters)
  #   parallel_rnn.to(my_device)

  #   parallel_rnn.setPrintLevel(print_level)
  #   parallel_rnn.setSkipDowncycle(True)
  #   parallel_rnn.setCFactor(cfactor)
  #   parallel_rnn.setNumRelax(nrelax)

  #   for i in range(1):

  #     y_parallel_hn,y_parallel_cn = parallel_rnn(x_block[i])

  #     comm.barrier()

  #     # send the final inference step to root
  #     if comm.Get_size()>1 and my_rank == comm.Get_size()-1:
  #       comm.send(y_parallel_hn.cpu(),0)
  #       comm.send(y_parallel_cn.cpu(),0)

  #     if my_rank==0:
  #       # recieve the final inference step
  #       if comm.Get_size()>1:
  #         parallel_hn = comm.recv(source=comm.Get_size()-1)
  #         parallel_cn = comm.recv(source=comm.Get_size()-1)
  #       else:
  #         parallel_hn = y_parallel_hn.cpu()
  #         parallel_cn = y_parallel_cn.cpu()

  #       print('cn values = ',torch.norm(y_serial_cn-parallel_cn).item()/torch.norm(y_serial_cn).item())
  #       print('hn values = ',torch.norm(y_serial_hn-parallel_hn).item()/torch.norm(y_serial_hn).item())
  #       self.assertTrue(torch.norm(y_serial_cn-parallel_cn).item()/torch.norm(y_serial_cn).item()<1e-6,'check cn')
  #       self.assertTrue(torch.norm(y_serial_hn-parallel_hn).item()/torch.norm(y_serial_hn).item()<1e-6,'check hn')
  # # forwardProp

  # def backwardProp(self,
  #                  max_levels = 1, # for testing parallel rnn
  #                  max_iters = 1, # for testing parallel rnn
  #                  sequence_length = 6, # total number of time steps for each sequence
  #                  input_size = 28, # input size for each time step in a sequence
  #                  hidden_size = 20,
  #                  num_layers = 1,
  #                  batch_size = 1,
  #                  tol=1e-6,
  #                  applications=1):

  #   comm      = MPI.COMM_WORLD
  #   num_procs = comm.Get_size()
  #   my_rank   = comm.Get_rank()

  #   my_device,my_host = getDevice(MPI.COMM_WORLD)

  #   Tf              = float(sequence_length)
  #   channels        = 1
  #   images          = 10
  #   image_size      = 28
  #   print_level     = 0
  #   nrelax          = 1
  #   cfactor         = 2
  #   num_batch = int(images / batch_size)

  #   # wait for serial processor
  #   comm.barrier()

  #   num_procs = comm.Get_size()

  #   # preprocess and distribute input data
  #   x_block = preprocess_distribute_input_data_parallel(my_rank,num_procs,num_batch,batch_size,channels,sequence_length,input_size,comm,my_device)

  #   num_steps = x_block[0].shape[1]

  #   basic_block_parallel = ImplicitGRUBlock(input_size, hidden_size)
  #   parallel_rnn = torchbraid.GRU_Parallel(comm,
  #                                          basic_block_parallel,
  #                                          num_steps,
  #                                          hidden_size,
  #                                          num_layers,
  #                                          Tf,
  #                                          max_fwd_levels=max_levels,
  #                                          max_bwd_levels=max_levels,
  #                                          max_iters=max_iters)
  #   parallel_rnn.to(my_device)
  #   parallel_rnn.setPrintLevel(print_level)
  #   parallel_rnn.setSkipDowncycle(True)
  #   parallel_rnn.setCFactor(cfactor)
  #   parallel_rnn.setNumRelax(nrelax)

  #   torch.manual_seed(20)
  #   rand_w = torch.randn([1,x_block[0].size(0),hidden_size],device=my_device)

  #   for i in range(applications):
  #     h_0 = torch.zeros(num_layers, x_block[i].size(0), hidden_size,requires_grad=True,device=my_device)
  #     c_0 = torch.zeros(num_layers, x_block[i].size(0), hidden_size,requires_grad=True,device=my_device)

  #     with torch.enable_grad():
  #       y_parallel_hn,y_parallel_cn = parallel_rnn(x_block[i],(h_0,c_0))

  #     comm.barrier()

  #     w_h = torch.zeros(y_parallel_hn.shape,device=my_device)
  #     w_c = torch.zeros(y_parallel_hn.shape,device=my_device)

  #     w_h[-1,:,:] = rand_w

  #     y_parallel_hn.backward(w_h)

  #     if i<applications-1:
  #       with torch.no_grad():
  #         for p in parallel_rnn.parameters():
  #           p += p.grad
  #       parallel_rnn.zero_grad()

  #   # compute serial solution
  #   #############################################

  #   if my_rank==0:

  #     torch.manual_seed(20)
  #     serial_rnn = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
  #     serial_rnn.to(my_device)
  #     image_all, x_block_all = preprocess_input_data_serial_test(num_procs,num_batch,batch_size,channels,sequence_length,input_size,my_device)

  #     for i in range(applications):
  #       y_serial_hn_0 = torch.zeros(num_layers, image_all[i].size(0), hidden_size,requires_grad=True,device=my_device)
  #       y_serial_cn_0 = torch.zeros(num_layers, image_all[i].size(0), hidden_size,requires_grad=True,device=my_device)

  #       with torch.enable_grad():
  #         q, (y_serial_hn, y_serial_cn) = serial_rnn(image_all[i],(y_serial_hn_0,y_serial_cn_0))

  #       w_q = torch.zeros(q.shape,device=my_device)
  #       w_q[:,-1,:] = rand_w.detach().clone()

  #       q.backward(w_q)

  #       if i<applications-1:
  #         with torch.no_grad():
  #           for p in serial_rnn.parameters():
  #             p += p.grad
  #         serial_rnn.zero_grad()
  #   # end if my_rank

  #   # now check the answers
  #   #############################################

  #   # send the final inference step to root
  #   if comm.Get_size()>1 and my_rank == comm.Get_size()-1:
  #     comm.send(y_parallel_hn.cpu(),0)
  #     comm.send(y_parallel_cn.cpu(),0)

  #   if my_rank==0:
  #     y_serial_cn = y_serial_cn.cpu()
  #     y_serial_hn = y_serial_hn.cpu()

  #     if comm.Get_size()>1:
  #       # recieve the final inference step
  #       parallel_hn = comm.recv(source=comm.Get_size()-1)
  #       parallel_cn = comm.recv(source=comm.Get_size()-1)
  #     else:
  #       parallel_hn = y_parallel_hn.cpu()
  #       parallel_cn = y_parallel_cn.cpu()

  #     print('\n\n')
  #     print(torch.norm(y_serial_cn-parallel_cn).item()/torch.norm(y_serial_cn).item(),'forward cn')
  #     print(torch.norm(y_serial_hn-parallel_hn).item()/torch.norm(y_serial_hn).item(),'forward hn')
  #     sys.stdout.flush()

  #     self.assertTrue(torch.norm(y_serial_cn-parallel_cn)/torch.norm(y_serial_cn)<tol,'cn value')
  #     self.assertTrue(torch.norm(y_serial_hn-parallel_hn)/torch.norm(y_serial_hn)<tol,'hn value')

  #     print(torch.norm(h_0.grad.cpu()-y_serial_hn_0.grad.cpu()).item(),'back soln hn')
  #     print(torch.norm(c_0.grad.cpu()-y_serial_cn_0.grad.cpu()).item(),'back soln cn')
  #     self.assertTrue(torch.norm(h_0.grad.cpu()-y_serial_hn_0.grad.cpu()).item()<tol)
  #     self.assertTrue(torch.norm(c_0.grad.cpu()-y_serial_cn_0.grad.cpu()).item()<tol)

  #     root_grads = [p.grad for p in serial_rnn.parameters()]
  #     root_grads = [r.cpu() for r in root_grads]
  #   else:
  #     root_grads = None

  #   ref_grads = comm.bcast(root_grads,root=0)
  #   for pa_grad,pb in zip(ref_grads,parallel_rnn.parameters()):
  #     if torch.norm(pa_grad).item()==0.0:
  #       print(my_rank,torch.norm(pa_grad.cpu()-pb.grad.cpu()).item().item(),'param grad')
  #       self.assertTrue(torch.norm(pa_grad.cpu()-pb.grad.cpu()).item()<1e1*tol,'param grad')
  #     else:
  #       print(my_rank,torch.norm(pa_grad.cpu()-pb.grad.cpu()).item()/torch.norm(pa_grad.cpu()).item(),'param grad')
  #       self.assertTrue(torch.norm(pa_grad.cpu()-pb.grad.cpu()).item()/torch.norm(pa_grad.cpu()).item()<1e1*tol,'param grad')
  # # forwardProp

if __name__ == '__main__':
  unittest.main()
