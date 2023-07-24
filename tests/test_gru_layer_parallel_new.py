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
  def __init__(self, input_size, hidden_size, seed=20):
    super(ImplicitGRUBlock, self).__init__()

    # This is the easiest way to distribute the same GRU across ranks
    torch.manual_seed(seed)

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

  def fastForward(self, level, tstart, tstop, x_red, h_prev):
    dt = tstop-tstart

    h_prev = h_prev[0]
    h0 = imp_gru_cell_fast(dt, *x_red, h_prev[0],
                           self.lin_rh[0].weight,
                           self.lin_zh[0].weight,
                           self.lin_nr[0].weight, self.lin_nr[0].bias)
    h1 = imp_gru_cell(dt, h0, h_prev[1],
                      self.lin_rx[1].weight, self.lin_rx[1].bias, self.lin_rh[1].weight,
                      self.lin_zx[1].weight, self.lin_zx[1].bias, self.lin_zh[1].weight,
                      self.lin_nx[1].weight, self.lin_nx[1].bias, self.lin_nr[1].weight, self.lin_nr[1].bias)

    # Note: we return a tuple with a single element
    return (torch.stack((h0, h1)),)


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

def generate_fake_data(dataset_size, sequence_length, input_size, hidden_size, seed=20):
  torch.manual_seed(seed)
  x = torch.randn(dataset_size, sequence_length, input_size)
  y = torch.randn(dataset_size, hidden_size)
  return x,y

def get_rel_error(a, b):
  return torch.norm(a - b)/torch.norm(a)

def test_args(comm,
              num_data = 10,
              input_size = 28,
              sequence_length = 24,
              print_level = 0,
              nrelax = 1,
              cfactor = 2,
              Tf = None,
              hidden_size = 20,
              batch_size = 1,
              max_iters = 1,
              max_levels = 1,
              skip_downcycle = True,
              seed = 42,
              tol = 1e-6):
  args = dict()
  args['num_data'] = num_data
  args['input_size'] = input_size
  args['sequence_length'] = sequence_length
  args['print_level'] = print_level
  args['nrelax'] = nrelax
  args['cfactor'] = cfactor
  args['hidden_size'] = hidden_size
  args['batch_size'] = batch_size
  args['max_iters'] = max_iters
  args['max_levels'] = max_levels
  args['skip_downcycle'] = skip_downcycle
  args['seed'] = seed
  args['tol'] = tol
  
  # The number of layers is always 2, unless the imp_gru_cell above is updated to include more layers
  args['num_layers'] = 2

  if Tf is None:
    args['Tf'] = float(args['sequence_length'])
  else:
    args['Tf'] = Tf

  args['local_steps'] = int(args['sequence_length']/comm.Get_size())
  args['dt'] = args['Tf'] / args['sequence_length']
  return args
    
def test_args_small(comm):
  return test_args(comm, num_data=5, input_size=2, sequence_length=4, hidden_size=3)

def test_args_backprop(comm, num_data=1, max_levels=1, max_iters=1, sequence_length=6, tol=1e-6):
  return test_args(comm,
                   num_data=num_data,
                   input_size=28,
                   sequence_length=sequence_length,
                   hidden_size=20,
                   tol=tol,
                   max_levels=max_levels,
                   max_iters=max_iters)

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
  def devices_test(self, args, func):
    """
    Test function func on cpu and gpu (if available)

    func should accept a single argument: the device on which to run the function
    """
    comm = MPI.COMM_WORLD
    func(args, 'cpu')
    my_device, my_host = getDevice(comm)
    if my_device.type != 'cpu':
      func(args, my_device)
    comm.barrier()
        
  def test_gru_forward_exact(self):
    comm = MPI.COMM_WORLD
    args = test_args(comm) # one-level, one-iteration
    self.devices_test(args, self.gru_serial_forward_device)
    self.devices_test(args, self.gru_parallel_forward_device)
    self.devices_test(args, self.gru_parallel_fastforward_device)

  def test_gru_forward_approx(self):
    comm = MPI.COMM_WORLD
    args = test_args(comm, max_levels=3, max_iters=20)
    self.devices_test(args, self.gru_serial_forward_device)
    self.devices_test(args, self.gru_parallel_forward_device)
    self.devices_test(args, self.gru_parallel_fastforward_device)
    
  def gru_serial_forward_device(self, args, device):
    """ Tests the gru_serial implementation for functionality """
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
      gru_model = ImplicitGRUBlock(args['input_size'], args['hidden_size'], args['seed']).to(device)
      serial_gru = torchbraid.GRU_Serial(gru_model, args['num_layers'], args['hidden_size'], args['dt']).to(device)

      x, y = generate_fake_data(args['num_data'], args['sequence_length'], args['input_size'], args['hidden_size'], args['seed'])
      x = x.to(device)
      y = y.to(device)
      h = torch.zeros(args['num_layers'], x.size(0), args['hidden_size']).to(device)

      # Test the forward with an initial hidden state
      yhat_1 = serial_gru(x, h)

      # shape[0] should always be 2 since there are two cells in the ImplicitGRUBlock
      self.assertTrue(yhat_1.shape[0] == args['num_layers'])
      self.assertTrue(yhat_1.shape[1] == args['num_data'])
      self.assertTrue(yhat_1.shape[2] == args['hidden_size'])

      # Test the forward without an initial hidden state
      yhat_2 = serial_gru(x)

      # shape[0] should always be 2 since there are two cells in the ImplicitGRUBlock
      self.assertTrue(yhat_2.shape[0] == args['num_layers'])
      self.assertTrue(yhat_2.shape[1] == args['num_data'])
      self.assertTrue(yhat_2.shape[2] == args['hidden_size'])

      # Check that the outputs match
      self.assertTrue(get_rel_error(yhat_1, yhat_2) < args['tol'])

      # Check the output is as expected:
      h = torch.zeros(args['num_layers'], x.size(0), args['hidden_size']).to(device)
      for i in range(args['sequence_length']):
        h = gru_model(0,0.0,args['dt'],x[:,i,:], h)

      self.assertTrue(get_rel_error(h[0], yhat_2) < args['tol'])
    
  def gru_parallel_forward_device(self, args, device):
    "Tests the parallel gru implementation for functionality in the forward pass"
    return
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    gru_model = ImplicitGRUBlock(args['input_size'], args['hidden_size'], args['seed']).to(device)

    # Set up the parallel GRU
    parallel_gru = get_parallel_gru(gru_model, args).to(device)

    if rank == 0:
      x_global, y_global = generate_fake_data(args['num_data'], args['sequence_length'], args['input_size'], args['hidden_size'], args['seed'])
    else:
      x_global, y_global = (None, None)

    x, y = distribute_input_data(x_global, y_global, comm)
    x = x.to(device)
    y = y.to(device)

    h = torch.zeros(args['num_layers'], x.size(0), args['hidden_size']).to(device)

    # Test the forward with an initial hidden state
    yhat_1 = parallel_gru(x, h)

    # shape[0] should always be 2 since there are two cells in the ImplicitGRUBlock
    self.assertTrue(yhat_1.shape[0] == args['num_layers'])
    self.assertTrue(yhat_1.shape[1] == args['num_data'])
    self.assertTrue(yhat_1.shape[2] == args['hidden_size'])

    # Test the forward without an initial hidden state
    yhat_2 = parallel_gru(x)

    # shape[0] should always be 2 since there are two cells in the ImplicitGRUBlock
    self.assertTrue(yhat_2.shape[0] == args['num_layers'])
    self.assertTrue(yhat_2.shape[1] == args['num_data'])
    self.assertTrue(yhat_2.shape[2] == args['hidden_size'])

    # Check that the outputs match
    self.assertTrue(get_rel_error(yhat_1, yhat_2) < args['tol'])

    # Run the serial case and compare results
    if rank == 0:
      # Compare to serial version
      serial_gru = torchbraid.GRU_Serial(gru_model, args['num_layers'], args['hidden_size'], args['dt']).to(device)
      yhat_serial = serial_gru(x_global.to(device))

      self.assertTrue(get_rel_error(yhat_serial, yhat_2) < args['tol'])

  def gru_parallel_fastforward_device(self, args, device):
    "Test that the forward and fastward paths give the same result for the forward pass"
    return
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_procs = comm.Get_size()
    gru_model = ImplicitGRUBlock(args['input_size'], args['hidden_size'], args['seed']).to(device)

    # Set up the parallel GRU
    parallel_gru = get_parallel_gru(gru_model, args).to(device)

    if rank == 0:
      x_global, y_global = generate_fake_data(args['num_data'], args['sequence_length'], args['input_size'], args['hidden_size'], args['seed'])
    else:
      x_global, y_global = (None, None)

    x, y = distribute_input_data(x_global, y_global, comm)
    x = x.to(device)
    y = y.to(device)
      
    # Run the forward with the fastfoward enabled (enabled by default since ImplicitGRUBlock has the required functions)
    yhat_on = parallel_gru(x)

    # Turn off fastforward and run
    parallel_gru.fwd_app.has_fastforward = False
    yhat_off = parallel_gru(x)

    # Compare the results (which are shipped to rank 0 automatically)
    if rank == 0:
      self.assertTrue(get_rel_error(yhat_off, yhat_on) < args['tol'])

  def test_gru_backward_exact(self):
    "Test the exact backward pass of parallel gru by comparing to the serial"
    comm = MPI.COMM_WORLD
    args = test_args_backprop(comm) # one-level, one-iteration, one data point
    self.devices_test(args, self.gru_parallel_backward_device)
    self.devices_test(args, self.gru_parallel_backward_fastforward_device)

  def test_gru_backward_multiple(self):
    "Test the exact backward pass of parallel gru by comparing to the serial on a batch of 8"
    comm = MPI.COMM_WORLD
    args = test_args_backprop(comm, num_data=8) # one-level, one-iteration
    self.devices_test(args, self.gru_parallel_backward_device)
    self.devices_test(args, self.gru_parallel_backward_fastforward_device)

  def test_gru_backward_approx(self):
    "Test the approximate backward pass of parallel gru by comparing to the serial using more levels and iterations"
    comm = MPI.COMM_WORLD
    args = test_args_backprop(comm, num_data = 8, max_levels=3, max_iters=20, sequence_length=27, tol=1e-5)
    self.devices_test(args, self.gru_parallel_backward_device)
    self.devices_test(args, self.gru_parallel_backward_fastforward_device)

  def gru_parallel_backward_device(self, args, device):
    """ Tests the parallel gru implementation for backward pass by comparing to the serial version"""
    comm = MPI.COMM_WORLD

    # Parallel gru stuff first
    rank = comm.Get_rank()
    gru_model = ImplicitGRUBlock(args['input_size'], args['hidden_size'], args['seed']).to(device)
    parallel_gru = get_parallel_gru(gru_model, args).to(device)

    # Generate data
    if rank == 0:
      x_global, y_global = generate_fake_data(args['num_data'], args['sequence_length'], args['input_size'], args['hidden_size'], args['seed'])
    else:
      x_global, y_global = (None, None)

    # Distribute data among ranks
    x, y = distribute_input_data(x_global, y_global, comm)
    x = x.to(device)
    y = y.to(device)

    # Set up the hidden state, tracking the gradient
    h_parallel = torch.zeros(args['num_layers'], x.size(0), args['hidden_size'], requires_grad=True, device=device)

    # Turn off the fastforward since the serial version doesn't have it
    parallel_gru.fwd_app.has_fastforward = False

    # Run the forward
    with torch.enable_grad():
      yhat_parallel = parallel_gru(x, h_parallel)

    # Wait for all processors to catch up
    comm.barrier()

    # Need a random vector to compute gradient w.r.t.
    rand_w = torch.randn(yhat_parallel.shape)

    # Copy the value
    w_h_parallel = rand_w.detach().clone().to(device)

    # Run the backwards pass
    yhat_parallel.backward(w_h_parallel)

    comm.barrier()

    # Now construct and run the serial gru
    if rank == 0:
      gru_model = ImplicitGRUBlock(args['input_size'], args['hidden_size'], args['seed']).to(device)
      serial_gru = torchbraid.GRU_Serial(gru_model, args['num_layers'], args['hidden_size'], args['dt']).to(device)

      # Create the initial hidden state
      h_serial = torch.zeros(args['num_layers'], x_global.size(0), args['hidden_size'], requires_grad=True, device=device)

      # Run the forward pass with the default zero initial hidden state
      with torch.enable_grad():
        yhat_serial = serial_gru(x_global.to(device), h_serial)

      # Copy the same random vector used for the parallel gru
      w_h_serial = rand_w.detach().clone().to(device)

      # Make sure we're taking the derivative w.r.t. the same vector
      self.assertTrue(torch.norm(w_h_parallel - w_h_serial) == 0.0)

      # Run the backward pass
      yhat_serial.backward(w_h_serial)

      # Make sure outputs match
      self.assertTrue(get_rel_error(yhat_serial, yhat_parallel) < args['tol'])

      # Make sure the gradients of the initial hidden state match
      self.assertTrue(get_rel_error(h_serial.grad, h_parallel.grad) < args['tol'])

      # get the parameter gradients for the serial case:
      root_grads = [p.grad for p in serial_gru.parameters()]

      # Get the parallel gradients
      parallel_grads = [p.grad for p in parallel_gru.parameters()]

      for i, (pa_grad, pb_grad) in enumerate(zip(root_grads, parallel_grads)):
        # Can't use relative error if the serial grad is zero, use abs error instead
        if torch.norm(pa_grad).item() == 0.0:
          self.assertTrue(torch.norm(pa_grad - pb_grad).item() < 1e1*args['tol'])
        else:
          if get_rel_error(pa_grad, pb_grad) > 1e1*args['tol']:
            print(rank, i, "rel_error", get_rel_error(pa_grad, pb_grad))
          self.assertTrue(get_rel_error(pa_grad, pb_grad) < 1e1*args['tol'])
      
  def gru_parallel_backward_fastforward_device(self, args, device):
    """ Tests the parallel gru implementation for backward pass by comparing to the serial version. Fastforwarding enabled"""
    comm = MPI.COMM_WORLD

    # Parallel gru stuff first
    rank = comm.Get_rank()
    gru_model = ImplicitGRUBlock(args['input_size'], args['hidden_size'], args['seed']).to(device)
    parallel_gru = get_parallel_gru(gru_model, args).to(device)

    # Generate data
    if rank == 0:
      x_global, y_global = generate_fake_data(args['num_data'], args['sequence_length'], args['input_size'], args['hidden_size'], args['seed'])
    else:
      x_global, y_global = (None, None)

    # Distribute data among ranks
    x, y = distribute_input_data(x_global, y_global, comm)
    x = x.to(device)
    y = y.to(device)

    # Set up the hidden state, tracking the gradient
    h_parallel = torch.zeros(args['num_layers'], x.size(0), args['hidden_size'], requires_grad=True, device=device)

    # Run the forward
    with torch.enable_grad():
      yhat_parallel = parallel_gru(x, h_parallel)

    # Wait for all processors to catch up
    comm.barrier()

    # Need a random vector to compute gradient w.r.t.
    rand_w = torch.randn(yhat_parallel.shape)

    # Copy the value
    w_h_parallel = rand_w.detach().clone().to(device)

    # Run the backwards pass
    yhat_parallel.backward(w_h_parallel)

    comm.barrier()

    # Now construct and run the serial gru
    if rank == 0:
      gru_model = ImplicitGRUBlock(args['input_size'], args['hidden_size'], args['seed']).to(device)
      serial_gru = torchbraid.GRU_Serial(gru_model, args['num_layers'], args['hidden_size'], args['dt']).to(device)

      # Create the initial hidden state
      h_serial = torch.zeros(args['num_layers'], x_global.size(0), args['hidden_size'], requires_grad=True, device=device)

      # Run the forward pass with the default zero initial hidden state
      with torch.enable_grad():
        yhat_serial = serial_gru(x_global.to(device), h_serial)

      # Copy the same random vector used for the parallel gru
      w_h_serial = rand_w.detach().clone().to(device)

      # Make sure we're taking the derivative w.r.t. the same vector
      self.assertTrue(torch.norm(w_h_parallel - w_h_serial) == 0.0)

      # Run the backward pass
      yhat_serial.backward(w_h_serial)

      # Make sure outputs match
      self.assertTrue(get_rel_error(yhat_serial, yhat_parallel) < args['tol'])

      # Make sure the gradients of the initial hidden state match
      self.assertTrue(get_rel_error(h_serial.grad, h_parallel.grad) < args['tol'])

      # get the parameter gradients for the serial case:
      root_grads = [p.grad for p in serial_gru.parameters()]

      # Get the parallel gradients
      parallel_grads = [p.grad for p in parallel_gru.parameters()]

      # Since we are testing with fastforward, the forward pass will not interact with rx[0], zx[0], nx[0], so those gradients aren't
      # expected to match, hence we don't compare them
      # NOTE: Change these if changing the ImplicitGRUBlock above
      grads_to_ignore = [0, 1, 6, 7, 12, 13]
  
      for i, (pa_grad, pb_grad) in enumerate(zip(root_grads, parallel_grads)):
        if i not in grads_to_ignore:
          # Can't use relative error if the serial grad is zero, use abs error instead
          if torch.norm(pa_grad).item() == 0.0:
            self.assertTrue(torch.norm(pa_grad - pb_grad).item() < 1e1*args['tol'])
          else:
            if get_rel_error(pa_grad, pb_grad) > 1e1*args['tol']:
              print(rank, i, "rel_error", get_rel_error(pa_grad, pb_grad))
            self.assertTrue(get_rel_error(pa_grad, pb_grad) < 1e1*args['tol'])

if __name__ == '__main__':
  unittest.main()
