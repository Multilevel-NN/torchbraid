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
from torchbraid.utils import l2_reg, getDevice

import faulthandler
faulthandler.enable()

from mpi4py import MPI

image_width = 5
ker_width = 3
target_size = 1

class OpenLayer(nn.Module):
  def __init__(self,channels):
    super(OpenLayer, self).__init__()
    self.conv = nn.Conv2d(3,channels,ker_width,padding=1)

  def forward(self, x):
    return F.relu(self.conv(x))
# end layer

class CloseLayer(nn.Module):
  def __init__(self,channels):
    super(CloseLayer, self).__init__()
    self.fc = nn.Linear(channels*image_width*image_width, target_size)

  def forward(self, x):
    x = torch.flatten(x, 1)
    output = self.fc(x)
    return output
# end layer


class StepLayer(nn.Module):
  def __init__(self,channels):
    super(StepLayer, self).__init__()
    self.conv1 = nn.Conv2d(channels,channels,ker_width,padding=1)

  def forward(self, x):
    return F.relu(self.conv1(x))
# end layer

class ParallelNet(nn.Module):
  def __init__(self,channels=4,local_steps=2,Tf=1.0,max_levels=1,max_iters=1,print_level=0):
    super(ParallelNet, self).__init__()

    self.rank = MPI.COMM_WORLD.Get_rank()
    self.numprocs = MPI.COMM_WORLD.Get_size()

    self.channels = channels
    step_layer = lambda: StepLayer(channels)

    m,h=getDevice(MPI.COMM_WORLD)
    
    self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD,step_layer,local_steps*self.numprocs,Tf,max_fwd_levels=max_levels,max_bwd_levels=max_levels,max_iters=max_iters)
    self.parallel_nn.setPrintLevel(print_level)
    self.parallel_nn.setCFactor(4)
    self.o = self.parallel_nn.comp_op() # get tool to build up composition neural networks

    # in this case, because OpenLayer/CloseLayer are classes, these return None on processors
    # away from rank==0...this might be too cute
    self.open_nn  = self.o(OpenLayer,channels)
    self.close_nn = self.o(CloseLayer,channels)

  def forward(self, x):
    o_ = self.o

    # here o_ is ensuring the gradients are handled yet no code is executed on rank!=0
    x = o_(self.open_nn,x)
    x = self.parallel_nn(x) 
    x = o_(self.close_nn,x)

    return x

  def copyParameterGradToRoot(self,device):
    # this will copy in a way consistent with the SerialNet
    comm     = self.parallel_nn.getMPIComm()
    my_rank  = self.parallel_nn.getMPIComm().Get_rank()
    num_proc = self.parallel_nn.getMPIComm().Get_size()
 
    params = [p.grad for p in list(self.parallel_nn.parameters())]

    if len(params)==0:
      return params

    if my_rank==0:
      for i in range(1,num_proc):
        remote_p = comm.recv(source=i,tag=77)
        remote_p = [p.to(device) for p in remote_p]
        params.extend(remote_p)

      return params + [p.grad for p in list(self.open_nn.parameters())] \
                    + [p.grad for p in list(self.close_nn.parameters())]
    else:
      params_cpu = [p.cpu() for p in params]
      comm.send(params_cpu,dest=0,tag=77)
      return None
  # end copyParametersToRoot
# end ParallelNet

class SerialNet(nn.Module):
  def __init__(self,serial_nn,open_nn,close_nn):
    super(SerialNet, self).__init__()

    self.serial_nn = serial_nn
    self.open_nn   = open_nn
    self.close_nn  = close_nn
 
  def forward(self, x):
    x = self.open_nn(x)
    x = self.serial_nn(x)
    x = self.close_nn(x)

    return x
# end SerialNet 

class TestTorchBraid(unittest.TestCase):

  def test_l2(self):
    my_rank = MPI.COMM_WORLD.Get_rank()
    procs = MPI.COMM_WORLD.Get_size()

    my_device,my_host = getDevice(MPI.COMM_WORLD)

    images   =  2
    channels = 1
    image_size = image_width
    data = torch.randn(images,3,image_size,image_size,device=my_device) 
    target = torch.randn(images,target_size,device=my_device)

    parallel_net = ParallelNet(channels=channels)
    parallel_net = parallel_net.to(my_device)

    # build and run the serial verson
    serial_layers = parallel_net.parallel_nn.buildSequentialOnRoot()
    if my_rank==0:
      serial_net = SerialNet(serial_layers,parallel_net.open_nn,parallel_net.close_nn)
      serial_net = serial_net.to(my_device)

      s_output = serial_net(data)
      s_loss = l2_reg(serial_net)

      print('serial value',s_loss)
      print('serial grad size',len(list(serial_net.parameters())))
    #######################

    MPI.COMM_WORLD.Barrier()

    parallel_net.eval()
    p_output = parallel_net(data)
    p_loss = l2_reg(parallel_net,MPI.COMM_WORLD)

    if my_rank==0:
      # Note that this is being computed (by default) in 32-bit floating
      # point. The sorting done in the l2 function is a result of this and
      # there are considerable failures if this is not sorted (e.g. hits
      # roundoff more frequently)
 
      self.assertTrue(s_loss>0.0)
      self.assertTrue(p_loss>0.0)
      self.assertTrue(abs(p_loss-s_loss)/s_loss <= 3e-7)

  def test_composite(self):
    my_rank = MPI.COMM_WORLD.Get_rank()
    procs = MPI.COMM_WORLD.Get_size()

    my_device,my_host = getDevice(MPI.COMM_WORLD)

    criterion = nn.MSELoss()
    criterion = criterion.to(my_device)

    images   =  2
    channels = 1
    image_size = image_width
    data = torch.randn(images,3,image_size,image_size,device=my_device) 
    target = torch.randn(images,target_size,device=my_device)

    parallel_net = ParallelNet(channels=channels)
    parallel_net = parallel_net.to(my_device)

    # build and run the serial verson
    serial_layers = parallel_net.parallel_nn.buildSequentialOnRoot()
    if my_rank==0:
      serial_net   = SerialNet(serial_layers,parallel_net.open_nn,parallel_net.close_nn)
      serial_net = serial_net.to(my_device)
      serial_net.train()
      serial_net.zero_grad()

      s_output = serial_net(data)
      s_loss = criterion(s_output, target) + l2_reg(serial_net)
      s_loss.backward()

      print('serial value',s_loss)
      print('serial grad size',len(list(serial_net.parameters())))
    #######################

    MPI.COMM_WORLD.Barrier()

    parallel_net.train()
    parallel_net.zero_grad()

    # do forward propagation
    p_output = parallel_net(data)

    # here o_ is ensuring the gradients are handled yet no code is executed on rank!=0
    o_ = parallel_net.o
    p_loss = o_(criterion,p_output, target) + l2_reg(parallel_net,MPI.COMM_WORLD)
    p_loss.backward()

    MPI.COMM_WORLD.Barrier()

    p_grads = parallel_net.copyParameterGradToRoot(my_device)
    if my_rank==0:
      s_grads = [p.grad for p in list(serial_net.parameters())]

      self.assertTrue(torch.norm(s_loss)>0.0)
      self.assertTrue(torch.norm(p_loss)>0.0)
      print('loss error: {} ?= {} (rel diff = {})'.format(p_loss,s_loss,(p_loss-s_loss)/s_loss))

      for s_grad,p_grad in zip(s_grads,p_grads):
        val = torch.norm(s_grad-p_grad).item()
        # check the error conditions for the gradient of the parameters
        print('GRAD ******** error grad in {}'.format(val))
        sys.stdout.flush()
        if val>1e-15:
          print(s_grad)
          print(p_grad)
          print(s_grad-p_grad)
        self.assertTrue(val<=1e-7)
    #self.assertTrue(False)
  # end test_linearNet_Exact
  

if __name__ == '__main__':
  #torch.set_default_dtype(torch.float64)
  unittest.main()
