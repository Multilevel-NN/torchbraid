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

class ExecLP:
  def __init__(self,rank):
    self.my_rank = rank

  def __call__(self,op,*args):
    if self.my_rank==0:
      return op(*args)

    value = torch.zeros(1)
    for a in args:
      value += torch.norm(a)
    return torch.zeros(1)*value

class ParallelNet(nn.Module):
  def __init__(self,channels=4,local_steps=2,Tf=1.0,max_levels=1,max_iters=1,print_level=0):
    super(ParallelNet, self).__init__()

    self.rank = MPI.COMM_WORLD.Get_rank()
    self.l = ExecLP(self.rank)

    self.channels = channels
    step_layer = lambda: StepLayer(channels)
    
    self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD,step_layer,local_steps,Tf,max_levels=max_levels,max_iters=max_iters)
    self.parallel_nn.setPrintLevel(print_level)
    self.parallel_nn.setCFactor(4)

    if self.rank==0:
      self.open_nn  = OpenLayer(channels)
      self.close_nn = CloseLayer(channels)
    else:
      self.open_nn  = None
      self.close_nn = None

  def forward(self, x):
    l_ = self.l

    x = l_(self.open_nn,x)
    x = self.parallel_nn(x)
    x = l_(self.close_nn,x)

    return x

  def copyParameterGradToRoot(self):
    # this will copy in a way consistent with the SerialNet
    comm     = self.parallel_nn.getMPIData().getComm()
    my_rank  = self.parallel_nn.getMPIData().getRank()
    num_proc = self.parallel_nn.getMPIData().getSize()
 
    params = [p.grad for p in list(self.parallel_nn.parameters())]

    if len(params)==0:
      return params

    if my_rank==0:
      for i in range(1,num_proc):
        remote_p = comm.recv(source=i,tag=77)
        params.extend(remote_p)

      return params + [p.grad for p in list(self.open_nn.parameters())] \
                    + [p.grad for p in list(self.close_nn.parameters())]
    else:
      comm.send(params,dest=0,tag=77)
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
  def test_composite(self):
    my_rank = MPI.COMM_WORLD.Get_rank()
    procs = MPI.COMM_WORLD.Get_size()

    criterion = nn.MSELoss()

    images   =  2
    channels = 1
    image_size = image_width
    data = torch.randn(images,3,image_size,image_size) 
    target = torch.randn(images,target_size)

    parallel_net = ParallelNet(channels=channels)

    # build and run the serial verson
    serial_layers = parallel_net.parallel_nn.buildSequentialOnRoot()
    if my_rank==0:
      serial_net   = SerialNet(serial_layers,parallel_net.open_nn,parallel_net.close_nn)

      s_output = serial_net(data)
      s_loss = criterion(s_output, target)
      s_loss.backward()

      print('serial value',s_loss)
      print('serial grad size',len(list(serial_net.parameters())))
      #print('serial grad')
      #for p in serial_net.parameters():
      #  print(p.grad)
      print('-----------------------------------')
    #######################

    MPI.COMM_WORLD.Barrier()

    parallel_net.train()
    parallel_net.zero_grad()

    l_ = parallel_net.l
    p_output = parallel_net(data)
    p_loss = l_(criterion,p_output, target)
    p_loss.backward()

    for r in range(procs):
      if r==my_rank:
        print('para %02d value' % my_rank,p_loss)
        print('para %02d grad size' % my_rank,len(list(parallel_net.parameters())))
        #print('para %02d grad' % my_rank)
        #for p in parallel_net.parameters():
        #  print(p.grad)
        print('-----------------------------------')
      MPI.COMM_WORLD.Barrier()

    p_grads = parallel_net.copyParameterGradToRoot()
    if my_rank==0:
      s_grads = [p.grad for p in list(serial_net.parameters())]

      self.assertTrue(torch.norm(s_loss)>0.0)
      self.assertTrue(torch.norm(p_loss)>0.0)

      for s_grad,p_grad in zip(s_grads,p_grads):
        # check the error conditions
        self.assertTrue(torch.norm(s_grad-p_grad)<=1e-15)
  # end test_linearNet_Exact

if __name__ == '__main__':
  unittest.main()
